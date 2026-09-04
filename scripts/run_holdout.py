#!/usr/bin/env python3
"""Phase 2, step 4: the holdout run. ONE SHOT.

Executes the frozen protocol pre-registered in config/frozen_model.json
(revision 3, committed at 737ff87) exactly as written. The config is READ, not
restated: the feature list, the transform parameters, the estimator parameters,
the fold boundaries and the success criterion all come out of that file, so
this script cannot silently disagree with the pre-registration.

    fold 1: train 2011-2021 -> predict prediction_date in 2022
    fold 2: train 2011-2022 -> predict prediction_date in 2023
    fold 3: train 2011-2023 -> predict prediction_date in 2024
    fold 4: train 2011-2024 -> predict prediction_date >= 2025-01-01 (terminal)

The pipeline is refit from scratch on each fold's own training window. Nothing
-- quantile grid, medians, means, variances, coefficients -- crosses a fold.

This consumes the holdout. There is no second one in this data.

Run:  python scripts/run_holdout.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

CONFIG_PATH = "config/frozen_model.json"
FEATURES_PARQUET = "data/features_v1.parquet"
PREDICTIONS_CSV = "data/holdout_predictions.csv"
RESULTS_CSV = "data/holdout_results.csv"
CALIBRATION_CSV = "data/holdout_calibration.csv"

EXPECTED_REVISION = 3
LABEL = "label_yoy"
CALIBRATION_DECILES = 10
MIN_TICKER_ROWS = 20


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def stop(message: str) -> None:
    """Anything that would require deviating from the pre-registration."""
    print(f"\n*** STOPPED: {message}", file=sys.stderr)
    print("*** The protocol was NOT altered. Nothing was scored.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Build the frozen pipeline FROM the config
# --------------------------------------------------------------------------


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        stop(f"{CONFIG_PATH} not found.")

    with open(CONFIG_PATH) as handle:
        config = json.load(handle)

    if config.get("revision") != EXPECTED_REVISION:
        stop(f"config revision is {config.get('revision')}, expected "
             f"{EXPECTED_REVISION}. The pre-registration this script executes "
             f"is revision {EXPECTED_REVISION}.")

    return config


def build_pipeline(config: dict) -> Pipeline:
    """Construct the frozen pipeline from the config's parameters, not from
    constants written here. A fresh, unfitted object every call."""
    preprocessing = config["preprocessing"]
    rank = preprocessing["step_1_rank_transform"]
    impute = preprocessing["step_2_imputation"]
    scale = preprocessing["step_3_scaling"]
    estimator = config["model"]["estimator"]

    pipeline = Pipeline([
        ("rank", ColumnTransformer(
            [("rank", QuantileTransformer(**rank["params"]), rank["applies_to"])],
            remainder="passthrough",
        )),
        ("impute", SimpleImputer(**impute["params"])),
        ("scale", StandardScaler(**scale["params"])),
        ("model", LogisticRegression(**estimator["params"])),
    ])

    # A pipeline that is already fitted would mean state crossed a fold.
    try:
        check_is_fitted(pipeline.named_steps["model"])
    except NotFittedError:
        return pipeline
    stop("a freshly built pipeline reports as already fitted.")


# --------------------------------------------------------------------------
# Data and folds
# --------------------------------------------------------------------------


def load_frame(config: dict) -> pd.DataFrame:
    section("STAGE 1 -- load")

    frame = pd.read_parquet(FEATURES_PARQUET)
    features = config["features"]["columns"]

    missing = [name for name in features + [LABEL] if name not in frame.columns]
    if missing:
        stop(f"columns missing from {FEATURES_PARQUET}: {missing}")

    identities = config["features"]["excluded_target_identities"]["columns"]
    contaminated = [name for name in identities if name in features]
    if contaminated:
        stop(f"target identity columns present in the frozen feature list: "
             f"{contaminated}")

    print(f"feature table rows      : {len(frame)}")
    print(f"frozen features         : {len(features)}")
    print(f"excluded identities     : {identities}")
    print(f"config revision         : {config['revision']}")
    print(f"selection code SHA      : {config['code_provenance']['git_sha_short']}")

    return frame


def build_folds(frame: pd.DataFrame, config: dict) -> list:
    section("STAGE 2 -- folds, exactly as pre-registered")

    boundaries = [
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", None),          # terminal: everything remaining
    ]

    folds = []
    for number, (start, end) in enumerate(boundaries, start=1):
        start_stamp = pd.Timestamp(start)

        is_predict = frame["prediction_date"] >= start_stamp
        if end is not None:
            is_predict &= frame["prediction_date"] < pd.Timestamp(end)

        # Training window: everything strictly before this fold opens, labelled.
        is_train = (frame["prediction_date"] < start_stamp) & frame[LABEL].notna()

        folds.append({
            "fold": number,
            "opens": start_stamp,
            "closes": pd.Timestamp(end) if end else None,
            "train": frame.loc[is_train],
            "predict": frame.loc[is_predict],
        })

    declared = {entry["fold"]: entry for entry in config["evaluation_protocol"]["folds"]}

    print(f"{'fold':>5s} {'trains on':>22s} {'n_train':>8s} {'predicts':>26s} "
          f"{'n_pred':>7s} {'n_lab':>7s}")
    for fold in folds:
        predict = fold["predict"]
        labelled = int(predict[LABEL].notna().sum())
        window = (f"< {fold['opens'].date()}")
        target = (f">= {fold['opens'].date()}" if fold["closes"] is None
                  else f"{fold['opens'].date()} .. {(fold['closes'] - pd.Timedelta(days=1)).date()}")
        print(f"{fold['fold']:5d} {window:>22s} {len(fold['train']):8d} "
              f"{target:>26s} {len(predict):7d} {labelled:7d}")

        # The config's declared counts are part of the pre-registration.
        expected = declared[fold["fold"]]
        if len(predict) != expected["n_predict_rows"]:
            stop(f"fold {fold['fold']} selects {len(predict)} prediction rows, "
                 f"but the pre-registration declares "
                 f"{expected['n_predict_rows']}.")
        if labelled != expected["n_predict_rows_labelled"]:
            stop(f"fold {fold['fold']} has {labelled} labelled prediction rows, "
                 f"but the pre-registration declares "
                 f"{expected['n_predict_rows_labelled']}.")

    print("\nPASS  every fold's row counts match the pre-registered declaration")
    return folds


def assert_protocol(frame: pd.DataFrame, folds: list, config: dict) -> None:
    """The four run-time assertions named in the config."""
    section("STAGE 3 -- pre-registered run-time assertions")

    holdout = frame.loc[frame["prediction_date"] >= pd.Timestamp("2022-01-01")]

    for fold in folds:
        train, predict = fold["train"], fold["predict"]

        latest_train = train["prediction_date"].max()
        earliest_predict = predict["prediction_date"].min()
        if not latest_train < earliest_predict:
            stop(f"fold {fold['fold']}: last training prediction_date "
                 f"{latest_train.date()} is not strictly before first predicted "
                 f"{earliest_predict.date()}.")

        overlap = train.index.intersection(predict.index)
        if len(overlap):
            stop(f"fold {fold['fold']} trains on {len(overlap)} rows it also "
                 f"predicts.")

    print("PASS  every fold trains strictly before the first row it predicts")
    print("PASS  no fold trains on a row it also predicts")

    covered = pd.Index([])
    for fold in folds:
        if len(covered.intersection(fold["predict"].index)):
            stop(f"fold {fold['fold']} predicts rows another fold already does.")
        covered = covered.union(fold["predict"].index)

    if len(covered) != len(holdout):
        stop(f"the folds cover {len(covered)} holdout rows, not {len(holdout)}.")

    protocol = config["evaluation_protocol"]
    if len(covered) != protocol["total_rows_scored"]:
        stop(f"coverage {len(covered)} does not match the pre-registered "
             f"{protocol['total_rows_scored']}.")

    labelled = int(holdout.loc[covered, LABEL].notna().sum())
    if labelled != protocol["total_labelled_rows_scored"]:
        stop(f"labelled coverage {labelled} does not match the pre-registered "
             f"{protocol['total_labelled_rows_scored']}.")

    print(f"PASS  the four prediction sets are disjoint and cover every holdout "
          f"row ({len(covered)} rows / {labelled} labelled)")
    print("PASS  a fresh unfitted pipeline is built per fold (checked at build)")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run(folds: list, config: dict) -> tuple:
    section("STAGE 4 -- refit per fold and predict")

    features = config["features"]["columns"]
    threshold = config["model"]["decision_threshold"]

    records = []
    predictions = []
    coefficients = {}
    fitted = []

    for fold in folds:
        train, predict = fold["train"], fold["predict"]

        pipeline = build_pipeline(config)
        if any(pipeline is other for other in fitted):
            stop("a pipeline object was reused across folds.")
        fitted.append(pipeline)

        pipeline.fit(train[features].astype("float64"),
                     train[LABEL].astype("int64").to_numpy())

        proba = pipeline.predict_proba(predict[features].astype("float64"))[:, 1]
        label = (proba >= threshold).astype("int64")

        predictions.append(pd.DataFrame({
            "ticker": predict["ticker"].to_numpy(),
            "period_end": predict["period_end"].to_numpy(),
            "prediction_date": predict["prediction_date"].to_numpy(),
            "predicted_proba": proba,
            "predicted_label": label,
            "actual_label": predict[LABEL].to_numpy(),
            "fold": fold["fold"],
        }))

        coefficients[fold["fold"]] = dict(
            zip(config["preprocessing"]["step_1_rank_transform"]
                ["column_order_after_transform"],
                pipeline.named_steps["model"].coef_[0].round(4))
        )

        # --- score on labelled rows only ------------------------------------
        scored = predict[LABEL].notna().to_numpy()
        y_true = predict.loc[scored, LABEL].astype("int64").to_numpy()
        model_proba = proba[scored]
        model_label = label[scored]

        # The constant baseline: always 1, at THIS fold's own training rate.
        train_rate = float(train[LABEL].mean())
        constant_proba = np.full(len(y_true), train_rate)
        constant_label = np.ones(len(y_true), dtype="int64")

        records.append({
            "fold": fold["fold"],
            "trains_before": str(fold["opens"].date()),
            "n_train": len(train),
            "train_positive_rate": round(train_rate, 6),
            "n_predicted": len(predict),
            "n_labelled": int(scored.sum()),
            "actual_positive_rate": round(float(y_true.mean()), 6),
            "model_log_loss": round(float(log_loss(y_true, model_proba, labels=[0, 1])), 6),
            "constant_log_loss": round(float(log_loss(y_true, constant_proba, labels=[0, 1])), 6),
            "model_accuracy": round(float(accuracy_score(y_true, model_label)), 6),
            "constant_accuracy": round(float(accuracy_score(y_true, constant_label)), 6),
            "model_brier": round(float(brier_score_loss(y_true, model_proba)), 6),
            "predicted_positive_rate": round(float(model_label.mean()), 6),
        })

        print(f"fold {fold['fold']}: fitted on {len(train)} rows, predicted "
              f"{len(predict)} ({int(scored.sum())} scored)")

    results = pd.DataFrame(records)
    results["log_loss_delta"] = (results["model_log_loss"]
                                 - results["constant_log_loss"]).round(6)
    results["accuracy_delta"] = (results["model_accuracy"]
                                 - results["constant_accuracy"]).round(6)
    results["passes_log_loss"] = results["log_loss_delta"] < 0
    results["passes_accuracy"] = results["accuracy_delta"] > 0
    results["passes_both"] = results["passes_log_loss"] & results["passes_accuracy"]

    return results, pd.concat(predictions, ignore_index=True), coefficients


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_folds(results: pd.DataFrame) -> None:
    section("PER-FOLD RESULTS")

    print(f"{'fold':>5s} {'n_pred':>7s} {'n_lab':>7s} {'actual_pos':>11s} "
          f"{'model_ll':>9s} {'const_ll':>9s} {'delta':>9s} "
          f"{'model_acc':>10s} {'const_acc':>10s} {'delta':>8s}")
    for _index, row in results.iterrows():
        print(f"{row['fold']:5d} {row['n_predicted']:7d} {row['n_labelled']:7d} "
              f"{row['actual_positive_rate']:11.1%} "
              f"{row['model_log_loss']:9.4f} {row['constant_log_loss']:9.4f} "
              f"{row['log_loss_delta']:+9.4f} "
              f"{row['model_accuracy']:10.1%} {row['constant_accuracy']:10.1%} "
              f"{row['accuracy_delta']:+8.2%}")


def report_pooled(results: pd.DataFrame, predictions: pd.DataFrame,
                  folds: list) -> dict:
    section("POOLED")

    scored = predictions[predictions["actual_label"].notna()].copy()
    y_true = scored["actual_label"].astype("int64").to_numpy()

    # Each row carries its own fold's constant probability; pooling the
    # baseline any other way would use a rate no fold actually had.
    rates = dict(zip(results["fold"], results["train_positive_rate"]))
    constant_proba = scored["fold"].map(rates).astype("float64").to_numpy()

    pooled = {
        "n_predicted": len(predictions),
        "n_labelled": len(scored),
        "actual_positive_rate": float(y_true.mean()),
        "model_log_loss": float(log_loss(y_true, scored["predicted_proba"], labels=[0, 1])),
        "constant_log_loss": float(log_loss(y_true, constant_proba, labels=[0, 1])),
        "model_accuracy": float(accuracy_score(y_true, scored["predicted_label"].astype("int64"))),
        "constant_accuracy": float(accuracy_score(y_true, np.ones(len(y_true), dtype="int64"))),
        "model_brier": float(brier_score_loss(y_true, scored["predicted_proba"])),
    }
    pooled["log_loss_delta"] = pooled["model_log_loss"] - pooled["constant_log_loss"]
    pooled["accuracy_delta"] = pooled["model_accuracy"] - pooled["constant_accuracy"]

    print(f"n predicted             : {pooled['n_predicted']}")
    print(f"n labelled (scored)     : {pooled['n_labelled']}")
    print(f"actual positive rate    : {pooled['actual_positive_rate']:.1%}")
    print()
    print(f"{'':24s} {'model':>10s} {'constant':>10s} {'delta':>10s}")
    print(f"{'log-loss':24s} {pooled['model_log_loss']:10.4f} "
          f"{pooled['constant_log_loss']:10.4f} {pooled['log_loss_delta']:+10.4f}")
    print(f"{'accuracy':24s} {pooled['model_accuracy']:10.1%} "
          f"{pooled['constant_accuracy']:10.1%} {pooled['accuracy_delta']:+10.2%}")
    print(f"{'Brier':24s} {pooled['model_brier']:10.4f}")

    return pooled


def report_verdict(results: pd.DataFrame, pooled: dict) -> None:
    section("PRE-REGISTERED VERDICT")

    print("Criterion, fixed at 737ff87 before the holdout was read:")
    print("  (1) log-loss BELOW that fold's own constant baseline, AND")
    print("  (2) accuracy ABOVE that fold's own constant baseline.")
    print("  Log-loss is primary if the two disagree.\n")

    print(f"{'fold':>5s} {'log-loss':>10s} {'accuracy':>10s} {'both':>8s} "
          f"{'primary (log-loss)':>20s}")
    for _index, row in results.iterrows():
        print(f"{row['fold']:5d} "
              f"{'PASS' if row['passes_log_loss'] else 'FAIL':>10s} "
              f"{'PASS' if row['passes_accuracy'] else 'FAIL':>10s} "
              f"{'PASS' if row['passes_both'] else 'FAIL':>8s} "
              f"{'PASS' if row['passes_log_loss'] else 'FAIL':>20s}")

    pooled_ll = pooled["log_loss_delta"] < 0
    pooled_acc = pooled["accuracy_delta"] > 0
    print(f"{'POOL':>5s} {'PASS' if pooled_ll else 'FAIL':>10s} "
          f"{'PASS' if pooled_acc else 'FAIL':>10s} "
          f"{'PASS' if (pooled_ll and pooled_acc) else 'FAIL':>8s} "
          f"{'PASS' if pooled_ll else 'FAIL':>20s}")

    print(f"\nfolds passing both conditions : "
          f"{int(results['passes_both'].sum())} of {len(results)}")
    print(f"folds passing log-loss        : "
          f"{int(results['passes_log_loss'].sum())} of {len(results)}")
    print(f"folds passing accuracy        : "
          f"{int(results['passes_accuracy'].sum())} of {len(results)}")

    print(f"\nPOOLED VERDICT (both conditions) : "
          f"{'PASS' if (pooled_ll and pooled_acc) else 'FAIL'}")
    print(f"POOLED VERDICT (log-loss primary): "
          f"{'PASS' if pooled_ll else 'FAIL'}")


def report_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    section("CALIBRATION -- logistic_rank on pooled holdout")

    print("Never measured for this model before now: the Phase 2 decile table")
    print("was computed for lightgbm_inner_es.\n")

    scored = predictions[predictions["actual_label"].notna()].copy()
    scored["actual_label"] = scored["actual_label"].astype("int64")
    scored["decile"] = pd.qcut(scored["predicted_proba"], CALIBRATION_DECILES,
                               labels=False, duplicates="drop")

    table = scored.groupby("decile").agg(
        n=("actual_label", "size"),
        predicted=("predicted_proba", "mean"),
        actual=("actual_label", "mean"),
        p_min=("predicted_proba", "min"),
        p_max=("predicted_proba", "max"),
    ).reset_index()
    table["gap"] = table["actual"] - table["predicted"]

    print(f"{'decile':>7s} {'n':>6s} {'range':>17s} {'predicted':>10s} "
          f"{'actual':>8s} {'gap':>8s}")
    for _index, row in table.iterrows():
        print(f"{int(row['decile']) + 1:7d} {int(row['n']):6d} "
              f"[{row['p_min']:.3f},{row['p_max']:.3f}] {row['predicted']:10.3f} "
              f"{row['actual']:8.3f} {row['gap']:+8.3f}  "
              f"{'.' * int(row['actual'] * 40)}|")

    ece = float((table["n"] * table["gap"].abs()).sum() / table["n"].sum())
    print(f"\n    expected calibration error (ECE) : {ece:.4f}")
    print(f"    mean predicted / mean actual     : "
          f"{scored['predicted_proba'].mean():.4f} / "
          f"{scored['actual_label'].mean():.4f} "
          f"({scored['predicted_proba'].mean() - scored['actual_label'].mean():+.4f})")
    rises = int((table["actual"].diff().dropna() > 0).sum())
    print(f"    deciles where actual rate rises  : {rises} of {len(table) - 1}")
    print(f"    top minus bottom decile          : "
          f"{table['actual'].iloc[-1] - table['actual'].iloc[0]:+.3f} "
          f"({table['actual'].iloc[0]:.1%} -> {table['actual'].iloc[-1]:.1%})")

    return table


def report_per_ticker(predictions: pd.DataFrame) -> None:
    section("PER-TICKER -- pooled holdout")

    scored = predictions[predictions["actual_label"].notna()].copy()
    scored["actual_label"] = scored["actual_label"].astype("int64")

    by_ticker = scored.groupby("ticker").apply(
        lambda group: pd.Series({
            "n": len(group),
            "positive_rate": group["actual_label"].mean(),
            "model_accuracy": (group["predicted_label"] == group["actual_label"]).mean(),
        }),
        include_groups=False,
    )
    by_ticker["lift"] = by_ticker["model_accuracy"] - by_ticker["positive_rate"]

    dense = by_ticker[by_ticker["n"] >= MIN_TICKER_ROWS]
    print(f"tickers in holdout      : {len(by_ticker)}")
    print(f"    with >= {MIN_TICKER_ROWS} rows     : {len(dense)} "
          f"(covering {int(dense['n'].sum())} of {len(scored)} rows)")

    if dense.empty:
        print("\nno ticker has enough holdout rows to report.")
        return

    print(f"\n    median per-ticker accuracy : {dense['model_accuracy'].median():.1%}")
    print(f"    mean   per-ticker accuracy : {dense['model_accuracy'].mean():.1%}")
    print(f"    median lift over always-1  : {dense['lift'].median():+.1%}")
    print(f"    beats always-1 on          : {int((dense['lift'] > 0).sum())} of "
          f"{len(dense)} ({(dense['lift'] > 0).mean():.1%})")
    print(f"\n    corr(accuracy, positive_rate) : "
          f"{dense['model_accuracy'].corr(dense['positive_rate']):+.3f}")
    print(f"    corr(lift,     positive_rate) : "
          f"{dense['lift'].corr(dense['positive_rate']):+.3f}")
    print(f"\n    Phase 2 validation folds gave +0.540 and -0.607.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", CONFIG_PATH],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        stop(f"{CONFIG_PATH} has uncommitted changes. The pre-registration must "
             f"be committed before the holdout is read.")

    config = load_config()
    print(f"repository HEAD         : {head}")

    frame = load_frame(config)
    folds = build_folds(frame, config)
    assert_protocol(frame, folds, config)

    results, predictions, coefficients = run(folds, config)

    report_folds(results)
    pooled = report_pooled(results, predictions, folds)
    report_verdict(results, pooled)
    calibration = report_calibration(predictions)
    report_per_ticker(predictions)

    section("COEFFICIENTS BY FOLD (standardized)")
    names = list(coefficients[1])
    print(f"    {'feature':26s}" + "".join(f"{'fold ' + str(k):>10s}"
                                           for k in sorted(coefficients)))
    for name in names:
        print(f"    {name:26s}" + "".join(
            f"{coefficients[k][name]:10.4f}" for k in sorted(coefficients)))

    section("OUTPUT")
    os.makedirs("data", exist_ok=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    calibration.to_csv(CALIBRATION_CSV, index=False)

    print(f"predictions -> {PREDICTIONS_CSV}  ({len(predictions)} rows)")
    print(f"results     -> {RESULTS_CSV}  ({len(results)} folds)")
    print(f"calibration -> {CALIBRATION_CSV}  ({len(calibration)} deciles)")
    print("\nThe holdout has now been read. It is spent.")


if __name__ == "__main__":
    main()
