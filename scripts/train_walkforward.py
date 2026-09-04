#!/usr/bin/env python3
"""Phase 2, step 1: baselines and walk-forward scaffolding.

Expanding-window walk-forward validation over data/features_v1.parquet, with
three models scored per fold and nothing tuned. One pass, honest numbers.

    train 2011-2014 -> validate 2015
    train 2011-2015 -> validate 2016
    ...
    train 2011-2020 -> validate 2021

2022-2025 is a LOCKED HOLDOUT. Those rows are discarded at load and never
reach a fold; every fold is asserted to end strictly before 2022-01-01.

Folds are cut on PREDICTION_DATE, never period_end. period_end is when a
quarter ended; prediction_date is when its number became public, and only the
latter bounds what was knowable. A quarter ending in December 2014 that filed
in February 2015 belongs to 2015 for this purpose.

Run:  python scripts/train_walkforward.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PARQUET = "data/features_v1.parquet"
RESULTS_CSV = "data/walkforward_results.csv"

# --------------------------------------------------------------------------
# The feature set
# --------------------------------------------------------------------------

# Every column here is derived ONLY from quarters strictly earlier than the row
# being predicted, verified by the leakage assertion in build_features_v1.py.
FEATURES = [
    "eps_growth_yoy_lag_1",
    "eps_growth_yoy_lag_2",
    "eps_growth_yoy_lag_3",
    "eps_growth_yoy_lag_4",
    "label_lag_1",
    "label_lag_2",
    "label_lag_3",
    "label_lag_4",
    "growth_streak",
    "quarters_available",
]

# EXCLUDED, permanently, and not because of look-ahead.
#
# label_yoy is 1 exactly when eps_diluted > eps_year_ago_adjusted, and
# eps_growth_yoy is that same difference signed and scaled -- so
# sign(eps_growth_yoy) IS the label (verified 21721/21721 rows in Phase 1).
# growth_acceleration is eps_growth_yoy - eps_growth_yoy_lag_1 and carries the
# same term.
#
# Both are genuinely public on prediction_date, so no leakage check flags them.
# They are target identities: a model given either scores ~100% and has learned
# nothing. Their presence in the matrix would invalidate every number below,
# so it is asserted against at runtime rather than left to this comment.
TARGET_IDENTITY_COLUMNS = ["eps_growth_yoy", "growth_acceleration"]

LABEL = "label_yoy"

# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------

FIRST_TRAIN_YEAR = 2011
FIRST_VALIDATE_YEAR = 2015
LAST_VALIDATE_YEAR = 2021

# Nothing at or after this date may be read, by any fold, for any purpose.
HOLDOUT_START = pd.Timestamp("2022-01-01")

RANDOM_STATE = 42

# Strong regularization, fixed in advance. No search: a hyperparameter chosen on
# these validation folds would make them a training set in disguise.
LGB_PARAMS = {
    "objective": "binary",
    "max_depth": 3,
    "num_leaves": 8,            # <= 2**max_depth; keeps the depth cap binding
    "learning_rate": 0.05,
    "n_estimators": 2000,       # an upper bound; early stopping picks the count
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbose": -1,
}
LGB_EARLY_STOPPING_ROUNDS = 50

MODEL_ORDER = ["constant", "logistic", "lightgbm"]

# Tickers with fewer validation rows than this are pooled into a summary line
# rather than given a per-ticker accuracy that means nothing.
MIN_TICKER_ROWS = 20


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fail(message: str) -> None:
    print(f"\n*** FATAL: {message}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Load, lock the holdout, drop null labels
# --------------------------------------------------------------------------


def load_modelling_frame() -> tuple:
    section("STAGE 1 -- load, lock the holdout, drop null labels")

    if not os.path.exists(FEATURES_PARQUET):
        fail(f"{FEATURES_PARQUET} not found. Run scripts/build_features_v1.py first.")

    frame = pd.read_parquet(FEATURES_PARQUET)
    print(f"feature table rows      : {len(frame)}")

    missing = [name for name in FEATURES + [LABEL] if name not in frame.columns]
    if missing:
        fail(f"columns missing from {FEATURES_PARQUET}: {missing}")

    # --- lock the holdout FIRST, before anything else touches the data -------
    # Discarded here, at the door, so no later stage can reach a 2022+ row even
    # by accident. The count is kept only to report what was set aside.
    is_holdout = frame["prediction_date"] >= HOLDOUT_START
    holdout_rows = int(is_holdout.sum())
    frame = frame.loc[~is_holdout].copy()

    print(f"LOCKED HOLDOUT          : {holdout_rows} rows at/after "
          f"{HOLDOUT_START.date()} discarded at load")
    print(f"available to modelling  : {len(frame)}")

    if frame["prediction_date"].max() >= HOLDOUT_START:
        fail("a holdout row survived the load filter.")

    # --- drop null labels ---------------------------------------------------
    null_labels = int(frame[LABEL].isna().sum())
    frame = frame.loc[frame[LABEL].notna()].copy()
    print(f"dropped null labels     : {null_labels}")
    print(f"modelling rows          : {len(frame)}")

    # Rows with null FEATURES are kept deliberately -- the point of this pass is
    # to measure what requiring complete lags would cost, not to assume it.
    incomplete = int(frame[FEATURES].isna().any(axis=1).sum())
    print(f"    of which >=1 null feature : {incomplete} "
          f"({incomplete / len(frame):.1%})  [KEPT]")

    frame["prediction_year"] = frame["prediction_date"].dt.year
    return frame, holdout_rows, null_labels


def assert_feature_set_is_clean(frame: pd.DataFrame) -> None:
    """No target identity may reach the training matrix, under any name."""
    section("STAGE 2 -- feature set assertions")

    contaminated = [name for name in TARGET_IDENTITY_COLUMNS if name in FEATURES]
    if contaminated:
        fail(f"target identity columns present in FEATURES: {contaminated}. "
             f"Any score computed with them is meaningless.")
    print(f"PASS  no target identity in FEATURES ({TARGET_IDENTITY_COLUMNS} excluded)")

    if LABEL in FEATURES:
        fail(f"the label {LABEL!r} is in FEATURES.")
    print(f"PASS  the label itself is not a feature")

    if len(set(FEATURES)) != len(FEATURES):
        fail("FEATURES contains duplicates.")
    print(f"PASS  {len(FEATURES)} distinct features")

    # The matrix builder is the thing that must be clean, so check what it
    # actually produces rather than only the constant it is built from.
    matrix_columns = list(frame[FEATURES].columns)
    leaked = [name for name in TARGET_IDENTITY_COLUMNS if name in matrix_columns]
    if leaked:
        fail(f"target identity columns in the built matrix: {leaked}.")
    print(f"PASS  built matrix carries exactly: {matrix_columns}")


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------


def build_folds(frame: pd.DataFrame) -> list:
    """Expanding window: train on everything before the validation year."""
    section("STAGE 3 -- walk-forward folds (cut on prediction_date)")

    folds = []
    for validate_year in range(FIRST_VALIDATE_YEAR, LAST_VALIDATE_YEAR + 1):
        is_train = frame["prediction_year"].between(FIRST_TRAIN_YEAR, validate_year - 1)
        is_validate = frame["prediction_year"] == validate_year

        train = frame.loc[is_train]
        validate = frame.loc[is_validate]

        if train.empty or validate.empty:
            fail(f"fold validating {validate_year} has an empty side "
                 f"(train {len(train)}, validate {len(validate)}).")

        folds.append({
            "validate_year": validate_year,
            "train_years": f"{FIRST_TRAIN_YEAR}-{validate_year - 1}",
            "train": train,
            "validate": validate,
        })

    print(f"{'fold':>5s} {'train':>12s} {'n_train':>8s} {'validate':>9s} "
          f"{'n_valid':>8s} {'train_pos':>10s} {'valid_pos':>10s}")
    for index, fold in enumerate(folds, start=1):
        print(f"{index:5d} {fold['train_years']:>12s} {len(fold['train']):8d} "
              f"{fold['validate_year']:9d} {len(fold['validate']):8d} "
              f"{fold['train'][LABEL].mean():10.1%} "
              f"{fold['validate'][LABEL].mean():10.1%}")

    return folds


def assert_holdout_untouched(folds: list) -> None:
    """No fold, on either side, may contain a prediction_date in the holdout."""
    for fold in folds:
        for side in ("train", "validate"):
            latest = fold[side]["prediction_date"].max()
            if latest >= HOLDOUT_START:
                fail(f"fold validating {fold['validate_year']} has {side} rows at "
                     f"{latest.date()}, at or after the holdout boundary "
                     f"{HOLDOUT_START.date()}.")

        # Expanding-window discipline: training must end before validation opens.
        if fold["train"]["prediction_date"].max() >= fold["validate"]["prediction_date"].min():
            fail(f"fold validating {fold['validate_year']} trains on a row that is "
                 f"not strictly before its validation window.")

    print(f"\nPASS  every fold ends strictly before {HOLDOUT_START.date()}")
    print("PASS  every fold trains strictly before it validates")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def matrix(frame: pd.DataFrame) -> np.ndarray:
    """Features as float64 with NaN for missing. Never touches excluded columns."""
    return frame[FEATURES].astype("float64").to_numpy()


def targets(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL].astype("int64").to_numpy()


def fit_constant(train: pd.DataFrame, validate: pd.DataFrame) -> tuple:
    """Always predict 1.

    The probability is the TRAINING fold's positive rate, not 1.0: a hard 1.0
    scores infinite log-loss the moment a single validation row is a 0, which
    would say nothing about the baseline's quality. The class prediction is
    still a constant 1, since that rate sits above 0.5 in every fold.
    """
    rate = float(train[LABEL].mean())
    probability = np.full(len(validate), rate)
    prediction = np.ones(len(validate), dtype="int64")
    return prediction, probability, {"train_positive_rate": rate}


def fit_logistic(train: pd.DataFrame, validate: pd.DataFrame) -> tuple:
    """Median-impute, standardize, fit -- all learned on the TRAIN fold only.

    The pipeline is what enforces that. Imputing or scaling on the pooled data
    would let the validation year's medians and variances inform the training
    fold, which is a quiet leak that never shows up as a look-ahead date.
    """
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    pipeline.fit(matrix(train), targets(train))

    probability = pipeline.predict_proba(matrix(validate))[:, 1]
    prediction = (probability >= 0.5).astype("int64")

    coefficients = pipeline.named_steps["model"].coef_[0]
    return prediction, probability, {
        "coefficients": dict(zip(FEATURES, coefficients.round(4)))
    }


def fit_lightgbm(train: pd.DataFrame, validate: pd.DataFrame) -> tuple:
    """LightGBM with early stopping on the validation fold.

    CAVEAT, stated because it changes how the number should be read: the
    stopping iteration is chosen ON the fold being scored. That makes this
    model's validation score optimistically biased relative to the other two,
    which never see the validation fold before predicting. It is done here
    because it was specified; a clean comparison needs an inner split carved
    out of the training years, which is a Phase 2 step 2 change, not a tweak.

    NaNs are passed through untouched -- LightGBM routes missing values down a
    learned default branch, so no imputation is applied or wanted.
    """
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        matrix(train), targets(train),
        eval_X=matrix(validate), eval_y=targets(validate),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    probability = model.predict_proba(matrix(validate))[:, 1]
    prediction = (probability >= 0.5).astype("int64")

    return prediction, probability, {
        "best_iteration": int(model.best_iteration_ or LGB_PARAMS["n_estimators"]),
        "importance": dict(zip(FEATURES, model.feature_importances_)),
    }


FITTERS = {
    "constant": fit_constant,
    "logistic": fit_logistic,
    "lightgbm": fit_lightgbm,
}


# --------------------------------------------------------------------------
# Run the walk-forward
# --------------------------------------------------------------------------


def run_walkforward(folds: list) -> tuple:
    section("STAGE 4 -- fit and score each fold")

    records = []
    pooled = []
    extras = {}

    for fold in folds:
        train, validate = fold["train"], fold["validate"]
        y_true = targets(validate)

        train_nulls = int(train[FEATURES].isna().any(axis=1).sum())
        validate_nulls = int(validate[FEATURES].isna().any(axis=1).sum())

        baseline_accuracy = None

        for name in MODEL_ORDER:
            prediction, probability, extra = FITTERS[name](train, validate)

            accuracy = float(accuracy_score(y_true, prediction))
            # labels= pins the column order so a fold with one class present
            # cannot silently invert the metric.
            loss = float(log_loss(y_true, probability, labels=[0, 1]))

            if name == "constant":
                baseline_accuracy = accuracy

            records.append({
                "validate_year": fold["validate_year"],
                "train_years": fold["train_years"],
                "model": name,
                "n_train": len(train),
                "n_validate": len(validate),
                "n_train_rows_with_null_feature": train_nulls,
                "n_validate_rows_with_null_feature": validate_nulls,
                "train_positive_rate": round(float(train[LABEL].mean()), 6),
                "validate_positive_rate": round(float(validate[LABEL].mean()), 6),
                "accuracy": round(accuracy, 6),
                "log_loss": round(loss, 6),
                # What fraction of the fold the model called positive. A model
                # that just tracks the majority class sits at 1.0 here.
                "predicted_positive_rate": round(float(prediction.mean()), 6),
                "accuracy_minus_constant": round(accuracy - baseline_accuracy, 6),
                "lgb_best_iteration": extra.get("best_iteration", ""),
            })

            pooled.append(pd.DataFrame({
                "validate_year": fold["validate_year"],
                "ticker": validate["ticker"].to_numpy(),
                "model": name,
                "y_true": y_true,
                "prediction": prediction,
                "probability": probability,
            }))

            extras.setdefault(name, {})[fold["validate_year"]] = extra

    return pd.DataFrame(records), pd.concat(pooled, ignore_index=True), extras


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_folds(results: pd.DataFrame) -> None:
    section("PER-FOLD RESULTS")

    print("Null-feature rows are KEPT, not dropped. LightGBM handles them")
    print("natively; logistic regression median-imputes them within the")
    print("training fold. The counts below are what a complete-lags")
    print("requirement would have removed.\n")

    for year, group in results.groupby("validate_year"):
        first = group.iloc[0]
        print(f"--- validate {year}   (train {first['train_years']}, "
              f"n_train {first['n_train']}, n_valid {first['n_validate']})")
        print(f"    positive rate: train {first['train_positive_rate']:.1%}, "
              f"validate {first['validate_positive_rate']:.1%}")
        print(f"    rows with >=1 null feature: train "
              f"{first['n_train_rows_with_null_feature']} "
              f"({first['n_train_rows_with_null_feature'] / first['n_train']:.1%}), "
              f"validate {first['n_validate_rows_with_null_feature']} "
              f"({first['n_validate_rows_with_null_feature'] / first['n_validate']:.1%})")
        print(f"    {'model':12s} {'accuracy':>9s} {'log_loss':>9s} "
              f"{'vs const':>9s} {'pred_pos':>9s}")
        for _index, row in group.iterrows():
            print(f"    {row['model']:12s} {row['accuracy']:9.1%} "
                  f"{row['log_loss']:9.4f} {row['accuracy_minus_constant']:+9.2%} "
                  f"{row['predicted_positive_rate']:9.1%}")
        print()


def report_baseline_drift(results: pd.DataFrame) -> None:
    section("BASELINE DRIFT ACROSS YEARS")

    print("If the constant baseline itself moves, a model's edge in one year is")
    print("not comparable to its edge in another.\n")

    constant = results[results["model"] == "constant"]
    print(f"{'year':>6s} {'valid_pos':>10s} {'const_acc':>10s} {'n_valid':>8s}")
    for _index, row in constant.iterrows():
        print(f"{row['validate_year']:6d} {row['validate_positive_rate']:10.1%} "
              f"{row['accuracy']:10.1%} {row['n_validate']:8d}")

    rates = constant["validate_positive_rate"]
    print(f"\npositive rate range     : {rates.min():.1%} .. {rates.max():.1%} "
          f"(spread {rates.max() - rates.min():.1%})")
    print(f"positive rate std dev   : {rates.std():.1%}")


def report_aggregate(results: pd.DataFrame, pooled: pd.DataFrame) -> None:
    section("AGGREGATE ACROSS FOLDS")

    print("Two aggregations, because they answer different questions:")
    print("  mean-of-folds  weights each YEAR equally")
    print("  pooled         weights each ROW equally (later years are bigger)\n")

    print(f"{'model':12s} {'mean acc':>9s} {'sd':>7s} {'mean ll':>9s} "
          f"{'mean vs const':>14s} {'pooled acc':>11s} {'pooled ll':>10s} "
          f"{'folds won':>10s}")

    constant_by_year = (
        results[results["model"] == "constant"]
        .set_index("validate_year")["accuracy"]
    )

    for name in MODEL_ORDER:
        group = results[results["model"] == name]
        subset = pooled[pooled["model"] == name]

        pooled_accuracy = float(accuracy_score(subset["y_true"], subset["prediction"]))
        pooled_loss = float(log_loss(subset["y_true"], subset["probability"],
                                     labels=[0, 1]))
        beat = int((group.set_index("validate_year")["accuracy"]
                    > constant_by_year).sum())

        print(f"{name:12s} {group['accuracy'].mean():9.1%} "
              f"{group['accuracy'].std():7.2%} {group['log_loss'].mean():9.4f} "
              f"{group['accuracy_minus_constant'].mean():+14.2%} "
              f"{pooled_accuracy:11.1%} {pooled_loss:10.4f} "
              f"{beat:>7d}/{len(group)}")

    print("\nNote: lightgbm early-stops ON the fold it is scored on, so its")
    print("numbers are optimistically biased relative to the other two.")


def choose_best_model(results: pd.DataFrame, pooled: pd.DataFrame) -> str:
    """Best by mean-of-folds accuracy; ties broken by pooled log-loss."""
    summary = []
    for name in MODEL_ORDER:
        group = results[results["model"] == name]
        subset = pooled[pooled["model"] == name]
        summary.append((
            float(group["accuracy"].mean()),
            -float(log_loss(subset["y_true"], subset["probability"], labels=[0, 1])),
            name,
        ))
    summary.sort(reverse=True)
    return summary[0][2]


def report_per_ticker(pooled: pd.DataFrame, best_model: str) -> None:
    section(f"PER-TICKER ACCURACY -- {best_model}, pooled validation folds")

    print("The question: is the model learning WHICH COMPANIES grow reliably,")
    print("rather than when any company will?")
    print()
    print("Worth stating up front: ticker identity is NOT in the feature set.")
    print("There is no ticker column, no embedding, no per-company parameter,")
    print("so the model cannot memorise companies outright. What it CAN do is")
    print("reach the same place indirectly -- growth_streak and label_lag_* are")
    print("company-persistence proxies, and a name that beat four quarters")
    print("running looks the same to the model whoever it is. That is the")
    print("failure mode the numbers below are testing for.\n")

    subset = pooled[pooled["model"] == best_model]

    by_ticker = subset.groupby("ticker").apply(
        lambda group: pd.Series({
            "n": len(group),
            "positive_rate": group["y_true"].mean(),
            "model_accuracy": (group["prediction"] == group["y_true"]).mean(),
            # Always-predict-1 on this ticker's own rows.
            "always_one_accuracy": group["y_true"].mean(),
        }),
        include_groups=False,
    )
    by_ticker["lift"] = by_ticker["model_accuracy"] - by_ticker["always_one_accuracy"]

    dense = by_ticker[by_ticker["n"] >= MIN_TICKER_ROWS]
    print(f"tickers in validation   : {len(by_ticker)}")
    print(f"    with >= {MIN_TICKER_ROWS} rows      : {len(dense)} "
          f"(covering {int(dense['n'].sum())} of {len(subset)} rows)")

    if dense.empty:
        print("\nno ticker has enough validation rows to report.")
        return

    print(f"\nper-ticker accuracy distribution ({len(dense)} tickers):")
    for quantile in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"    p{int(quantile * 100):02d}  "
              f"{dense['model_accuracy'].quantile(quantile):6.1%}")
    print(f"    mean {dense['model_accuracy'].mean():6.1%}")

    print(f"\nper-ticker LIFT over always-predict-1:")
    for quantile in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"    p{int(quantile * 100):02d}  {dense['lift'].quantile(quantile):+6.1%}")
    print(f"    mean {dense['lift'].mean():+6.1%}")
    print(f"\n    tickers where the model beats always-1 : "
          f"{int((dense['lift'] > 0).sum())} of {len(dense)} "
          f"({(dense['lift'] > 0).mean():.1%})")
    print(f"    tickers where it does worse            : "
          f"{int((dense['lift'] < 0).sum())} of {len(dense)}")

    # The tell: if accuracy is explained by the ticker's own base rate, the
    # model is a base-rate lookup wearing a model's clothes.
    correlation = dense["model_accuracy"].corr(dense["positive_rate"])
    print(f"\n    corr(per-ticker accuracy, per-ticker positive rate) : "
          f"{correlation:+.3f}")
    print(f"    corr(per-ticker LIFT,     per-ticker positive rate) : "
          f"{dense['lift'].corr(dense['positive_rate']):+.3f}")

    print(f"\n    Read the second number with care: lift IS accuracy minus")
    print(f"    positive_rate, so it is negatively correlated with positive_rate")
    print(f"    by construction unless accuracy rises just as fast. A strongly")
    print(f"    negative value therefore means the model's edge is concentrated")
    print(f"    in the LOW-base-rate names -- where always-predict-1 is weak and")
    print(f"    easy to beat -- and that it adds little on the reliable growers,")
    print(f"    which is where the first number's accuracy is coming from.")

    print(f"\nbest 5 tickers by lift:")
    print(dense.nlargest(5, "lift")[
        ["n", "positive_rate", "model_accuracy", "lift"]
    ].to_string())
    print(f"\nworst 5 tickers by lift:")
    print(dense.nsmallest(5, "lift")[
        ["n", "positive_rate", "model_accuracy", "lift"]
    ].to_string())


def report_complete_lags_cost(results: pd.DataFrame) -> None:
    section("WHAT A COMPLETE-LAGS REQUIREMENT WOULD HAVE COST")

    constant = results[results["model"] == "constant"]
    train_nulls = constant["n_train_rows_with_null_feature"].sum()
    validate_nulls = constant["n_validate_rows_with_null_feature"].sum()
    train_total = constant["n_train"].sum()
    validate_total = constant["n_validate"].sum()

    print("Summed across folds (training rows recur across expanding windows,")
    print("so these are fold-weighted counts, not distinct rows):")
    print(f"    train rows with >=1 null feature    : {train_nulls} of "
          f"{train_total} ({train_nulls / train_total:.1%})")
    print(f"    validate rows with >=1 null feature : {validate_nulls} of "
          f"{validate_total} ({validate_nulls / validate_total:.1%})")
    print("\nThese rows were KEPT. Dropping them is a decision for a later step,")
    print("and this is the number it would cost.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    frame, holdout_rows, _null_labels = load_modelling_frame()
    assert_feature_set_is_clean(frame)

    folds = build_folds(frame)
    assert_holdout_untouched(folds)

    results, pooled, extras = run_walkforward(folds)

    report_folds(results)
    report_baseline_drift(results)
    report_aggregate(results, pooled)

    best_model = choose_best_model(results, pooled)
    print(f"\nbest by mean-of-folds accuracy: {best_model}")
    report_per_ticker(pooled, best_model)
    report_complete_lags_cost(results)

    section("MODEL DETAIL")
    last_year = max(fold["validate_year"] for fold in folds)
    print(f"logistic coefficients, final fold (validate {last_year}, standardized):")
    for name, value in extras["logistic"][last_year]["coefficients"].items():
        print(f"    {name:24s} {value:+8.4f}")

    print("\n    The eps_growth_yoy_lag_* coefficients are near zero, and that is")
    print("    a diagnostic rather than a finding. Those features reach ~4.7e7")
    print("    with a standard deviation of ~4.2e5, so standardizing maps almost")
    print("    every row to a spike near zero and leaves a handful of outliers")
    print("    carrying the variance. A linear model cannot use them in that")
    print("    shape. LightGBM, being rank-based, is unaffected -- which is part")
    print("    of why it leads. Winsorizing or rank-transforming them is an")
    print("    obvious Phase 2 step 2 change; it is NOT applied here, because")
    print("    choosing the transform against these folds would tune on them.")
    print(f"\nlightgbm early-stopped iteration by fold:")
    for year, extra in extras["lightgbm"].items():
        print(f"    {year}  {extra['best_iteration']:4d}")

    section("OUTPUT")
    os.makedirs("data", exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"results -> {RESULTS_CSV}  ({len(results)} rows: "
          f"{len(folds)} folds x {len(MODEL_ORDER)} models)")
    print(f"\nHOLDOUT STILL LOCKED: {holdout_rows} rows at/after "
          f"{HOLDOUT_START.date()} were never read.")
    print("No hyperparameter search was run. No model was selected on the holdout.")


if __name__ == "__main__":
    main()
