#!/usr/bin/env python3
"""Phase 2: baselines and walk-forward validation.

Expanding-window walk-forward over data/features_v1.parquet. No tuning: every
hyperparameter is fixed in advance and nothing is chosen against fold results.

    train 2011-2014 -> validate 2015
    train 2011-2015 -> validate 2016
    ...
    train 2011-2020 -> validate 2021

2022-2025 is a LOCKED HOLDOUT. Those rows are discarded at load and never
reach a fold; every fold is asserted to end strictly before 2022-01-01.

Folds are cut on PREDICTION_DATE, never period_end. period_end is when a
quarter ended; prediction_date is when its number became public, and only the
latter bounds what was knowable.

STEP 2 -- TWO STRUCTURAL CORRECTIONS
------------------------------------
Both fix a way the step-1 numbers were biased. Neither is a choice made
against fold results, and neither is a hyperparameter:

1. INNER VALIDATION SPLIT. In step 1, LightGBM early-stopped on the same fold
   it was scored on, so the stopping iteration was chosen using the answer.
   Now the LAST year of each training window is held out as an inner
   validation set, the model trains on the years before it, and the outer fold
   stays untouched until scoring. This is a correction to the protocol; it
   would be required whatever the numbers had come out as.

2. RANK TRANSFORM on eps_growth_yoy_lag_*. Those features reach ~4.7e7 with a
   standard deviation of ~4.2e5, so standardizing collapsed almost every row
   to a spike near zero and left a linear model unable to use them. Ranking is
   fitted on the training window ONLY and applied to the validation fold, so
   the validation distribution never informs the transform. Applied to the
   logistic pipeline alone -- a rank transform is strictly monotone, and trees
   split on thresholds, so it is a no-op for LightGBM except for the
   information a quantile grid would throw away.

Both the biased and the corrected variants are run, so the size of each bias
is visible rather than asserted.

Run:  python scripts/train_walkforward.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

FEATURES_PARQUET = "data/features_v1.parquet"
RESULTS_CSV = "data/walkforward_results.csv"
CALIBRATION_CSV = "data/walkforward_calibration.csv"

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

# The heavy-tailed ones. Rank-transformed for the linear model; see fix 2.
RANK_FEATURES = [
    "eps_growth_yoy_lag_1",
    "eps_growth_yoy_lag_2",
    "eps_growth_yoy_lag_3",
    "eps_growth_yoy_lag_4",
]
PASSTHROUGH_FEATURES = [name for name in FEATURES if name not in RANK_FEATURES]

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
# nothing. Asserted against at runtime rather than left to this comment.
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

# Quantile grid for the rank transform. Fixed, not tuned; large enough that the
# grid is never the binding constraint on the smallest training window (5,400).
RANK_N_QUANTILES = 1000

# Variants. The *_biased entries are diagnostics, kept only so the size of each
# correction is measurable. They are excluded from the freeze decision.
CONSTANT = "constant"
LOGISTIC_BIASED = "logistic_raw"
LOGISTIC = "logistic_rank"
LGBM_BIASED = "lightgbm_outer_es"
LGBM = "lightgbm_inner_es"

MODEL_ORDER = [CONSTANT, LOGISTIC_BIASED, LOGISTIC, LGBM_BIASED, LGBM]

# What step 1 reported, and what a freeze decision may consider.
BIASED_VARIANTS = [LOGISTIC_BIASED, LGBM_BIASED]
CORRECTED_VARIANTS = [CONSTANT, LOGISTIC, LGBM]

BEFORE_AFTER = [
    ("Fix 1 -- early stopping", LGBM_BIASED, LGBM),
    ("Fix 2 -- rank transform", LOGISTIC_BIASED, LOGISTIC),
]

CALIBRATION_DECILES = 10

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
    is_holdout = frame["prediction_date"] >= HOLDOUT_START
    holdout_rows = int(is_holdout.sum())
    frame = frame.loc[~is_holdout].copy()

    print(f"LOCKED HOLDOUT          : {holdout_rows} rows at/after "
          f"{HOLDOUT_START.date()} discarded at load")
    print(f"available to modelling  : {len(frame)}")

    if frame["prediction_date"].max() >= HOLDOUT_START:
        fail("a holdout row survived the load filter.")

    null_labels = int(frame[LABEL].isna().sum())
    frame = frame.loc[frame[LABEL].notna()].copy()
    print(f"dropped null labels     : {null_labels}")
    print(f"modelling rows          : {len(frame)}")

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
    print("PASS  the label itself is not a feature")

    if len(set(FEATURES)) != len(FEATURES):
        fail("FEATURES contains duplicates.")
    print(f"PASS  {len(FEATURES)} distinct features")

    matrix_columns = list(design(frame).columns)
    leaked = [name for name in TARGET_IDENTITY_COLUMNS if name in matrix_columns]
    if leaked:
        fail(f"target identity columns in the built matrix: {leaked}.")
    print(f"PASS  built matrix carries exactly: {matrix_columns}")

    if set(RANK_FEATURES) - set(FEATURES):
        fail("RANK_FEATURES names a column that is not a feature.")
    print(f"PASS  rank transform targets only: {RANK_FEATURES}")


# --------------------------------------------------------------------------
# Folds, and the inner split that fixes early stopping
# --------------------------------------------------------------------------


def inner_split(train: pd.DataFrame) -> tuple:
    """Hold out the LAST year of the training window for early stopping.

    The point of fix 1: the stopping iteration has to be chosen on data the
    outer fold has not seen and that the scoring never touches. The last
    training year is the natural choice -- it is the closest in time to the
    outer fold, so it is the most representative of what the model will face.
    """
    inner_validate_year = int(train["prediction_year"].max())

    inner_train = train.loc[train["prediction_year"] < inner_validate_year]
    inner_validate = train.loc[train["prediction_year"] == inner_validate_year]

    return inner_train, inner_validate, inner_validate_year


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

        inner_train, inner_validate, inner_year = inner_split(train)
        if inner_train.empty or inner_validate.empty:
            fail(f"fold validating {validate_year} cannot form an inner split.")

        folds.append({
            "validate_year": validate_year,
            "train_years": f"{FIRST_TRAIN_YEAR}-{validate_year - 1}",
            "train": train,
            "validate": validate,
            "inner_train": inner_train,
            "inner_validate": inner_validate,
            "inner_validate_year": inner_year,
        })

    print("The inner split is fix 1: LightGBM early-stops on inner_valid, which")
    print("is carved out of the TRAINING window, so the outer fold stays")
    print("untouched until it is scored.\n")

    print(f"{'fold':>5s} {'train':>12s} {'n_train':>8s} {'inner_tr':>9s} "
          f"{'inner_val':>10s} {'n_inner_v':>10s} {'validate':>9s} {'n_valid':>8s}")
    for index, fold in enumerate(folds, start=1):
        print(f"{index:5d} {fold['train_years']:>12s} {len(fold['train']):8d} "
              f"{len(fold['inner_train']):9d} {fold['inner_validate_year']:10d} "
              f"{len(fold['inner_validate']):10d} {fold['validate_year']:9d} "
              f"{len(fold['validate']):8d}")

    return folds


def assert_split_discipline(folds: list) -> None:
    """The holdout, the expanding window, and the inner split, all checked."""
    for fold in folds:
        for side in ("train", "validate", "inner_train", "inner_validate"):
            latest = fold[side]["prediction_date"].max()
            if latest >= HOLDOUT_START:
                fail(f"fold validating {fold['validate_year']} has {side} rows at "
                     f"{latest.date()}, at or after the holdout boundary "
                     f"{HOLDOUT_START.date()}.")

        if fold["train"]["prediction_date"].max() >= fold["validate"]["prediction_date"].min():
            fail(f"fold validating {fold['validate_year']} trains on a row that is "
                 f"not strictly before its validation window.")

        # Fix 1's whole point: the early-stopping set must sit strictly inside
        # the training window and strictly before the scored fold.
        if fold["inner_train"]["prediction_date"].max() >= fold["inner_validate"]["prediction_date"].min():
            fail(f"fold validating {fold['validate_year']} has an inner train set "
                 f"that is not strictly before its inner validation set.")

        if fold["inner_validate"]["prediction_date"].max() >= fold["validate"]["prediction_date"].min():
            fail(f"fold validating {fold['validate_year']} early-stops on rows that "
                 f"are not strictly before the fold it is scored on.")

        pooled = len(fold["inner_train"]) + len(fold["inner_validate"])
        if pooled != len(fold["train"]):
            fail(f"fold validating {fold['validate_year']}: inner split loses rows "
                 f"({pooled} vs {len(fold['train'])}).")

    print(f"\nPASS  every fold ends strictly before {HOLDOUT_START.date()}")
    print("PASS  every fold trains strictly before it validates")
    print("PASS  every inner split is strictly inside its training window")
    print("PASS  early stopping never sees the fold it is scored on")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """Features as float64 with NaN for missing, names preserved."""
    return frame[FEATURES].astype("float64")


def targets(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL].astype("int64").to_numpy()


def logistic_pipeline(rank_transform: bool) -> tuple:
    """Median-impute, standardize, fit -- optionally rank-transform first.

    Every step is inside one Pipeline, which is what guarantees fix 2's
    requirement: fit() sees only the training window, so the quantile grid, the
    medians and the variances are all learned there and merely APPLIED to the
    validation fold. Fitting any of them on pooled data would leak the
    validation distribution backwards without ever looking like a look-ahead.
    """
    steps = []
    if rank_transform:
        # QuantileTransformer ignores NaN when fitting and preserves it, so it
        # composes with the imputer that follows.
        steps.append(("rank", ColumnTransformer(
            [("rank", QuantileTransformer(
                output_distribution="normal",
                n_quantiles=RANK_N_QUANTILES,
                subsample=None,
                random_state=RANDOM_STATE,
            ), RANK_FEATURES)],
            remainder="passthrough",
        )))
        order = RANK_FEATURES + PASSTHROUGH_FEATURES
    else:
        order = list(FEATURES)

    steps += [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
    return Pipeline(steps), order


def fit_constant(fold: dict) -> tuple:
    """Always predict 1.

    The probability is the TRAINING fold's positive rate, not 1.0: a hard 1.0
    scores infinite log-loss the moment a single validation row is a 0, which
    would say nothing about the baseline's quality. The class prediction is
    still a constant 1, since that rate sits above 0.5 in every fold.
    """
    rate = float(fold["train"][LABEL].mean())
    probability = np.full(len(fold["validate"]), rate)
    prediction = np.ones(len(fold["validate"]), dtype="int64")
    return prediction, probability, {"train_positive_rate": rate}


def fit_logistic(fold: dict, rank_transform: bool) -> tuple:
    pipeline, order = logistic_pipeline(rank_transform)
    pipeline.fit(design(fold["train"]), targets(fold["train"]))

    probability = pipeline.predict_proba(design(fold["validate"]))[:, 1]
    prediction = (probability >= 0.5).astype("int64")

    coefficients = pipeline.named_steps["model"].coef_[0]
    return prediction, probability, {
        "coefficients": dict(zip(order, coefficients.round(4)))
    }


def fit_lightgbm(fold: dict, inner_early_stopping: bool) -> tuple:
    """LightGBM, early-stopping either honestly (inner) or not (outer).

    inner_early_stopping=True is the corrected path: train on inner_train, stop
    on inner_validate, never touch the outer fold until scoring.

    inner_early_stopping=False reproduces the step-1 bias on purpose -- it
    trains on the whole window and stops on the fold it is about to be scored
    on. It is kept only to measure how much that was worth, and is excluded
    from the freeze decision.

    NaNs are passed through untouched -- LightGBM routes missing values down a
    learned default branch, so no imputation is applied or wanted. The rank
    transform is deliberately NOT applied: it is strictly monotone and trees
    split on thresholds, so it could only subtract information via the quantile
    grid.
    """
    if inner_early_stopping:
        train = fold["inner_train"]
        stop_on = fold["inner_validate"]
    else:
        train = fold["train"]
        stop_on = fold["validate"]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        design(train), targets(train),
        eval_X=design(stop_on), eval_y=targets(stop_on),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    probability = model.predict_proba(design(fold["validate"]))[:, 1]
    prediction = (probability >= 0.5).astype("int64")

    return prediction, probability, {
        "best_iteration": int(model.best_iteration_ or LGB_PARAMS["n_estimators"]),
        "n_train_used": len(train),
        "importance": dict(zip(FEATURES, model.feature_importances_)),
    }


FITTERS = {
    CONSTANT: fit_constant,
    LOGISTIC_BIASED: lambda fold: fit_logistic(fold, rank_transform=False),
    LOGISTIC: lambda fold: fit_logistic(fold, rank_transform=True),
    LGBM_BIASED: lambda fold: fit_lightgbm(fold, inner_early_stopping=False),
    LGBM: lambda fold: fit_lightgbm(fold, inner_early_stopping=True),
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
            prediction, probability, extra = FITTERS[name](fold)

            accuracy = float(accuracy_score(y_true, prediction))
            # labels= pins the column order so a fold with one class present
            # cannot silently invert the metric.
            loss = float(log_loss(y_true, probability, labels=[0, 1]))

            if name == CONSTANT:
                baseline_accuracy = accuracy

            records.append({
                "validate_year": fold["validate_year"],
                "train_years": fold["train_years"],
                "inner_validate_year": fold["inner_validate_year"],
                "model": name,
                "corrected": name in CORRECTED_VARIANTS,
                "n_train": len(train),
                "n_train_used": extra.get("n_train_used", len(train)),
                "n_validate": len(validate),
                "n_train_rows_with_null_feature": train_nulls,
                "n_validate_rows_with_null_feature": validate_nulls,
                "train_positive_rate": round(float(train[LABEL].mean()), 6),
                "validate_positive_rate": round(float(validate[LABEL].mean()), 6),
                "accuracy": round(accuracy, 6),
                "log_loss": round(loss, 6),
                "brier": round(float(brier_score_loss(y_true, probability)), 6),
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
    section("PER-FOLD RESULTS -- all variants")

    print("Null-feature rows are KEPT, not dropped. LightGBM handles them")
    print("natively; logistic regression median-imputes them within the")
    print("training fold.\n")

    for year, group in results.groupby("validate_year"):
        first = group.iloc[0]
        print(f"--- validate {year}   (train {first['train_years']}, "
              f"n_train {first['n_train']}, inner_val {first['inner_validate_year']}, "
              f"n_valid {first['n_validate']})")
        print(f"    positive rate: train {first['train_positive_rate']:.1%}, "
              f"validate {first['validate_positive_rate']:.1%}")
        print(f"    {'model':20s} {'log_loss':>9s} {'accuracy':>9s} "
              f"{'brier':>8s} {'vs const':>9s}")
        for _index, row in group.iterrows():
            mark = "   " if row["corrected"] else " * "
            print(f"{mark} {row['model']:20s} {row['log_loss']:9.4f} "
                  f"{row['accuracy']:9.1%} {row['brier']:8.4f} "
                  f"{row['accuracy_minus_constant']:+9.2%}")
        print()

    print("* = biased diagnostic variant, excluded from the freeze decision.")


def report_before_after(results: pd.DataFrame, extras: dict) -> None:
    section("BEFORE / AFTER EACH CORRECTION")

    for title, before, after in BEFORE_AFTER:
        print(f"\n{title}:  {before}  ->  {after}")
        print(f"    {'year':>6s} {'ll before':>10s} {'ll after':>10s} "
              f"{'delta':>8s} {'acc before':>11s} {'acc after':>10s} {'delta':>8s}")

        left = results[results["model"] == before].set_index("validate_year")
        right = results[results["model"] == after].set_index("validate_year")

        for year in left.index:
            loss_delta = right.loc[year, "log_loss"] - left.loc[year, "log_loss"]
            accuracy_delta = right.loc[year, "accuracy"] - left.loc[year, "accuracy"]
            print(f"    {year:6d} {left.loc[year, 'log_loss']:10.4f} "
                  f"{right.loc[year, 'log_loss']:10.4f} {loss_delta:+8.4f} "
                  f"{left.loc[year, 'accuracy']:11.1%} "
                  f"{right.loc[year, 'accuracy']:10.1%} {accuracy_delta:+8.2%}")

        loss_before, loss_after = left["log_loss"].mean(), right["log_loss"].mean()
        accuracy_before = left["accuracy"].mean()
        accuracy_after = right["accuracy"].mean()
        print(f"    {'mean':>6s} {loss_before:10.4f} {loss_after:10.4f} "
              f"{loss_after - loss_before:+8.4f} {accuracy_before:11.1%} "
              f"{accuracy_after:10.1%} {accuracy_after - accuracy_before:+8.2%}")

    # Fix 1 costs training rows as well as removing the bias; both matter.
    print(f"\nFix 1 also shrinks the training set (the last year becomes the")
    print(f"early-stopping set). Rows actually trained on:")
    print(f"    {'year':>6s} {'before':>8s} {'after':>8s} "
          f"{'iters before':>13s} {'iters after':>12s}")
    biased = results[results["model"] == LGBM_BIASED].set_index("validate_year")
    fixed = results[results["model"] == LGBM].set_index("validate_year")
    for year in biased.index:
        print(f"    {year:6d} {biased.loc[year, 'n_train_used']:8d} "
              f"{fixed.loc[year, 'n_train_used']:8d} "
              f"{biased.loc[year, 'lgb_best_iteration']:13} "
              f"{fixed.loc[year, 'lgb_best_iteration']:12}")


def report_coefficients(extras: dict, folds: list) -> None:
    section("FIX 2 -- LOGISTIC COEFFICIENTS BEFORE / AFTER THE RANK TRANSFORM")

    print("Standardized coefficients, averaged over the seven folds. The")
    print("question is whether eps_growth_yoy_lag_* come alive once the heavy")
    print("tail is removed.\n")

    years = [fold["validate_year"] for fold in folds]

    def mean_coefficients(model_name):
        frames = [pd.Series(extras[model_name][year]["coefficients"]) for year in years]
        return pd.concat(frames, axis=1).mean(axis=1)

    before = mean_coefficients(LOGISTIC_BIASED)
    after = mean_coefficients(LOGISTIC)

    print(f"    {'feature':26s} {'before':>9s} {'after':>9s} {'|after|/|before|':>18s}")
    for name in FEATURES:
        ratio = (abs(after[name]) / abs(before[name])
                 if abs(before[name]) > 1e-12 else float("inf"))
        flag = "  <-- rank-transformed" if name in RANK_FEATURES else ""
        print(f"    {name:26s} {before[name]:+9.4f} {after[name]:+9.4f} "
              f"{ratio:18.1f}{flag}")

    rank_before = before[RANK_FEATURES].abs().mean()
    rank_after = after[RANK_FEATURES].abs().mean()
    other_before = before[PASSTHROUGH_FEATURES].abs().mean()
    other_after = after[PASSTHROUGH_FEATURES].abs().mean()

    print(f"\n    mean |coef| on rank-transformed features : "
          f"{rank_before:.4f} -> {rank_after:.4f}  "
          f"({rank_after / rank_before:.1f}x)")
    print(f"    mean |coef| on the untouched features    : "
          f"{other_before:.4f} -> {other_after:.4f}  "
          f"({other_after / other_before:.1f}x)")


def report_aggregate(results: pd.DataFrame, pooled: pd.DataFrame) -> None:
    section("AGGREGATE ACROSS FOLDS")

    print("mean-of-folds weights each YEAR equally; pooled weights each ROW.\n")
    print(f"{'model':20s} {'mean ll':>9s} {'sd':>7s} {'pooled ll':>10s} "
          f"{'mean acc':>9s} {'pooled acc':>11s} {'mean brier':>11s}")

    for name in MODEL_ORDER:
        group = results[results["model"] == name]
        subset = pooled[pooled["model"] == name]
        pooled_loss = float(log_loss(subset["y_true"], subset["probability"],
                                     labels=[0, 1]))
        pooled_accuracy = float(accuracy_score(subset["y_true"], subset["prediction"]))
        mark = "   " if name in CORRECTED_VARIANTS else " * "
        print(f"{mark}{name:20s} {group['log_loss'].mean():9.4f} "
              f"{group['log_loss'].std():7.4f} {pooled_loss:10.4f} "
              f"{group['accuracy'].mean():9.1%} {pooled_accuracy:11.1%} "
              f"{group['brier'].mean():11.4f}")

    print("\n* = biased diagnostic, not eligible to be frozen on.")


def report_per_fold_logloss(results: pd.DataFrame) -> None:
    section("PER-FOLD LOG-LOSS -- every variant, side by side")

    table = results.pivot(index="validate_year", columns="model", values="log_loss")
    table = table[MODEL_ORDER]

    header = "".join(f"{name:>20s}" for name in MODEL_ORDER)
    print(f"{'year':>6s}{header}")
    for year, row in table.iterrows():
        print(f"{year:6d}" + "".join(f"{row[name]:20.4f}" for name in MODEL_ORDER))
    print(f"{'mean':>6s}" + "".join(f"{table[name].mean():20.4f}"
                                    for name in MODEL_ORDER))
    print(f"{'wins':>6s}" + "".join(
        f"{int((table[name] == table.min(axis=1)).sum()):20d}"
        for name in MODEL_ORDER))
    print("\n'wins' counts folds where that variant had the lowest log-loss of all.")


def choose_freeze_model(results: pd.DataFrame, pooled: pd.DataFrame) -> str:
    """Lowest mean-of-folds log-loss among the CORRECTED variants only."""
    section("WHICH MODEL TO FREEZE ON (judged on log-loss)")

    ranking = []
    for name in CORRECTED_VARIANTS:
        group = results[results["model"] == name]
        subset = pooled[pooled["model"] == name]
        ranking.append({
            "model": name,
            "mean_log_loss": group["log_loss"].mean(),
            "sd": group["log_loss"].std(),
            "pooled_log_loss": float(log_loss(subset["y_true"],
                                              subset["probability"], labels=[0, 1])),
            "folds_beating_constant": int(
                (group.set_index("validate_year")["log_loss"] <
                 results[results["model"] == CONSTANT]
                 .set_index("validate_year")["log_loss"]).sum()
            ),
        })

    ranking.sort(key=lambda row: row["mean_log_loss"])

    print("Biased variants are excluded: their scores were obtained by looking")
    print("at the fold they were scored on.\n")
    print(f"{'rank':>5s} {'model':20s} {'mean ll':>9s} {'sd':>8s} "
          f"{'pooled ll':>10s} {'beats const':>12s}")
    for position, row in enumerate(ranking, start=1):
        print(f"{position:5d} {row['model']:20s} {row['mean_log_loss']:9.4f} "
              f"{row['sd']:8.4f} {row['pooled_log_loss']:10.4f} "
              f"{row['folds_beating_constant']:>9d}/7")

    best = ranking[0]["model"]
    runner_up = ranking[1]["model"]
    margin = ranking[1]["mean_log_loss"] - ranking[0]["mean_log_loss"]

    # Mean-of-folds hides whether the winner wins CONSISTENTLY. The folds are
    # shared, so the paired per-fold difference is the honest comparison.
    print(f"\nhead to head, {best} vs {runner_up} (paired by fold):")
    left = results[results["model"] == best].set_index("validate_year")["log_loss"]
    right = results[results["model"] == runner_up].set_index("validate_year")["log_loss"]
    difference = right - left        # positive = the winner is better that year

    print(f"    {'year':>6s} {'winner':>9s} {'runner-up':>10s} {'diff':>9s}")
    for year in left.index:
        marker = "  <-- runner-up better" if difference[year] < 0 else ""
        print(f"    {year:6d} {left[year]:9.4f} {right[year]:10.4f} "
              f"{difference[year]:+9.4f}{marker}")

    wins = int((difference > 0).sum())
    standard_error = float(difference.std() / np.sqrt(len(difference)))
    t_statistic = float(difference.mean() / standard_error) if standard_error else 0.0

    print(f"\n    mean paired difference : {difference.mean():+.4f}")
    print(f"    standard error         : {standard_error:.4f}")
    print(f"    t                      : {t_statistic:+.2f} on "
          f"{len(difference) - 1} df")
    print(f"    folds won              : {wins} of {len(difference)}")

    print(f"\nFREEZE ON (lowest mean log-loss): {best}")
    print(f"    margin over {runner_up}: {margin:.4f}")

    if abs(t_statistic) < 2.0:
        print(f"\n    BUT the two are NOT separated by this evidence: |t| = "
              f"{abs(t_statistic):.2f} on {len(difference) - 1} df, and the")
        print(f"    winner takes only {wins} of {len(difference)} folds. On seven")
        print(f"    folds a {margin:.4f} margin is noise, not a ranking.")
        print(f"    Tie-breakers that do NOT depend on these folds:")
        print(f"      - {LOGISTIC} is deterministic, has no early-stopping")
        print(f"        iteration to carry forward, and needs no inner split, so")
        print(f"        it trains on the year closest to the scored fold.")
        print(f"      - {LGBM} gives up its most recent training year to the")
        print(f"        inner split -- the year most like the fold it must predict.")
        print(f"      - lower fold-to-fold sd is the more durable property here:")
        for row in ranking[:2]:
            print(f"          {row['model']:20s} sd {row['sd']:.4f}")

    return best


def report_calibration(pooled: pd.DataFrame, best_model: str) -> pd.DataFrame:
    section(f"CALIBRATION -- {best_model}, pooled validation folds")

    print("Task B ranks on predicted probability, so this matters twice over:")
    print("ranking needs only MONOTONICITY (a higher score must mean a higher")
    print("true rate), but any absolute use of the number -- a threshold, a")
    print("position size, an expected-value calculation -- needs the level to")
    print("be right too. Both are visible below.\n")

    subset = pooled[pooled["model"] == best_model].copy()

    subset["decile"] = pd.qcut(
        subset["probability"], CALIBRATION_DECILES, labels=False, duplicates="drop"
    )

    table = subset.groupby("decile").agg(
        n=("y_true", "size"),
        predicted=("probability", "mean"),
        actual=("y_true", "mean"),
        p_min=("probability", "min"),
        p_max=("probability", "max"),
    ).reset_index()
    table["gap"] = table["actual"] - table["predicted"]

    print(f"{'decile':>7s} {'n':>6s} {'range':>17s} {'predicted':>10s} "
          f"{'actual':>8s} {'gap':>8s}")
    for _index, row in table.iterrows():
        bar_position = int(row["actual"] * 40)
        print(f"{int(row['decile']) + 1:7d} {int(row['n']):6d} "
              f"[{row['p_min']:.3f},{row['p_max']:.3f}] {row['predicted']:10.3f} "
              f"{row['actual']:8.3f} {row['gap']:+8.3f}  "
              f"{'.' * bar_position}|")

    # Expected calibration error: the size of the miss, weighted by bucket.
    ece = float((table["n"] * table["gap"].abs()).sum() / table["n"].sum())
    overall_predicted = float(subset["probability"].mean())
    overall_actual = float(subset["y_true"].mean())

    print(f"\n    expected calibration error (ECE) : {ece:.4f}")
    print(f"    mean predicted / mean actual     : "
          f"{overall_predicted:.4f} / {overall_actual:.4f} "
          f"({overall_predicted - overall_actual:+.4f})")
    print(f"    Brier score                      : "
          f"{brier_score_loss(subset['y_true'], subset['probability']):.4f}")

    # Monotonicity is the property ranking actually needs.
    increases = int((table["actual"].diff().dropna() > 0).sum())
    print(f"\n    deciles where actual rate rises  : {increases} of "
          f"{len(table) - 1}")
    spread = table["actual"].iloc[-1] - table["actual"].iloc[0]
    print(f"    top decile minus bottom decile   : {spread:+.3f} "
          f"({table['actual'].iloc[0]:.1%} -> {table['actual'].iloc[-1]:.1%})")

    if ece > 0.05:
        print(f"\n    !!! ECE above 0.05: the LEVEL is not trustworthy. Ranking")
        print(f"    !!! may still be, if the column above is monotone.")
    else:
        print(f"\n    ECE within 0.05; the level is usable, not just the order.")

    return table


def report_per_ticker(pooled: pd.DataFrame, best_model: str) -> None:
    section(f"PER-TICKER ACCURACY -- {best_model}, pooled validation folds")

    print("Ticker identity is NOT in the feature set, so nothing can be")
    print("memorised outright -- but growth_streak and label_lag_* are")
    print("company-persistence proxies, which reach the same place indirectly.\n")

    subset = pooled[pooled["model"] == best_model]

    by_ticker = subset.groupby("ticker").apply(
        lambda group: pd.Series({
            "n": len(group),
            "positive_rate": group["y_true"].mean(),
            "model_accuracy": (group["prediction"] == group["y_true"]).mean(),
            "always_one_accuracy": group["y_true"].mean(),
        }),
        include_groups=False,
    )
    by_ticker["lift"] = by_ticker["model_accuracy"] - by_ticker["always_one_accuracy"]

    dense = by_ticker[by_ticker["n"] >= MIN_TICKER_ROWS]
    print(f"tickers with >= {MIN_TICKER_ROWS} validation rows : {len(dense)} of "
          f"{len(by_ticker)}")

    if dense.empty:
        return

    print(f"    median per-ticker accuracy : {dense['model_accuracy'].median():.1%}")
    print(f"    median lift over always-1  : {dense['lift'].median():+.1%}")
    print(f"    beats always-1 on          : {int((dense['lift'] > 0).sum())} of "
          f"{len(dense)} ({(dense['lift'] > 0).mean():.1%})")
    print(f"\n    corr(accuracy, positive_rate) : "
          f"{dense['model_accuracy'].corr(dense['positive_rate']):+.3f}")
    print(f"    corr(lift,     positive_rate) : "
          f"{dense['lift'].corr(dense['positive_rate']):+.3f}")
    print("\n    Lift IS accuracy minus positive_rate, so the second is")
    print("    negatively correlated by construction. A strongly negative value")
    print("    means the edge sits in the low-base-rate names, where")
    print("    always-predict-1 is weak, and adds little on reliable growers.")


def report_baseline_drift(results: pd.DataFrame) -> None:
    section("BASELINE DRIFT ACROSS YEARS")

    constant = results[results["model"] == CONSTANT]
    print(f"{'year':>6s} {'valid_pos':>10s} {'const_acc':>10s} {'const_ll':>10s}")
    for _index, row in constant.iterrows():
        print(f"{row['validate_year']:6d} {row['validate_positive_rate']:10.1%} "
              f"{row['accuracy']:10.1%} {row['log_loss']:10.4f}")

    rates = constant["validate_positive_rate"]
    print(f"\npositive rate range     : {rates.min():.1%} .. {rates.max():.1%} "
          f"(spread {rates.max() - rates.min():.1%}, sd {rates.std():.1%})")
    print("\nAccuracy-minus-constant is not comparable across these years.")
    print("Log-loss does not depend on where the class boundary falls, which is")
    print("why the freeze decision is made on it.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    frame, holdout_rows, _null_labels = load_modelling_frame()
    assert_feature_set_is_clean(frame)

    folds = build_folds(frame)
    assert_split_discipline(folds)

    results, pooled, extras = run_walkforward(folds)

    report_folds(results)
    report_before_after(results, extras)
    report_coefficients(extras, folds)
    report_per_fold_logloss(results)
    report_aggregate(results, pooled)
    report_baseline_drift(results)

    best_model = choose_freeze_model(results, pooled)
    calibration = report_calibration(pooled, best_model)
    report_per_ticker(pooled, best_model)

    section("OUTPUT")
    os.makedirs("data", exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    calibration.insert(0, "model", best_model)
    calibration.to_csv(CALIBRATION_CSV, index=False)

    print(f"results     -> {RESULTS_CSV}  ({len(results)} rows: "
          f"{len(folds)} folds x {len(MODEL_ORDER)} variants)")
    print(f"calibration -> {CALIBRATION_CSV}  ({len(calibration)} deciles)")
    print(f"\nHOLDOUT STILL LOCKED: {holdout_rows} rows at/after "
          f"{HOLDOUT_START.date()} were never read.")
    print("No hyperparameter search was run. Both changes in this step are")
    print("structural corrections, not choices made against fold results.")


if __name__ == "__main__":
    main()
