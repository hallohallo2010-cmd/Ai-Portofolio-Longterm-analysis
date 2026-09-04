# Ai-Portofolio-Longterm-analysis

Can a company's own earnings history tell you whether its next quarter will
beat the year-ago one?

This repository is a research study, not a trading system. Everything below
the "Pre-registered decisions" heading was fixed **before any model existed**,
and is recorded here so that the study cannot be quietly re-specified after
seeing a result.

---

## The research question

For a member of the S&P 500, at the moment a quarterly EPS figure becomes
public, predict whether that quarter's diluted EPS **beat the same quarter one
year earlier**.

The prediction is made on `prediction_date`, defined as `filed_date + 1 day`.
That is the first moment a portfolio could have acted on the number, so it is
the only defensible time axis in the study. Every feature must be knowable
strictly before it.

Scope discipline: Phase 0 built the panel and the label and nothing else.
Phase 1 builds features. No model is trained and no train/test split is made
until the feature set is fixed.

---

## Pre-registered decisions

These were decided on the data's structure and on documented sourcing
constraints, before any predictive result was computed. Each is enforced in
code by an assertion in `scripts/build_eps_panel.py`, not merely asserted here.

### The label: `label_yoy`

```
label_yoy = 1  if  eps_diluted > eps_year_ago_adjusted
          = 0  otherwise
          = null  if the comparison cannot be made honestly
```

Strictly greater — a flat quarter is not a beat.

**Why year-over-year rather than a consensus-estimate surprise.** The obvious
alternative label is "did EPS beat the analyst consensus". It is rejected here
for three reasons:

1. *Point-in-time consensus is not freely available.* Vendor estimate
   histories are licensed, and the freely scrapeable versions are the
   **current** consensus, not the consensus as it stood the day before the
   filing. Using them would embed a look-ahead that is invisible in the
   output.
2. *Consensus is itself a forecast.* Predicting it means modelling analyst
   behaviour on top of company fundamentals, which is a different — and
   noisier — question than the one asked here.
3. *A YoY comparison is reconstructible from the same primary source as the
   EPS figure itself.* Both sides of the comparison come from SEC EDGAR, so
   the label inherits the point-in-time guarantees described below rather than
   depending on a second source with its own revision history.

The cost of this choice is stated plainly: a YoY beat is an easier and more
autocorrelated target than a consensus surprise, and the baseline is
correspondingly high (see below). This study is about whether earnings history
carries signal, not about whether that signal is tradeable after costs.

### `STUDY_START = 2011-01-01`

XBRL tagging was phased in across 2009–2011. Before it settled, a quarter's
*earliest* XBRL record is frequently a later comparative rather than its own
original filing: the number was public on time in HTML, but was not tagged
until a subsequent filing repeated it. Measured on the development basket, 42
of the 48 periods whose first filing lands more than 90 days after period end
have a period end before 2011.

Because `prediction_date` is derived from `filed_date`, those rows would carry
a filing date up to a year later than the market's actual knowledge date. The
panel therefore starts in 2011. Quarters before this date are still fetched —
a surviving 2010 quarter is needed as the year-ago lookup for an in-window 2011
quarter — but they never reach the output.

### `STUDY_END = 2025-12-31`

Two reasons, both about trustworthiness rather than availability:

1. Index membership is reconstructed from a source pinned at 2025-05-27 (see
   below), so quarters after 2025 cannot be gated against a constituent list
   anyone can vouch for.
2. The most recent quarters are the ones most likely to be restated later, and
   a restatement changes what the panel believes was true.

Rows past this bound are dropped and logged to `data/dropped_periods.csv`,
never silently kept.

### The `MIN(filed)` rule

A period appears in EDGAR once per filing that reported it. The panel keeps
the **earliest** filing for each `(ticker, period_end)`.

This is the single most important correctness rule in the study. Later filings
repeat a quarter as a comparative, and across a stock split they repeat it with
a **split-adjusted** value that did not exist at the time. Taking the latest
filing — or any aggregate over filings — would import a number nobody could
have seen on `prediction_date`. Keeping `MIN(filed)` is what makes each row
point-in-time correct.

Two supporting rules follow from it:

- **Filing lag ≤ 120 days.** A quarter first filed later than this did not
  become public on a normal reporting schedule, so its `filed_date` cannot be
  trusted as the moment the market learned the number. Observed range in the
  panel: 9 to 111 days, median 34.
- **The year-ago quarter is matched by date, not by row offset.** Target −365
  days, tolerance ±45 days, nearest match. Fiscal quarters are ragged (4-4-5
  calendars, 52/53-week years), and a single missing quarter would make a
  positional shift silently compare the wrong periods.

### Splits are inferred from restatements, not from a price feed

No external split feed is used. When a later filing restates a period's EPS by
almost exactly an integer factor (≥2, within 2%), that is a split — real
restatements do not land on 4.000. The effective date is never stated but is
bounded: a filing made on date D reports in the units current on D, so the
split falls in `(last pre-split filing, first post-split filing]`.

Each of the two filings is then placed on its own side of that window. If the
split intervenes between them, the year-ago EPS is divided by the factor
(compounding if two splits intervene). If either filing lands *inside* the
undecidable window, the row is flagged `split_ambiguous` and **its label is
nulled rather than guessed**.

`eps_diluted` is never adjusted. Only the year-ago side is.

### The membership gate

A row survives only if its ticker was in the index **on its `prediction_date`**
— not on `period_end`, because `prediction_date` is when the decision would be
taken.

This is what separates the panel from a survivors-only one. A company's 2013
quarters do not belong here if it joined the index in 2016, and a company that
left in 2015 must still contribute the quarters it was a member for.

### Hard dependency: pinned Wikipedia revision `1292523673`

Membership intervals are reconstructed from Wikipedia's S&P 500 added/removed
changes table. **That table was removed from the live page in mid-2025.** The
study therefore depends on a pinned revision:

```
WIKIPEDIA_CHANGES_OLDID = "1292523673"   # 2025-05-27
```

372 change rows, 308 of them inside the study window.

Both the current-constituents table and the changes table are read from *the
same* pinned revision. Mixing a live current table with a pinned changes table
leaves a blind spot for every index change between the revision and today,
which previously misclassified 21 departed tickers as still current.

This is a genuine single point of failure and is treated as one:

- If the pinned revision stops yielding a changes table, `src/index_membership.py`
  **aborts**. It does not fall back to today's constituent list — that
  substitution is exactly the survivorship bias the gate exists to prevent.
- If fewer than 150 changes are found in the window, the run aborts. Real S&P
  500 turnover is 20–25 names a year; far below that means the table is a stub
  and the reconstruction would be silently incomplete.

Anyone reproducing this study should expect to re-pin, and should treat a
changed revision id as a change to the study rather than a maintenance detail.

### The baseline the model must beat

The label is not balanced, and the relevant baseline is **not 50%**.

| group | labelled rows | positive rate |
|---|---:|---:|
| all rows | 21,753 | **60.3%** |
| never removed from the index | 17,314 | 61.7% |
| removed at some point | 4,439 | 54.8% |

A constant "always predict 1" classifier scores **60.3%**. Any model is judged
against that number, pre-registered here so it cannot be replaced with 50%
after the fact.

The 6.9-point survivorship gap is the reason the membership gate exists. Names
that were eventually removed from the index beat their year-ago quarter
markedly less often. A survivors-only panel would have shown 61.7% and taught a
model an optimism the real index never had.

---

## The panel

`data/eps_panel.parquet` — 23,576 rows, 716 tickers, one row per
`(ticker, period_end)`.

- `period_end` spans 2011-01-01 → 2025-12-31
- `prediction_date` runs to 2026-02-28 (Q4 2025 filings land in early 2026)
- median 34 quarters per ticker (min 1, max 60)
- 21,753 rows carry a label; 1,823 are null — 834 with no year-ago match
  within ±45 days, 989 with an unresolvable split

| column | meaning |
|---|---|
| `ticker`, `cik`, `cik_span_used` | identity; `cik_span_used` records which CIK span served the row |
| `period_end` | fiscal quarter end |
| `filed_date` | when the number became public (EDGAR `filed`) |
| `prediction_date` | `filed_date + 1 day`; the decision moment |
| `eps_diluted` | diluted EPS **as first filed** |
| `n_filings_seen` | how many filings reported this period |
| `label_yoy` | the label; nullable `Int64` |
| `in_index_at_prediction` | membership gate result (always true in the output) |
| `is_removed_name` | ticker left the index at some point |
| `period_end_year_ago`, `prediction_date_year_ago`, `eps_year_ago` | the matched comparison quarter |
| `split_factor_applied`, `eps_year_ago_adjusted` | split adjustment, year-ago side only |
| `split_ambiguous`, `split_contaminated` | split could not be pinned / row touched by a split |
| `filing_lag_days` | `filed_date − period_end` |

The year-ago columns are **label provenance, not features**. They exist so the
±45-day match and the split adjustment can be audited and the unadjusted label
rebuilt.

Corporate successions split a company's filing history across CIKs. A plain
CIK override *replaces* rather than extends, trading recent years for old ones
(DIS −7.5yr, CI −7.8yr, LIN −7.8yr, MDT −11.4yr). `src/data_loader.CIK_SPANS`
uses date-ranged, non-overlapping spans so both halves are kept.

---

## Phase 1, step 1: EPS-derived features

`data/features_v1.parquet` — the panel carried through unchanged, plus 12
features derived from EPS alone. No prices, no new sources, no model, no split.

| feature | definition |
|---|---|
| `eps_growth_yoy` | `(eps_diluted − eps_year_ago_adjusted) / abs(eps_year_ago_adjusted)` |
| `label_lag_1..4` | the label from the 1–4 prior quarters |
| `eps_growth_yoy_lag_1..4` | `eps_growth_yoy` from the 1–4 prior quarters |
| `growth_acceleration` | `eps_growth_yoy − eps_growth_yoy_lag_1` |
| `growth_streak` | consecutive prior quarters with `label == 1`, capped at 8 |
| `quarters_available` | prior quarters for this ticker |

Nulls are left null — never filled, never interpolated. A zero year-ago base is
nulled rather than allowed to produce infinity (32 rows).

**The leakage rule.** Every lag must come from a row whose `prediction_date` is
strictly earlier than the row's own. Lags are attached by an explicit merge on
a rank derived from `prediction_date` within ticker — never by a positional
shift, and never on `period_end`, because a late filing can invert the two
orderings. The rule is asserted on the built file, and the assertion is
fire-tested against injected violations on every run so it cannot silently
degrade into a check of nothing. Each lag's source `prediction_date` is carried
into the output as an audit column so the check is reproducible by anyone.

As of the current panel the two orderings never actually disagree — zero
inversions, zero ties across all 23,576 rows. The rule is enforced anyway, and
the point is that this is verified per build rather than assumed.

### Two of these features are not legal predictors

`label_yoy` is 1 exactly when `eps_diluted > eps_year_ago_adjusted`, and
`eps_growth_yoy` is the signed, scaled version of that same difference. So
`sign(eps_growth_yoy)` **is** the label — verified at build time, 21,721 of
21,721 rows, zero mismatches. `growth_acceleration` carries `eps_growth_yoy` as
a term and is disqualified for the same reason.

This is not a look-ahead: both are genuinely public on `prediction_date`. It is
a target identity. A model handed either column scores ~100% and has learned
nothing. Both are still built — `eps_growth_yoy` is what the lags are lags of,
and both were specified — but the build labels them explicitly:

```
not legal predictors of label_yoy : eps_growth_yoy, growth_acceleration
safe to model on                  : eps_growth_yoy_lag_1..4, label_lag_1..4,
                                    growth_streak, quarters_available
```

### What the prior-only features are worth

Measured in-sample, so these are optimistic ceilings, not held-out estimates.
Against the 60.3% majority-class baseline:

| feature | Spearman | best single-cut accuracy |
|---|---:|---:|
| `label_lag_1` | 0.253 | 64.2% |
| `eps_growth_yoy_lag_1` | 0.240 | 65.0% |
| `growth_streak` | 0.224 | 62.9% |
| `label_lag_2` | 0.142 | 59.7% |
| `label_lag_4` | −0.086 | 59.6% |
| `quarters_available` | −0.027 | 60.3% |

Signal decays fast with lag and turns **negative** by lag 4 — a year-ago beat
is mildly evidence *against* a beat now, which is what a YoY comparison against
a strong base quarter should do. `growth_streak` is monotone in the right
direction: a 0 streak beats 46.9% of the time, an 8 streak 75.7%.

Pearson is reported too, because it was asked for, but it is near-useless here:
`eps_growth_yoy` has a max of 4.7e7 and a standard deviation of 4.2e5, so a
handful of near-zero-base rows dominate it. Pearson reads 0.006 on a feature
that reproduces the label exactly; Spearman reads 0.848.

### Coverage cost

Requiring `label_lag_1..4` all present keeps 18,406 of 23,576 rows (78.1%),
18,235 of them labelled, across 635 of 716 tickers. The loss is systematic, not
random: it removes the first four quarters of every ticker's history, so 2011
alone accounts for 1,541 of the 5,170 dropped rows. Surviving rows are slightly
cleaner than the panel as a whole (18.7% removed names vs 21.0%), which is a
mild survivorship re-introduction worth remembering when reading any result on
that subset.

---

## Phase 2: baselines and walk-forward validation

No tuning. Every hyperparameter is fixed in advance and nothing is chosen
against fold results.

### The locked holdout — pre-registered

**2022-01-01 onward is a locked holdout: 5,898 rows that have not been read.**
`scripts/train_walkforward.py` discards them at load, before any other stage
touches the data, and asserts that every fold — outer *and* inner — ends
strictly before the boundary. No model has been selected on them and no
hyperparameter has been searched anywhere.

### The split

Expanding window, cut on `prediction_date` — never `period_end`, since a
quarter ending in December 2014 that filed in February 2015 was not knowable
until 2015.

```
train 2011-2014 -> validate 2015      ...      train 2011-2020 -> validate 2021
```

16,232 modelling rows after dropping 1,446 null labels. Rows with null
*features* are kept (19.3%); LightGBM routes NaN natively and logistic
regression median-imputes within each training fold only.

Ten features, all prior-quarter only: `eps_growth_yoy_lag_1..4`,
`label_lag_1..4`, `growth_streak`, `quarters_available`. `eps_growth_yoy` and
`growth_acceleration` are excluded as target identities, asserted at runtime
against the built matrix.

### Step 2: two structural corrections

**Both are corrections to the protocol, not choices made against fold
results.** Each fixes a way the step-1 numbers were biased, and each would have
been required whatever the numbers had come out as. Neither is a
hyperparameter, and no variant was selected by comparing outcomes. The biased
versions are still run, so the size of each bias is measured rather than
asserted.

**Fix 1 — inner validation split.** In step 1, LightGBM early-stopped on the
same fold it was scored on, choosing its stopping iteration using the answer.
Now the **last year of each training window** is held out as an inner
validation set, the model trains on the years before it, and the outer fold is
untouched until scoring. Asserted: the inner split sits strictly inside the
training window, and early stopping never sees the scored fold.

**Fix 2 — rank transform on `eps_growth_yoy_lag_*`.** Those features reach
4.7e7 with σ ≈ 4.2e5, so standardizing collapsed nearly every row to a spike
near zero and left the linear model unable to use them. A quantile transform is
fitted on the **training window only** and applied to the validation fold,
inside a `Pipeline` so the boundary is enforced by construction. Applied to the
logistic model alone: a rank transform is strictly monotone and trees split on
thresholds, so for LightGBM it is a no-op except for what a quantile grid would
throw away.

### How much each bias was worth

**Fix 1 — LightGBM lost most of its lead.** The step-1 advantage was largely
the bias:

| | mean log-loss | mean accuracy |
|---|---:|---:|
| `lightgbm_outer_es` (biased) | 0.6192 | 65.5% |
| `lightgbm_inner_es` (corrected) | **0.6275** | 64.9% |
| cost of the correction | +0.0083 | −0.59pp |

The correction also costs training rows — the last year becomes the
early-stopping set. A useful internal check: the inner-split model for fold *Y*
trains on exactly the same rows as the biased model for fold *Y−1*, and the two
report identical stopping iterations, which is what the implementation should
produce.

**Fix 2 — the rank transform helped, and the features came alive.** Better
log-loss in 6 of 7 folds:

| | mean log-loss | mean accuracy |
|---|---:|---:|
| `logistic_raw` | 0.6494 | 64.4% |
| `logistic_rank` | **0.6281** | 65.0% |
| gain | −0.0213 | +0.66pp |

Mean standardized coefficients across folds:

| feature | before | after | ratio |
|---|---:|---:|---:|
| `eps_growth_yoy_lag_1` | +0.047 | +0.121 | 2.6× |
| `eps_growth_yoy_lag_2` | +0.047 | +0.071 | 1.5× |
| `eps_growth_yoy_lag_3` | +0.021 | −0.082 | 3.9× |
| `eps_growth_yoy_lag_4` | −0.020 | **−0.394** | 20.1× |
| `label_lag_1` | +0.387 | +0.273 | 0.7× |
| `label_lag_4` | −0.212 | +0.084 | 0.4× |

Mean |coefficient| on the rank-transformed features rises 5.0× (0.034 → 0.167)
while the untouched features fall to 0.7×. They did come alive — and
`eps_growth_yoy_lag_4` becomes the **largest** coefficient in the model,
absorbing the mean-reversion signal that `label_lag_4` had been carrying as a
crude binary. The negative sign is consistent with Phase 1: a strong quarter
four back is evidence against a beat now.

### Per-fold log-loss, all variants

| year | constant | logistic_raw* | **logistic_rank** | lightgbm_outer_es* | **lightgbm_inner_es** |
|---|---:|---:|---:|---:|---:|
| 2015 | 0.7081 | 0.6841 | 0.6676 | 0.6482 | 0.6448 |
| 2016 | 0.7034 | 0.6739 | 0.6162 | 0.6057 | 0.6313 |
| 2017 | 0.6548 | 0.6281 | 0.6092 | 0.5938 | 0.6059 |
| 2018 | 0.6333 | 0.6304 | 0.6161 | 0.6106 | 0.6253 |
| 2019 | 0.6869 | 0.6657 | 0.6424 | 0.6383 | 0.6378 |
| 2020 | 0.7351 | 0.6836 | 0.6874 | 0.6843 | 0.7005 |
| 2021 | 0.6142 | 0.5802 | 0.5581 | 0.5532 | 0.5470 |
| **mean** | 0.6766 | 0.6494 | **0.6281** | 0.6192 | **0.6275** |

`*` = biased diagnostic, excluded from the freeze decision.

Both corrected models beat the constant on log-loss in **all seven folds**.

### Which model to freeze on

Lowest mean log-loss is `lightgbm_inner_es` (0.6275) over `logistic_rank`
(0.6281) — **a margin of 0.0006, which is noise.** Paired by fold:

- mean paired difference **+0.0006**, standard error 0.0052
- **t = +0.12 on 6 df**
- LightGBM wins **4 of 7** folds

The two are not separated by this evidence. On tie-breakers that do not depend
on these folds, **`logistic_rank` is the better thing to freeze on**:

- it is deterministic, with no early-stopping iteration to carry forward;
- it needs no inner split, so it trains on the year closest to the fold it must
  predict — LightGBM gives that year up;
- it has lower fold-to-fold spread (sd 0.0425 vs 0.0460);
- it wins 2020, the regime-break year, by the largest single-fold margin either
  model achieves.

The script reports the log-loss winner as asked and prints the tie-break
reasoning alongside it. The choice between two statistically indistinguishable
models on parsimony grounds is a judgement, and is recorded as one.

### Calibration

Deciles of predicted probability, pooled validation folds, `lightgbm_inner_es`:

| decile | n | predicted | actual | gap |
|---|---:|---:|---:|---:|
| 1 | 1084 | 0.329 | 0.373 | +0.043 |
| 2 | 1084 | 0.423 | 0.403 | −0.020 |
| 3 | 1082 | 0.479 | 0.477 | −0.002 |
| 4 | 1083 | 0.541 | 0.506 | −0.035 |
| 5 | 1083 | 0.601 | 0.560 | −0.040 |
| 6 | 1083 | 0.648 | 0.653 | +0.004 |
| 7 | 1083 | 0.690 | 0.693 | +0.003 |
| 8 | 1083 | 0.731 | 0.735 | +0.004 |
| 9 | 1083 | 0.764 | 0.799 | +0.035 |
| 10 | 1084 | 0.820 | 0.790 | −0.031 |

- **ECE 0.0218**, mean predicted 0.6026 vs mean actual 0.5988 (+0.0038)
- Brier 0.2190
- monotone in **8 of 9** decile steps
- top decile minus bottom: **+0.417** (37.3% → 79.0%)

Good enough for Task B on both counts. Ranking needs only monotonicity, and the
one inversion is the 9→10 step (0.799 → 0.790) — the top two deciles are
effectively tied rather than ordered, so treat the top ~20% as one bucket
rather than trusting the split between them. The level is usable too: ECE well
under 0.05, with the largest miss a 4-point under-confidence in the bottom
decile. The mid-range deciles (4 and 5) are the mildly over-confident ones.

### Is it just learning which companies grow?

Ticker identity is not in the feature set, so nothing can be memorised
outright; `growth_streak` and `label_lag_*` are company-persistence proxies
that reach the same place indirectly.

340 tickers with ≥20 validation rows: median accuracy 64.3%, median lift over
always-predict-1 +3.6pp, beating it on 185 of 340 (54.4%).
`corr(accuracy, positive_rate)` = **+0.540**; `corr(lift, positive_rate)` =
**−0.607**. The second is partly arithmetic — lift *is* accuracy minus positive
rate — but the direction holds: the edge sits in the low-base-rate names where
always-predict-1 is weak, and adds little on the reliable growers. Much of the
headline accuracy on those names is the base rate, not the model.

### The baseline is not stable

| year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive rate | 54.0% | 53.5% | 64.5% | 69.2% | 57.2% | **46.5%** | **75.1%** |

A 28.6-point spread, σ = 10.0pp — the COVID collapse and rebound. The 60.3%
pre-registered from the whole panel is an average over years that look nothing
like each other. Accuracy-minus-constant is therefore not comparable across
folds, which is why the freeze decision is made on log-loss: it does not depend
on where the class boundary happens to fall.

---

## Frozen model (pre-registered)

**`logistic_rank` is frozen.** The full specification — model class, exact
feature list and order, rank-transform spec, imputation rule, training window,
and the git SHA of the code that produced it — is committed in
[`config/frozen_model.json`](config/frozen_model.json).

**That file was committed before the holdout was read for the first time.**
Anything in it that changes after the holdout has been read invalidates the
holdout as a test.

### Why this model

Chosen over `lightgbm_inner_es` on **structural grounds, not on a log-loss
margin**. The two are statistically indistinguishable:

| | mean log-loss | sd across folds |
|---|---:|---:|
| `lightgbm_inner_es` | 0.6275 | 0.0460 |
| **`logistic_rank`** | **0.6281** | **0.0425** |
| constant (always 1) | 0.6766 | 0.0438 |

Paired by fold: mean difference **0.0006**, standard error 0.0052,
**t = 0.12 on 6 df**, LightGBM winning 4 of 7 folds. That is a coin flip, not a
ranking. The tie-breakers are all properties that hold independently of these
folds: `logistic_rank` is deterministic, carries no early-stopping iteration
forward, needs no inner split (so it trains on the year closest to the fold it
must predict, which LightGBM gives up), has the lower fold-to-fold spread, and
wins 2020 — the regime-break year — by the largest single-fold margin either
model achieves.

Walk-forward mean log-loss **0.6281 vs 0.6766** for the constant baseline,
beating it in **7 of 7** folds.

### The bias-correction finding

This is the result worth carrying forward, and it is a caution about method
rather than a fact about earnings.

In Phase 2 step 1, LightGBM appeared to lead logistic regression by **0.030
log-loss** (0.6192 vs 0.6494). After two structural corrections that gap fell
to **0.0006**. Essentially all of the apparent advantage was measurement error,
in two parts:

1. **Early-stopping leakage.** LightGBM early-stopped on the very fold it was
   scored on, choosing its stopping iteration using the answer. Holding out the
   last training year as an inner validation set instead cost it **+0.0083**
   log-loss.
2. **An unscaled feature.** `eps_growth_yoy_lag_*` reach 4.7e7 with σ ≈ 4.2e5,
   so standardizing collapsed them to a spike near zero and the linear model
   could not use them at all. Rank-transforming them — fitted on the training
   window only — gained logistic **0.0213** log-loss and raised the mean
   |coefficient| on those four features by **5.0×**. `eps_growth_yoy_lag_4`
   went from −0.020 to −0.394 and became the largest coefficient in the model.

Neither correction was a choice made against fold results; both would have been
required whatever the numbers had been. The lesson generalises: a tree model
beating a linear one by a margin like this is worth suspecting before it is
worth believing, because the two most likely explanations — a leaky validation
protocol and a feature the linear model cannot physically use — both favour the
tree for reasons that have nothing to do with the data.

### Calibration

Measured on pooled validation folds: **ECE 0.0218**, mean predicted 0.6026 vs
mean actual 0.5988, monotone across 8 of 9 decile steps, top-minus-bottom
spread +0.417.

**Deciles 9 and 10 are inverted** (0.799 vs 0.790). Since Task B ranks on
predicted probability, **treat the top 20% as one bucket** rather than trusting
the ordering between them. Everything below decile 9 is cleanly ordered.

One caveat recorded in the config rather than smoothed over: that decile table
was computed for `lightgbm_inner_es`, the log-loss winner, before
`logistic_rank` was chosen on structural grounds. `logistic_rank`'s own
calibration has not been separately tabulated and should be measured on the
holdout run.

### Amendment (pre-hoc): fit-once → expanding walk-forward

**Revision 2 of the pre-registration, committed as its own commit before any
holdout row was read or scored.** The amendment is timestamped ahead of the
run so it is evidently pre-hoc rather than a response to a result, and
revision 1 is retained in the config rather than overwritten.

The holdout is now evaluated as an expanding walk-forward, refitting each fold:

```
train 2011-2021 -> predict 2022      train 2011-2023 -> predict 2024
train 2011-2022 -> predict 2023      train 2011-2024 -> predict 2025
```

Same frozen config throughout — same model, same ten features, same rank
transform, same imputation rule. **No config changes between folds.** The whole
pipeline is refit from scratch on each fold's own training window, so no
quantile grid, median, mean, variance or coefficient is ever carried across
folds.

Why this replaces fit-once:

- **Fit-once measures signal decay, not the research question.** A model held
  fixed from 2021 while the world moves is being tested on how fast it goes
  stale, which is a different question from whether earnings history carries
  signal.
- **Walk-forward matches the deployment protocol in Phase 3 step 4.** The
  holdout should test the thing that would actually be run, not a
  simplification of it.
- **2022 is a regime break.** A model frozen at 2021 conflates staleness with
  signal failure: if it did badly on 2024 there would be no way to tell whether
  the signal had stopped working or the model had simply never seen the new
  regime.

What the amendment does *not* touch: the model, the features, the transform,
the imputation rule, the success criterion, the expanding-window rule, or the
`prediction_date` cut. It changes only **when the model is refit**.

One consequence recorded rather than discovered later: the protocol
**consumes** the holdout — once fold 2 trains on 2022 those labels have been
read, so this is a one-shot test with no second holdout available in this data.

### Amendment 3 (pre-hoc): fold 4 scores the remainder

**Revision 3, committed before the run.** Fold 4's prediction set becomes **all
remaining holdout rows** (`prediction_date >= 2025-01-01`) rather than
`prediction_year` 2025, admitting the 70 rows (62 labelled) whose `period_end`
is Q4 2025 but whose filing landed in early 2026.

Why: the revision-2 exclusion was drawn on **filing lag, not on the research
question**. A Q4-2025 quarter is in scope; that its filing happened to land in
January or February 2026 says something about the filer's calendar, not about
whether the study should score it. The panel's convention is that a row
*belongs* to its `period_end` while `prediction_date` governs *knowability*, and
excluding a 2025 quarter for when it was filed confuses the two. Making fold 4
the terminal "everything remaining" fold **removes** the exclusion rather than
relocating it — with nothing excluded, there is no filing-lag boundary left to
justify.

The obvious form of this amendment — defining fold 4 as `period_end in 2025` —
was checked and **rejected**: it is not a superset of `prediction_year` 2025. It
would admit the 70 wanted rows but **orphan 81 rows (75 labelled)** with
`period_end` 2024 filed in early 2025, which sit outside fold 3, outside the new
fold 4, and outside fold 4's training window. Net −11 rows, trading 62
explicitly excluded rows for 75 falling through a gap — the same problem one
year earlier. Applying `period_end` boundaries to all four folds orphans 96 rows
at the front for the same structural reason.

Folds 1–3 are unchanged, as is every fold's training window. **No look-ahead is
introduced:** fold 4 still trains only on `prediction_date < 2025-01-01` and its
earliest predicted row is 2025-01-03. Widening the terminal fold forward in time
cannot create one, because the added rows are *later* than everything trained
on.

The holdout is now scored **in full** — 5,898 rows, 5,521 labelled, every row in
exactly one fold, no exclusions and no orphans:

| fold | trains on | predicts | rows | labelled |
|---|---|---|---:|---:|
| 1 | 2011–2021 | `prediction_date` 2022 | 1,454 | 1,357 |
| 2 | 2011–2022 | `prediction_date` 2023 | 1,448 | 1,359 |
| 3 | 2011–2023 | `prediction_date` 2024 | 1,461 | 1,365 |
| 4 | 2011–2024 | `prediction_date` ≥ 2025-01-01 | **1,535** | **1,440** |
| | | **total** | **5,898** | **5,521** |

### Pre-registered success criterion

Stated before the holdout is read, and **unchanged** by the amendment above —
only its granularity is specified further. **Both conditions must hold:**

1. **log-loss** of `logistic_rank` **below** the constant baseline's log-loss;
2. **accuracy** of `logistic_rank` **above** the constant baseline's accuracy.

Evaluated **per-fold (all four) and pooled** across all scored rows.

The constant baseline predicts class 1 always, with probability equal to **each
fold's own training positive rate**. Fold 1 uses 0.610461 (2011–2021); folds
2–4 use rates that include holdout years and are unknown at freeze time. The
baseline may never use the rate of the year it is predicting — that would score
the baseline with the answer.

**The holdout base rate is unknown and may differ materially from 60.3%.**
Across the seven validation folds the yearly positive rate ranged from 46.5% to
75.1% — a 28.6-point spread, σ = 10.0pp — so a single figure is not a stable
property of this data. Accuracy-vs-constant is therefore computed against
**each fold's own constant baseline**, not against 60.3%, not against the
training window's 61.0%, and not against one pooled figure. Judging the model
against 60.3% would reward or punish it for where each year's base rate
happened to land, which is not something the model controls.

If the two conditions disagree, **log-loss is primary** — it does not depend on
where the class boundary falls, which is exactly why it was chosen for the
freeze decision. Passing on some folds and not others is a real possible
outcome and will not be reported as a pass.

Only row counts and date spans were inspected to write the pre-registration;
the holdout label distribution was not computed.

---

## Phase 3: the live forward test — scoring rule (pre-registered 2026-09-04)

Fixed on **2026-09-04**, the same day `data/live_predictions_2026-09-04.csv` was
committed and **before any quarter it predicts had been filed**. No outcome
existed when this was written, and none has been read since.

The 2022–2025 holdout is spent (§ *The locked holdout*). The forward test is the
replacement, and its one advantage over a backtest — that the predictions were
recorded before the answers existed — is destroyed by any discretion about
*when to look*. Scoring early, seeing a bad number and waiting for more filings
is indistinguishable, in the committed record, from scoring once at a fixed
point. So the trigger is mechanical and is written down here in advance.

### When a scoring run may happen

`scripts/score_forward.py` runs **once per committed prediction file**, and only
when **every row** in that file is either *scoreable* or *excluded*:

- **Scoreable** — the ticker has filed a quarter whose `prediction_date`
  (`filed_date + 1 day`) falls after the file's `cut_date`, and that quarter
  yields a non-null `label_yoy` under the panel's own construction rules:
  `MIN(filed)`, the 120-day filing-lag filter, the ±45-day year-ago match, split
  adjustment from restatement evidence, and the `STUDY_START` floor.
- **Excluded** — either the filing arrived but its label is null under those
  same rules (no year-ago quarter within ±45 days, or a split that cannot be
  pinned to one side of its window), or **no qualifying filing has appeared by
  `expected_period_end + 120 days`**, both values read from the committed
  prediction file.
- **Outstanding** — anything else. A run is permitted only when the outstanding
  count is **zero**.

**The resolution test is on the filing date, not on `period_end`.** This is not
a detail. A prediction is for *the next quarter the company files after the
cut*, and for 114 of the 474 rows in the 2026-09-04 file that quarter **ends
before the cut date**. MSFT is one: its last filed quarter ended 2026-03-31, the
file records an `expected_period_end` of 2026-07-01, and the cut is 2026-09-04 —
the quarter being predicted closes two months *before* the prediction was even
made, and is simply not filed yet. A test worded as "a filing with
`period_end` after `cut_date`" would leave those 114 rows permanently
unresolvable and silently discard 24% of the file, keeping the subset that
happens to report on a calendar-year rhythm. Filing date is also the exact
complement of the rule that generated the file: `run_forward.py` treated
everything with `prediction_date <= cut_date` as history, so everything after it
is the forward window.

The 120-day exclusion clock is keyed to `expected_period_end`, which is
**already committed inside the prediction file** and therefore cannot be moved
after the fact. 120 days is the panel's own `MAX_FILING_LAG_DAYS` — the
threshold past which a filing is treated as not having arrived on a normal
reporting schedule. `expected_period_end` is an estimate, and where the estimate
is poor the clock is generous; it errs toward waiting rather than toward
discarding a row.

### Partial scoring is not permitted

A run that cannot resolve every row **scores nothing**. It reports which rows are
outstanding and exits. There is no partial result, no interim figure, and no
"provisional" number — the companies that report first are not a random sample
of the universe, so an early subset is a biased one, and a sequence of partial
looks is the discretion this rule exists to remove.

### The result is committed whatever it is

The scored output and the run that produced it are committed **in the same
commit**, unedited, whether the model beats the baseline, matches it, or loses to
it. A failing forward test is a result and will be recorded as one. Metrics are
those `scripts/score_forward.py` already computes — accuracy and log-loss against
a constant baseline whose probability is the prediction file's own
`training_positive_rate`, never the rate of the period being scored — fixed by
committed code before any outcome existed. The run also asserts that the
prediction file's commit predates every filing it scores against; if that fails,
nothing is scored.

### What must change in the code before the first run

`scripts/score_forward.py` as committed does **not** yet implement this rule, and
recording that here is part of the pre-registration:

1. It accepts `--rescore`, which permits a second run over the same file. That
   flag must be removed, or restricted to a run that scored nothing.
2. It writes a result with rows still unfiled and labels it a partial score.
   That path must become a refusal.
3. It has no exclusion clock; `expected_period_end + 120 days` does not exist in
   the code yet.

Those are code changes, not changes to this rule. This entry is text-only and
deliberately precedes them, so the rule cannot be shaped around whatever the
implementation happened to make easy.

---

## Layout

```
scripts/build_eps_panel.py     Phase 0: the panel and the label
scripts/build_features_v1.py   Phase 1: EPS-derived features
scripts/train_walkforward.py   Phase 2: baselines and walk-forward validation
src/data_loader.py             all SEC EDGAR access; CIK resolution
src/index_membership.py        membership reconstruction from the pinned revision
data/eps_panel.parquet         the panel (tracked)
data/features_v1.parquet       the feature table (tracked)
data/walkforward_results.csv   per-fold, per-variant scores (tracked)
data/walkforward_calibration.csv  decile calibration of the best model (tracked)
config/frozen_model.json       the pre-registered frozen model (tracked)
```

`src/data_loader.py` is the single source of truth for EDGAR access. Scripts
must not build EDGAR URLs, set headers, or sleep for rate limiting on their own.

## Running it

Set `SEC_CONTACT_EMAIL` in `src/data_loader.py` to a real address first —
EDGAR rejects requests without one, and every script refuses to call EDGAR
while it is the placeholder.

```
pip install -r requirements.txt
python scripts/build_eps_panel.py      # ~900 EDGAR calls; caches to data/fact_cache/
python scripts/build_features_v1.py    # offline; reads the panel only
python scripts/train_walkforward.py    # offline; locks the 2022+ holdout at load
```

