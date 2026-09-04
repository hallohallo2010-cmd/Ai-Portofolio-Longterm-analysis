# Task A — Results

Can a company's own quarterly earnings history predict whether its next
reported EPS beats the year-ago quarter?

Short answer: yes, by a small but consistent margin. On a pre-registered
holdout of 5,521 labelled quarters spanning 2022–2025, a ten-feature logistic
model scored **0.6402 log-loss against a 0.6821 constant baseline** and
**64.8% accuracy against 58.0%**, passing both pre-registered conditions on
all four folds and pooled.

Every number below is followed by the committed file it comes from. Two
exceptions are flagged explicitly in §3.

---

## 1. The research question, and why the baseline is 60.3%

For an S&P 500 member, at the moment a quarterly diluted EPS figure becomes
public, predict whether it **beat the same quarter one year earlier**.

The prediction is made on `prediction_date` = `filed_date + 1 day` — the first
moment a portfolio could have acted on the number. Every feature must be
knowable strictly before it.

The label is not balanced. Across the panel, 13,110 of 21,753 labelled
quarters are beats: **60.3%**. A classifier that always predicts 1 therefore
scores 60.3%, not 50%. Reporting "our model gets 64%" against a 50% baseline
would be claiming 14 points of skill where there are 4.

The 60.3% figure is itself unstable. Yearly positive rates across the seven
validation folds ranged from **46.5% (2020) to 75.1% (2021)** — a 28.6-point
spread, σ = 10.0pp (`data/walkforward_results.csv`, `validate_positive_rate`).
So accuracy-minus-baseline is not comparable across years, and the study uses
log-loss — which does not depend on where the class boundary falls — as its
primary metric.

*Source: `data/eps_panel.parquet` (23,576 rows, 716 tickers);
`data/walkforward_results.csv`.*

---

## 2. Data construction

**Source.** SEC EDGAR XBRL, `us-gaap/EarningsPerShareDiluted`, via the
companyconcept API with a companyfacts fallback. The fallback is not cosmetic:
verified cases (ABT CIK 1800, AMT CIK 1053507) return HTTP 200 with an *empty*
record list from companyconcept while companyfacts holds hundreds of records
for the same concept and CIK (`src/data_loader.py`).

**The `MIN(filed)` rule.** A quarter appears in EDGAR once per filing that
reported it. The panel keeps the **earliest** filing for each
`(ticker, period_end)`. This is the single most important correctness rule
here. Later filings repeat a quarter as a comparative, and across a stock split
they repeat it with a **split-adjusted** value that did not exist at the time.
Taking the latest filing — or any aggregate over filings — imports a number
nobody could have seen on `prediction_date`
(`scripts/build_eps_panel.py`, `dedupe_to_first_filing`).

**The re-tagging discovery, and why the study starts in 2011.** XBRL tagging
was phased in across 2009–2011. Before it settled, a quarter's *earliest* XBRL
record is frequently a later comparative rather than its own original filing:
the number was public on time in HTML but was not tagged until a subsequent
filing repeated it. Measured on the development basket, **42 of the 48 periods
whose first filing lands more than 90 days after period end have a period end
before 2011**. Because `prediction_date` is derived from `filed_date`, those
rows would carry a filing date up to a year later than the market's actual
knowledge date. `STUDY_START = 2011-01-01` follows from that measurement, not
from convenience (`scripts/build_eps_panel.py`, lines 51–59).

Two supporting filters: quarters first filed more than **120 days** after
period end are dropped as not having become public on a normal reporting
schedule (observed range in the surviving panel: **9 to 111 days, median 34**);
and the year-ago quarter is matched **by date** — target −365 days, tolerance
±45, nearest — never by row offset, since fiscal quarters are ragged and one
missing quarter would make a positional shift compare the wrong periods.

**Splits, inferred from restatement evidence alone.** No external split feed is
used. When a later filing restates a period's EPS by almost exactly an integer
factor (≥2, within 2%), that is a split — real restatements do not land on
4.000. The effective date is never stated but *is* bounded: a filing made on
date D reports in the units current on D, so the split falls in
`(last pre-split filing, first post-split filing]`. Each of the two filings is
then placed on its own side of that window. If the split intervenes, the
year-ago EPS is divided by the factor — **241 rows**. If either filing lands
*inside* the undecidable window, the row is flagged `split_ambiguous` and its
label is **nulled rather than guessed** — **989 rows**. `eps_diluted` itself is
never adjusted (`scripts/build_eps_panel.py`, `detect_split_events` /
`apply_split_adjustment`; counts from `data/eps_panel.parquet`).

**Result:** 23,576 rows, 716 tickers, `period_end` 2011-01-01 to 2025-12-31.
21,753 carry a label; 1,823 are null — 834 with no year-ago match within ±45
days, 989 with an unresolvable split.

---

## 3. Survivorship

A panel built from today's S&P 500 constituent list would be a survivors-only
panel and would overstate everything. Three problems had to be solved.

**Ever-constituent reconstruction.** Membership intervals are rebuilt from
Wikipedia's added/removed changes table, so a company contributes exactly the
quarters it was a member for — a name that joined in 2016 contributes no 2013
quarters, and a name removed in 2015 still contributes what it had. A ticker
can hold several disjoint intervals, because companies leave and rejoin, so a
single first/last pair would wrongly cover the gap. Rows are gated on
membership at **`prediction_date`**, not `period_end`, because that is when the
decision would be taken (`src/index_membership.py`, `build_intervals` /
`is_member_at`; `scripts/build_eps_panel.py`, `gate_on_membership`).

That table was removed from the live Wikipedia page in mid-2025, so the study
depends on **pinned revision 1292523673** (2025-05-27). Both the current and
changes tables are read from the *same* revision — mixing a live current table
with a pinned changes table previously misclassified 21 departed tickers as
still current. If the pinned revision stops yielding a changes table the code
**aborts** rather than falling back to today's list, which is exactly the
survivorship substitution the gate exists to prevent
(`src/index_membership.py`).

**The CIK resolution failure.** SEC's `company_tickers.json` lists only
*current* registrants. Delisted and acquired companies — precisely the removed
names that carry the survivorship signal — are simply absent. Resolving tickers
through that map alone silently drops them.

Three repairs, all committed:

- **Name-based recovery.** Companies were re-matched by company name against
  the full EDGAR filer index. **151 CIKs recovered**, all 151 with real EPS
  history: 94 exact matches, 55 resolved from ambiguous candidates by tenure
  overlap, 2 fuzzy (`src/recovered_ciks.csv`, 151 rows). Two name matches were
  verified as *wrong* and permanently barred from re-proposal — CA/"CYA
  Technologies" and HAR/"Thurman International" — because a high similarity
  score on a short name means nothing (`src/data_loader.REJECTED_MATCHES`).
- **Date-ranged CIK spans.** Corporate reorganizations split a filing history
  across CIKs. A plain override *replaces* rather than extends, trading recent
  years for old ones: DIS −7.5yr, CI −7.8yr, LIN −7.8yr, MDT −11.4yr. **24
  tickers** carry non-overlapping `(cik, valid_from, valid_to)` spans so both
  halves survive (`src/data_loader.CIK_SPANS`).
- **Three outright overrides** where the map points at the wrong filer
  entirely, including ticker *reuse* — MMI now returns Marcus & Millichap
  rather than Motorola Mobility (`src/data_loader.CIK_OVERRIDES`).

> **Two figures below are not reproducible from a committed artifact.** The
> reported collapse in no-history tickers, **60.4% → 12.8%**, and the **31
> tickers still unrecoverable**, are emitted to console by
> `scripts/recover_ciks.py` (`RESULT of ... NO_CIK tickers` block, lines
> 314–322) and written to `data/unresolved_tickers.csv`, which is gitignored.
> The script that computes them is committed; the run log is not. A reader
> cannot currently verify these two numbers without re-running the recovery
> against EDGAR. Every other number in this document can be checked against a
> tracked file.

**What survivorship correction bought.** In the final panel, names that were
eventually removed from the index beat their year-ago quarter markedly less
often than names that never left:

| group | labelled rows | positive rate |
|---|---:|---:|
| never removed | 17,314 | **61.7%** |
| removed at some point | 4,439 | **54.8%** |
| all | 21,753 | 60.3% |

A **6.9-point gap**. A survivors-only panel would have shown 61.7% and taught a
model an optimism the real index never had (`data/eps_panel.parquet`,
`is_removed_name` × `label_yoy`).

---

## 4. The label, and what it does and does not mean

```
label_yoy = 1  if  eps_diluted > eps_year_ago_adjusted   (strictly greater)
          = 0  otherwise
          = null  where the comparison cannot be made honestly
```

A flat quarter is not a beat.

**Why year-over-year rather than consensus surprise.** The obvious alternative
label is "did EPS beat the analyst consensus". It was rejected for three
reasons. Point-in-time consensus is not freely available — vendor estimate
histories are licensed, and freely scrapeable versions give the *current*
consensus, not the consensus as it stood the day before the filing, which would
embed an invisible look-ahead. Consensus is itself a forecast, so predicting it
means modelling analyst behaviour on top of company fundamentals. And a YoY
comparison is reconstructible from the *same primary source* as the EPS figure,
so the label inherits the point-in-time guarantees above instead of depending on
a second source with its own revision history.

**What that changes about interpretation.** A YoY beat is an easier and more
autocorrelated target than a consensus surprise, and the baseline is
correspondingly high. **This experiment predicts the direction of earnings
growth, not market-relevant surprise.** A company that grows earnings every
quarter is highly predictable here and carries no information the market did
not already have. Nothing in this document supports a claim about returns.

---

## 5. Model selection, and a correction worth more than the model

Ten features, all derived only from quarters strictly earlier than the row being
predicted: `eps_growth_yoy_lag_1..4`, `label_lag_1..4`, `growth_streak`,
`quarters_available`. The leakage rule is enforced structurally — lags are
attached by an explicit merge on a rank derived from `prediction_date` within
ticker, never a positional shift and never on `period_end` — asserted on the
built file, and the assertion is fire-tested against injected violations on
every run (`scripts/build_features_v1.py`).

Two columns are **excluded as target identities**: `eps_growth_yoy` and
`growth_acceleration`. `label_yoy` is 1 exactly when
`eps_diluted > eps_year_ago_adjusted`, and `eps_growth_yoy` is that same
difference signed and scaled, so `sign(eps_growth_yoy)` *is* the label —
verified 21,721 of 21,721 rows, zero mismatches. This is not a look-ahead; both
are public on `prediction_date`. It is an identity, and a model given either
scores ~100% having learned nothing. The exclusion is asserted at runtime
against the built matrix (`scripts/train_walkforward.py`).

Selection used expanding-window walk-forward inside 2011–2021: train 2011..Y−1,
validate Y, for Y = 2015…2021.

**The finding.** In the first pass, LightGBM appeared to beat logistic
regression by **0.030 log-loss** (0.6192 vs 0.6494). After two structural
corrections that gap fell to **0.0006**. Essentially the entire apparent
advantage was measurement error, in two parts:

1. **Early-stopping leakage.** LightGBM early-stopped on the very fold it was
   scored on, choosing its stopping iteration using the answer. Holding out the
   last training year as an *inner* validation set instead cost it **+0.0083**
   log-loss (0.6192 → 0.6275).
2. **An unscaled feature.** `eps_growth_yoy_lag_*` reach 4.7e7 with σ ≈ 4.2e5,
   so standardizing collapsed nearly every row to a spike near zero and the
   linear model could not use them at all. Rank-transforming them — fitted on
   the training window only — gained logistic **0.0213** log-loss
   (0.6494 → 0.6281) and raised mean |coefficient| on those four features
   **5.0×**. `eps_growth_yoy_lag_4` went from −0.020 to −0.394 and became the
   largest coefficient in the model.

Neither correction was a choice made against fold results; both would have been
required whatever the numbers had been. The lesson generalises: a tree model
beating a linear one by a margin like this is worth suspecting before it is
worth believing, because the two most likely explanations — a leaky validation
protocol, and a feature the linear model cannot physically use — both favour the
tree for reasons that have nothing to do with the data.

| variant | mean log-loss | sd |
|---|---:|---:|
| constant | 0.6766 | 0.0438 |
| `logistic_raw` (biased) | 0.6494 | 0.0384 |
| `lightgbm_outer_es` (biased) | 0.6192 | 0.0422 |
| **`logistic_rank`** | **0.6281** | 0.0425 |
| **`lightgbm_inner_es`** | **0.6275** | 0.0460 |

*Source: `data/walkforward_results.csv`.*

**Why `logistic_rank` was chosen.** Not on the log-loss margin. Paired by fold,
the difference from `lightgbm_inner_es` is **0.0006** with standard error
0.0052 — **t = 0.12 on 6 df**, LightGBM taking 4 of 7 folds. That is a coin
flip. The tie-breakers are properties that hold independently of these folds:
`logistic_rank` is deterministic, carries no early-stopping iteration forward,
needs no inner split (so it trains on the year closest to the fold it must
predict, which LightGBM gives up), has lower fold-to-fold spread, and wins 2020
— the regime break — by the largest single-fold margin either model achieves.
Choosing between two indistinguishable models on parsimony grounds is a
judgement, and is recorded as one
(`config/frozen_model.json`, `model._selection_rationale`).

Both corrected models beat the constant on log-loss in **7 of 7** folds.

---

## 6. Pre-registration

The full specification — model class with estimator parameters written out
explicitly, the ten features in order, the rank-transform spec, the imputation
rule, the training-window definition, the fold boundaries, the success
criterion, and pinned library versions — is committed in
`config/frozen_model.json`, **before the holdout was read**.

| commit | what |
|---|---|
| `8dcf0c6` | selection code (`scripts/train_walkforward.py`) that produced the numbers in §5 |
| `03068a2` | freeze, revision 1 — `logistic_rank`, success criterion, fit-once protocol |
| `1f28de2` | amendment, revision 2 — fit-once → expanding walk-forward |
| `737ff87` | amendment, revision 3 — fold 4 scores the remainder; the protocol executed |
| `4e2dbb7` | the holdout run and its results |

Both amendments were made **before** any holdout row was read, each as its own
commit so the timestamp evidences it as pre-hoc, and revisions 1 and 2 are
retained in the config's `amendments` array rather than overwritten.

- **Revision 2** changed only *when* the model is refit: fit-once measures
  signal decay rather than the research question, walk-forward matches the
  intended deployment protocol, and a model frozen at 2021 would conflate
  staleness with signal failure across the 2022 regime break.
- **Revision 3** widened fold 4 to all remaining rows. The literal form first
  considered — "fold 4 predicts `period_end` in 2025" — was checked against the
  data and **rejected**: it is not a superset of `prediction_year` 2025 and
  would have orphaned 81 rows (75 labelled) into no fold at all. The rejected
  alternative and its arithmetic are recorded in the config.

**Success criterion, fixed in advance:** log-loss **below** and accuracy
**above** that fold's *own* constant baseline; evaluated per fold and pooled;
log-loss primary if the two disagree. The constant baseline predicts class 1
always at the fold's own **training** positive rate — never the rate of the year
it predicts, which would score the baseline using the answer. Accuracy is
compared against each fold's own constant, explicitly **not** against 60.3%,
since the holdout base rate was unknown at freeze time and the validation folds
had ranged 46.5%–75.1%.

---

## 7. Holdout result

Executed once, at `737ff87`, reading the config rather than restating it. All
four pre-registered run-time assertions passed: every fold trains strictly
before the first row it predicts, no fold trains on a row it predicts, the four
prediction sets are disjoint and cover all 5,898 holdout rows, and a fresh
unfitted pipeline is built per fold.

| fold | predicts | n pred | n lab | actual pos | model ll | const ll | Δ | model acc | const acc | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2022 | 1454 | 1357 | 55.5% | 0.6535 | 0.6935 | −0.0400 | 63.7% | 55.5% | +8.18 |
| 2 | 2023 | 1448 | 1359 | 54.7% | 0.6426 | 0.6961 | −0.0535 | 62.9% | 54.7% | +8.24 |
| 3 | 2024 | 1461 | 1365 | 60.0% | 0.6351 | 0.6730 | −0.0379 | 66.7% | 60.0% | +6.74 |
| 4 | 2025+ | 1535 | 1440 | 61.5% | 0.6300 | 0.6667 | −0.0367 | 65.7% | 61.5% | +4.17 |

**Pooled** over 5,521 scored rows, actual positive rate 58.0%:

| | model | constant | delta |
|---|---:|---:|---:|
| log-loss | **0.6402** | 0.6821 | **−0.0419** |
| accuracy | **64.8%** | 58.0% | **+6.79pp** |
| Brier | 0.2236 | | |

**Verdict: PASS on 4 of 4 folds and pooled, on both conditions.** The two
conditions never disagreed, so the log-loss-primary rule was not invoked.
Holdout log-loss (0.6402) sits close to the selection-time walk-forward mean
(0.6281).

*Source: `data/holdout_results.csv`, `data/holdout_predictions.csv` (5,898
rows), `scripts/run_holdout.py`.*

---

## 8. Limitations

**The edge concentrates in the names where the baseline is weakest.** Per
ticker on the pooled holdout, `corr(lift over always-predict-1, per-ticker
positive rate)` = **−0.611**, against **−0.607** in validation. Part of that is
arithmetic — lift *is* accuracy minus positive rate — but it held out almost
exactly, and the direction is clear: the model adds most where always-predict-1
is weak, and little on reliable growers, where most of the headline accuracy is
the base rate rather than the model.
`corr(accuracy, positive rate)` = **+0.463** (validation: +0.540). Ticker
identity is not in the feature set, so nothing is memorised outright, but
`growth_streak` and `label_lag_*` are company-persistence proxies that reach the
same place indirectly. *(Computed from `data/holdout_predictions.csv`; the
per-ticker block in `scripts/run_holdout.py` reports nothing at its ≥20-row
threshold because a four-year holdout gives at most 17 rows per ticker.)*

**Calibration degrades exactly where the model is most confident.** Pooled
holdout, ECE **0.0378**, mean predicted 0.5857 vs actual 0.5798:

| decile | predicted | actual | gap |
|---|---:|---:|---:|
| 1 | 0.278 | 0.325 | +0.048 |
| 8 | 0.723 | 0.708 | −0.015 |
| 9 | 0.758 | 0.710 | −0.048 |
| 10 | **0.824** | **0.736** | **−0.088** |

The top three deciles are effectively flat — 0.708, 0.710, 0.736 realized
against 0.723, 0.758, 0.824 predicted. The model is **overconfident precisely
where it is most confident**, and the ordering between deciles 8–10 carries
almost no information. Any application that ranks on predicted probability and
acts on the top slice is acting on the least reliable part of the distribution.
Monotone in 8 of 9 decile steps overall; top minus bottom +0.410
(`data/holdout_calibration.csv`).

**Accuracy lift is largest in the years the constant was weakest.** Fold 1
(+8.18pp) and fold 2 (+8.24pp) are the years with the lowest actual positive
rates (55.5%, 54.7%); fold 4 (+4.17pp) has the highest (61.5%). Accuracy
improvement over a constant baseline is partly a function of how far that
year's base rate sits from the model's operating point, not purely of skill.
This is why log-loss was pre-registered as primary
(`data/holdout_results.csv`).

**The label is not consensus surprise.** As set out in §4, this predicts the
direction of earnings growth. A predictable grower is easy here and carries no
information the market lacked. Nothing here measures surprise.

**Known data gaps.** 31 tickers remain unrecoverable (see the caveat in §3 —
this figure is not reproducible from a committed artifact). **1,823 of 23,576
panel rows carry null labels** — 834 with no year-ago match within ±45 days,
989 with a split that could not be pinned to one side of its window
(`data/eps_panel.parquet`). Those rows are dropped from training and scoring
rather than imputed. The membership reconstruction depends on a single pinned
Wikipedia revision, which is a genuine single point of failure.

---

## 9. What this does not show

- **No returns.** No price data was used anywhere in this study. Nothing here
  says a portfolio built on these predictions would have made money.
- **No costs.** No spreads, commissions, market impact, or borrow.
- **No market-relative claim.** The label is a YoY earnings comparison, not a
  surprise relative to expectations. A correct prediction is not evidence of
  information the market did not have.
- **No Task B.** No ranking, portfolio construction, or position sizing has been
  evaluated. The calibration finding above is a caution for any such work, not a
  result from it.
- **One holdout, now spent.** The 2022–2025 period has been read. Any further
  model iteration measured against it would no longer be an out-of-sample test.

The defensible claim is narrow: **on 5,521 out-of-sample quarters, prior
earnings history predicted the direction of the next year-over-year EPS
comparison better than the base rate, by 0.042 log-loss and 6.8 accuracy
points, under a protocol fixed in advance.**
