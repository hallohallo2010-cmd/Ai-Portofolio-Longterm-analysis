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

## Layout

```
scripts/build_eps_panel.py     Phase 0: the panel and the label
scripts/build_features_v1.py   Phase 1: EPS-derived features
src/data_loader.py             all SEC EDGAR access; CIK resolution
src/index_membership.py        membership reconstruction from the pinned revision
data/eps_panel.parquet         the panel (tracked)
data/features_v1.parquet       the feature table (tracked)
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
```
