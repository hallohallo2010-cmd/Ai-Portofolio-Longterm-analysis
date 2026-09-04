# Handoff

Orientation for a session starting cold. Read `RESULTS.md` for the findings;
this file is about where things stand and what to do next.

## Current state

**Task A is complete and its holdout is spent.** A frozen logistic model was
tested once on 2022–2025 at commit `4e2dbb7` and passed both pre-registered
conditions on all four folds and pooled: log-loss 0.6402 against a 0.6821
constant baseline, accuracy 64.8% against 58.0%, over 5,521 labelled quarters.

`RESULTS.md` is the writeup — research question, data construction,
survivorship handling, model selection, pre-registration, the holdout table,
limitations, and what the work does not show. Every figure in it cites a
tracked file.

## Key files

| file | what it is |
|---|---|
| `data/eps_panel.parquet` | The panel. 23,576 rows, 716 tickers, one row per `(ticker, period_end)`, `period_end` 2011-01-01 → 2025-12-31. Point-in-time diluted EPS as **first filed**, the `label_yoy` target, and label provenance (year-ago match, split adjustment). Treat as read-only. |
| `config/frozen_model.json` | The pre-registration, revision 3. Model class and estimator params written out explicitly, the ten features in order, rank-transform spec, imputation rule, fold boundaries, success criterion, pinned library versions. Revisions 1 and 2 are retained in its `amendments` array, all marked PRE-HOC. |
| `src/data_loader.py` | Every EDGAR call. CIK resolution (overrides, date-ranged spans, name-recovered CIKs), the rate limiter, the User-Agent. Scripts must not build EDGAR URLs or sleep for throttling themselves. |
| `scripts/run_holdout.py` | The holdout runner. Reads the config rather than restating it, enforces four run-time assertions, writes `data/holdout_{predictions,results,calibration}.csv`. The template for how a scored evaluation should be structured. |

Also: `scripts/build_eps_panel.py` (Phase 0), `scripts/build_features_v1.py`
(Phase 1 features + leakage assertion), `scripts/train_walkforward.py`
(selection).

## The frozen model

`logistic_rank`: a scikit-learn `Pipeline` of `QuantileTransformer`
(normal output, 1000 quantiles) applied to the four `eps_growth_yoy_lag_*`
columns only → median `SimpleImputer` → `StandardScaler` →
`LogisticRegression` (l2, C=1.0, lbfgs, max_iter 2000, seed 42), threshold 0.5.
Ten features, all from strictly earlier quarters: `eps_growth_yoy_lag_1..4`,
`label_lag_1..4`, `growth_streak`, `quarters_available`. `eps_growth_yoy` and
`growth_acceleration` are permanently excluded as **target identities** —
`sign(eps_growth_yoy)` *is* the label — and the exclusion is asserted at
runtime. The whole pipeline is refit per fold on that fold's training window
only; nothing crosses a fold boundary.

**The 2022–2025 holdout cannot be reused.** It has been read. Any further model
iteration measured against it is no longer an out-of-sample test, and no amount
of care in the code changes that. If you need a fresh test, it has to come from
data that did not exist when the model was frozen — which is what the forward
test below is for.

## What's next: the live forward test

Run the frozen config forward on quarters nobody has seen yet.

Each earnings season: pull fresh filings from EDGAR, build features for the new
quarters, generate predictions with the frozen config **unchanged**, and commit
those predictions with a timestamp **before the outcomes are known**. Score them
once the year-ago comparisons resolve. The commit timestamp is the whole point
— it is what makes the test real rather than a re-description of history.

**Two quarters closes it.** That is enough to say whether the holdout result
survives contact with genuinely unseen data.

**One run is on record.** `data/live_predictions_2026-09-04.csv` — 474
predictions at the 2026-09-04 cut, committed before any of those quarters was
filed. 519 members at the cut, 45 skipped (32 no usable history, 2 under four
quarters, 11 silent past 220 days), coverage 97.3%. Predicted positive rate
80.6% at threshold 0.5 against a 60.7% training positive rate, well above the
68.2% the model predicted on the holdout. Treat that gap as selection before
signal: the forward universe is 474 currently-active constituents, and in the
panel the never-removed group runs 61.7% positive against 54.8% for names that
were eventually removed. One more quarterly run closes the test.

Do not tune anything during the forward test. If the model needs changing, that
is a new experiment with a new pre-registration, not an amendment to this one.

### The forward runner: `scripts/run_forward.py`

`scripts/run_forward.py` is that incremental path. It predicts and **never
scores**; scoring is a separate script and a separate commit, still to be
written.

**What it predicts is not the newly filed quarter.** This is the one design point
worth understanding before reading the code. `label_yoy` compares a quarter
against the same quarter a year earlier, and the year-ago quarter is already in
the panel — so a label becomes resolvable the *instant* a new filing lands.
There is no waiting period to hide behind. Predicting a quarter you have just
downloaded is predicting something already determined, by a process that has
already touched the number determining it.

So the runner predicts each member's **next, as-yet-unfiled quarter**. All ten
frozen features are lagged and none need the quarter's own EPS, so they can be
built before the filing exists. Fresh filings are still fetched — they extend
the history and the training window — but they are inputs, not targets. This is
what makes the run-time assertion satisfiable rather than decorative: a predicted
row has no `eps_diluted`, and STAGE 6 proves it rather than assuming it.

Four refusals, each of which aborts without writing:

- a predicted row carrying an EPS figure or a label;
- a predicted ticker with a filing dated **after the cut** — which is what a
  backdated `--as-of` looks like from the inside;
- a cut date that already has a prediction file (one cut, one file — re-rolling
  a prediction is how a forward test becomes a search over predictions);
- fewer than 50% of eligible members ending up predictable, which at a real cut
  means the EDGAR fetch under-delivered rather than that the index went quiet.

Two further guards worth knowing about. Tickers silent for more than 220 days
(≈ one quarter + the panel's 120-day filing-lag cap) are skipped as having
missed a reporting cycle — without that, the pinned revision's stale "current"
list would produce confident predictions for companies acquired after 2025-05-27
that will never report again, and the rows that *could* be scored would be the
survivors of continued filing. And the panel is opened read-only: its sha256 is
taken at start and re-checked at exit.

**Membership is frozen at the pinned revision — decided, not defaulted.** The
gate stays on **Wikipedia revision 1292523673** (2025-05-27) and the live page is
never consulted as a fallback. The forward universe is therefore the index as it
stood on that date, and it drifts from the real index at roughly the real
turnover rate, **~20-25 names a year**, compounding every quarter the test runs.
Every prediction file states this in its header. The run also refuses outright
past `MEMBERSHIP_OPEN_END` (2026-12-31), where every membership interval closes
and the gate would silently return an empty universe.

## Before anything touches EDGAR

`SEC_CONTACT_EMAIL` in `src/data_loader.py` is set to a real address, so EDGAR
calls work. `sec_headers()` still aborts if it is ever put back to a
placeholder. Throttling is 0.5 s per request, handled centrally; a full panel
rebuild is roughly 900 calls and caches to `data/fact_cache/`, a forward run
about 500 and caches to `data/forward_cache_<cut>/`.

**`lxml` matters more than its one line in `requirements.txt` suggests.**
`pandas.read_html` needs it to parse the pinned Wikipedia revision, so without
it `src/index_membership.py` fails on the first membership call — it was missing
from the requirements until the first forward run hit exactly that. Anything
that fetches needs it; nothing offline does.

Offline work needs none of this: the panel, the features, and every result file
are committed, so `build_features_v1.py`, `train_walkforward.py` and
`run_holdout.py` all run without network access.

## sklearn 1.10 will break the frozen config

`config/frozen_model.json` writes `"penalty": "l2"` explicitly into the
estimator params. scikit-learn deprecated `penalty` in 1.8 and **removes it in
1.10**, so on 1.10 the frozen pipeline stops constructing and
`scripts/run_holdout.py` becomes unrunnable — as does `scripts/run_forward.py`,
which builds the pipeline through the same function. Both already emit the
`FutureWarning` on 1.9.0, the pinned version.

This is a known future break and is **not to be fixed by changing the config.**
The config's own note says the estimator params were written out explicitly "so
a future default change cannot silently alter the frozen model"; a parameter
*removal* defeats that, and editing the file to route around it would edit a
pre-registration after its holdout was read, which is the one thing the freeze
forbids. The holdout result stands on the committed
`data/holdout_{predictions,results,calibration}.csv` regardless of whether the
run can be reproduced on a future library.

If reproduction is needed later, pin the environment (`scikit-learn==1.9.0`,
per the config's `environment` block) rather than touch the config.

## Known gaps

- **No price data anywhere.** Nothing in this repository supports a claim about
  returns. The label is an earnings-growth comparison, not a market surprise.
- **Task B has not started.** No ranking, portfolio construction, or position
  sizing. Note the §8 calibration finding first: the model is overconfident
  exactly where it is most confident (decile 10 predicts 0.824, realizes 0.736),
  and the ordering across the top three deciles carries almost no information.
  That matters for anything that ranks on predicted probability.
- **The recovery run outputs are untracked and unrecoverable.**
  `data/unresolved_tickers.csv` and the `scripts/recover_ciks.py` console log do
  not exist on disk or in history. The script is committed; its output is not,
  and it cannot be regenerated offline. `RESULTS.md` deliberately omits the two
  figures that depended on them. If you re-run the recovery with EDGAR access,
  commit the log and those figures become citable.
- **EPS coverage stops early for some active constituents.** The panel and the
  forward run both draw on `us-gaap/EarningsPerShareDiluted`, and for a number
  of names that concept simply stops carrying quarterly-duration facts while the
  company goes on filing: VRSN ends 2013-09-30 (9 panel rows), GD 2019-09-29,
  KIM and WEC 2020-12-31, BKR 2022-09-30. Verified against EDGAR at the
  2026-09-04 cut — the fetch returns the full history and it genuinely ends
  there, so this is a tagging/concept gap, not a fetch failure. It predates the
  forward test and is part of why the panel holds 716 tickers. In the forward
  run these names are caught by the 220-day silence guard (11 skipped at that
  cut), which is the right outcome for the wrong reason: they are excluded as
  "stopped reporting" when they have actually stopped reporting *under this
  concept*. Recovering them would mean a second concept with its own
  point-in-time properties, which is a data-construction change, not a patch.
- **`data/dropped_periods.csv` is likewise untracked** — the drop log from the
  panel build is gitignored and absent.
