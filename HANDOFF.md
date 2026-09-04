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

`src/data_loader.py:29` is `SEC_CONTACT_EMAIL = "REPLACE_ME@example.com"`.
EDGAR requires a real contact address in the User-Agent, and `sec_headers()`
(line 168) aborts on the placeholder — so **a fresh container hits this on the
first EDGAR call**. Set it to a real address before running anything that
fetches. Throttling is 0.5 s per request, handled centrally; a full panel
rebuild is roughly 900 calls and caches to `data/fact_cache/`.

Offline work needs none of this: the panel, the features, and every result file
are committed, so `build_features_v1.py`, `train_walkforward.py` and
`run_holdout.py` all run without network access.

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
- **`data/dropped_periods.csv` is likewise untracked** — the drop log from the
  panel build is gitignored and absent.
