# Test Lab: live scoring by the frozen system

## What it is
A dashboard page (`/testlab`) and backend (`src/vaultic/live`, `services/api/testlab.py`) that score
one transaction live with the frozen final-run system: MVAF over the five views and E12 decisions
(D182-D189). Three modes: a real held-out test transaction (label hidden until "Reveal truth", with
a session scoreboard), a custom transaction (no ground truth), and a batch CSV.

## Why we built it
A live demonstration must run the evaluated system, not an approximation (D182). It also lets an
examiner see one decision, its view weights and its reasons end to end.

## How it works (simple)
A history store is built at startup. For a transaction at time t, every feature is computed from
stored transactions strictly before t, with the research pipeline's own feature code. Before it
scores anything, a parity gate recomputes 2,000 random real test transactions online and requires
the features and scores to equal the frozen offline ones; otherwise the page refuses to score and
names the failing features (D183, D189).

## Key results (with sources)
- Parity gate: 2,000 real test transactions, largest feature difference 3.0e-8 (float32 rounding),
  MVAF score 3.3e-16, B5 0.0, no failing column; tolerances features 1e-6, scores 1e-12
  (`research/decisions.md`, D183 and D189).
- Latency on the idle laptop, 200 sequential live scores: features + scoring p50 472 ms, p95 577 ms
  (max 726); explanation p50 344 ms, p95 426 ms; full HTTP request p50 825 ms, p95 973 ms; 200 of 200
  matched the frozen run (`research/decisions.md`, D182).
- The session scoreboard is a tally of the user's picks, not recall (page wording, D186).

## Limitations
- Product layer only: no research number comes from the Test Lab.
- Custom transactions fill the fields the form does not ask for by fixed rules (D184), so they are
  not real transactions and have no ground truth.
- Batch CSV metrics are for the uploaded file only, no confidence intervals, not the evaluation
  (D185).
- Runs on 127.0.0.1 only; startup builds the history store (minutes).

## Screenshots to take
1. Real mode after "Reveal truth": the result card (risk, decision, braid of view weights, reasons)
   and the "Matches frozen run ✓" badge.
2. The status banner with "Parity passed" and the latency line.
3. Batch mode with the confusion matrix and the "not the research evaluation" label.
