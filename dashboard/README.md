# Evidence Room (Vaultic dashboard)

A fraud investigator's console for Vaultic. Design brief, tokens and wireframes:
[`docs/DESIGN.md`](../docs/DESIGN.md). Scope decision: D47 in `research/decisions.md`.

**Every number on these screens is synthetic mock data** until the research pipeline produces
real scores. The top bar and the model lab say so.

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173. Other scripts: `npm test` (Vitest), `npm run typecheck`,
`npm run build`.

## Data: one adapter

All data goes through the `Api` interface in `src/api/types.ts`. Its row schemas mirror the
Python side:
- `ViewPredictionRow` mirrors the D42 view table;
- `DecisionRecord` and `CostModel` mirror the D43, D45 and D46 decision engine;
- `Reason` mirrors the D38 reason codes.

`src/api/index.ts` is the only file that picks the implementation. To use the FastAPI backend,
write `createHttpApi()` implementing the same interface and change that one line.

The mock (`src/api/mock`) is seeded and generated in the browser. It enforces the as-of rules in
the data layer, and `src/api/mock/mockApi.test.ts` tests them:
- nothing after the as-of time is returned;
- a label is returned only after `its_time + L`;
- network fraud counts use matured labels only;
- missing views stay null, with weight 0.

## Pages

| Page | What it shows |
|---|---|
| Overview | Today's transactions, alerts, expected fraud stopped and loss, queue, cost breakdown |
| Live monitor | Replay from the as-of time at ×1 / ×100 / ×1000, with the inline braid |
| Cases | Three-pane console: alerts (J/K), case (gauge, large braid, reasons, narrative, F/L/U, print) and evidence (timeline, network, counterfactual with locked fields) |
| Fraud network | Rings, expand an entity, time slider |
| Analyst queue | R1–R4 side by side, fraud value caught on the latest day with known labels |
| Drift monitor | PSI, score and disagreement drift, alarm timeline that moves with the label delay, champion–challenger history |
| Model lab | PR and calibration curves, ablation, fusion, cost sensitivity (mock numbers) |
| Scenario simulator | Live risk, braid and reasons as inputs change |
| Reports and settings | Print report, role, admin-only thresholds, audit log |
| Test Lab | Live scoring of one transaction by the frozen system: real held-out transactions (label hidden until revealed), custom transactions, batch CSV; parity-gated (see Running the Test Lab) |

Keyboard: Ctrl+K palette, J/K, F/L/U, G then a page letter, `[` / `]` for the as-of time, `?` for help.

## Tests

- `src/components/Braid.test.tsx`: a missing view is a dashed gap, available weights sum to 1,
  thickness follows the weight, end caps follow the score, and hovering shows reasons.
- `src/api/mock/mockApi.test.ts`: the as-of and label-delay rules, missing views, and admin-only
  thresholds.

Pages were checked in a browser in both themes, at desktop and tablet widths. Playwright is not
installed: the browser checks were done by hand with Claude's built-in browser pane, not as
automated tests.

## Running on real data

Two servers, each in its own PowerShell window, both started from the repository root (the
folder with `src/`, `services/` and `dashboard/`). The bundle files must already exist on drive E
(see the next section for how they are made).

**First time only** — install the dashboard's packages, with npm's cache on drive E:

```powershell
cd dashboard
npm ci --cache "E:\dev-cache\npm"
cd ..
```

**1. Start the demo API** (window 1; serves the final run's test period on port 8000):

```powershell
$env:PYTHONPATH = "src"
$env:VAULTIC_DEMO_BUNDLE = "E:\old work\fyp 11\fraud-detection\data\interim\demo_final_bundle.parquet"
.\.venv\Scripts\python.exe -m uvicorn services.api.demo_server:app --port 8000
```

It refuses to start unless the bundle carries the guarded export's test-data marker (D154).
Check it: `curl http://127.0.0.1:8000/api/meta` should report `"source": "IEEE-CIS test period, final run"`.

**2. Start the dashboard on the API** (window 2; Vite dev server on port 5173):

```powershell
cd dashboard
$env:VITE_DATA_SOURCE = "http"
npm run dev
```

**3. Open** <http://localhost:5173>. The top bar must say "IEEE-CIS test period, final run"
(without `VITE_DATA_SOURCE=http` it says "Synthetic mock data").

**Stop:** press `Ctrl+C` in each window. If a window is gone and a port stays busy, stop whatever
listens on the two ports:

```powershell
Get-NetTCPConnection -LocalPort 8000, 5173 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## Running the Test Lab

The Test Lab page (`/testlab`, G then T) scores one transaction live with the frozen final-run
system (MVAF over the five views, E12 decisions): every feature is computed from the stored
history strictly before the transaction, with the research pipeline's own feature code. It needs
the real-data setup above (bundle and data files on drive E) and the frozen stage-2 system cache
(`E:\dev-cache\stage2\system_dev.joblib` or `E:\dev-cache\testlab\system.joblib`; built
automatically on first start if neither exists, about 25 minutes). The caches live under `VAULTIC_CACHE_DIR` (default `E:/dev-cache`):
`testlab/` for the system and the parity results.

**1. Start the backend with the Test Lab** (window 1, repository root):

```powershell
$env:PYTHONPATH = "src"
$env:VAULTIC_DATA_DIR = "E:/old work/fyp 11/fraud-detection/data"
$env:VAULTIC_DEMO_BUNDLE = "E:\old work\fyp 11\fraud-detection\data\interim\demo_final_bundle.parquet"
$env:VAULTIC_TESTLAB = "1"
.\.venv\Scripts\python.exe -m uvicorn services.api.demo_server:app --port 8000
```

The server answers at once; the Test Lab loads in the background (history store about 4
minutes; on the very first start also the parity check, about 20 minutes). Follow it with
`curl http://127.0.0.1:8000/api/live/status`: `loading`, then `ready`, or `refused` with the
features that failed parity.

**2. Start the dashboard on the API** (window 2): `cd dashboard`, `$env:VITE_DATA_SOURCE = "http"`,
`npm run dev`, then open <http://localhost:5173/testlab>.

**Parity gate.** Before it scores anything, the Test Lab recomputes the features of 2,000 random
real test transactions online and requires them, and the live MVAF and B5 scores, to equal the
offline frozen features and the stored final-run scores within 1e-6. If they do not, the page
says so and refuses to score. The result is cached in `E:\dev-cache\testlab` and recomputed
only when the data, the feature code or the system cache change.

**Modes.**
- *Real test transaction*: pick a random held-out test transaction (fraud / legitimate / cold
  start / any); its label stays hidden until "Reveal truth". The session scoreboard counts caught,
  missed, false alarms and correct passes (an alert is an E12 decision of step-up or higher), and
  each live score shows "matches frozen run ✓" when it equals the stored final-run score.
- *Build your own transaction*: custom input, no ground truth. An existing customer's masked
  fields come from their latest earlier transaction and D1 is set so the transaction is theirs; a
  new customer gets an unseen card and D1 = 0.
- *Batch test (CSV)*: download the template (IEEE-CIS columns), upload up to 1,000 rows in the
  test period; each row is scored independently against the stored history before its own time.
  With an `isFraud` column: PR-AUC, recall at the E12 alert and at MVAF's E10 cost threshold
  (0.881), and the confusion matrix. Bad files are refused with the reason.

**Latency.** With the backend ready and the machine otherwise idle,
`python scripts/bench_testlab_latency.py` scores the same 200 test transactions (seeds 0-199) and
prints p50 / p95 of the scoring latency, the explanation and the HTTP request, and how many scores
matched the frozen run.

**Stop:** `Ctrl+C` in both windows (or the `Get-NetTCPConnection` command above).

## Real data: the final-run demo backend

The dashboard can read the EXP-200-final test period instead of the mock (labelled
"IEEE-CIS test period, final run" in the top bar):

1. On phase3-run, write the demo bundle (reproduces MVAF's final-run scores and E12 decisions,
   and stops if they differ): `python -m vaultic.reports.demo_export`
   (writes `<data>/interim/demo_final_bundle.parquet` and `.json`, kept on drive E).
2. Start the backend from the repository root:
   `VAULTIC_DEMO_BUNDLE=<path>/demo_final_bundle.parquet uvicorn services.api.demo_server:app --port 8000`
3. Start the dashboard with `VITE_DATA_SOURCE=http` (optional `VITE_API_URL`, default
   `http://127.0.0.1:8000`): `VITE_DATA_SOURCE=http npm run dev`.

Without `VITE_DATA_SOURCE=http` the synthetic mock is used, as before.

Demo notes (D122): the Overview explains why no Block decisions appear (in the cost model a Block
costs a review plus the false-positive penalty, a Hold only the review, so Block is never cheaper);
Settings has an optional **what-if cost mode** that re-chooses the thresholds on the calibration days
(147-150) for other costs and applies them to the test period, labelled on every page as
"illustrative, not the evaluated system" (needs `demo_final_calibration.parquet` next to the bundle).
The what-if mode is stateless (D128): the setting lives in the browser tab and is sent as query
parameters (`c_fp`, `c_rev`, `c_step`, `step_up_success_rate`, `legit_step_up_friction`,
`hold_success_rate`) with every request; every decision, day summary and network in the response
carries a `whatIf` flag, decisions show a What-if badge, and the banner appears only when what-if
figures were served;
the scenario simulator is labelled "Illustrative model, not the evaluated system". The as-of clock
takes its range from the data source (test days 151-182 with the backend). Reasons are view-level
(MVAF gate weight x view logit), not feature-level SHAP. The what-if simulator keeps the mock's
illustrative model, because the final-run models score recorded transactions only. Threshold
changes, feedback and the audit log live in the backend's memory for the session; drift alarms,
injected drifts and retraining events are empty because none were run on the final run.
