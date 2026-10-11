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

Keyboard: Ctrl+K palette, J/K, F/L/U, G then a page letter, `[` / `]` for the as-of time, `?` for help.

## Tests

- `src/components/Braid.test.tsx`: a missing view is a dashed gap, available weights sum to 1,
  thickness follows the weight, end caps follow the score, and hovering shows reasons.
- `src/api/mock/mockApi.test.ts`: the as-of and label-delay rules, missing views, and admin-only
  thresholds.

Pages were checked in a browser in both themes, at desktop and tablet widths. Playwright is not
installed: the browser checks were done by hand with Claude's built-in browser pane, not as
automated tests.

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

Without `VITE_DATA_SOURCE=http` the synthetic mock is used, as before. Reasons are view-level
(MVAF gate weight x view logit), not feature-level SHAP. The what-if simulator keeps the mock's
illustrative model, because the final-run models score recorded transactions only. Threshold
changes, feedback and the audit log live in the backend's memory for the session; drift alarms,
injected drifts and retraining events are empty because none were run on the final run.
