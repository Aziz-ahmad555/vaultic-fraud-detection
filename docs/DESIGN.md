# The Evidence Room — dashboard design

A fraud investigator's console for Vaultic. The user is an analyst working through a
queue of alerts. Every screen has to answer two questions: *what is the evidence?* and
*how sure are we?* It is a working tool, built like a well-kept case file: ruled paper,
labelled exhibits, a clear chain of evidence. It is not a showroom.

Data: until the research pipeline produces real scores, every screen runs on **synthetic
mock data**. The mock uses the exact schemas of the view-prediction table (D42) and the
decision engine (D43), and every page says it is mock data (decision D47).

---

## 1. Tokens

### Core palette (5 named colours)

| Token | Light | Dark | Use |
|---|---|---|---|
| **Ledger** | `#E6EBF0` | `#2B3643` (vault steel) | page background: cool grey ledger paper; dark is steel blue-grey, not near-black |
| **Sheet** | `#F4F6F9` | `#34414F` | panes and tables (the "paper" on the desk) |
| **Ink** | `#17212B` | `#E9EEF3` | primary text, primary buttons, focus ring |
| **Graphite** | `#4B5866` | `#B3C0CC` | secondary text, input borders, axes |
| **Rule** | `#B9C3CD` | `#55657A` | hairline rules and grid lines (decorative only) |

Checked contrast (WCAG 2.1):
- Ink on Ledger: 13.6 (light) / 10.5 (dark).
- Graphite on Ledger: 6.1 / 6.6. On Sheet: 6.7 / 5.6.
- Rules are decorative. Interactive boundaries use Graphite (≥ 3:1).

### View colours: one per evidence view, used everywhere

The same colour marks a view in the braid, charts, reason codes, graph edges and legends.

| View | Light | Dark | Contrast on Sheet (light / dark) |
|---|---|---|---|
| Tabular | `#1F5FA8` blue | `#86B6EE` | 5.95 / 4.93 |
| Behavioral | `#9A5B00` amber | `#E8A852` | 5.01 / 5.04 |
| Temporal | `#00735C` bluish green | `#5ACBAE` | 5.38 / 5.25 |
| Relational | `#8B3D83` reddish purple | `#DDA0D6` | 6.27 / 5.00 |
| Anomaly | `#566400` olive | `#B9C657` | 6.03 / 5.60 |

- The hues follow the Okabe–Ito colour-blind-safe set, darkened (light theme) or lightened
  (dark theme) until each passes 4.5:1 as text.
- Colour is never the only cue: braid strands are always in the same order (tabular on top
  to anomaly at the bottom), and every view is also named in text.

### Risk scale: a single hue (vermilion)

Light `#FBEAE3 → #F2BCA5 → #E2875F → #C4501D → #8A3210`; dark `#4A3B37 → … → #F6B595`.
- Used only for risk (gauge, risk cells, heat-map).
- The risk number is always printed next to its colour.
- Vermilion is used for nothing else.

### Decisions: a restrained blue-grey ramp, from Allow to Block

| Decision | Light fill / text | Dark fill / text | Text contrast (light / dark) |
|---|---|---|---|
| Allow | `#E3E8ED` / Ink (outlined) | `#3B4959` / Ink | 13.2 / 7.9 |
| Monitor | `#CBD4DD` / Ink | `#4D5D6F` / Ink | 10.9 / 5.8 |
| Step-up | `#A3B2C1` / Ink | `#5E7186` / white | 7.5 / 5.0 |
| Hold | `#5D7187` / white | `#A9B9C9` / `#17212B` | 5.0 / 8.1 |
| Block | `#1E2C3A` / white | `#E3EAF1` / `#17212B` | 14.2 / 13.4 |

Decisions get darker as they get more severe. Severity reads from lightness and the word,
never from a red/green pair.

### Type

- **Public Sans**: all UI. Numbers use `font-variant-numeric: tabular-nums`.
- **Newsreader**: case narratives only, so the narrative reads as a written finding and not
  as UI chrome.

| Step | Size / line height | Use |
|---|---|---|
| xs | 12 / 16 | axis ticks, footnotes |
| sm | 13 / 18 | table cells, badges |
| base | 14 / 20 | body, controls |
| md | 16 / 24 | pane titles |
| lg | 20 / 28 | page titles |
| xl | 28 / 34 | single key figures (overview) |
| narrative | 18 / 29 (Newsreader) | case narrative |

Weights: 400 and 600 only. Labels are sentence case: no ALL-CAPS, no letter-spaced
eyebrows, no monospace data labels.

### Space, shape, depth

- **Spacing** on a 4 px base: 4, 8, 12, 16, 24, 32, 48.
- **Radius** 2 px for controls and badges; 0 for panes, which are ruled like paper.
- **Depth** comes from Sheet against Ledger plus 1 px rules. Shadows appear only on
  floating layers (palette, popovers, toasts), and differ per layer.
- **Focus**: a 2 px Ink outline with 2 px offset on every interactive element, always visible.

### Motion

- Only in response to the user: 120 ms ease-out for panel and row state changes.
- **One** orchestrated moment: the live stream. New transactions settle into the top of the
  list, and the braid strands draw in left to right (300 ms, staggered by view).
- `prefers-reduced-motion: reduce` turns every transition off; the stream then updates in
  place.

---

## 2. Signature element: the evidence braid

For each transaction:
- **Five strands**, one per view, in fixed order: tabular, behavioral, temporal, relational,
  anomaly.
- **Thickness** = that view's MVAF gate weight. The weights of the available views sum to 1,
  so the braid always has the same total thickness.
- **A missing view** keeps its lane: the strand is drawn as a thin dashed gap with no fill,
  so missing evidence is *visible as missing* and never as zero (rule 11).
- **Strand end** (right cap) = that view's fraud probability, shown as the view colour's
  lightness, from pale (low) to full (high).
- **Inline braid** (tables): 72 × 20 px, no labels.
- **Case braid**: full width with view names and weights. Hovering or focusing a strand
  shows that view's reasons; the arrow keys move between strands.

```
tabular     ═══════════════════════════════■   w .31  p .82
behavioral  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━■   w .24  p .77
temporal    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    not available (no history)
relational  ████████████████████████████████■   w .38  p .91
anomaly     ───────────────────────────────▫   w .07  p .40
```

---

## 3. Layout and wireframes

### Shell

A global bar holds:
- the **as-of time machine**;
- the label-delay setting;
- search / command palette (Ctrl+K);
- the theme toggle and role switch;
- the mock-data notice.

The left rail holds the navigation.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Vaultic · Evidence Room   As of [◂ Day 141 · 14:00 ▸] [⏵]   Label delay L [30 d]      │
│                           Synthetic mock data            [Search or jump… Ctrl+K]  ◐  │
├────────┬─────────────────────────────────────────────────────────────────────────────┤
│Overview│                                                                             │
│Live    │                         page                                                │
│Queue   │                                                                             │
│Cases   │                                                                             │
│Network │                                                                             │
│Drift   │                                                                             │
│Lab     │                                                                             │
│Simulate│                                                                             │
│Reports │                                                                             │
└────────┴─────────────────────────────────────────────────────────────────────────────┘
```

The **label delay curtain**: wherever labels appear, transactions newer than as-of − L sit
behind a hatched band headed "Label not known yet (L = 30 days)". Their outcome is never
shown, even though the mock knows it.

### 1 Overview

```
┌ Today, day 141 ───────────────────────────────────────────────────────────────────────┐
│ Transactions 2,384   Alerts 61   Fraud value stopped $18,240   Estimated loss $6,910  │
│ Queue 37 waiting                                                                       │
├──────────────────────────────────────────┬───────────────────────────────────────────┤
│ Decisions today (stacked bar, decision   │ Cost breakdown (missed fraud | false       │
│ ramp)  Allow ▇▇▇▇▇▇▇ Monitor ▇ …         │ positives | reviews | step-ups), $ and %   │
├──────────────────────────────────────────┴───────────────────────────────────────────┤
│ Risk by hour (bars in risk hue, counts printed)        Views available today (5 bars) │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 2 Live monitor (the one orchestrated motion moment)

```
┌ Replay from as-of   [⏸ Pause] Speed (•)×1 ( )×100 ( )×1000   Filter [decision ▾] [Search]┐
├────────┬────────┬──────┬──────────┬───────────┬────────────────────────────┬──────────┤
│ Time   │ Amount │ Risk │ Set      │ Decision  │ Top reason                 │ Evidence │
│ 14:02  │ $940.00│  87  │ fraud    │ Block     │ Amount 25.0x usual (+20)   │ ≡≡≡ ≡    │
│ 14:02  │  $12.50│   4  │ legit    │ Allow     │ Card seen 41 times …       │ ≡≡ ≡≡    │
└────────┴────────┴──────┴──────────┴───────────┴────────────────────────────┴──────────┘
```

### 3 Case investigation: three panes (queue | case | evidence), keyboard-first

```
┌ Queue (J/K) ──────┬ Case 3115204 ─────────────────────────────┬ Evidence ─────────────┐
│ ▸ 3115204  87 Blk │ Risk [gauge 87]  Decision Block            │ Customer timeline     │
│   3115188  74 Hld │ Conformal set: fraud    Disagreement 0.08  │ ·  ·· · ··· ·   ◆     │
│   3115170  69 Hld │ ┌ Evidence braid (large, hover strand) ──┐ │ Mini network          │
│   …               │ └───────────────────────────────────────┘ │  (card)─(uid)─(device)│
│                   │ Reasons (signed points, view colour bar)   │ Counterfactual        │
│                   │ +20 Amount is 25.0x usual (behavioral)     │ Amount  [====|----]   │
│                   │ +12 Unusual value in masked counter C13    │ Device  [known ▾]     │
│                   │  −6 Card seen 41 times before              │ [lock] Card, address, │
│                   │ Narrative (Newsreader)                     │   history: locked     │
│                   │ [Mark as fraud F] [Mark as legit L] [U]    │ Risk would be 31      │
│                   │ [Print or save as PDF]                     │                       │
└───────────────────┴────────────────────────────────────────────┴───────────────────────┘
```

### 4 Fraud network

```
┌ Entity network, as of day 141 · 14:00 ──────────────────────────────── Legend: views ─┐
│                                                                                       │
│        (device)━━(uid)━━(card)        rings outlined in risk hue, labelled            │
│            ┃                ┃         "Ring 3: 5 customers, 4 confirmed frauds"       │
│        (email)          (address)                                                    │
│                                                                                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Time  [|────────────────●───────]  day 128 … 150   Only edges that existed by then    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 5 Analyst queue: policies side by side

```
┌ Today's K = [100] cases                        Compare: [R1] [R2] [R3] [R4]             ┐
├ R1 top-K risk ───┬ R2 expected loss ─┬ R3 + uncertainty ─┬ R4 + disagreement ──────────┤
│ fraud value $…   │ fraud value $…    │ fraud value $…    │ fraud value $…               │
│ list (braid)     │ list              │ list              │ list                         │
└──────────────────┴───────────────────┴───────────────────┴─────────────────────────────┘
```

### 6 Drift monitor

```
┌ PSI per feature (bars, 0.2 alert rule)  ┬ Score drift / disagreement drift (lines)     ┐
├──────────────────────────────────────────┴───────────────────────────────────────────┤
│ Timeline: ADWIN ● Page-Hinkley ◆ alarms · injected drift ▮ start · retrains ▲ (accepted│
│ / rejected with CI)                     Label delay L: (0) (7) (•30) (60)            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Champion–challenger history table (day, trigger, PR-AUC both, gain CI, cost, outcome)│
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 7 Model lab

```
┌ Runs [x] B5  [x] MVAF  [ ] F3 …   ┬ PR curves            ┬ Calibration (reliability) ┐
├───────────────────────────────────┴──────────────────────┴───────────────────────────┤
│ Ablation table (B5 → + behavioral → … → MVAF) with CIs │ Fusion F1–F7 vs MVAF          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Cost sensitivity heat-map: C_FP × step-up success (risk hue, values printed)          │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 8 Scenario simulator

```
┌ Inputs ───────────────────────┬ Result ───────────────────────────────────────────────┐
│ Amount        [====|-----]    │ Risk 64   Decision Hold                               │
│ Device        (new) (known)   │ Braid (live)                                          │
│ Velocity 24 h [==|-------]    │ Reasons (signed points)                               │
│ Shared entities [=|------]    │                                                       │
│ Views: [x]tab [x]beh [ ]tem … │                                                       │
└───────────────────────────────┴──────────────────────────────────────────────────────┘
```

### 9 Reports and settings

```
┌ Reports: [Print or save daily report as PDF]   ┬ Role (analyst | admin)               ┐
├───────────────────────────────────────────────┴───────────────────────────────────────┤
│ Decision thresholds (admin only; analysts see them read-only with a lock and a reason)│
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Audit log: time, actor, action, before → after                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Responsive:**
- At ≥ 1280 px all three case panes show.
- At 768–1279 px (tablet) the evidence pane folds under the case pane, and the left rail
  becomes an icon rail with labels in tooltips and the accessible name.

---

## 4. Keyboard

| Key | Action |
|---|---|
| Ctrl+K | command palette (pages, cases by id, actions) |
| J / K | next / previous case |
| F / L / U | mark as fraud / legit / unsure (case page) |
| G then O / M / Q / C / N / D / B / S / R | go to page |
| [ / ] | as-of back / forward one hour |
| ? | shortcut help |
| Esc | close the palette or dialog |

Shortcuts are ignored while typing in an input.

---

## 5. Copy

- Sentence case everywhere. Actions say what they do: "Mark as fraud", "Print or save as
  PDF", "Pause replay". No arrows or chevrons in button labels.
- Toasts confirm in the past tense: "Marked as fraud", "Threshold saved".
- Errors say what happened and how to fix it: "Couldn't load the queue. The mock API threw
  an error; reload the page."
- Empty states invite an action: "No alerts before day 128 at this as-of time. Move the
  as-of date forward to see alerts."
- Masked features keep honest wording ("Unusual value in masked counter C13"), the same
  library as `explain/reason_codes.py`.
- Never "AI", "insights", "magic", "smart", or "real-time" without a latency number.

---

## 6. Check against generic-AI tells

| Tell | Status |
|---|---|
| Cream background with a terracotta accent | Avoided: cool grey Ledger, no warm accent; vermilion is reserved for risk |
| Black background with a neon accent | Avoided: dark theme is steel blue-grey, colours are muted and AA-checked |
| Identical rounded cards with one shadow | Avoided: ruled panes, 0 radius, no card grid; shadows only on floating layers |
| Gradient washes, glass, glow | None |
| ALL-CAPS eyebrow labels, letter-spaced kickers | None; sentence case |
| Monospace for data labels | None; Public Sans with tabular numerals |
| "→" or "✨" on buttons, emoji icons | None; the lock icon for immutable fields is an inline SVG |
| Vague AI copy ("insights", "smart") | None; copy names evidence, numbers and actions |
| Decorative animation on load | None; motion only on user action, plus the live stream |
| Fake precision | Probabilities as 0–100 integers; CIs shown where they exist; masked data says it is masked; mock data says it is mock |
| Red/green as the only signal | Decisions use a lightness ramp and words; views use fixed order and names |
