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
All five sit at **equal perceived lightness** (OKLCH L 0.52 light / 0.79 dark) with the same
muted chroma (C 0.09), so no view looks more important than another. Only the hue differs:

| View | Hue (OKLCH h) | Light | Dark | Text contrast on Sheet (light / dark) |
|---|---|---|---|---|
| Tabular | 285 (indigo) | `#64619B` | `#B4B3F3` | 5.2 / 5.3 |
| Behavioral | 105 (olive) | `#716C26` | `#C3BE79` | 5.0 / 5.5 |
| Temporal | 185 (teal) | `#087970` | `#72CEC2` | 4.9 / 5.6 |
| Relational | 325 (plum) | `#835685` | `#D8A7DB` | 5.3 / 5.2 |
| Anomaly | 145 (green) | `#467748` | `#96CA97` | 4.9 / 5.6 |

How these were chosen: an exhaustive search over hue sets on a 10° grid, at chroma 0.09–0.12.
- Hues from 0° to 80° were excluded: the crimson risk hue and every orange or terracotta.
- Each set had to stay inside the sRGB gamut and pass AA as text on both Sheets.
- Among sets with a minimum normal-vision OKLab distance of at least 0.06, the winner is the
  set with the largest worst-case distance under simulated deuteranopia, protanopia and
  tritanopia (Machado et al. 2009). For this set:
  - normal vision: minimum distance 0.062;
  - colour-blind: worst case 0.032.
- A set with 0.076 normal-vision distance was rejected because its 85° member reads as ochre.

The two closest hues (green and teal) are never neighbours in the braid. Colour is also never
the only cue:
- strands keep a fixed order (tabular on top to anomaly at the bottom);
- every strand and reason names its view in text.

### Risk scale: a single hue (crimson, OKLCH h 15), 8 steps

- **Light** `#FFE8E9 → #F6CBCD → #EAAFB3 → #DD9499 → #D27C83 → #AF4E59 → #A23345 → #950C31`.
  Higher risk is deeper and darker.
- **Dark** `#4C3738 → … → #FEA9AF`. Higher risk is brighter and more saturated, so it still
  stands out on the steel background.
- Each step has its own text colour (`--risk-N-fg`), checked at 5.0:1 or better. The middle
  lightness band, where neither ink nor white reaches 4.5:1, is skipped.
- The step is chosen from p^0.6, which spreads the many low risks over the light steps so
  that 14 and 27 no longer look the same.
- Used only for risk: the end of the braid cord, risk cells, the heat-map and confirmed-fraud
  marks. The number is always printed with its colour.

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
| xs | 12 / 16 | axis ticks, chart values, braid meta, footnotes |
| sm | 13 / 18 | table cells, badges, braid view names, reasons |
| base | 14 / 20 | body, controls |
| md | 16 / 24 | pane titles |
| lg | 20 / 28 | page titles |
| xl | 28 / 34 | single key figures (overview) |
| narrative | 18 / 29 (Newsreader) | case narrative |

Weights: 400 and 600 only. Labels are sentence case: no ALL-CAPS, no letter-spaced
eyebrows, no monospace data labels. Charts and the braid use the same steps: no 11 px or
other off-scale text.

### Space, shape, depth

- **Spacing** on a 4 px base: 4, 8, 12, 16, 20, 28, 48. The analyst-console density is:
  - table rows 5 px vertical padding;
  - pane heads 8 × 12 px;
  - case sections 12 × 16 px.
- **Scrollbars**: thin (8 px), in the Rule colour, Graphite on hover, no track; they follow
  the theme.
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

## 2. Signature element: the evidence braid (a convergence)

For each transaction, the braid shows how five pieces of evidence become one score:
- **Five strands** start apart on the left, one lane per view, in fixed order: tabular,
  behavioral, temporal, relational, anomaly.
- **Start labels.** Each strand is labelled with its view name, its own score, and its
  weight as a percentage.
- **The merge.** The strands flow (Sankey-style bands) and merge into **one cord** on the right.
- **Thickness = gate weight.** A strand's thickness is its MVAF gate weight, constant along its
  length. At the merge, the cord is exactly the stack of the available weights, which sum to 1.
  On the case braid the cord is 72 px, so a 0.41 and a 0.26 strand differ by about 11 px.
- **The cord ends in the final risk number**, coloured on the risk scale. The separate risk
  gauge was removed, so there is one place to read the risk.
- **A missing view** is a thin dashed ghost strand in its view colour. It stops halfway, before
  the merge, and is labelled "not available". It never feeds the cord, so missing evidence is
  visibly missing and never zero (rule 11).
- **Inline braid** (tables): 80 × 22 px. It has the same convergence, without labels or the
  risk box; the row's risk cell carries the number.
- **Interaction.** Hovering or focusing a strand shows that view's reasons, and the arrow keys
  move between strands.

```
Tabular         ======\
score 44  20%          \=====
Behavioral      ========\=====\
score 35  22%                   ########  +------+
Temporal        ########################  |  32  |  risk, on the risk scale
score 20  29%                   ########  +------+
Relational      ========//=====//
score 52  17%          //=====
Anomaly         - - - - -          dashed ghost, stops before the merge
not available
```

---

## 3. Layout and wireframes

### Shell

A **single-row** global bar:
- **Left:** the as-of time machine, the label-delay setting and the mock-data notice.
- **Right:** Search (Ctrl K), a theme icon toggle (moon or sun, with an accessible name) and
  help.

The brand sits at the top of the left rail, above the navigation.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ As of [◂ ──●── ▸] Day 141, 14:00  Label delay [30 days]  Synthetic mock data  Search Ctrl K  ◐  ? │
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
┌ Alerts today (J/K) ┬ Case 3105477  [Step-up] ────────────────────┬ Evidence ─────────────┐
│ $53.70        32 Su│ $53.70   Day 141, 05:40                      │ Customer timeline     │
│ 05:40 Case 3105477 │ Customer  u8cw     Product      S            │ .  .. . ... .   o     │
│ $321.60       27 Hd│ Conformal set  Uncertain (both classes)      │ Linked entities       │
│ 03:04 Case 3105445 │ Disagreement 0.11  Expected loss $17.19      │  (card)-(uid)-(device)│
│ ...                │ Outcome  Not known yet (labels after 30 days)│ Counterfactual        │
│                    │ Evidence braid: five labelled strands merge  │ Amount  [====|----]   │
│                    │ into one cord that ends in the risk [ 32 ]   │ Device  [Known|New]   │
│                    │ Reasons (signed points, view name at right)  │ Locked                │
│                    │ +28 Amount is 5.1x usual        Temporal     │ [lock] Card ...       │
│                    │ Narrative (Newsreader)                       │ Risk would be 21      │
│                    │ [Mark as fraud F] [Mark as legit L] [U]      │                       │
│                    │ [Print or save as PDF]                       │                       │
└────────────────────┴──────────────────────────────────────────────┴───────────────────────┘
```

### 4 Fraud network

```
┌ Entity network, as of Day 141, 14:00 ───────────────────────────────── Legend: views ─┐
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
│ Timeline: ADWIN ● and Page-Hinkley ◆ alarms, injected drift start, retrains ▲ (accepted│
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
| Cream background with a terracotta accent | Avoided: cool grey Ledger, no orange or terracotta anywhere; crimson is reserved for risk |
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


---

## 7. Revision log

### Design review 1, Cases page (2026-10-08)

| Change | Why |
|---|---|
| Braid rebuilt as a convergence: labelled strands merge into one cord that ends in the risk number; missing views are dashed ghosts that stop before the merge | Straight lanes showed weights but not that the evidence *combines* into one score; the merge makes the fusion legible and makes weight differences visible as width at one point |
| Risk gauge removed | Two places showed the risk; the number now sits where the evidence ends |
| Only the decision is a badge; conformal set, disagreement, expected loss and outcome are label–value text | A row of equal badges flattened the hierarchy; the decision is the one actionable state, the rest is evidence |
| View colours retuned to equal OKLCH lightness, muted chroma, no orange or terracotta, checked for colour-blind distance and AA | The Okabe–Ito-based set differed in lightness (some views looked heavier) and used amber, which competed with the risk hue |
| Risk scale: 8 crimson steps on p^0.6 with per-step AA text, replacing 5 vermilion steps | Queue risks 14–27 all landed on one orange; the scale now shows higher as deeper, and is not orange |
| Header is one row; the brand moved to the rail | Two header rows wasted height in a dense console |
| No middle-dot meta strings: amount is prominent, time secondary, customer and product are labelled fields or links | "day 148 · 02:09 · customer u880 · $162.56" hid the amount and mixed field types |
| Thin, theme-styled scrollbars; braid and chart text on the type scale; denser spacing | Default scrollbars and off-scale 11 px text broke the system; an analyst console should show more per screen |
| Default case: the richest of today's alerts (all five views, at least 5 earlier transactions, shared entities). If no alert qualifies, today's richest transaction, with a note that it is not an alert | The first alert was often a cold start with missing views, which hid most of the console |
| Mock data: fraud-ring members are active accounts; customer activity is skewed; warm-up days kept as history | Rich cases need history; timelines showed "cold start" because pre-period transactions were dropped |
