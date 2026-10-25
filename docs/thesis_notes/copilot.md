# Local-LLM investigation copilot: backend (D222-D224h)

Stage 1 of Sprint 2 (roadmap Phase 15, our Phase C (c)): the backend only. The UI (chat panel,
citation chips, rule-based next checks) waits for D226; the evaluation E20 waits for its own
pre-registration (D225). The copilot is a COULD item (D31) taken up at Aziz's request.

## What it is
A question-answering helper for one case on the analyst's laptop. An analyst asks "why was this
flagged?", "what is it linked to?", "has this customer done this before?" or "what evidence is
missing?", and gets a short list of facts, each citing the case evidence line it came from. A
local open-weight model (Qwen2.5-7B-Instruct, Q4_K_M, Apache-2.0; D223) runs in Ollama on
127.0.0.1; no case evidence leaves the laptop. The copilot is read-only: it cannot change a
decision, a label, a threshold or a setting. Nothing it says is a research number, and it is
labelled "not the evaluated detection system".

## Why we built it
The case page already shows the evidence (decision, view scores, reasons, history, network), but an
analyst has to read several panels to answer a simple question. A language model can summarise them,
but a model on its own invents facts. The design question was therefore how to let a small local
model help without letting it state anything the evidence does not support.

## How it works (simple)
1. **Refusal rules first** (no model involved). Requests to change something ("change the decision
   to allow", "mark this as legit"), to see another customer's data that is not linked to the case,
   or to ignore the rules are refused with a fixed message before any tool or model runs.
2. **A fixed router picks the tools** from the question's wording (why flagged: case + reasons;
   links: case + network; history: case + customer history; missing or uncertain: case + view scores
   + reasons; also similar cases and the counterfactual). The model never chooses tools; each extra
   tool round would cost another slow prompt pass on this CPU (D223).
3. **Read-only tools** read through the demo backend's own functions, so they see exactly what the
   case page sees: nothing after the as-of time, and a label only after its delay (label-delay
   curtain). Reasons are the frozen system's live top-5 feature reasons when the Test Lab is running
   (parity-gated), otherwise the stored view-level reasons, labelled "feature reasons not available:
   the Test Lab is off" (or "is loading" / "is refused" while it starts). Similar cases are a new
   helper: earlier cases with the same decision, nearest by the calibrated view scores, labelled
   "similar by scores, not by investigation outcome", shown only as counts (how many, how many with a
   matured fraud or legit label), never other customers' transaction ids or labels (D224d).
   Counterfactuals always say "not available" (E15 stores them per alert index without a
   transaction id).
4. **Trimmed evidence.** Only the chosen tools' results are sent, as short lines with handles
   [E1], [E2], ... (about 700 tokens at most per question, 1,000 for a summary).
5. **The model answers in structured output** (D224c): Ollama's JSON-schema mode forces a list of
   at most 5 facts, each with exactly one handle chosen from the handles given in this turn; the
   prose the analyst reads is rendered by code ("<fact> [Ek]."), not written by the model. The
   evidence block is called data in the rules; strings from the data are cleaned (no brackets or
   new lines, at most 60 characters) and a string that reads like an instruction is withheld. While
   the answer streams, it is shown as "draft, not yet checked".
6. **A deterministic citation checker** then keeps a fact only if (D224a, D224d):
   - every number is a whole value of its cited line (a time such as 03:15 is one value) and
     belongs to a field whose name shares a word with the fact, so a number cannot borrow another
     field's value; ids (digits, "_", "."), e-mail domains and other names appear there exactly;
   - a decision word in any form ("held", "stepped up", "monitoring") cites the case's own
     decision line and matches it (about an earlier transaction: that transaction's line, by id;
     about earlier transactions in general: the customer-history summary line, with its count,
     e.g. "their decisions: hold 2" supports "2 earlier transactions were held", D224e). The rule
     is conservative (D224h): a fact with any case reference ("it", "this", "that one", "the
     case", "the transaction", or "both", "too", "also", "as well", "as was", "so was", "like",
     "unlike", "same as", "similar to", "following", "after", "given", "with") gets no support
     from the history line for any decision word, only from the case's own decision line; without
     one, the history line supports a decision word only in "<N> earlier transactions ...
     <decision>", N directly attached to "earlier transactions" and equal to the line's count;
   - a claim that the case is fraud or legitimate ("definitely fraud", "not legitimate") cites the
     case's own label line, which states a known label of the same polarity;
   - it does not claim that anything was changed (updated, set to, dismissed, escalated, closed ...).
   Removed facts are counted with the reason; an empty list, or nothing kept, reads "Not available
   from the case evidence."
7. **One audit entry per question** (D208): model, route, evidence ids, kept and removed counts,
   refusal kind, outcome and seconds, written once whether the answer finishes, is refused, the model
   is not running, or the client disconnects (fail-open, D209).
8. **Cached demo summaries.** Five demo cases picked by fixed rules (highest-risk hold, a hold
   without a graph score, a cold-start step-up, the most uncertain step-up, the highest-risk step-up)
   are summarised once; a cached summary is served only when the evidence, the rules and the checker
   version match, the current checker runs again on the stored draft, and it is labelled "cached,
   generated at <time> by <model>". Requests with an invalid body or label delay and tool errors
   are audited too (D224d).

Endpoints (mounted with `VAULTIC_COPILOT=1`): `GET /api/copilot/status`, `POST /api/copilot/ask`
(NDJSON token stream), `POST /api/copilot/summary`. Code: `src/vaultic/copilot/`,
`services/api/copilot.py`.

## Key results / numbers (with sources)
- Model speed on the laptop (i5-8265U, 15.9 GB RAM, CPU only; D223, `docs/plans/copilot_plan.md`
  progress section; not research numbers): uncached first question with trimmed evidence (about 525
  tokens in, 160 out) 76.1-78.6 s, with the full evidence (about 1,880 tokens) 184.8-186.9 s; prompt
  reading about 14 tokens/s, generation about 4 tokens/s; the loaded model holds about 5 GB.
- Cached demo summaries with the real model (Qwen2.5-7B Q4, Test Lab off, evidence 617-965 tokens
  estimated; `E:/dev-cache/copilot/demo_summaries.json`, not committed; not research numbers):
  77.7-124.7 s per summary; 4 of the 5 summaries kept 4-6 sentences with 0-1 removed; the fifth
  (the hold without a graph score) had no valid citations, so all 7 sentences were removed and it
  reads "Not available from the case evidence." Cause (D224b, from the stored drafts, no new run):
  the evidence was complete and identical in all three generations, and the checker was right; the
  model wrote the sentences without handles and then one line listing every handle, starting with
  "<E1>", imitating the format line "Format: <one fact> [E2]." in the rules. The format line now
  has no placeholder, a handle-only line is named as such, and a cached summary is served only under
  the rules it was made with, so the five summaries above are no longer served.
- Fourth generation, once, under the D224b rules (2026-10-25, Aziz's go-ahead, kept as it came out,
  no re-run; `E:/dev-cache/copilot/demo_summaries.json`, earlier versions v1-v3 beside it):
  63.3-119.9 s per summary; sentences written / kept / removed: highest-risk hold 8 / 0 / 8,
  hold without a graph score 5 / 0 / 4 (plus the instructed "not available" sentence), cold-start
  step-up 5 / 0 / 4 (plus the instructed sentence), uncertain step-up 10 / 1 / 9, highest-risk
  step-up 8 / 0 / 8; every removal reason was "no citation". Four of the five demo summaries now
  read "Not available from the case evidence." The placeholder-free format line fixed the
  handle-dump but the model stopped citing almost entirely: the D224b wording did worse than D224a's
  (4 of 5 usable). The checker held: no uncited sentence was shown.
- The first real-model runs changed the checker (D224a): the model cited ranges such as [E8]-[E10],
  added the "not available" sentence without a citation, and, after an example sentence ("The
  decision is hold [E1].") was added to the rules, copied it for two step-up cases. The checker now
  expands ranges, ignores the instructed phrase and removes a decision word that the cited lines do
  not contain; the rules carry a content-free format line instead of an example.
- **Structured output, dev-set comparison (D224c; 2026-10-25; one generation per variant and task,
  no re-runs; `E:/dev-cache/copilot/dev_compare_20261025-040441.json`, not committed; design work on
  10 dev cases excluded from E20, not research numbers).** Per case one summary and one "Why was this
  transaction flagged?" (20 generations per variant):

  | variant | facts written | kept | removed | kept-fact rate | generations with no kept fact | invalid JSON | total seconds |
  |---|---|---|---|---|---|---|---|
  | V1 schema only (at most 5) | 52 | 48 | 4 | 0.923 | 4 (3 empty lists) | 0 | 1,258.8 |
  | **V2 schema + one instruction** | 55 | 52 | 3 | **0.945** | 2 (2 empty lists) | 0 | 1,255.8 |
  | V3 schema, at most 4 facts | 44 | 41 | 3 | 0.932 | 4 (3 empty lists) | 0 | 646.4 |

  V2 was chosen by the fixed rule and is the default; its five demo summaries are the cached ones
  (44 of 47 summary facts kept across the 10 dev summaries; every demo summary has kept facts).
  Removal reasons: a number not in the cited line (a points value or expected loss cited to the wrong
  line, a transaction id with a dropped digit) and, under V1 and V3, a handle written inside the fact
  text. All generations everywhere were valid JSON. Per-task timing for V2: summaries 83.9-128.9 s
  (median 103.5 s), questions 4.0-25.9 s (median 22.6 s, few facts per answer). Under V1 and V3
  three of ten "why flagged" answers were an empty list ("not available"), under V2 two.
  Compared with the prose answers (1 sentence kept of 36 under D224b), structured output with an
  enum of handles removes the citation problem; what remains is the meaning of a fact.
- **Review fixes (D224d) re-checked on the stored dev drafts** (the same 60 generations, no model
  run; `E:/dev-cache/copilot/dev_recheck_D224d_20261025-223959.json`): kept-fact rate V1 0.923 ->
  0.904 (47/52), V2 0.945 -> 0.909 (50/55), V3 0.932 -> 0.909 (40/44). V2 and V3 now tie (10/11
  each). Newly removed: "the last transaction was allowed" citing the customer line (a correct
  catch), and "6 earlier transactions, all of which were held" citing the customer-history summary
  line in V1 and V2 (V3: "history of 6 transactions", removed by the field binding).
- **History summary lines (D224e) and a correction (D224f).** D224e lets the customer-history summary
  line support decision words about earlier transactions when the fact gives the line's count;
  it was decided after seeing that the rule touches the V2/V3 tie, on the principle that the checker
  should not remove true facts. Reading the line then showed the "6 ... all held" fact is not true:
  the line says "6 earlier transactions in total (all periods); 2 of them in the test-period data
  shown here; their decisions: hold 2", so only 2 are known to be held (D224f corrects the earlier
  "appears correct"). The count rule removes it, and the re-check under D224e
  (`E:/dev-cache/copilot/dev_recheck_D224e_20261025-231757.json`) gives the same rates as under
  D224d: V1 0.904 (47/52), V2 0.909 (50/55), V3 0.909 (40/44). Applying D224c's rule as written
  (highest rate; ties: fewer removed, then faster): V2 and V3 tie, V3 has fewer removed facts (4 vs
  5), so **V3 (schema, at most 4 facts) is the default**. Its five demo summaries from the dev run are
  the cached ones (4 facts each, none removed, 31.6-60.8 s when generated); the V2 cache, re-checked
  under D224e (4-5 facts each, none removed), is kept beside it.
- **Clause-by-clause attribution (D224g)**, after a narrow review: pronoun and comparison claims
  ("It was held, like the 6 earlier transactions") are case claims; a wrong history count is removed
  by the count rule with its own reason. Re-check of the stored dev drafts
  (`E:/dev-cache/copilot/dev_recheck_D224g_20261026-004333.json`, no model): unchanged, V1 0.904
  (47/52), V2 0.909 (50/55), V3 0.909 (40/44); D224c's rule still picks V3. The cached V3 summaries
  keep all their facts (4 each) under checker version D224g.
- **Conservative decision rule (D224h)**, after a further narrow review in which D224g's clause
  parsing still kept 12 of the reviewer's case claims ("After 6 earlier transactions it was held",
  "The case and the 6 earlier transactions were held", "6 earlier transactions over 2 periods were
  held" against "hold 2" ...). Rule-based clause parsing cannot cover all English phrasings, and a
  false removal is harmless (the case page's key facts show the truth) while a false keep is harmful,
  so any case reference now rules out the history line. All probes of the review (probe_d224g.py,
  probe_prior.py) are tests. Re-check of the stored dev drafts
  (`E:/dev-cache/copilot/dev_recheck_D224h_20261026-013611.json`, no model): unchanged, V1 0.904
  (47/52), V2 0.909 (50/55), V3 0.909 (40/44), no fact newly removed; D224c's rule still picks V3;
  the cached V3 summaries keep all their facts (4 each) under checker version D224h.
- Tests (all with a scripted model, so CI never needs Ollama): `tests/test_copilot_check.py` and
  `tests/test_copilot_api.py` (181 tests: checker reject cases incl. the review's probe cases,
  refusals, injection in a tool result, read-only guarantee, one audit entry per request incl.
  invalid ones and tool errors, Test Lab off, loading and on, model not running, proxy-free
  client, evidence budget, cached summaries).

## Limitations
- **Decision claims are checked by a conservative rule, not by understanding** (D224h). Over-strict
  removals are accepted: a true history fact is removed when it also contains a case-reference word
  ("6 earlier transactions, all held, with a median amount of 1331.00") or gives its count another
  way ("6 earlier transactions in total, 2 of them held"). Unusual phrasings may still slip through:
  the checker is rule-based, and E20 measures what it misses.
- **Not checked: negation, comparisons and numbers written as words.** "The device was not seen
  before" citing a line that says "device seen before yes" is kept, and so is "over one thousand
  dollars"; a comparison ("higher than usual") is not verified. E20 measures these with human labels
  (D225).
- More generally, the checker verifies numbers, ids, handles, decision words, change claims and
  label claims, not the meaning of a free-text sentence: a cited sentence without those can still
  misread its evidence (seen once: "the reasons source is unavailable" for a line saying feature
  reasons are not available).
- The field binding of numbers is word overlap: a correct fact that names a field differently
  ("history of 6 transactions") is removed, and a wrong one that shares a word could be kept.
- With prose answers the model's citation discipline depended strongly on the prompt wording (one
  of five demo summaries lost every sentence in the first three generations, four of five under
  D224b) and it imitated examples and placeholders in the rules; structured output (D224c) fixed
  this on the dev set, but the dev set is small (10 cases, one generation per variant).
- Some kept facts are correct but unhelpful: a fact may copy a whole evidence line, or restate a line
  loosely ("the reasons source is unavailable" for a line saying feature reasons are not
  available); "why flagged" answers are short (0-2 facts), and two of ten were "not available".
- The router uses keywords; an unusual wording can pick the default tools (case + reasons).
- Refusal rules are patterns; they are tested on the planned adversarial types, not on every possible
  wording.
- Cached summaries are of the evidence at the case's decision time with the Test Lab off; with the
  Test Lab on, the evidence differs and the summary is generated live.
- A 7B model on this CPU is slow (about 1-1.5 minutes for a first question); summaries of other
  cases take longer.
