# Local-LLM investigation copilot: backend (D222-D224b)

Stage 1 of Sprint 2 (roadmap Phase 15, our Phase C (c)): the backend only. The UI (chat panel,
citation chips, rule-based next checks) waits for D226; the evaluation E20 waits for its own
pre-registration (D225). The copilot is a COULD item (D31) taken up at Aziz's request.

## What it is
A question-answering helper for one case on the analyst's laptop. An analyst asks "why was this
flagged?", "what is it linked to?", "has this customer done this before?" or "what evidence is
missing?", and gets a short answer in which every sentence cites the case evidence it came from. A
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
   the Test Lab is off". Similar cases are a new helper: earlier cases with the same decision,
   nearest by the calibrated view scores, labelled "similar by scores, not by investigation outcome".
   Counterfactuals always say "not available" (E15 stores them per alert index without a
   transaction id).
4. **Trimmed evidence.** Only the chosen tools' results are sent, as short lines with handles
   [E1], [E2], ... (about 700 tokens at most per question, 1,000 for a summary; answers capped at 120
   tokens). This keeps a first question under the 90-second rule (D223).
5. **The model answers** from the evidence block, which the rules call data; strings from the data
   are cleaned (no brackets or new lines, at most 60 characters) and a string that reads like an
   instruction is withheld. While the answer streams, it is shown as "draft, not yet checked".
6. **A deterministic citation checker** then keeps a sentence only if it cites handles given in this
   turn and every number and id in it appears in the cited lines (rounding allowed; small integers
   exact), and every decision word (allow, monitor, step-up, hold, block) appears in them (D224a).
   Claims that something was changed are always removed; a claim that the case is fraud or
   legitimate needs a cited known label. Removed sentences are counted with the reason; if nothing is
   left, the answer is "Not available from the case evidence."
7. **One audit entry per question** (D208): model, route, evidence ids, kept and removed counts,
   refusal kind, outcome and seconds, written once whether the answer finishes, is refused, the model
   is not running, or the client disconnects (fail-open, D209).
8. **Cached demo summaries.** Five demo cases picked by fixed rules (highest-risk hold, a hold
   without a graph score, a cold-start step-up, the most uncertain step-up, the highest-risk step-up)
   are summarised once; a cached summary is served only when the evidence is identical and is
   labelled "cached, generated at <time> by <model>".

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
  the rules it was made with, so the five summaries above are generated live until the cache is
  regenerated (with Aziz's go-ahead; the fix is not yet verified on the real model).
- The first real-model runs changed the checker (D224a): the model cited ranges such as [E8]-[E10],
  added the "not available" sentence without a citation, and, after an example sentence ("The
  decision is hold [E1].") was added to the rules, copied it for two step-up cases. The checker now
  expands ranges, ignores the instructed phrase and removes a decision word that the cited lines do
  not contain; the rules carry a content-free format line instead of an example.
- Tests (all with a scripted model, so CI never needs Ollama): `tests/test_copilot_check.py` and
  `tests/test_copilot_api.py` (58 tests: checker reject cases, refusals, injection in a tool
  result, read-only guarantee, one audit entry per question, Test Lab off and on, model not
  running, evidence budget, cached summaries).

## Limitations
- The checker verifies numbers, ids, handles, decision words, change claims and label claims, not
  the meaning of a free-text sentence: a cited sentence without those can still misread its
  evidence (seen once: "the reasons source is unavailable" for a line saying feature reasons are not
  available). E20 measures this with human labels (D225).
- The model's citation discipline varies between cases and prompt wordings: in each of the three
  generations one of the five demo summaries lost every sentence (a different case in the first
  and third), and the model imitates examples and placeholders in the rules. The checker makes that visible ("not available") instead of showing uncited text.
- The router uses keywords; an unusual wording can pick the default tools (case + reasons).
- Refusal rules are patterns; they are tested on the planned adversarial types, not on every possible
  wording.
- Cached summaries are of the evidence at the case's decision time with the Test Lab off; with the
  Test Lab on, the evidence differs and the summary is generated live.
- A 7B model on this CPU is slow (about 1-1.5 minutes for a first question); summaries of other
  cases take longer.
