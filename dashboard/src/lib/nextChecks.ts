/**
 * Suggested next checks for a case (D226): fixed rules on the case's own fields, never the language
 * model. Each check names the field it comes from, so the analyst can see why it is suggested.
 */
import type { AsOf, Transaction } from "../api/types";
import { VIEWS, VIEW_LABEL } from "../api/types";

export interface NextCheck {
  id: string;
  text: string;
  because: string;
}

export const HIGH_DISAGREEMENT = 0.3;

export function nextChecks(t: Transaction, asOf: AsOf): NextCheck[] {
  const out: NextCheck[] = [];
  const d = t.decision;
  if (t.row.m_graph === 0)
    out.push({ id: "graph", text: "Check the linked cards, e-mails and devices by hand on the fraud network page.", because: `the ${VIEW_LABEL.graph.toLowerCase()} view has no score for this case` });
  const missing = VIEWS.filter((v) => v !== "graph" && t.row[`m_${v}`] === 0);
  if (missing.length)
    out.push({ id: "missing", text: `Weigh the decision knowing that ${missing.map((v) => VIEW_LABEL[v].toLowerCase()).join(" and ")} evidence is missing.`, because: "missing views carry no weight (never filled)" });
  if (d.conformal_set === "uncertain" || d.conformal_set === "empty")
    out.push({ id: "uncertain", text: "Treat the risk as uncertain: confirm with the customer (step-up) before a final call.", because: `the conformal set is ${d.conformal_set}` });
  if (d.disagreement >= HIGH_DISAGREEMENT)
    out.push({ id: "disagree", text: "Read each view's reason: the views point in different directions.", because: `view disagreement is ${d.disagreement.toFixed(2)} (at least ${HIGH_DISAGREEMENT})` });
  if (t.inputs.historyCount === 0)
    out.push({ id: "cold", text: "Verify the customer's identity details: there is no earlier behaviour to compare with.", because: "cold start: no earlier transactions" });
  if (t.inputs.deviceKnown === false)
    out.push({ id: "device", text: "Check the device: this customer has not used it before.", because: "device seen before: no" });
  if (t.inputs.sharedEntities !== null && t.inputs.sharedEntities > 0)
    out.push({ id: "shared", text: "Review the other customers on this card.", because: `${t.inputs.sharedEntities} other customer${t.inputs.sharedEntities > 1 ? "s" : ""} used the card in the 7 days before` });
  if (t.row.label === null)
    out.push({ id: "label", text: "Do not rely on an outcome label yet.", because: `the label is not known before the ${asOf.labelDelayDays}-day label delay` });
  return out;
}
