import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createHttpApi } from "../api/http/httpApi";
import { getIdentity, identityHeaders, setIdentity } from "../api/identity";
import { createMockApi } from "../api/mock/mockApi";
import type { Api, AuditEntry } from "../api/types";
import { AuditPanel } from "./AuditPanel";

const ENTRIES: AuditEntry[] = [
  { id: 2, time: "2026-10-18T10:01:00+00:00", analyst: "Aziz", role: "admin", action: "Marked transaction 7 as legit", target: "tx:7", before: "fraud", after: "legit", detail: "was fraud" },
  { id: 1, time: "2026-10-18T10:00:00+00:00", analyst: "unknown", role: "unknown", action: "Exported the audit log", target: "audit_log", detail: "1 entries (CSV)" },
];

function fakeApi(over: Partial<Api> = {}): Api {
  return {
    auditLog: vi.fn(async () => ENTRIES),
    auditVerify: vi.fn(async () => ({ ok: true, entries: 2, first_bad: null, reason: null })),
    auditExport: vi.fn(async () => "id,time\n1,x\n"),
    ...over,
  } as unknown as Api;
}

describe("audit panel (D208)", () => {
  it("shows analyst, role, target and before → after for every entry", async () => {
    render(<AuditPanel api={fakeApi()} toast={() => undefined} />);
    const rows = within(await screen.findByRole("table", { name: "Audit log entries" })).getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Aziz");
    expect(rows[1]).toHaveTextContent("admin");
    expect(rows[1]).toHaveTextContent("tx:7");
    expect(rows[1]).toHaveTextContent("fraud → legit");
    expect(rows[2]).toHaveTextContent("unknown");
  });

  it("filters by analyst and action, checks the chain and exports", async () => {
    const api = fakeApi();
    const toast = vi.fn();
    globalThis.URL.createObjectURL = vi.fn(() => "blob:x");
    globalThis.URL.revokeObjectURL = vi.fn();
    render(<AuditPanel api={api} toast={toast} />);
    await screen.findByRole("table", { name: "Audit log entries" });
    await userEvent.type(screen.getByLabelText("Analyst"), "Aziz");
    await userEvent.type(screen.getByLabelText("Action contains"), "Marked");
    await userEvent.click(screen.getByRole("button", { name: "Filter" }));
    await waitFor(() => expect(api.auditLog).toHaveBeenLastCalledWith({ analyst: "Aziz", action: "Marked", limit: 500 }));
    await userEvent.click(screen.getByRole("button", { name: "Check the chain" }));
    expect(await screen.findByTestId("audit-check")).toHaveTextContent("Chain intact: 2 entries");
    await userEvent.click(screen.getByRole("button", { name: "Export CSV" }));
    await waitFor(() => expect(api.auditExport).toHaveBeenCalled());
    expect(toast).toHaveBeenCalledWith("Audit log exported (the export is recorded in the log)");
  });

  it("says plainly when the chain is broken", async () => {
    const api = fakeApi({ auditVerify: vi.fn(async () => ({ ok: false, entries: 4, first_bad: 5, reason: "entry 5 does not match the chain" })) });
    render(<AuditPanel api={api} toast={() => undefined} />);
    await userEvent.click(await screen.findByRole("button", { name: "Check the chain" }));
    expect(await screen.findByTestId("audit-check")).toHaveTextContent("Chain broken at entry 5: entry 5 does not match the chain");
  });
});

describe("acting analyst on every request (D208)", () => {
  it("sends the analyst and role headers with every backend request", async () => {
    setIdentity("  Aziz Ahmad ", "admin");
    const seen: Record<string, string>[] = [];
    const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
      seen.push(init?.headers as Record<string, string>);
      return new Response(JSON.stringify([]), { status: 200 });
    });
    const api = createHttpApi("http://x", fetchImpl);
    await api.auditLog();
    await api.feedback(7, "fraud", "admin");
    await api.auditEvent!("report_printed", "day 151");
    for (const h of seen) expect(h).toMatchObject({ "X-Vaultic-Analyst": "Aziz Ahmad", "X-Vaultic-Role": "admin" });
    expect(fetchImpl.mock.calls[2][0]).toBe("http://x/api/audit/event");
    setIdentity("", "analyst");
    expect(identityHeaders()).toEqual({ "X-Vaultic-Role": "analyst" }); // no name: the server records "unknown"
    setIdentity("Zoë", "analyst");
    expect(identityHeaders()["X-Vaultic-Analyst"]).toBe("Zo%C3%AB"); // headers must be ASCII
    setIdentity("", "analyst");
  });

  it("the mock records the named analyst, events and exports", async () => {
    window.localStorage.clear();
    setIdentity("Aziz", "analyst");
    const api = createMockApi();
    await api.feedback(9, "fraud", "analyst");
    await api.auditEvent!("whatif_applied", "c_fp 5");
    const csv = await api.auditExport!();
    const log = await api.auditLog();
    expect(log.map((e) => e.action)).toEqual(["Exported the audit log", "Applied a what-if cost setting", "Marked transaction 9 as fraud"]);
    expect(log.every((e) => e.analyst === "Aziz")).toBe(true);
    expect(log[2]).toMatchObject({ target: "tx:9", before: null, after: "fraud" });
    expect(csv.split("\n")[0]).toBe("time,analyst,role,action,target,detail");
    expect((await api.auditLog({ action: "what-if" })).length).toBe(1);
    expect(getIdentity().analyst).toBe("Aziz");
    setIdentity("", "analyst");
  });
});
