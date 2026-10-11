import { describe, expect, it } from "vitest";
import { createHttpApi } from "./httpApi";

type Call = { url: string; init?: RequestInit };

function fakeFetch(responses: Record<string, { status?: number; body: unknown }>, calls: Call[]) {
  return async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const path = url.replace("http://api.test", "").split("?")[0];
    const r = responses[`${init?.method ?? "GET"} ${path}`];
    if (!r) throw new Error(`unexpected request ${url}`);
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  };
}

const asOf = { time: 151 * 86400 + 43200, labelDelayDays: 30 };

describe("createHttpApi", () => {
  it("sends the as-of time and label delay with every as-of request", async () => {
    const calls: Call[] = [];
    const api = createHttpApi("http://api.test/", fakeFetch({ "GET /api/alerts": { body: [] } }, calls));
    await api.alerts(asOf);
    const url = new URL(calls[0].url);
    expect(url.pathname).toBe("/api/alerts");
    expect(url.searchParams.get("time")).toBe(String(asOf.time));
    expect(url.searchParams.get("L")).toBe("30");
  });

  it("maps stream, queues and network parameters", async () => {
    const calls: Call[] = [];
    const api = createHttpApi(
      "http://api.test",
      fakeFetch(
        {
          "GET /api/stream": { body: [] },
          "GET /api/queues": { body: [] },
          "GET /api/network": { body: { nodes: [], edges: [], rings: [] } },
        },
        calls,
      ),
    );
    await api.stream(asOf, 10, 20);
    await api.queues(asOf, 50);
    await api.network(asOf, "uid:u 1");
    expect(new URL(calls[0].url).searchParams.get("start")).toBe("10");
    expect(new URL(calls[0].url).searchParams.get("end")).toBe("20");
    expect(new URL(calls[1].url).searchParams.get("k")).toBe("50");
    expect(new URL(calls[2].url).searchParams.get("focus")).toBe("uid:u 1");
  });

  it("surfaces the server's error message (e.g. analysts cannot change thresholds)", async () => {
    const api = createHttpApi(
      "http://api.test",
      fakeFetch({ "PUT /api/thresholds": { status: 403, body: { detail: "Only admins can change thresholds." } } }, []),
    );
    await expect(
      api.saveThresholds({ t_monitor: 0.1, t_step: 0.2, t_hold: 0.3, t_block: 0.4, t_dis: null }, "analyst"),
    ).rejects.toThrow("Only admins can change thresholds.");
  });

  it("keeps the illustrative simulator for score() without calling the server", async () => {
    const calls: Call[] = [];
    const api = createHttpApi("http://api.test", fakeFetch({}, calls));
    const r = await api.score({
      amount: 120,
      usualAmount: 60,
      deviceKnown: true,
      velocity24h: 1,
      sharedEntities: 0,
      historyCount: 5,
      maskedC13: 1,
      available: { tabular: true, behavioral: true, temporal: true, graph: false, anomaly: true },
    });
    expect(r.p).toBeGreaterThanOrEqual(0);
    expect(r.p).toBeLessThanOrEqual(1);
    expect(calls).toHaveLength(0);
  });
});

describe("createHttpApi: data-source notes and the what-if cost mode", () => {
  it("maps the backend's notes", async () => {
    const api = createHttpApi(
      "http://api.test",
      fakeFetch({ "GET /api/meta": { body: { source: "IEEE-CIS test period, final run", no_block_note: "No Block decisions: ...", whatif_available: true } } }, []),
    );
    const info = await api.info!();
    expect(info).toEqual({ source: "IEEE-CIS test period, final run", noBlockNote: "No Block decisions: ...", whatIfAvailable: true });
  });

  it("turns the what-if cost mode on with PUT and off with DELETE", async () => {
    const calls: Call[] = [];
    const status = { active: true, available: true, cost: {}, thresholds: null, note: "illustrative" };
    const api = createHttpApi(
      "http://api.test",
      fakeFetch({ "PUT /api/whatif": { body: status }, "DELETE /api/whatif": { body: { ...status, active: false } } }, calls),
    );
    const cost = { c_fp: 2, c_rev: 1, c_step: 0.1, step_up_success_rate: 0.9, legit_step_up_friction: 0 };
    expect((await api.setWhatIf!(cost, "analyst")).active).toBe(true);
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ cost, role: "analyst" });
    expect((await api.setWhatIf!(null, "analyst")).active).toBe(false);
    expect(calls[1].init?.method).toBe("DELETE");
  });
});
