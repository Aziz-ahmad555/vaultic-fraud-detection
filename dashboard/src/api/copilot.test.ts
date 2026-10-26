import { describe, expect, it } from "vitest";
import { createHttpCopilot, NOT_ENABLED, readStream, type CopilotEvent } from "./copilot";

const enc = new TextEncoder();

/** A Response whose body arrives in the given chunks (split anywhere, even inside a line). */
function streamed(chunks: string[], status = 200): Response {
  const body = new ReadableStream({
    start(c) {
      chunks.forEach((s) => c.enqueue(enc.encode(s)));
      c.close();
    },
  });
  return new Response(body, { status, headers: { "Content-Type": "application/x-ndjson" } });
}

const FINAL = { type: "final", case: 1, kind: "question", route: "why", model: "m", note: "n", reasons_source: null, evidence: [], answer: "a", kept: [], removed: [], seconds: 1, audit: "ok" };

describe("copilot client (D226)", () => {
  it("reads an NDJSON stream split across chunks and resolves with the final event", async () => {
    const lines = [{ type: "start", evidence: [], model: "m", reasons_source: null }, { type: "progress", phase: "writing", drafted: 1, shown: 1, removed: 0 }, { type: "fact", fact: { text: "a [E1].", citations: ["E1"] } }, FINAL].map((e) => JSON.stringify(e)).join("\n") + "\n";
    const seen: CopilotEvent[] = [];
    const final = await readStream(streamed([lines.slice(0, 17), lines.slice(17, 60), lines.slice(60)]), (e) => seen.push(e));
    expect(seen.map((e) => e.type)).toEqual(["start", "progress", "fact", "final"]);
    expect(final.answer).toBe("a");
  });

  it("turns an error event into a thrown error, and a stream without a final into one too", async () => {
    await expect(readStream(streamed([JSON.stringify({ type: "error", message: "model failed" }) + "\n"]), () => {})).rejects.toThrow("model failed");
    await expect(readStream(streamed([JSON.stringify({ type: "start", evidence: [], model: "m", reasons_source: null })]), () => {})).rejects.toThrow("stopped before the checked answer");
  });

  it("sends the question with the as-of time and label delay, and reports the backend's 503 message", async () => {
    const calls: { url: string; body: unknown }[] = [];
    const fetchImpl = async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : null });
      if (url.endsWith("/api/copilot/summary")) return new Response(JSON.stringify({ detail: "The local language model is not running: start it with E:\\ollama\\start_ollama.cmd." }), { status: 503 });
      return streamed([JSON.stringify(FINAL) + "\n"]);
    };
    const cop = createHttpCopilot("http://x", fetchImpl);
    await cop.ask(7, "Why?", { time: 100, labelDelayDays: 30 }, () => {});
    expect(calls[0]).toEqual({ url: "http://x/api/copilot/ask", body: { case_id: 7, question: "Why?", time: 100, L: 30, stream: true } });
    await expect(cop.summary(7, { time: 100, labelDelayDays: 30 }, () => {})).rejects.toMatchObject({ status: 503, message: expect.stringContaining("start_ollama.cmd") });
  });

  it("says the copilot is not enabled when the backend has no copilot endpoints", async () => {
    const cop = createHttpCopilot("http://x", async () => new Response("{}", { status: 404 }));
    await expect(cop.status()).rejects.toThrow(NOT_ENABLED);
    const down = createHttpCopilot("http://x", async () => {
      throw new TypeError("fetch failed");
    });
    await expect(down.status()).rejects.toThrow("not reachable");
  });
});
