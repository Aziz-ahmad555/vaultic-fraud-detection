import { afterEach, describe, expect, it, vi } from "vitest";
import { configureAuth, getAuth, loadMode, login, logout, resetAuthForTests, withAuth } from "./auth";
import { identityHeaders, setIdentity } from "./identity";

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

function backend(mode: "jwt" | "demo", opts: { refreshOk?: boolean } = {}) {
  const calls: { url: string; init?: RequestInit }[] = [];
  let n = 0;
  const f = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    if (url.endsWith("/api/auth/mode")) return json(200, { mode, demoLabel: mode === "demo" ? "Demo mode: no login; names and roles are self-declared" : null });
    if (url.endsWith("/api/auth/refresh")) {
      return opts.refreshOk ? json(200, { accessToken: `fresh-${++n}`, expiresIn: 900, user: { username: "analyst1", role: "analyst" } }) : json(401, { detail: "invalid refresh token" });
    }
    if (url.endsWith("/api/auth/login")) {
      const body = JSON.parse(String(init?.body));
      if (body.password !== "right-password") return json(401, { detail: "invalid username or password" });
      return json(200, { accessToken: "first", expiresIn: 900, user: { username: body.username, role: "admin" } });
    }
    if (url.endsWith("/api/auth/logout")) return json(200, { loggedOut: true });
    return json(200, {});
  });
  return { f, calls };
}

afterEach(() => resetAuthForTests());

describe("login session (D235)", () => {
  it("demo mode keeps the self-declared headers and shows the demo label", async () => {
    const { f } = backend("demo");
    configureAuth("http://api", f);
    await loadMode();
    expect(getAuth()).toMatchObject({ mode: "demo", ready: true, user: null });
    expect(getAuth().demoLabel).toMatch(/no login/);
    setIdentity("Zoe", "admin");
    expect(identityHeaders()).toEqual({ "X-Vaultic-Role": "admin", "X-Vaultic-Analyst": "Zoe" });
  });

  it("with login on, only the bearer token is sent, never the self-declared headers", async () => {
    const { f, calls } = backend("jwt");
    configureAuth("http://api", f);
    await loadMode(); // the silent refresh fails: no session yet
    expect(getAuth()).toMatchObject({ mode: "jwt", ready: true, user: null });
    setIdentity("mallory", "admin");
    expect(identityHeaders()).toEqual({});
    await expect(login("analyst1", "wrong")).rejects.toThrow("invalid username or password");
    const user = await login("analyst1", "right-password");
    expect(user).toEqual({ username: "analyst1", role: "admin" });
    expect(identityHeaders()).toEqual({ Authorization: "Bearer first" });
    const loginCall = calls.find((c) => c.url.endsWith("/api/auth/login"));
    expect(loginCall?.init?.credentials).toBe("include"); // the refresh cookie comes back
    expect(window.localStorage.getItem("vaultic_access")).toBeNull(); // the token stays in memory
  });

  it("a 401 gets one silent refresh and one retry with the new token", async () => {
    const { f } = backend("jwt", { refreshOk: true });
    configureAuth("http://api", f);
    await loadMode();
    expect(getAuth().user?.username).toBe("analyst1");
    const api = vi.fn(async (_url: string, init?: RequestInit) =>
      (init?.headers as Record<string, string>)?.Authorization === "Bearer fresh-2" ? json(200, { ok: true }) : json(401, { detail: "the session has expired" }),
    );
    const res = await withAuth(api)("http://api/api/meta", { headers: { Authorization: "Bearer fresh-1" } });
    expect(res.status).toBe(200);
    expect(api).toHaveBeenCalledTimes(2);
    const refreshCalls = f.mock.calls.filter(([u]) => String(u).endsWith("/api/auth/refresh"));
    expect(refreshCalls).toHaveLength(2); // one at start-up, one after the 401
    expect((refreshCalls[1][1]?.headers as Record<string, string>)["X-Vaultic-CSRF"]).toBe("1");
  });

  it("when the refresh fails the session ends (the login page appears)", async () => {
    const { f } = backend("jwt");
    configureAuth("http://api", f);
    await loadMode();
    await login("analyst1", "right-password");
    const api = vi.fn(async () => json(401, { detail: "the session has expired" }));
    const res = await withAuth(api)("http://api/api/meta");
    expect(res.status).toBe(401);
    expect(api).toHaveBeenCalledTimes(1);
    expect(getAuth().user).toBeNull();
  });

  it("logout ends the session even if the backend cannot be reached", async () => {
    const { f } = backend("jwt");
    configureAuth("http://api", f);
    await loadMode();
    await login("analyst1", "right-password");
    f.mockImplementationOnce(async () => {
      throw new Error("offline");
    });
    await expect(logout()).rejects.toThrow("offline");
    expect(getAuth().user).toBeNull();
    expect(identityHeaders()).toEqual({});
  });

  it("a backend without login routes (the plain no-Docker demo, D236) is the demo mode", async () => {
    configureAuth("http://api", async () => json(404, { detail: "Not Found" }));
    await loadMode();
    expect(getAuth()).toMatchObject({ mode: "demo", ready: true, user: null });
    expect(getAuth().demoLabel).toMatch(/no login/);
  });

  it("demo mode passes 401s through without refreshing", async () => {
    const { f } = backend("demo");
    configureAuth("http://api", f);
    await loadMode();
    const api = vi.fn(async () => json(401, {}));
    expect((await withAuth(api)("http://api/api/meta")).status).toBe(401);
    expect(f.mock.calls.some(([u]) => String(u).endsWith("/api/auth/refresh"))).toBe(false);
  });
});
