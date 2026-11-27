import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { configureAuth, resetAuthForTests, setMockMode } from "../api/auth";

const { AuthGate, CACHE_OFF } = await import("./AuthGate");

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function app() {
  return render(
    <AuthGate>
      <p>the pages</p>
    </AuthGate>,
  );
}

afterEach(() => resetAuthForTests());

describe("AuthGate (D235)", () => {
  it("the no-login demo mode shows the pages under a visible demo banner", async () => {
    configureAuth("http://api", async () => json(200, { mode: "demo", demoLabel: "Demo mode: no login; names and roles are self-declared" }));
    app();
    expect(await screen.findByText("the pages")).toBeTruthy();
    expect(screen.getByRole("note").textContent).toMatch(/Demo mode: no login/);
  });

  it("the mock data source is the demo mode too", async () => {
    setMockMode();
    app();
    expect(await screen.findByText("the pages")).toBeTruthy();
    expect(screen.getByRole("note")).toBeTruthy();
  });

  it("with login on, the pages appear only after a successful login, without the demo banner", async () => {
    configureAuth("http://api", async (url: string, init?: RequestInit) => {
      if (url.endsWith("/api/auth/mode")) return json(200, { mode: "jwt", demoLabel: null });
      if (url.endsWith("/api/auth/refresh")) return json(401, { detail: "invalid refresh token" });
      const body = JSON.parse(String(init?.body));
      return body.password === "right-password"
        ? json(200, { accessToken: "t", expiresIn: 900, user: { username: "admin1", role: "admin" } })
        : json(401, { detail: "invalid username or password" });
    });
    app();
    expect(await screen.findByRole("heading", { name: "Log in to Vaultic" })).toBeTruthy();
    expect(screen.queryByText("the pages")).toBeNull();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin1" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    expect((await screen.findByRole("alert")).textContent).toBe("invalid username or password");
    expect((screen.getByLabelText("Password") as HTMLInputElement).value).toBe(""); // cleared
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "right-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    await waitFor(() => expect(screen.getByText("the pages")).toBeTruthy());
    expect(screen.queryByRole("note")).toBeNull();
  });

  it("shows a small cache-off note only while the backend reports Redis down (D244)", async () => {
    const redis = "down";
    configureAuth("http://api", async (url: string) => {
      if (url.endsWith("/api/health")) return json(200, { api: "up", redis, postgres: "unused" });
      return json(200, { mode: "demo", demoLabel: "Demo mode: no login; names and roles are self-declared" });
    });
    app();
    expect(await screen.findByText(CACHE_OFF)).toBeTruthy();
  });

  it("no cache note with Redis up, off, a backend without /api/health, or the mock data source", async () => {
    for (const health of [json(200, { api: "up", redis: "up", postgres: "up" }), json(200, { api: "up", redis: "off", postgres: "unused" }), json(404, { detail: "Not Found" })]) {
      resetAuthForTests();
      configureAuth("http://api", async (url: string) => (url.endsWith("/api/health") ? health.clone() : json(200, { mode: "demo", demoLabel: "x" })));
      const { unmount } = app();
      expect(await screen.findByText("the pages")).toBeTruthy();
      await new Promise((r) => setTimeout(r, 20));
      expect(screen.queryByText(CACHE_OFF)).toBeNull();
      unmount();
    }
    resetAuthForTests();
    setMockMode();
    const { unmount } = app();
    expect(await screen.findByText("the pages")).toBeTruthy();
    expect(screen.queryByText(CACHE_OFF)).toBeNull();
    unmount();
  });
});
