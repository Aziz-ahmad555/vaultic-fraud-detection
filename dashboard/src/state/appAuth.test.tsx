import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { configureAuth, login, resetAuthForTests } from "../api/auth";

vi.mock("../api", () => ({
  DATA_SOURCE: "http",
  api: { range: async () => ({ first: 151 * 86400, last: 183 * 86400 - 1 }) },
}));

const { AppProvider, useApp } = await import("./app");
const { getIdentity } = await import("../api/identity");

function Who() {
  const { role, analyst } = useApp();
  return <p>{`${analyst}|${role}`}</p>;
}

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => resetAuthForTests());

describe("AppProvider with login on (D235)", () => {
  it("takes the role and name from the logged-in user, not from the browser's saved settings", async () => {
    window.localStorage.setItem("evidence-room-prefs-v1", JSON.stringify({ role: "analyst", analyst: "self-declared" }));
    configureAuth("http://api", async (url: string) =>
      url.endsWith("/api/auth/login") ? json(200, { accessToken: "t", expiresIn: 900, user: { username: "admin1", role: "admin" } }) : json(200, { mode: "jwt", demoLabel: null }),
    );
    const { loadMode } = await import("../api/auth");
    await loadMode().catch(() => undefined);
    await login("admin1", "right-password");
    render(
      <AppProvider>
        <Who />
      </AppProvider>,
    );
    await waitFor(() => expect(screen.getByText("admin1|admin")).toBeTruthy());
    expect(getIdentity()).toEqual({ analyst: "admin1", role: "admin" });
  });
});
