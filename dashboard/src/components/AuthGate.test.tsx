import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { configureAuth, resetAuthForTests, setMockMode } from "../api/auth";

const { AuthGate } = await import("./AuthGate");

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
});
