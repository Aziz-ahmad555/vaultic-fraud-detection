/**
 * Login session (D235, Sprint 3 step 3.2). The backend tells us its mode (GET /api/auth/mode):
 * - "jwt": login is on. The short-lived access token is kept in memory only (never localStorage) and
 *   sent as a bearer token. The refresh token is an httpOnly cookie the page cannot read. On a 401
 *   the request is retried once after a silent refresh, otherwise the login page is shown.
 * - "demo": the no-login demo mode (and the mock data source). Names and roles are self-declared,
 *   the D208 headers are sent, and the dashboard says so on screen.
 */
import type { Role } from "./types";

export type AuthMode = "jwt" | "demo";
export interface SessionUser {
  username: string;
  role: Role;
}
export interface AuthState {
  mode: AuthMode;
  /** set once the backend's mode is known (always true with the mock data source) */
  ready: boolean;
  user: SessionUser | null;
  demoLabel: string;
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export const DEMO_LABEL = "Demo mode: no login; names and roles are self-declared";
const CSRF = { "X-Vaultic-CSRF": "1" };

let state: AuthState = { mode: "demo", ready: false, user: null, demoLabel: DEMO_LABEL };
let token: string | null = null;
let refreshing: Promise<boolean> | null = null;
const listeners = new Set<() => void>();
let base = "http://127.0.0.1:8000";
let baseFetch: Fetch = (input, init) => fetch(input, init);

function set(next: Partial<AuthState>): void {
  state = { ...state, ...next };
  for (const fn of listeners) fn();
}

export function configureAuth(baseUrl: string, fetchImpl?: Fetch): void {
  base = baseUrl.replace(/\/$/, "");
  if (fetchImpl) baseFetch = fetchImpl;
}

export function getAuth(): AuthState {
  return state;
}

export function accessToken(): string | null {
  return token;
}

export function subscribeAuth(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** The mock data source has no backend: demo mode, ready at once. */
export function setMockMode(): void {
  token = null;
  set({ mode: "demo", ready: true, user: null, demoLabel: DEMO_LABEL });
}

async function detail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string };
    return body.detail || `${res.status}`;
  } catch {
    return `${res.status}`;
  }
}

function startSession(body: { accessToken: string; user: SessionUser }): void {
  token = body.accessToken;
  set({ user: body.user });
}

export function endSession(): void {
  token = null;
  set({ user: null });
}

/** Ask the backend which mode it runs in; with login on, try to resume a session silently. */
export async function loadMode(): Promise<AuthState> {
  const res = await baseFetch(`${base}/api/auth/mode`);
  if (res.status === 404) {
    // the plain no-Docker demo backend (services/api/demo_server.py, unchanged, D236) has no login
    token = null;
    set({ mode: "demo", user: null, demoLabel: DEMO_LABEL, ready: true });
    return state;
  }
  if (!res.ok) throw new Error(await detail(res));
  const body = (await res.json()) as { mode: AuthMode; demoLabel: string | null };
  if (body.mode === "jwt") {
    set({ mode: "jwt", demoLabel: "" });
    await refreshSession();
  } else {
    token = null;
    set({ mode: "demo", user: null, demoLabel: body.demoLabel || DEMO_LABEL });
  }
  set({ ready: true });
  return state;
}

export async function login(username: string, password: string): Promise<SessionUser> {
  const res = await baseFetch(`${base}/api/auth/login`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const msg = await detail(res);
    const wait = res.headers.get("Retry-After");
    throw new Error(res.status === 429 && wait ? `${msg} (about ${Math.ceil(Number(wait) / 60)} min)` : msg);
  }
  const body = (await res.json()) as { accessToken: string; user: SessionUser };
  startSession(body);
  return body.user;
}

/** One silent refresh at a time; false (and the session ended) if it fails. */
export function refreshSession(): Promise<boolean> {
  if (!refreshing) {
    refreshing = (async () => {
      try {
        const res = await baseFetch(`${base}/api/auth/refresh`, { method: "POST", credentials: "include", headers: CSRF });
        if (!res.ok) {
          endSession();
          return false;
        }
        startSession((await res.json()) as { accessToken: string; user: SessionUser });
        return true;
      } catch {
        endSession();
        return false;
      } finally {
        refreshing = null;
      }
    })();
  }
  return refreshing;
}

export async function logout(): Promise<void> {
  try {
    await baseFetch(`${base}/api/auth/logout`, { method: "POST", credentials: "include", headers: CSRF });
  } finally {
    endSession();
  }
}

/** Wrap a client's fetch: with login on, a 401 gets one silent refresh and one retry with the new
 *  token; if that fails the session ends and the login page appears. */
export function withAuth(fetchImpl: Fetch): Fetch {
  return async (input, init) => {
    const res = await fetchImpl(input, init);
    if (res.status !== 401 || state.mode !== "jwt" || input.includes("/api/auth/")) return res;
    if (!(await refreshSession())) return res;
    const headers = { ...((init?.headers as Record<string, string>) ?? {}), Authorization: `Bearer ${token}` };
    return fetchImpl(input, { ...init, headers });
  };
}

/** Test helper: back to the initial state. */
export function resetAuthForTests(): void {
  token = null;
  refreshing = null;
  state = { mode: "demo", ready: false, user: null, demoLabel: DEMO_LABEL };
  listeners.clear();
}
