import { useEffect, useState, useSyncExternalStore, type FormEvent, type ReactNode } from "react";
import { getAuth, loadMode, login, subscribeAuth, type AuthState } from "../api/auth";
import { ErrorState, Loading } from "./States";

/** The login session as React state (D235). */
export function useAuth(): AuthState {
  return useSyncExternalStore(subscribeAuth, getAuth, getAuth);
}

/**
 * Login on (jwt): the app (and every data request) starts only after a login. AppProvider takes the
 * role and name from the logged-in user. No-login demo mode: the app starts at once, under a visible
 * "Demo mode" banner (D235, Q8). It sits outside AppProvider, so no request is sent before the
 * session is known.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!auth.ready) loadMode().catch((e: Error) => setError(e.message));
  }, [auth.ready]);

  if (error) return <ErrorState what="the login settings" error={new Error(error)} />;
  if (!auth.ready) return <Loading what="the login settings" />;
  if (auth.mode === "jwt" && !auth.user) return <LoginPage />;
  return (
    <>
      {auth.mode === "demo" && (
        <p className="demo-banner xs" role="note">
          {auth.demoLabel}
        </p>
      )}
      {children}
    </>
  );
}

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username.trim(), password);
    } catch (err) {
      setError((err as Error).message);
      setPassword("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login" aria-labelledby="login-title">
      <form className="pane login-form" onSubmit={submit}>
        <div className="pane-head">
          <h1 id="login-title">Log in to Vaultic</h1>
        </div>
        <div className="pane-body">
          <label className="field">
            <span>Username</span>
            <input type="text" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required maxLength={60} />
          </label>
          <label className="field">
            <span>Password</span>
            <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>
          {error && (
            <p className="login-error" role="alert">
              {error}
            </p>
          )}
          <button type="submit" className="btn btn-primary" disabled={busy || !username || !password}>
            {busy ? "Logging in" : "Log in"}
          </button>
          <p className="muted small">Local demo login with role checks. Test users come from the seed command; there is no TLS on localhost.</p>
        </div>
      </form>
    </main>
  );
}
