/**
 * Who is acting.
 * - With login on (D235): the logged-in user, proven by the bearer access token. The self-declared
 *   headers are not sent; the backend ignores them anyway.
 * - In the no-login demo mode (D208): a named analyst and a role, self-declared in Settings and
 *   remembered in this browser, sent with every request so the audit log records them. An empty
 *   name is sent as nothing and the server records "unknown".
 */
import { accessToken, getAuth } from "./auth";
import type { Role } from "./types";

let current: { analyst: string; role: Role } = { analyst: "", role: "analyst" };

export function setIdentity(analyst: string, role: Role): void {
  current = { analyst: analyst.trim(), role };
}

export function getIdentity(): { analyst: string; role: Role } {
  return current;
}

export function identityHeaders(): Record<string, string> {
  if (getAuth().mode === "jwt") {
    const t = accessToken();
    return t ? { Authorization: `Bearer ${t}` } : {};
  }
  const h: Record<string, string> = { "X-Vaultic-Role": current.role };
  // header values must be ASCII; a non-ASCII name is percent-encoded rather than dropped
  if (current.analyst) h["X-Vaultic-Analyst"] = /^[\x20-\x7e]*$/.test(current.analyst) ? current.analyst : encodeURIComponent(current.analyst);
  return h;
}
