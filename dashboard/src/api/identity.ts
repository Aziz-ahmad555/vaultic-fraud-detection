/**
 * Who is acting (D208): a named analyst and a role, sent with every request to the backend so the
 * audit log records them. Set from Settings (remembered in this browser). There is no sign-in yet;
 * an empty name is sent as nothing and the server records "unknown".
 */
import type { Role } from "./types";

let current: { analyst: string; role: Role } = { analyst: "", role: "analyst" };

export function setIdentity(analyst: string, role: Role): void {
  current = { analyst: analyst.trim(), role };
}

export function getIdentity(): { analyst: string; role: Role } {
  return current;
}

export function identityHeaders(): Record<string, string> {
  const h: Record<string, string> = { "X-Vaultic-Role": current.role };
  // header values must be ASCII; a non-ASCII name is percent-encoded rather than dropped
  if (current.analyst) h["X-Vaultic-Analyst"] = /^[\x20-\x7e]*$/.test(current.analyst) ? current.analyst : encodeURIComponent(current.analyst);
  return h;
}
