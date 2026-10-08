import type { ReactNode } from "react";

export function Loading({ what }: { what: string }) {
  return (
    <div className="state" role="status">
      <p className="muted">Loading {what}…</p>
    </div>
  );
}

export function ErrorState({ what, error, onRetry }: { what: string; error: Error; onRetry?: () => void }) {
  return (
    <div className="state" role="alert">
      <h2>Couldn't load {what}</h2>
      <p>
        {error.message} {onRetry ? "Try again; if it keeps failing, reload the page." : "Reload the page to try again."}
      </p>
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="state">
      <h2>{title}</h2>
      {children && <p className="muted">{children}</p>}
      {action}
    </div>
  );
}
