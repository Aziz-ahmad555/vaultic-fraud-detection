/** A fail-open action whose audit entry could not be written (D209): the result is shown anyway. */
export function AuditFailed() {
  return (
    <p className="xs m0" role="status" data-testid="audit-failed">
      Not recorded in the audit log: the audit write failed (the result is shown anyway).
    </p>
  );
}
