"""Phase B attack scenarios (product layer, D190-D196): synthetic attacks injected into a replay of
real legitimate test-period traffic. Synthetic rows live only in memory and in scenario outputs
under the dev cache; they never enter research data or any research number. Every scenario metric
is labelled 'synthetic scenario - not the evaluation'."""

LABEL = "synthetic scenario - not the evaluation"
SYNTHETIC_ID_START = 1_000_000_000  # every synthetic TransactionID is >= this; real ones are below


def require_no_synthetic(transaction_ids) -> None:
    """Refuse any synthetic TransactionID. Used only inside the scenarios (on the replay's real
    rows); the research pipeline does not call it. Synthetic rows stay out of research data by
    construction: they exist only in memory and in outputs under the dev cache."""
    import numpy as np

    ids = np.asarray(transaction_ids)
    if len(ids) and (ids >= SYNTHETIC_ID_START).any():
        raise RuntimeError("a synthetic scenario row reached research data")
