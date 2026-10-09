"""The five evidence views and their feature columns (D89).

  tabular     B5's features WITHOUT the label-derived ones: label-free, transaction-level
              (raw IEEE-CIS columns with V-reduction + the label-free point-in-time uid features)
  behavioral  the behavioral features (features/behavioral.py) + the label-derived customer
              features: the customer view; missing for cold-start uids (no history)
  relational  the graph features (features/graph.py, setting C) incl. neighbour fraud rates;
              available only with non-hub relational evidence (D52)
  anomaly     the forward-chained anomaly scores (views/anomaly_build.py, D68)
  temporal    XGBoost on the flattened sequence inputs (D88); missing without history

The label-derived features moved from the tabular to the behavioral view after the ablation
D65 showed they carry almost all of B5's margin, and only for uids with history.
"""

from __future__ import annotations

# Label-derived base features (found by perturbation, reports/label_features.py): two depend on
# label VALUES, one on label TIMING. All respect the label delay L.
LABEL_DERIVED = ("uid_fraud_known", "uid_fraud_rate_known", "uid_n_labels_known")

VIEW_FEATURE_SETS = {
    "tabular": "tabular_view",  # features/sets.py
    "behavioral": "behavioral_view",
    "graph": "graph_only",
    "anomaly": "anomaly_only",
    "temporal": "sequence_only",
}


def tabular_columns(b5_columns) -> list[str]:
    """The tabular view's columns: B5's design-matrix columns minus the label-derived ones."""
    return [c for c in b5_columns if c not in LABEL_DERIVED]


def behavioral_columns(behavioral_file_columns) -> list[str]:
    """The behavioral view's columns: the behavioral features + the label-derived features."""
    cols = [c for c in behavioral_file_columns if c != "TransactionID"]
    return cols + [c for c in LABEL_DERIVED if c not in cols]
