"""Vaultic: adaptive multi-view framework for explainable, uncertainty- and drift-aware fraud
detection."""

# LightGBM must be loaded before scikit-learn: on Windows, importing scikit-learn first makes
# LightGBM crash with "access violation reading 0x0" when it builds a Dataset (native
# runtime clash; reproduced and recorded in research/decisions.md D22).
try:
    import lightgbm  # noqa: F401
except ImportError:  # optional dependency
    pass
