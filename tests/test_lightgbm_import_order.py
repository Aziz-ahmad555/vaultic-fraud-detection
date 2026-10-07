"""Regression test for the LightGBM crash when scikit-learn is imported first (D22)."""

import subprocess
import sys

import pytest

from vaultic.paths import REPO_ROOT

pytest.importorskip("lightgbm")

SCRIPT = """
import vaultic.eval.run  # imports scikit-learn, as the harness does
import numpy as np
from vaultic.views.tabular import make_model
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 20)).astype("float32")
y = (X[:, 0] > 1).astype("int8")
make_model("lightgbm", {"n_estimators": 10, "num_leaves": 31}, 0).fit(X, y)
print("ok")
"""


def test_lightgbm_trains_after_harness_imports():
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
