"""Shared test setup: the audit log (D208) of every test goes to a temporary file, never to the
real log on drive E. On Windows with E:/dev-cache (D239): temporary files go to drive E, never C, and
the run stops before any test if drive C or E has less than 10 GB free."""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

DEV_CACHE = Path("E:/dev-cache")


def _disk_space():
    spec = importlib.util.spec_from_file_location(
        "disk_space", Path(__file__).resolve().parents[1] / "tools" / "disk_space.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_configure(config):
    if os.name != "nt" or not DEV_CACHE.exists():
        return  # Linux CI and other machines: unchanged
    tmp, root = DEV_CACHE / "tmp", DEV_CACHE / "pytest"
    tmp.mkdir(exist_ok=True)
    root.mkdir(exist_ok=True)
    # pytest's own temporary directories (tmp_path) and Python's tempfile: drive E (D239)
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(root))
    for key in ("TEMP", "TMP"):
        os.environ[key] = str(tmp)
    tempfile.tempdir = str(tmp)
    bad = _disk_space().problems()
    if bad:
        pytest.exit("not starting the tests: " + "; ".join(bad), returncode=3)


@pytest.fixture(autouse=True, scope="session")
def _temporary_audit_db(tmp_path_factory):
    old = os.environ.get("VAULTIC_AUDIT_DB")
    os.environ["VAULTIC_AUDIT_DB"] = str(tmp_path_factory.mktemp("audit") / "audit.sqlite")
    yield
    if old is None:
        os.environ.pop("VAULTIC_AUDIT_DB", None)
    else:
        os.environ["VAULTIC_AUDIT_DB"] = old
