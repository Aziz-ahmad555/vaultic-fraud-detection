"""Shared test setup: the audit log (D208) of every test goes to a temporary file, never to the
real log on drive E."""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _temporary_audit_db(tmp_path_factory):
    old = os.environ.get("VAULTIC_AUDIT_DB")
    os.environ["VAULTIC_AUDIT_DB"] = str(tmp_path_factory.mktemp("audit") / "audit.sqlite")
    yield
    if old is None:
        os.environ.pop("VAULTIC_AUDIT_DB", None)
    else:
        os.environ["VAULTIC_AUDIT_DB"] = old
