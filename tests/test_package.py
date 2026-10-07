import importlib

import pytest

SUBPACKAGES = [
    "data",
    "features",
    "views",
    "fusion",
    "trust",
    "explain",
    "drift",
    "learning",
    "eval",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(f"vaultic.{name}")
    assert module.__doc__
