"""The free-space guard and the temp redirect (D239)."""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("disk_space", ROOT / "tools/disk_space.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def test_a_drive_below_the_floor_or_missing_is_reported():
    free = {"C:\\": 9.9, "E:\\": 10.0}
    assert ds.problems(usage=free.__getitem__) == [
        "C:\\ has 9.9 GB free, below the 10 GB floor (D239)"
    ]

    def missing(d):
        raise OSError(d)

    assert ds.problems(drives=("Z:\\",), usage=missing) == ["Z:\\ is not available"]
    assert ds.problems(usage=lambda d: 50.0) == []


@pytest.mark.skipif(
    os.name != "nt" or not Path("E:/dev-cache").exists(), reason="Windows with E:/dev-cache only"
)
def test_temporary_files_of_the_tests_go_to_drive_e(tmp_path):
    assert str(tmp_path).upper().startswith("E:"), tmp_path
    assert tempfile.gettempdir().upper().startswith("E:")
    assert os.environ["TEMP"].upper().startswith("E:") and os.environ["TMP"].upper().startswith(
        "E:"
    )
