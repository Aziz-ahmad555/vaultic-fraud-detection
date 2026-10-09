"""Warnings are captured, grouped and flagged, never silenced (item 1 after D61)."""

import json
import warnings

from vaultic.eval.warn_capture import capture_warnings


# categories are matched by name; importing sklearn.exceptions here, before LightGBM is loaded,
# breaks later LightGBM tests: scikit-learn must not load before LightGBM (D22)
class ConvergenceWarning(UserWarning):
    pass


def test_capture_groups_counts_and_flags(capsys):
    with capture_warnings() as caught:
        for _ in range(3):
            warnings.warn("lbfgs failed to converge", ConvergenceWarning, stacklevel=1)
        warnings.warn("old api", DeprecationWarning, stacklevel=1)
        warnings.warn("plain", UserWarning, stacklevel=1)
    s = caught.summary()
    assert s["n_warnings"] == 5 and s["n_ignored"] == 1
    assert s["flagged"] == {"ConvergenceWarning": 3}
    assert {g["category"] for g in s["groups"]} == {"ConvergenceWarning", "UserWarning"}
    assert caught.flag_text() == "WARNINGS: ConvergenceWarning x3"
    err = capsys.readouterr().err
    assert "[warning x3] ConvergenceWarning" in err and "old api" not in err
    json.dumps(s)  # JSON-ready


def test_nothing_flagged_gives_empty_text(tmp_path):
    with capture_warnings(echo=False) as caught:
        warnings.warn("plain", UserWarning, stacklevel=1)
    assert caught.flag_text() == "" and caught.summary()["flagged"] == {}
    caught.write_log(tmp_path / "w.log")
    assert "UserWarning" in (tmp_path / "w.log").read_text(encoding="utf-8")


def test_origin_is_machine_independent():
    with capture_warnings(echo=False) as caught:
        warnings.warn("x", UserWarning, stacklevel=1)
    (origin,) = [g["origin"] for g in caught.summary()["groups"]]
    assert ":" in origin and "\\" not in origin and not origin.startswith(("C:", "E:", "/"))
