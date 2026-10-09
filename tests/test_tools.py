import importlib.util

from vaultic.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "check", REPO_ROOT / "tools" / "check_no_data_files.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_blocks_data_models_and_secrets():
    for path in [
        "data/raw/train_transaction.csv",
        "x/merged.parquet",
        "a.db",
        "m.joblib",
        ".env",
        "legacy/fyp1/.env",
        "data/raw/anything.bin",
    ]:
        assert check.main([path]) == 1, path


def test_allows_code_and_examples():
    assert (
        check.main(
            [
                ".env.example",
                "src/vaultic/data/load.py",
                "data/raw/.gitkeep",
                "data/raw/.gitignore",
                "data/raw/train_transaction.csv.dvc",
                "experiments/configs/splits.yaml",
            ]
        )
        == 0
    )


def test_graph_availability_script_reports_no_test_period():
    """Review R2 (D84): the D52 availability script prints training and validation only."""
    import ast

    src = (REPO_ROOT / "tools" / "graph_availability.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    strings = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert {"train", "validation"} <= strings and "test" not in strings
