# Separate environment for DiCE counterfactuals (research/decisions.md D41).
# dice-ml needs xgboost and lightgbm; pinned to the main venv's versions. Kept apart from
# .venv-torch because xgboost 2.0.3 imported before torch breaks torch's DLL loading on
# Windows, and apart from .venv, which the Phase 2 chain uses.
# Run from the repo (or worktree) root:  powershell -File tools\setup_dice_venv.ps1
$ErrorActionPreference = "Stop"
$venv = ".venv-dice"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3.11 -m venv $venv
}
$py = "$venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
$constraints = "numpy==1.26.4`npandas==2.2.3`nscikit-learn==1.3.2`nscipy==1.17.1`nnetworkx==3.6.1`nPyYAML==6.0.3`npytest==9.1.1`nxgboost==2.0.3`nlightgbm==4.7.0`n"
Set-Content -Path "$venv\constraints.txt" -Value $constraints -Encoding ascii
& $py -m pip install -c "$venv\constraints.txt" numpy pandas scikit-learn scipy networkx PyYAML pytest xgboost lightgbm dice-ml
& $py -m pip check
& $py -m pip freeze | Set-Content -Path "requirements-dice.txt" -Encoding ascii
& $py -c "import xgboost, lightgbm, dice_ml; print('xgboost', xgboost.__version__, 'lightgbm', lightgbm.__version__)"
