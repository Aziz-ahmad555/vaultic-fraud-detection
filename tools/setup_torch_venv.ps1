# Separate CPU-only PyTorch environment for the Phase 5 GRU (research/decisions.md D39).
# Kept apart from .venv, which the Phase 2 chain uses: nothing here touches that venv.
# Run from the repo (or worktree) root:  powershell -File tools\setup_torch_venv.ps1
$ErrorActionPreference = "Stop"
$venv = ".venv-torch"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3.11 -m venv $venv
}
$py = "$venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# Same pins as requirements.txt for the shared libraries, so results match the main venv.
$constraints = "numpy==1.26.4`npandas==2.2.3`nscikit-learn==1.3.2`nscipy==1.17.1`nnetworkx==3.6.1`nPyYAML==6.0.3`npytest==9.1.1`n"
Set-Content -Path "$venv\constraints.txt" -Value $constraints -Encoding ascii
& $py -m pip install -c "$venv\constraints.txt" numpy pandas scikit-learn scipy networkx PyYAML pytest
& $py -m pip install -c "$venv\constraints.txt" torch --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -c "$venv\constraints.txt" dice-ml
& $py -m pip freeze | Set-Content -Path "requirements-torch.txt" -Encoding ascii
& $py -c "import torch, numpy, dice_ml; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
