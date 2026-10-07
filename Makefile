# Run from the repository root with the virtual environment active.
# On Windows without make: powershell -File tools/paper_results.ps1
PYTHON ?= python
export PYTHONPATH := src

.PHONY: paper-results paper-results-list test

paper-results:  ## regenerate every paper table and figure from existing harness runs
	$(PYTHON) -m vaultic.reports.paper_results

paper-results-list:
	$(PYTHON) -m vaultic.reports.paper_results --list

test:
	$(PYTHON) -m pytest -q
