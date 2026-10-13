#!/usr/bin/env bash
# Full reproduction of all results and figures in the paper (single CPU core: several hours).
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
python tests/test_smoke.py
python scripts/run_experiments.py all
python scripts/make_figures.py
python scripts/make_diagrams.py
echo "Done: results/ and figures/ regenerated."
