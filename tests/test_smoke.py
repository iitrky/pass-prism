"""Fast smoke tests:  python -m pytest -q   (or python tests/test_smoke.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pass_prism import run, SCHEMES, AISLE


def test_all_schemes_run_default():
    keys = ('prism', 'nopred', 'nomap', 'hung', 'reactive', 'ddpg', 'oracle', 'gtrack', 'mimo', 'fixed', 'blind', 'genie')
    out = run({'warm': 20, 'T': 30}, seed=0, schemes=keys)
    for k in keys:
        assert out[k].shape == (12,) and np.all(np.isfinite(out[k])) and np.all(out[k] >= 0)


def test_aisle_scenario_runs():
    out = run({**AISLE, 'warm': 20, 'T': 30}, seed=1, schemes=('prism', 'ddpg_aisle', 'oracle'))
    assert all(np.isfinite(v).all() for v in out.values())


def test_reproducible():
    a = run({'warm': 10, 'T': 20}, seed=3, schemes=('prism',))['prism']
    b = run({'warm': 10, 'T': 20}, seed=3, schemes=('prism',))['prism']
    assert np.allclose(a, b)


if __name__ == '__main__':
    test_all_schemes_run_default(); test_aisle_scenario_runs(); test_reproducible(); print('all smoke tests passed')
