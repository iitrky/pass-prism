"""PRISM: predictive coverage placement of pinching antennas with finite movement speed.

Importing this package registers all schemes (PRISM, ablations, baselines, DDPG,
Oracle-PRISM) in ``SCHEMES`` and exposes ``run`` and ``Env``.
"""
from .extensions import *  # noqa: F401,F403
from .simulator import DEFAULT, Env, run, metrics  # noqa: F401
from .extensions import AISLE, SCHEMES, train_ddpg, metrics_ci, load_actor, save_actor  # noqa: F401

__version__ = "1.0.0"
