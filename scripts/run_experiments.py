"""Reproduce every experiment in the paper.

Usage:
  python scripts/run_experiments.py <experiment> [--seeds N] [--quick]
  experiments: train_default train_aisle ablation aisle aisle_speed power vpa length vuser users humans all
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import argparse, json, time, numpy as np
from pass_prism import *
from pass_prism.extensions import save_actor

DEFAULT['warm'] = 200; DEFAULT['T'] = 600          # 20 s warm-up + 60 s measurement
OUT = 'results'
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def sweep(name, param, values, schemes, seeds, base=None):
    out = {'param': param, 'values': values, 'seeds': seeds, 'res': {k: {q: [] for q in ['sum', 'sum_std', 'ci', 'p5', 'jain']} for k in schemes}}
    for v in values:
        res = {k: [] for k in schemes}
        for s in range(seeds):
            over = dict(base or {}); over[param] = v
            o = run(over, seed=100 + s, schemes=tuple(schemes))
            for k in o: res[k].append(o[k])
        for k in schemes:
            m = metrics_ci(res[k])
            for q in ['sum', 'sum_std', 'ci', 'p5', 'jain']:
                out['res'][k][q].append(m[q])
        log(name, v, {k: (round(out['res'][k]['sum'][-1], 2), round(out['res'][k]['ci'][-1], 2)) for k in schemes})
    json.dump(out, open(f'{OUT}/{name}.json', 'w'))


def ablation(name, keys, seeds, base=None):
    prm = dict(DEFAULT); prm.update(base or {})
    res = {k: [] for k in keys}; tm = {k: 0.0 for k in keys}; trace = None
    for s in range(seeds):
        env = Env(prm, 100 + s); env.step()
        Sc = {k: SCHEMES[k](env) for k in keys}
        acc = {k: np.zeros(prm['K']) for k in keys}; tr = {k: [] for k in keys}
        for t in range(prm['warm'] + prm['T']):
            env.step()
            for k, sc in Sc.items():
                t0 = time.perf_counter(); sc.act(); tm[k] += time.perf_counter() - t0
                r = sc.serve()
                if t >= prm['warm']: acc[k] += r
                if s == 0: tr[k].append(float(r.sum()))
        for k in keys: res[k].append(acc[k] / prm['T'])
        if s == 0: trace = {k: tr[k] for k in keys}
        log(name, 'seed', s, {k: round(float(acc[k].sum() / prm['T']), 2) for k in ['prism', 'reactive'] if k in keys})
    nsl = seeds * (prm['warm'] + prm['T'])
    out = {k: {**metrics_ci(res[k]), 'ms_per_slot': 1e3 * tm[k] / nsl} for k in keys}
    out['trace'] = trace; out['seeds'] = seeds; out['warm'] = prm['warm']
    json.dump(out, open(f'{OUT}/{name}.json', 'w'))


def train(tag, base):
    actor, hist = train_ddpg(base, episodes=EPISODES, T_ep=400, seed=0, log=log)
    os.makedirs(MODELS, exist_ok=True); save_actor(actor, f'{MODELS}/ddpg_{tag}.npz')
    json.dump(hist, open(f'{OUT}/ddpg_hist_{tag}.json', 'w'))


ap = argparse.ArgumentParser()
ap.add_argument('exp')
ap.add_argument('--seeds', type=int, default=None, help='override number of realizations')
ap.add_argument('--quick', action='store_true', help='short smoke run (2 seeds, 100+100 slots)')
args = ap.parse_args()
import pass_prism.extensions as E
MODELS, EPISODES = 'models', 150
if args.quick:            # never overwrite the paper's results/models in quick mode
    DEFAULT['warm'] = 100; DEFAULT['T'] = 100
    OUT, MODELS, EPISODES = 'results_quick', 'models_quick', 2
    os.makedirs(OUT, exist_ok=True)
E.MODEL_DIR = os.path.join(ROOT, MODELS)
ORDER = ['train_default', 'ablation', 'train_aisle', 'aisle', 'aisle_speed', 'power', 'vpa', 'length', 'vuser', 'users', 'humans']
EXPS = ORDER if args.exp == 'all' else [args.exp]
for exp in EXPS:
    log('===', exp, time.strftime('%H:%M:%S'))
    S20 = args.seeds or (2 if args.quick else 20)
    S10 = args.seeds or (2 if args.quick else 8)
    NA = args.seeds or (2 if args.quick else 12)
    NS = args.seeds or (2 if args.quick else 6)
    DEFKEYS = ['prism', 'nopred', 'nomap', 'hung', 'reactive', 'ddpg', 'oracle', 'gtrack', 'mimo', 'fixed', 'blind', 'genie']
    if exp == 'train_default': train('default', {})
    if exp == 'train_aisle': train('aisle', AISLE)
    if exp == 'ablation': ablation('ablation', DEFKEYS, S20)
    if exp == 'aisle': ablation('aisle', ['prism', 'nopred', 'nomap', 'reactive', 'ddpg_aisle', 'oracle', 'mimo', 'fixed'], NA, AISLE)
    if exp == 'aisle_speed': sweep('aisle_speed', 'v_user', [0.5, 1.5, 3.0], ['prism', 'nopred', 'nomap', 'reactive'], NS, AISLE)
    if exp == 'power': sweep('power', 'P_dBm', [-5, 0, 10, 20, 25], ['prism', 'reactive', 'ddpg', 'mimo', 'fixed', 'genie'], S10)
    if exp == 'vpa': sweep('vpa', 'v_pa', [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], ['prism', 'reactive', 'gtrack', 'mimo', 'fixed'], S10)
    if exp == 'length': sweep('length', 'L', [20.0, 30.0, 45.0, 60.0, 90.0], ['prism', 'reactive', 'mimo', 'fixed', 'genie'], S10)
    if exp == 'vuser': sweep('vuser', 'v_user', [0.5, 1.5, 3.0, 5.0], ['prism', 'nopred', 'reactive', 'mimo'], S10)
    if exp == 'users': sweep('users', 'K', [8, 12, 20], ['prism', 'reactive', 'mimo', 'fixed'], S10)
    if exp == 'humans': sweep('humans', 'n_human', [0, 20, 40], ['prism', 'nomap', 'reactive', 'mimo'], S10)
