"""Revision (v3) experiments. Usage: python scripts/run_v3.py <exp> [--seeds N]"""
import os, sys, json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from pass_prism import *
from pass_prism.extensions import DDPGScheme, metrics_ci, save_actor, load_actor
from pass_prism.v3 import *
import pass_prism.extensions as E
DEFAULT['warm'] = 200; DEFAULT['T'] = 600
OUT = 'results/v3'; os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)
SCHEMES['td3'] = lambda e: DDPGScheme(e, load_actor('td3_default'))
SCHEMES['td3_aisle'] = lambda e: DDPGScheme(e, load_actor('td3_aisle'))


def main_cmp(name, keys, seeds, base=None):
    prm = dict(DEFAULT); prm.update(base or {})
    res = {k: [] for k in keys}; tm = {k: 0.0 for k in keys}; s0 = 0
    if os.path.exists(f'{OUT}/{name}.json'):                      # resume from checkpoint
        old = json.load(open(f'{OUT}/{name}.json')); s0 = old['seeds']
        for k in keys:
            res[k] = list(np.array(old[k]['all']).reshape(s0, prm['K']))
            tm[k] = old[k]['ms_per_slot'] * s0 * (prm['warm'] + prm['T']) / 1e3
        log(name, 'resuming at seed', s0)
    for s in range(s0, seeds):
        env = Env(prm, 100 + s); env.step()
        Sc = {k: SCHEMES[k](env) for k in keys}; acc = {k: np.zeros(prm['K']) for k in keys}
        for t in range(prm['warm'] + prm['T']):
            env.step()
            for k, sc in Sc.items():
                t0 = time.perf_counter(); sc.act(); tm[k] += time.perf_counter() - t0
                r = sc.serve()
                if t >= prm['warm']: acc[k] += r
        for k in keys: res[k].append(acc[k] / prm['T'])
        log(name, 'seed', s, {k: round(float(res[k][-1].sum()), 2) for k in keys})
        out = {k: {**metrics_ci(res[k]), 'ms_per_slot': 1e3 * tm[k] / ((s + 1) * (prm['warm'] + prm['T']))} for k in keys}
        out['seeds'] = s + 1
        json.dump(out, open(f'{OUT}/{name}.json', 'w'))          # checkpoint after every seed


def sweep(name, grid, seeds, base=None):
    out = {'points': [], 'seeds': seeds}
    for over, keys in grid:
        res = {k: [] for k in keys}
        for s in range(seeds):
            o = run({**(base or {}), **over}, seed=100 + s, schemes=tuple(keys))
            for k in o: res[k].append(o[k])
        pt = {'over': over, 'res': {k: {q: metrics_ci(v)[q] for q in ['sum', 'ci', 'p5', 'jain', 'seed_sums']} for k, v in res.items()}}
        out['points'].append(pt); json.dump(out, open(f'{OUT}/{name}.json', 'w'))
        log(name, over, {k: round(pt['res'][k]['sum'], 2) for k in keys})


ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('--seeds', type=int, default=None)
a = ap.parse_args(); log('===', a.exp, time.strftime('%H:%M:%S'))
if a.exp == 'train_td3_default':
    A, h = train_td3({}, episodes=200, log=log); save_actor(A, 'models/td3_default.npz'); json.dump(h, open(f'{OUT}/td3_hist_default.json', 'w'))
if a.exp == 'train_td3_aisle':
    A, h = train_td3(AISLE, episodes=200, log=log); save_actor(A, 'models/td3_aisle.npz'); json.dump(h, open(f'{OUT}/td3_hist_aisle.json', 'w'))
if a.exp == 'main':
    main_cmp('main', ['prism3', 'prism3_nomap', 'prism3_nopred', 'oracle', 'reactive', 'td3', 'gtrack', 'mimo', 'fixed', 'genie',
                      'prismM2', 'prismM3', 'prismM3_nomap', 'reactM3', 'fixedM3'], a.seeds or 20)
if a.exp == 'aisle':
    main_cmp('aisle', ['prism3', 'prism3_nomap', 'prism3_nopred', 'oracle', 'or_pos', 'or_blk', 'reactive', 'ddpg_aisle', 'mimo',
                       'prismM3', 'prismM3_nomap', 'reactM3', 'fixedM3'], a.seeds or 10, AISLE)
if a.exp == 'dp':
    rows = []
    for s in range(a.seeds or 20):
        rows.append(dp_experiment(100 + s, T=600)); log('dp', s, {k: round(v, 3) for k, v in rows[-1].items()})
        json.dump(rows, open(f'{OUT}/dp.json', 'w'))
if a.exp == 'mv':
    grid = [({'v_pa': v}, [f'prismM{M}', f'reactM{M}']) for M in (1, 2, 3, 4) for v in (0.5, 2.0)]
    grid = [({**o, 'M_tag': k[0][-1]}, k) for o, k in grid]
    sweep('mv', grid, a.seeds or 6)
if a.exp == 'mv2':          # remaining antenna-count x speed points
    done = {(p['over']['v_pa'], p['over']['M_tag']) for p in json.load(open(f'{OUT}/mv.json'))['points']} if os.path.exists(f'{OUT}/mv.json') else set()
    prev = json.load(open(f'{OUT}/mv.json')) if os.path.exists(f'{OUT}/mv.json') else {'points': [], 'seeds': 6}
    for M in (1, 2, 3, 4):
        for v in (0.5, 2.0):
            if (v, str(M)) in done: continue
            keys = [f'prismM{M}', f'reactM{M}']; res = {k: [] for k in keys}
            for s in range(6):
                o = run({'v_pa': v}, seed=100 + s, schemes=tuple(keys))
                for k in o: res[k].append(o[k])
            prev['points'].append({'over': {'v_pa': v, 'M_tag': str(M)}, 'res': {k: {q: metrics_ci(r)[q] for q in ['sum', 'ci', 'p5', 'jain', 'seed_sums']} for k, r in res.items()}})
            json.dump(prev, open(f'{OUT}/mv.json', 'w')); log('mv', M, v, {k: round(prev['points'][-1]['res'][k]['sum'], 2) for k in keys})
if a.exp == 'length3':
    sweep('length3', [({'L': L}, ['prism3', 'prismM2', 'reactive', 'mimo']) for L in (20.0, 30.0, 45.0, 60.0, 90.0)], a.seeds or 6)
