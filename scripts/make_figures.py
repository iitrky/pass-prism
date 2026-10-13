"""Regenerate Figs. 3-11 from results/*.json into figures/."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.3, 'lines.markersize': 4.5,
                     'legend.fontsize': 6.5})
ST = {'prism': ('PRISM (proposed)', '#c00000', 'o', '-'), 'reactive': ('Reactive tracking PASS', '#1f77b4', 's', '--'),
      'gtrack': ('Genie tracking (speed-limited)', '#9467bd', '^', '-.'), 'mimo': ('Conventional MIMO BS', '#2ca02c', 'D', ':'),
      'fixed': ('Fixed PASS (midpoint)', '#7f7f7f', 'v', '--'), 'blind': ('Blind roaming PASS', '#8c564b', 'x', ':'),
      'genie': ('Ideal PASS ($v_{\\max}=\\infty$, genie)', 'k', '*', '-'), 'nopred': ('PRISM w/o prediction', '#ff7f0e', 'P', '--'),
      'nomap': ('PRISM w/o ORBM', '#17becf', 'h', '--'), 'ddpg': ('DDPG PASS', '#e377c2', '<', '-.'),
      'ddpg_aisle': ('DDPG PASS', '#e377c2', '<', '-.'), 'oracle': ('Oracle-PRISM (clairvoyant)', '#bcbd22', '>', '-'),
      'hung': ('Predictive one-to-one', '#aec7e8', 'p', ':')}
R = 'results/'
def load(n): return json.load(open(f'{R}{n}.json'))
def line(ax, d, q, keys=None, xlab='', ylab=''):
    for k in (keys or d['res']):
        l, c, m, ls = ST[k]; y = np.array(d['res'][k][q])
        if q == 'sum':
            ax.errorbar(d['values'], y, yerr=d['res'][k]['ci'], ls=ls, marker=m, color=c, label=l, capsize=2, elinewidth=0.8)
        else:
            ax.plot(d['values'], y, ls=ls, marker=m, color=c, label=l)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
def save(f, n): f.tight_layout(); f.savefig(f'figures/{n}.png', dpi=300); plt.close(f)
SR, P5 = 'Average sum rate (bit/s/Hz)', '5th-percentile user rate (bit/s/Hz)'

d = load('power'); f, a = plt.subplots(1, 2, figsize=(7.2, 2.8))
line(a[0], d, 'sum', xlab='Transmit power $P$ (dBm)', ylab=SR); a[0].set_title('(a)', fontsize=8); a[0].legend(loc='upper left')
line(a[1], d, 'p5', xlab='Transmit power $P$ (dBm)', ylab=P5); a[1].set_title('(b)', fontsize=8); save(f, 'fig3_power')

d = load('vpa'); f, a = plt.subplots(figsize=(3.6, 2.7))
line(a, d, 'sum', xlab='Maximum PA speed $v_{\\max}$ (m/s)', ylab=SR); a.set_xscale('log', base=2)
a.axhline(load('ablation')['genie']['sum'], color='k', lw=1, label=ST['genie'][0]); a.legend(); save(f, 'fig4_vpa')

d = load('length'); f, a = plt.subplots(1, 2, figsize=(7.2, 2.8))
line(a[0], d, 'sum', xlab='Hall length $L$ (m)', ylab=SR); a[0].legend(); a[0].set_title('(a)', fontsize=8)
m = np.array(d['res']['mimo']['sum'])
for k in ['prism', 'reactive', 'fixed', 'genie']:
    l, c, mk, ls = ST[k]; a[1].plot(d['values'], 100 * (np.array(d['res'][k]['sum']) / m - 1), ls=ls, marker=mk, color=c, label=l)
a[1].axhline(0, color='g', lw=1); a[1].set_xlabel('Hall length $L$ (m)'); a[1].set_ylabel('Gain over conventional MIMO (%)')
a[1].set_title('(b)', fontsize=8); a[1].legend(); save(f, 'fig5_length')

d = load('vuser'); f, a = plt.subplots(figsize=(3.6, 2.7))
line(a, d, 'sum', xlab='Mean user speed (m/s)', ylab=SR); a.legend(); save(f, 'fig6_vuser')

d = load('users'); f, a = plt.subplots(1, 2, figsize=(7.2, 2.7))
line(a[0], d, 'sum', xlab='Number of users $K$', ylab=SR); a[0].legend(); a[0].set_title('(a)', fontsize=8)
line(a[1], d, 'p5', xlab='Number of users $K$', ylab=P5); a[1].set_title('(b)', fontsize=8); save(f, 'fig7_users')

d = load('humans'); f, a = plt.subplots(figsize=(3.6, 2.7))
line(a, d, 'sum', xlab='Number of moving human blockers', ylab=SR); a.legend(); save(f, 'fig8_humans')

A = load('ablation'); f, a = plt.subplots(1, 2, figsize=(7.2, 2.8))
for k in ['prism', 'reactive', 'ddpg', 'gtrack', 'mimo', 'fixed', 'genie']:
    x = np.sort(A[k]['all']); l, c, mm, ls = ST[k]; a[0].plot(x, np.arange(1, len(x) + 1) / len(x), ls=ls, color=c, label=l)
a[0].set_xlabel('Per-user average rate (bit/s/Hz)'); a[0].set_ylabel('CDF'); a[0].legend(fontsize=6); a[0].set_title('(a)', fontsize=8)
tr = A['trace']; W = 50
for k in ['prism', 'reactive', 'ddpg', 'gtrack', 'mimo']:
    y = np.convolve(tr[k], np.ones(W) / W, 'valid'); l, c, mm, ls = ST[k]
    a[1].plot(np.arange(len(y)) * 0.1, y, ls=ls, color=c, label=l, lw=1)
a[1].axvline(A['warm'] * 0.1, color='k', lw=0.8, ls=':')
a[1].set_xlabel('Time (s)'); a[1].set_ylabel('Sum rate, 5-s moving avg.'); a[1].legend(fontsize=6); a[1].set_title('(b)', fontsize=8)
save(f, 'fig9_cdf_trace')

# ---- Fig 10: warehouse-aisle scenario
B = load('aisle'); sp = load('aisle_speed')
f, a = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1.5, 1]})
ks = ['oracle', 'prism', 'nomap', 'nopred', 'reactive', 'ddpg_aisle', 'mimo', 'fixed']
vals = [B[k]['sum'] for k in ks]; ci = [B[k]['ci'] for k in ks]
a[0].barh(range(len(ks)), vals, xerr=ci, color=[ST[k][1] for k in ks], capsize=2, alpha=0.85)
a[0].set_yticks(range(len(ks))); a[0].set_yticklabels([ST[k][0] for k in ks], fontsize=6.5); a[0].invert_yaxis()
a[0].set_xlabel(SR); a[0].set_title('(a) Warehouse-aisle scenario', fontsize=8)
line(a[1], sp, 'sum', xlab='Mean AGV speed (m/s)', ylab=SR); a[1].legend(fontsize=6); a[1].set_title('(b)', fontsize=8)
save(f, 'fig10_aisle')

# ---- Fig 11: DDPG training curves
f, a = plt.subplots(figsize=(3.6, 2.5))
for tag, c in [('default', '#e377c2'), ('aisle', '#7b3294')]:
    h = np.array(json.load(open(f'{R}ddpg_hist_{tag}.json'))); k = 5
    a.plot(np.arange(len(h)), h, color=c, alpha=0.25, lw=0.8)
    a.plot(np.arange(k - 1, len(h)), np.convolve(h, np.ones(k) / k, 'valid'), color=c, label=f'{tag} scenario')
a.set_xlabel('Training episode (400 slots each)'); a.set_ylabel('Avg. reward (sum rate / 10)'); a.legend()
save(f, 'fig11_ddpg_training')
print('ok')
