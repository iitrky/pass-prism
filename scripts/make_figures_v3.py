"""Figures for the revised (v3) manuscript, from results/v3/*.json."""
import os, sys, json, numpy as np, matplotlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8, 'axes.grid': True,
                     'grid.alpha': 0.3, 'lines.linewidth': 1.3, 'lines.markersize': 4.5, 'legend.fontsize': 6.5})
R = 'results/v3/'; os.makedirs('figures/v3', exist_ok=True)
J = lambda n: json.load(open(R + n + '.json'))
save = lambda f, n: (f.tight_layout(), f.savefig(f'figures/v3/{n}.png', dpi=300), plt.close(f))
SR = 'Average sum rate (bit/s/Hz)'

# Fig. 3: learning-based baselines
f, a = plt.subplots(figsize=(3.5, 2.4))
for path, lab, c in [(R + 'td3_hist_default.json', 'TD3 (default hall)', '#7b3294'), ('results/ddpg_hist_default.json', 'DDPG (default hall)', '#e377c2')]:
    h = np.array(json.load(open(path))); k = 5
    a.plot(h, color=c, alpha=0.25, lw=0.7); a.plot(np.arange(k - 1, len(h)), np.convolve(h, np.ones(k) / k, 'valid'), color=c, label=lab)
a.set_xlabel('Training episode (400 slots each)'); a.set_ylabel('Avg. reward (sum rate/10)'); a.legend()
save(f, 'fig3_drl_training')

# Fig. 4: number of PAs per waveguide x PA speed
d = J('mv'); pts = {(float(p['over']['v_pa']), int(p['over']['M_tag'])): p['res'] for p in d['points']}
Ms = [1, 2, 3, 4]; f, ax = plt.subplots(2, 1, figsize=(3.5, 4.6))
for v, ls in [(0.5, '--'), (2.0, '-')]:
    for kind, c, mk in [('prism', '#c00000', 'o'), ('react', '#1f77b4', 's')]:
        ys = [pts[(v, M)][f'{kind}M{M}'] if (v, M) in pts else None for M in Ms]
        ok = [i for i, y in enumerate(ys) if y]
        lab = ('PRISM' if kind == 'prism' else 'Reactive tracking') + f', $v_{{\\max}}$={v} m/s'
        ax[0].errorbar([Ms[i] for i in ok], [ys[i]['sum'] for i in ok], yerr=[ys[i]['ci'] for i in ok], ls=ls, marker=mk, color=c, label=lab, capsize=2)
        ax[1].plot([Ms[i] for i in ok], [ys[i]['p5'] for i in ok], ls=ls, marker=mk, color=c, label=lab)
for i, yl in enumerate([SR, '5th-percentile user rate (bit/s/Hz)']):
    ax[i].set_xlabel('PAs per waveguide $M$'); ax[i].set_ylabel(yl); ax[i].set_xticks(Ms); ax[i].set_title(f'({"ab"[i]})', fontsize=8)
ax[0].legend(); save(f, 'fig4_multi_pa')

# Fig. 5: hall length
d = J('length3'); Ls = [p['over']['L'] for p in d['points']]
ST = {'prism3': ('PRISM ($M$=1)', '#c00000', 'o', '-'), 'prismM2': ('PRISM ($M$=2)', '#8b0000', 'D', '-'),
      'reactive': ('Reactive tracking PASS', '#1f77b4', 's', '--'), 'mimo': ('Conventional MIMO BS', '#2ca02c', '^', ':')}
f, ax = plt.subplots(2, 1, figsize=(3.5, 4.6))
for k, (l, c, m, ls) in ST.items():
    y = np.array([p['res'][k]['sum'] for p in d['points']]); e = [p['res'][k]['ci'] for p in d['points']]
    ax[0].errorbar(Ls, y, yerr=e, ls=ls, marker=m, color=c, label=l, capsize=2)
    if k != 'mimo':
        mm = np.array([p['res']['mimo']['sum'] for p in d['points']])
        ax[1].plot(Ls, 100 * (y / mm - 1), ls=ls, marker=m, color=c, label=l)
ax[1].axhline(0, color='g', lw=1)
ax[0].set_xlabel('Hall length $L$ (m)'); ax[0].set_ylabel(SR); ax[0].legend(); ax[0].set_title('(a)', fontsize=8)
ax[1].set_xlabel('Hall length $L$ (m)'); ax[1].set_ylabel('Gain over conventional MIMO (%)'); ax[1].legend(); ax[1].set_title('(b)', fontsize=8)
save(f, 'fig5_length')

# Fig. 6: warehouse aisle
B = J('aisle')
names = [('oracle', 'Oracle (true future positions + blockage)'), ('or_blk', 'Oracle, true blockage only'), ('or_pos', 'Oracle, true positions only'),
         ('prismM3', 'PRISM ($M$=3)'), ('prismM3_nomap', 'PRISM ($M$=3) w/o map'), ('reactM3', 'Reactive tracking ($M$=3)'), ('fixedM3', 'Fixed PASS ($M$=3)'),
         ('prism3', 'PRISM ($M$=1)'), ('prism3_nomap', 'PRISM ($M$=1) w/o map'), ('prism3_nopred', 'PRISM ($M$=1) w/o prediction'),
         ('reactive', 'Reactive tracking ($M$=1)'), ('ddpg_aisle', 'DDPG'), ('mimo', 'Conventional MIMO BS')]
cols = ['#bcbd22'] * 3 + ['#8b0000', '#c06060', '#1f77b4', '#7f7f7f'] + ['#c00000', '#e08080', '#f0b0b0', '#6baed6', '#e377c2', '#2ca02c']
f, a = plt.subplots(figsize=(3.5, 3.9))
a.barh(range(len(names)), [B[k]['sum'] for k, _ in names], xerr=[B[k]['ci'] for k, _ in names], color=cols, capsize=2, alpha=0.9)
a.set_yticks(range(len(names))); a.set_yticklabels([n for _, n in names], fontsize=5.6); a.invert_yaxis(); a.set_xlabel(SR)
save(f, 'fig6_aisle')
print('ok')
