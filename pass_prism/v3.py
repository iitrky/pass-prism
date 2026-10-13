"""Revision (v3) extensions:
 1. Waveguide-aware online radio blockage map (WA-ORBM)
 2. Multiple pinching antennas (PAs) per waveguide, each confined to its own segment
 3. TD3 baseline (twin critics, delayed actor, target smoothing), NumPy
 4. Exact speed-limited optimum by dynamic programming (single-waveguide, max-rate case)
 5. Oracle decomposition (true future positions only / true blockage only)
"""
import copy, numpy as np
from scipy.optimize import linear_sum_assignment
from . import simulator as S
from .simulator import DEFAULT, Env, Scheme, PRISM, los_gain, blockage_loss_dB, pf_zf_greedy, SCHEMES
from .extensions import MLP, ddpg_state, AISLE, bca_place, OraclePRISM, save_actor

DXB = np.array([-10.0, -4.0, -1.5, 1.5, 4.0, 10.0])          # signed along-waveguide offset bins (m)


# ============================================================ 1. WA-ORBM
class WAMap:
    """LoS-probability map indexed by (waveguide n, user cell, signed PA-user offset along x).
    PAs are confined to known waveguides, so the waveguide index is the natural key: it separates
    links that share an azimuth but cross different obstacles (e.g. shelf rows between aisles)."""
    def __init__(self, p):
        self.p = p
        self.ux = int(np.ceil(p['L'] / p['u_cell'])); self.uy = int(np.ceil(p['W'] / p['u_cell']))
        self.S = np.zeros((p['N'], self.ux * self.uy, len(DXB) + 1)); self.C = np.zeros_like(self.S)
        self.c0 = p.get('c0', 3.0)

    def idx(self, n, pax, u):
        ix = np.clip((u[..., 0] / self.p['u_cell']).astype(int), 0, self.ux - 1)
        iy = np.clip((u[..., 1] / self.p['u_cell']).astype(int), 0, self.uy - 1)
        return n, ix * self.uy + iy, np.searchsorted(DXB, pax - u[..., 0])

    def lookup(self, n, pax, u):
        a, b, c = self.idx(n, pax, u)
        return (self.S[a, b, c] + self.c0) / (self.C[a, b, c] + self.c0)

    def update(self, n, pax, u, los, f):
        a, b, c = self.idx(n, pax, u)
        self.S *= f; self.C *= f
        np.add.at(self.S, (a, b, c), los); np.add.at(self.C, (a, b, c), 1.0)


def _los_labels(env, last_H, pa_x, pa_wg, Mfac=1.0):
    """LoS/NLoS labels from CSI: |h|^2 vs modelled LoS gain at estimated user positions."""
    g0 = los_gain(env, pa_x[:, None], env.yw[pa_wg][:, None], env.meas[None]) / Mfac
    att = 10 * np.log10(np.clip(np.abs(last_H) ** 2 / g0, 1e-6, 10.0))
    return att


class PRISM3(PRISM):
    """PRISM with the waveguide-aware map (single PA per waveguide)."""
    name = 'PRISM-WA'

    def __init__(self, env, use_pred=True, use_map=True):
        super().__init__(env, use_pred=use_pred, use_map=False)
        self.use_map3 = use_map
        self.name = 'PRISM' + ('' if use_map else ' w/o map') + ('' if use_pred else ' w/o prediction')
        self.wam = WAMap(env.p)
        self.kS, self.kC = -20.0, 1.0

    def lookup(self, pax, pay, u):           # called by PRISM.act through use_map
        n = np.rint((pay * 2 * self.p['N'] / self.p['W'] - 1) / 2).astype(int)
        n = np.broadcast_to(n, np.broadcast_shapes(np.shape(pax), np.shape(u)[:-1]))
        return self.wam.lookup(n, pax, u)

    def act(self):
        self.use_map = self.use_map3
        super().act()
        self.use_map = False                  # keep base-class map update disabled

    def serve(self):
        r = Scheme.serve(self)
        if self.use_map3:
            p, env = self.p, self.env
            att = _los_labels(env, self.last_H.T, self.x, np.arange(p['N']))      # (N,K)
            los = (att > -10).astype(float)
            if (los == 0).any():
                self.kS = p['forget'] * self.kS + att[los == 0].sum(); self.kC = p['forget'] * self.kC + (los == 0).sum()
            n = np.repeat(np.arange(p['N']), p['K'])
            self.wam.update(n, np.repeat(self.x, p['K']), np.tile(env.meas, (p['N'], 1)), los.ravel(), p['forget'])
        return r


# ============================================================ 2. multiple PAs per waveguide
_orig_step = S.Env.step


def _step(self):
    _orig_step(self)
    if not hasattr(self, 'rng2'):
        self.rng2 = np.random.default_rng(int(abs(self.users.p.sum()) * 1e6) % (1 << 31))   # no draws from self.rng
    Mx = self.p.get('M_max', 4)
    K, N = self.p['K'], self.p['N']
    self.gm = (self.rng2.normal(size=(K, N, Mx)) + 1j * self.rng2.normal(size=(K, N, Mx))) / np.sqrt(2)


S.Env.step = _step


def multi_channel(env, X):
    """X: (N,M) PA positions. Equal power split over the M PAs of a waveguide.
    Returns H (K,N): coherent sum over PAs of each waveguide, plus per-PA channels (K,N,M)."""
    p = env.p
    N, M = X.shape
    pa = np.stack([X.ravel(), np.repeat(env.yw, M)], -1)
    G, r = env.ls_gain(pa, env.users.p, X.ravel())                      # (NM,K)
    Kr = 10 ** (p['Kr_dB'] / 10)
    phi = np.exp(-1j * 2 * np.pi / env.lam * (r + p['neff'] * X.ravel()[:, None]))
    g = env.gm[:, :, :M].transpose(1, 2, 0).reshape(N * M, -1)          # (NM,K)
    h = np.sqrt(G / M) * (np.sqrt(Kr / (Kr + 1)) * phi + np.sqrt(1 / (Kr + 1)) * g)
    hm = h.reshape(N, M, -1)
    return hm.sum(1).T, hm.transpose(2, 0, 1)


class MultiBase(Scheme):
    def __init__(self, env, M):
        super().__init__(env)
        p = env.p
        self.M = M
        self.seg = p['L'] / M
        self.lo = np.arange(M) * self.seg; self.hi = self.lo + self.seg
        self.X = np.tile(self.lo + self.seg / 2, (p['N'], 1))
        self.x = self.X[:, 0]                                  # for compatibility

    def serve(self):
        env, p = self.env, self.p
        H, Hm = multi_channel(env, self.X)
        r = pf_zf_greedy(H, 1 / self.Rbar, env.P, env.sigma2)
        self.Rbar = (1 - 1 / p['tc']) * self.Rbar + r / p['tc']
        self.last_H, self.last_Hm = H, Hm
        return r


class FixedMulti(MultiBase):
    def __init__(self, env, M):
        super().__init__(env, M); self.name = f'Fixed PASS (M={M})'
    def act(self): pass


class ReactiveMulti(MultiBase):
    """Each PA tracks, within its segment, the user with the largest PF-weighted LoS rate."""
    def __init__(self, env, M):
        super().__init__(env, M); self.name = f'Reactive tracking PASS (M={M})'

    def act(self):
        env, p = self.env, self.p
        z = env.meas; w = 1 / self.Rbar; st = p['v_pa'] * p['dt']
        for n in range(p['N']):
            for m in range(self.M):
                tx = np.clip(z[:, 0], self.lo[m], self.hi[m])
                g = los_gain(env, tx, env.yw[n], z)
                k = int(np.argmax(w * np.log2(1 + env.P * g / (self.M * env.sigma2))))
                self.X[n, m] += np.clip(tx[k] - self.X[n, m], -st, st)


class PRISMMulti(MultiBase):
    """PRISM coverage placement over N*M PA units, each confined to its segment."""
    def __init__(self, env, M, use_pred=True, use_map=True):
        super().__init__(env, M)
        p = env.p
        self.name = f'PRISM (M={M})'
        self.use_pred, self.use_map = use_pred, use_map
        self.kf = S.CVKalman(env.meas, p['dt'], rstd=max(p['pos_err'], 1e-3))
        self.H = p['H']; hs = np.arange(1, self.H + 1); self.hs = hs
        self.disc = p['gamma'] ** hs / np.sum(p['gamma'] ** hs)
        nc = int(round(self.seg / p['dx_cand'])) + 1
        self.cand = np.stack([np.linspace(self.lo[m], self.hi[m], nc) for m in range(M)])   # (M,C)
        self.wam = WAMap(p); self.kS, self.kC = -20.0, 1.0

    def act(self):
        env, p = self.env, self.p
        N, M = p['N'], self.M
        self.kf.update(env.meas)
        if self.use_pred:
            U = self.kf.predict(self.H, p['L'], p['W']); hs, disc = self.hs, self.disc
        else:
            U = env.meas[:, None]; hs, disc = np.array([1]), np.array([1.0])
        st = p['v_pa'] * p['dt']
        X = self.X.ravel(); wg = np.repeat(np.arange(N), M); cand = np.tile(self.cand, (N, 1))   # (NM,C)
        xt = X[:, None, None] + np.clip(cand[:, :, None] - X[:, None, None], -st * hs, st * hs)   # (NM,C,Hh)
        g = los_gain(env, xt[:, None], env.yw[wg][:, None, None, None], U[None, :, None]) / M      # (NM,K,C,Hh)
        snr = env.P * g / env.sigma2
        if self.use_map:
            pi = self.wam.lookup(np.broadcast_to(wg[:, None, None, None], g.shape), xt[:, None], U[None, :, None])
            kap = 10 ** ((self.kS / self.kC) / 10)
            rate = pi * np.log2(1 + snr) + (1 - pi) * np.log2(1 + kap * snr)
        else:
            rate = np.log2(1 + snr)
        w = 1 / self.Rbar
        Uu = N * M
        cidx = np.argmin(np.abs(cand - X[:, None]), 1)
        for _ in range(p.get('n_bca', 3)):
            changed = False
            for u in range(Uu):
                cur = rate[np.arange(Uu), :, cidx, :]
                other = np.max(np.delete(cur, u, 0), 0)
                obj = w @ (np.maximum(rate[u], other[:, None, :]) @ disc)
                obj[cidx[u]] *= (1 + 1e-6)
                b = int(np.argmax(obj))
                if b != cidx[u]:
                    cidx[u] = b; changed = True
            if not changed:
                break
        tgt = cand[np.arange(Uu), cidx]
        self.X = (X + np.clip(tgt - X, -st, st)).reshape(N, M)

    def serve(self):
        r = super().serve()
        if self.use_map:
            p, env = self.p, self.env
            N, M = p['N'], self.M
            X = self.X.ravel(); wg = np.repeat(np.arange(N), M)
            hm = self.last_Hm.transpose(1, 2, 0).reshape(N * M, -1)                  # per-PA channel (NM,K)
            att = _los_labels(env, hm, X, wg, Mfac=M)
            los = (att > -10).astype(float)
            if (los == 0).any():
                self.kS = p['forget'] * self.kS + att[los == 0].sum(); self.kC = p['forget'] * self.kC + (los == 0).sum()
            self.wam.update(np.repeat(wg, p['K']), np.repeat(X, p['K']), np.tile(env.meas, (N * M, 1)), los.ravel(), p['forget'])
        return r


# ============================================================ 3. TD3 (NumPy)
def train_td3(prm_over=None, episodes=300, T_ep=400, seed=0, log=print):
    prm = dict(DEFAULT); prm.update(prm_over or {})
    rng = np.random.default_rng(seed)
    N, K = prm['N'], prm['K']; sd = N + 3 * K
    A = MLP([sd, 128, 128, N], rng, 'tanh'); At = MLP([sd, 128, 128, N], rng, 'tanh'); At.copy_from(A)
    Q = [MLP([sd + N, 128, 128, 1], rng) for _ in range(2)]; Qt = [MLP([sd + N, 128, 128, 1], rng) for _ in range(2)]
    for q, qt in zip(Q, Qt): qt.copy_from(q)
    cap = 200000; Sb = np.zeros((cap, sd)); Ab = np.zeros((cap, N)); Rb = np.zeros(cap); S2 = np.zeros((cap, sd))
    ptr = full = 0; gamma, tau, B, it = 0.97, 0.005, 128, 0
    hist, best, best_w = [], -1e9, None
    from .extensions import DDPGScheme
    for ep in range(episodes):
        env = Env(prm, 7000 + seed * 1000 + ep); env.step()
        noise = max(0.1, 0.5 * (1 - ep / (0.5 * episodes)))
        ag = DDPGScheme(env, A, noise, rng)
        tot = 0.0
        for t in range(T_ep):
            env.step(); ag.act(); r = ag.serve()
            rew = float(r.sum()) / 10.0
            if t > 20:
                Sb[ptr], Ab[ptr], Rb[ptr], S2[ptr] = ag.s, ag.a, rew, ddpg_state(env, ag.x, ag.Rbar)
                ptr = (ptr + 1) % cap; full = min(full + 1, cap)
            tot += rew
            if full > 5000:
                it += 1
                idx = rng.integers(0, full, B); s, a, rr, ss = Sb[idx], Ab[idx], Rb[idx], S2[idx]
                a2 = np.clip(At.forward(ss) + np.clip(rng.normal(0, 0.2, (B, N)), -0.5, 0.5), -1, 1)
                y = rr[:, None] + gamma * np.minimum(Qt[0].forward(np.c_[ss, a2]), Qt[1].forward(np.c_[ss, a2]))
                for q in Q:
                    pred = q.forward(np.c_[s, a]); gq, _ = q.backward((pred - y) / B); q.adam(gq, 3e-4)
                if it % 2 == 0:
                    ap = A.forward(s); Q[0].forward(np.c_[s, ap]); _, gin = Q[0].backward(-np.ones((B, 1)) / B)
                    ga, _ = A.backward(gin[:, sd:]); A.adam(ga, 1e-4)
                    At.copy_from(A, tau); [qt.copy_from(q, tau) for q, qt in zip(Q, Qt)]
        hist.append(tot / T_ep)
        if ep >= episodes // 2 and np.mean(hist[-5:]) > best:     # keep best late-training actor
            best = np.mean(hist[-5:]); best_w = ([w.copy() for w in A.W], [b.copy() for b in A.b])
        if ep % 10 == 0: log(f'td3 ep {ep} noise {noise:.2f} reward {tot / T_ep:.3f}')
    if best_w is not None:
        A.W, A.b = best_w
    return A, hist


# ============================================================ 4. DP speed-limited optimum
def dp_experiment(seed, prm_over=None, K=4, T=600, warm=0, H=10, log=None):
    """Single waveguide, single PA, max-rate scheduling (one user per slot), true large-scale
    channel incl. blockage. Compares the exact speed-limited optimum (offline DP, clairvoyant)
    with online placement rules. Returns time-average rates (bit/s/Hz)."""
    prm = dict(DEFAULT); prm.update({'N': 1, 'K': K}); prm.update(prm_over or {})
    env = Env(prm, seed); env.step()
    st = prm['v_pa'] * prm['dt']
    grid = np.arange(0, prm['L'] + 1e-9, st)                                     # step = one move per slot
    G = len(grid); yw = env.yw[0]
    R = np.empty((T, G)); Ufull = np.empty((T, K, 2)); meas = np.empty((T, K, 2))
    for t in range(T):
        env.step()
        c, rr, h, Lb = env.obs
        pa = np.c_[grid, np.full(G, yw)]
        bl = blockage_loss_dB(pa[:, None], env.users.p[None], c, rr, h, Lb, prm['d'], prm['hu'])
        g = los_gain(env, grid[:, None], yw, env.users.p[None]) * 10 ** (-bl / 10)
        R[t] = np.max(np.log2(1 + env.P * g / env.sigma2), 1)
        Ufull[t] = env.users.p; meas[t] = env.meas
    # exact DP (Viterbi): value-to-go over positions, moves of -1/0/+1 grid steps
    start = G // 2
    # reachable from the start only: forward pass with the start constraint
    acc = np.full(G, -np.inf); acc[start] = 0.0
    for t in range(T):
        acc = np.maximum(np.maximum(np.r_[acc[1:], -np.inf], np.r_[-np.inf, acc[:-1]]), acc) if t > 0 else acc
        acc = acc + R[t]
    dp = np.max(acc) / T
    genie = R.max(1).mean()                                       # infinite speed
    fixed = R[:, start].mean()
    # online rules
    kf = S.CVKalman(meas[0], prm['dt'], rstd=prm['pos_err'])
    hs = np.arange(1, H + 1); disc = 0.9 ** hs / np.sum(0.9 ** hs)
    i_pred = i_track = start; tot_pred = tot_track = 0.0
    for t in range(T):
        z = meas[t]; kf.update(z); Up = kf.predict(H, prm['L'], prm['W'])
        # predictive: target maximizing discounted max-rate along the speed-limited path (LoS model)
        xi = grid[i_pred]
        xt = xi + np.clip(grid[:, None] - xi, -st * hs, st * hs)                         # (G,H)
        gg = los_gain(env, xt[:, None, :], yw, Up[None])                                 # (G,K,H)
        val = np.max(np.log2(1 + env.P * gg / env.sigma2), 1) @ disc
        i_pred = int(np.clip(i_pred + np.sign(np.argmax(val) - i_pred), 0, G - 1))
        # reactive: move toward the currently best user (LoS model, measured positions)
        gz = los_gain(env, z[:, 0], yw, z)
        tgt = z[int(np.argmax(gz)), 0]; i_track = int(np.clip(i_track + np.sign(np.rint((tgt - grid[i_track]) / st)), 0, G - 1))
        tot_pred += R[t, i_pred]; tot_track += R[t, i_track]
    return dict(dp=dp, genie=genie, fixed=fixed, predictive=tot_pred / T, reactive=tot_track / T)


# ============================================================ 5. oracle decomposition
class OraclePos(PRISM3):
    """True future positions, learned (WA) blockage map."""
    name = 'Oracle positions only'
    def act(self):
        env = self.env
        fut = copy.deepcopy(env); U = np.empty((self.p['K'], self.H, 2))
        for h in range(self.H):
            fut.step(); U[:, h] = fut.users.p
        self.kf.predict = lambda H, L, W, U=U: U
        super().act()


class OracleBlk(OraclePRISM):
    """True future blockage evaluated at Kalman-predicted positions."""
    name = 'Oracle blockage only'
    def __init__(self, env):
        super().__init__(env)
        self.kf3 = S.CVKalman(env.meas, env.p['dt'], rstd=max(env.p['pos_err'], 1e-3))
    def act(self):
        self.kf3.update(self.env.meas)
        Up = self.kf3.predict(self.H, self.p['L'], self.p['W'])
        orig, env, U = copy.deepcopy, self.env, Up
        p, step = self.p, self.p['v_pa'] * self.p['dt']
        futs = orig(env); obs = []
        for h in range(self.H):
            futs.step(); obs.append(futs.obs)
        hs = self.hs
        xt = self.x[:, None, None] + np.clip(self.cand[None, :, None] - self.x[:, None, None], -step * hs, step * hs)
        g = los_gain(env, xt[:, None], env.yw[:, None, None, None], U[None, :, None])
        bl = np.empty_like(g)
        for h in range(self.H):
            c, rr, hh, Lb = obs[h]
            for n in range(p['N']):
                ux, inv = np.unique(np.round(xt[n, :, h], 6), return_inverse=True)
                pa = np.c_[ux, np.full(len(ux), env.yw[n])]
                bl[n, :, :, h] = blockage_loss_dB(pa[None, :], U[:, h][:, None], c, rr, hh, Lb, p['d'], p['hu'])[:, inv]
        rate = np.log2(1 + env.P * g * 10 ** (-bl / 10) / env.sigma2)
        bca_place(self, rate, self.disc, step)


SCHEMES.update({
    'prism3': lambda e: PRISM3(e),
    'prism3_nomap': lambda e: PRISM3(e, use_map=False),
    'prism3_nopred': lambda e: PRISM3(e, use_pred=False),
    'or_pos': lambda e: OraclePos(e),
    'or_blk': lambda e: OracleBlk(e),
})
for _M in (1, 2, 3, 4):
    SCHEMES[f'prismM{_M}'] = (lambda M: lambda e: PRISMMulti(e, M))(_M)
    SCHEMES[f'prismM{_M}_nomap'] = (lambda M: lambda e: PRISMMulti(e, M, use_map=False))(_M)
    SCHEMES[f'reactM{_M}'] = (lambda M: lambda e: ReactiveMulti(e, M))(_M)
    SCHEMES[f'fixedM{_M}'] = (lambda M: lambda e: FixedMulti(e, M))(_M)
