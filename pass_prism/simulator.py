# PRISM simulator core. See README.md.
"""
PRISM-PASS simulator
Predictive Receding-horizon pinching-antenna roaming with an Online Radio
Blockage Map (ORBM) for multi-user, multi-waveguide pinching-antenna systems
with finite antenna movement speed.

All results in the manuscript are produced by this file (seeded, reproducible).
Author: Gregory Arul Dalton
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

C0 = 3e8

DEFAULT = dict(
    fc=28e9, L=60.0, W=12.0, d=3.0, hu=1.0, N=4, K=12,
    P_dBm=10.0, B=100e6, NF=9.0, alpha_dB=0.08, neff=1.4, Kr_dB=10.0,
    dt=0.1, v_user=1.5, v_pa=2.0, n_shelf=6, n_human=10,
    shelf_r=0.8, shelf_h=2.2, shelf_loss=30.0,
    human_r=0.3, human_h=1.75, human_loss=20.0, v_human=1.0,
    pos_err=0.2, H=10, gamma=0.9, dx_cand=0.5,
    pa_cell=1.0, u_cell=1.5, forget=0.998, tc=50.0,
    warm=300, T=600,
)


# ---------------------------------------------------------------- geometry
def blockage_loss_dB(pa, ue, obs_c, obs_r, obs_h, obs_L, d, hu):
    """pa: (...,2) PA xy; ue: (...,2) user xy (broadcastable). Returns loss dB (...)."""
    pa = np.asarray(pa); ue = np.asarray(ue)
    seg = ue - pa                                   # (...,2)
    seg2 = np.sum(seg ** 2, -1)[..., None] + 1e-12  # (...,1)
    rel = obs_c - pa[..., None, :]                  # (...,M,2)
    t = np.clip(np.sum(rel * seg[..., None, :], -1) / seg2, 0, 1)   # (...,M)
    close = pa[..., None, :] + t[..., None] * seg[..., None, :]
    dist = np.linalg.norm(close - obs_c, axis=-1)
    z = d + t * (hu - d)
    blk = (dist < obs_r) & (z < obs_h)
    return np.minimum(np.sum(blk * obs_L, -1), 40.0)


class RWP:
    """Random-waypoint mobility in a rectangle."""
    def __init__(self, n, L, W, v, rng, vspread=0.5):
        self.rng, self.L, self.W = rng, L, W
        self.p = np.c_[rng.uniform(0, L, n), rng.uniform(0, W, n)]
        self.wp = np.c_[rng.uniform(0, L, n), rng.uniform(0, W, n)]
        self.v = v * rng.uniform(1 - vspread, 1 + vspread, n)

    def step(self, dt):
        d = self.wp - self.p
        dist = np.linalg.norm(d, axis=1) + 1e-9
        stepl = self.v * dt
        arrive = dist <= stepl
        self.p = np.where(arrive[:, None], self.wp, self.p + d / dist[:, None] * stepl[:, None])
        n = arrive.sum()
        if n:
            self.wp[arrive] = np.c_[self.rng.uniform(0, self.L, n), self.rng.uniform(0, self.W, n)]
        return self.p


class Env:
    def __init__(self, prm, seed):
        self.p = prm
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.lam = C0 / prm['fc']
        self.eta = (self.lam / (4 * np.pi)) ** 2
        self.sigma2 = 10 ** ((-174 + 10 * np.log10(prm['B']) + prm['NF'] - 30) / 10)
        self.P = 10 ** ((prm['P_dBm'] - 30) / 10)
        N = prm['N']
        self.yw = prm['W'] * (2 * np.arange(N) + 1) / (2 * N)
        self.shelf = np.c_[rng.uniform(3, prm['L'] - 3, prm['n_shelf']),
                           rng.uniform(2, prm['W'] - 2, prm['n_shelf'])]
        self.users = RWP(prm['K'], prm['L'], prm['W'], prm['v_user'], rng)
        self.humans = RWP(prm['n_human'], prm['L'], prm['W'], prm['v_human'], rng, 0.3)
        self.meas = self.users.p.copy()

    def obstacles(self):
        p = self.p
        c = np.r_[self.shelf, self.humans.p] if p['n_human'] else self.shelf
        ns, nh = p['n_shelf'], p['n_human']
        r = np.r_[np.full(ns, p['shelf_r']), np.full(nh, p['human_r'])]
        h = np.r_[np.full(ns, p['shelf_h']), np.full(nh, p['human_h'])]
        Lb = np.r_[np.full(ns, p['shelf_loss']), np.full(nh, p['human_loss'])]
        return c, r, h, Lb

    def step(self):
        p = self.p
        self.users.step(p['dt'])
        if p['n_human']:
            self.humans.step(p['dt'])
        self.meas = self.users.p + self.rng.normal(0, p['pos_err'], self.users.p.shape)
        K, N = p['K'], p['N']
        self.g = (self.rng.normal(size=(K, N)) + 1j * self.rng.normal(size=(K, N))) / np.sqrt(2)
        self.obs = self.obstacles()

    # large-scale gain for PAs at (x,yw) to users (true positions)
    def ls_gain(self, pa_xy, ue_xy, wg_len):
        p = self.p
        dx = pa_xy[:, None, :] - ue_xy[None, :, :]
        r = np.sqrt(np.sum(dx ** 2, -1) + (p['d'] - p['hu']) ** 2)
        c, rr, h, Lb = self.obs
        bl = blockage_loss_dB(pa_xy[:, None, :], ue_xy[None, :, :], c, rr, h, Lb, p['d'], p['hu'])
        wl = 10 ** (-p['alpha_dB'] * wg_len / 10)[:, None]
        return self.eta / r ** 2 * wl * 10 ** (-bl / 10), r  # (N,K)

    def channel(self, pa_xy, wg_len, phase_len):
        """Returns H (K,N) complex."""
        p = self.p
        G, r = self.ls_gain(pa_xy, self.users.p, wg_len)
        Kr = 10 ** (p['Kr_dB'] / 10)
        phi = np.exp(-1j * 2 * np.pi / self.lam * (r + p['neff'] * phase_len[:, None]))
        small = np.sqrt(Kr / (Kr + 1)) * phi + np.sqrt(1 / (Kr + 1)) * self.g.T
        return (np.sqrt(G) * small).T, G


# ------------------------------------------------------------- scheduling
def pf_zf_greedy(H, w, P, s2):
    K, N = H.shape
    S, best_u = [], 0.0
    rates = np.zeros(K)
    while len(S) < N:
        cand = [k for k in range(K) if k not in S]
        if not cand:
            break
        idx = np.array([S + [k] for k in cand])            # (C,s+1)
        Hs = H[idx]                                        # (C,s+1,N)
        Gm = Hs @ np.conj(np.transpose(Hs, (0, 2, 1)))
        try:
            inv = np.linalg.inv(Gm)
        except np.linalg.LinAlgError:
            break
        dg = np.real(np.diagonal(inv, axis1=1, axis2=2))
        dg = np.where(dg > 0, dg, np.inf)
        r = np.log2(1 + (P / idx.shape[1]) / (s2 * dg))    # (C,s+1)
        u = np.sum(r * w[idx], 1)
        j = int(np.argmax(u))
        if u[j] <= best_u:
            break
        best_u = u[j]; S = list(idx[j]); rsel = r[j]
    if S:
        rates[S] = rsel
    return rates


# -------------------------------------------------------------- Kalman
class CVKalman:
    def __init__(self, z0, dt, q=0.5, rstd=0.2):
        K = z0.shape[0]
        self.x = np.c_[z0, np.zeros((K, 2))]
        self.Pm = np.tile(np.diag([rstd ** 2, rstd ** 2, 1, 1]), (K, 1, 1))
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.F = F
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.Q = q ** 2 * G @ G.T
        self.Hm = np.eye(2, 4)
        self.R = rstd ** 2 * np.eye(2)

    def update(self, z):
        x = self.x @ self.F.T
        P = self.F @ self.Pm @ self.F.T + self.Q
        S = self.Hm @ P @ self.Hm.T + self.R
        Kg = P @ self.Hm.T @ np.linalg.inv(S)
        y = z - x[:, :2]
        self.x = x + np.einsum('kij,kj->ki', Kg, y)
        self.Pm = (np.eye(4) - Kg @ self.Hm) @ P

    def predict(self, H, L, W):
        out = np.empty((self.x.shape[0], H, 2))
        x = self.x.copy()
        for h in range(H):
            x = x @ self.F.T
            out[:, h] = x[:, :2]
        out[..., 0] = np.clip(out[..., 0], 0, L); out[..., 1] = np.clip(out[..., 1], 0, W)
        return out


# -------------------------------------------------------------- schemes
class Scheme:
    name = 'base'
    def __init__(self, env):
        p = env.p
        self.env, self.p = env, p
        self.Rbar = np.ones(p['K'])
        self.log = []

    def pa_xy(self):
        return np.c_[self.x, self.env.yw]

    def serve(self):
        env, p = self.env, self.p
        H, G = env.channel(self.pa_xy(), self.x, self.x)
        r = pf_zf_greedy(H, 1 / self.Rbar, env.P, env.sigma2)
        self.Rbar = (1 - 1 / p['tc']) * self.Rbar + r / p['tc']
        self.last_H = H
        return r


class FixedPASS(Scheme):
    name = 'Fixed PASS (midpoint)'
    def __init__(self, env):
        super().__init__(env); self.x = np.full(env.p['N'], env.p['L'] / 2)
    def act(self): pass


class ConvMIMO(Scheme):
    """Co-located N-element ULA at ceiling centre (no waveguide)."""
    name = 'Conventional MIMO BS'
    def __init__(self, env):
        super().__init__(env)
        p = env.p
        self.xy = np.c_[np.full(p['N'], p['L'] / 2),
                        p['W'] / 2 + (np.arange(p['N']) - (p['N'] - 1) / 2) * env.lam / 2]
    def act(self): pass
    def serve(self):
        env, p = self.env, self.p
        G, r = env.ls_gain(self.xy, env.users.p, np.zeros(p['N']))
        Kr = 10 ** (p['Kr_dB'] / 10)
        phi = np.exp(-1j * 2 * np.pi / env.lam * r)
        H = (np.sqrt(G) * (np.sqrt(Kr / (Kr + 1)) * phi + np.sqrt(1 / (Kr + 1)) * env.g.T)).T
        rr = pf_zf_greedy(H, 1 / self.Rbar, env.P, env.sigma2)
        self.Rbar = (1 - 1 / p['tc']) * self.Rbar + rr / p['tc']
        return rr


def los_gain(env, x, yw, ue):
    """Free-space + waveguide gain, no blockage. x:(...), ue broadcast."""
    p = env.p
    r2 = (x - ue[..., 0]) ** 2 + (yw - ue[..., 1]) ** 2 + (p['d'] - p['hu']) ** 2
    return env.eta / r2 * 10 ** (-p['alpha_dB'] * x / 10)


class Reactive(Scheme):
    """Myopic tracking: speed-limited move toward assigned user's current measured x."""
    name = 'Reactive tracking PASS'
    def __init__(self, env):
        super().__init__(env); self.x = np.full(env.p['N'], env.p['L'] / 2)
    def act(self):
        env, p = self.env, self.p
        z = env.meas
        step = p['v_pa'] * p['dt']
        xn = self.x[:, None] + np.clip(z[None, :, 0] - self.x[:, None], -step, step)   # (N,K)
        g = los_gain(env, xn, env.yw[:, None], z[None])
        A = (1 / self.Rbar)[None] * np.log2(1 + env.P * g / env.sigma2)
        rows, cols = linear_sum_assignment(-A)
        self.x[rows] = xn[rows, cols]


class GeniePASS(Scheme):
    """Infinite speed + perfect position and blockage knowledge (upper-bound style)."""
    name = 'Ideal PASS (genie, v=inf)'
    def __init__(self, env):
        super().__init__(env)
        p = env.p
        self.x = np.full(p['N'], p['L'] / 2)
        self.cand = np.arange(0, p['L'] + 1e-9, p['dx_cand'])
    def act(self):
        env, p = self.env, self.p
        N, K, Cn = p['N'], p['K'], len(self.cand)
        pa = np.stack(np.broadcast_arrays(self.cand[None, :], env.yw[:, None]), -1)   # (N,C,2)
        ue = env.users.p
        c, rr, h, Lb = env.obs
        bl = blockage_loss_dB(pa[:, None], ue[None, :, None], c, rr, h, Lb, p['d'], p['hu'])  # (N,K,C)
        g = los_gain(env, self.cand[None, None], env.yw[:, None, None], ue[None, :, None]) * 10 ** (-bl / 10)
        u = np.log2(1 + env.P * g / env.sigma2)
        bc = np.argmax(u, -1)
        A = (1 / self.Rbar)[None] * np.max(u, -1)
        rows, cols = linear_sum_assignment(-A)
        self.x[rows] = self.cand[bc[rows, cols]]


class PRISM(Scheme):
    """Proposed: KF trajectory prediction + ORBM + receding-horizon speed-limited
    roaming + Hungarian PA-user association + PF-ZF scheduling."""
    name = 'PRISM (proposed)'
    def __init__(self, env, use_pred=True, use_map=True, H=None, mode='bca'):
        super().__init__(env)
        p = env.p
        self.use_pred, self.use_map = use_pred, use_map
        self.H = p['H'] if H is None else H
        if not use_pred: self.name = 'PRISM w/o prediction'
        if not use_map: self.name = 'PRISM w/o blockage map'
        self.x = np.full(p['N'], p['L'] / 2)
        self.cand = np.arange(0, p['L'] + 1e-9, p['dx_cand'])
        self.kf = CVKalman(env.meas, p['dt'], rstd=max(p['pos_err'], 1e-3))
        # direction-aware ORBM: (user cell, azimuth sector, horizontal-distance bin)
        self.ux = int(np.ceil(p['L'] / p['u_cell'])); self.uy = int(np.ceil(p['W'] / p['u_cell']))
        self.na, self.dbins = p.get('n_az', 12), np.array(p.get('dbins', [2.0, 5.0, 10.0]))
        shape = (self.ux * self.uy, self.na, len(self.dbins) + 1)
        self.S = np.zeros(shape); self.Cnt = np.zeros(shape)
        self.gS, self.gC = 0.0, 1.0
        self.thr = p.get('los_thr', -10.0); self.kS, self.kC = -20.0, 1.0
        self.c0 = p.get('c0', 3.0)
        self.assign = None; self.hyst = p.get('hyst', 0.2)
        self.mode = mode; self.n_bca = p.get('n_bca', 3)
        hs = np.arange(1, self.H + 1)
        self.disc = p['gamma'] ** hs / np.sum(p['gamma'] ** hs)
        self.hs = hs

    def ue_idx(self, u):
        ix = np.clip((u[..., 0] / self.p['u_cell']).astype(int), 0, self.ux - 1)
        iy = np.clip((u[..., 1] / self.p['u_cell']).astype(int), 0, self.uy - 1)
        return ix * self.uy + iy

    def dir_idx(self, pax, pay, u):
        dx = pax - u[..., 0]; dy = pay - u[..., 1]
        az = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * self.na).astype(int) % self.na
        db = np.searchsorted(self.dbins, np.hypot(dx, dy))
        return az, db

    def lookup(self, pax, pay, u):
        ui = self.ue_idx(u); az, db = self.dir_idx(pax, pay, u)
        S = self.S[ui, az, db]; C = self.Cnt[ui, az, db]
        return (S + self.c0 * 1.0) / (C + self.c0)          # posterior-mean LoS probability

    def act(self):
        env, p = self.env, self.p
        N = p['N']
        self.kf.update(env.meas)
        if self.use_pred:
            U = self.kf.predict(self.H, p['L'], p['W'])            # (K,H,2)
            hs = self.hs
        else:
            U = np.repeat(env.meas[:, None], 1, 1); hs = np.array([1])
        disc = self.disc if self.use_pred else np.array([1.0])
        step = p['v_pa'] * p['dt']
        # PA trajectories toward each candidate: (N,C,Hh)
        xt = self.x[:, None, None] + np.clip(self.cand[None, :, None] - self.x[:, None, None],
                                             -step * hs, step * hs)
        g = los_gain(env, xt[:, None], env.yw[:, None, None, None], U[None, :, None])   # (N,K,C,Hh)
        snr0 = env.P * g / env.sigma2
        if self.use_map:
            pi = self.lookup(xt[:, None], env.yw[:, None, None, None], U[None, :, None])
            kap = 10 ** ((self.kS / self.kC) / 10)
            rate = pi * np.log2(1 + snr0) + (1 - pi) * np.log2(1 + kap * snr0)
        else:
            rate = np.log2(1 + snr0)
        g = snr0 * env.sigma2 / env.P
        if self.mode == 'hungarian':
            u = rate @ disc               # (N,K,C)
            bc = np.argmax(u, -1)
            A = (1 / self.Rbar)[None] * np.max(u, -1)
            if self.assign is not None and self.hyst > 0:
                A[np.arange(N), self.assign] *= (1 + self.hyst)
            rows, cols = linear_sum_assignment(-A)
            self.assign = cols.copy()
            tgt = self.cand[bc[rows, cols]]
            self.x[rows] = self.x[rows] + np.clip(tgt - self.x[rows], -step, step)
            return
        # coverage-oriented best-server utility, block-coordinate ascent over waveguides
        w = 1 / self.Rbar
        snr = rate                                                         # expected rate (N,K,C,Hh)
        cidx = np.argmin(np.abs(self.cand[None, :] - self.x[:, None]), 1)   # start: stay
        for _ in range(self.n_bca):
            changed = False
            for n in range(N):
                cur = snr[np.arange(N), :, cidx, :]                           # (N,K,Hh)
                other = np.max(np.delete(cur, n, 0), 0) if N > 1 else np.zeros_like(cur[0])
                val = np.maximum(snr[n], other[:, None, :]) @ disc   # (K,C)
                obj = w @ val                                                  # (C,)
                obj[cidx[n]] *= (1 + 1e-6)
                b = int(np.argmax(obj))
                if b != cidx[n]:
                    cidx[n] = b; changed = True
            if not changed:
                break
        tgt = self.cand[cidx]
        self.x = self.x + np.clip(tgt - self.x, -step, step)

    def serve(self):
        r = super().serve()
        if self.use_map:
            self.update_map()
        return r

    def update_map(self):
        env, p = self.env, self.p
        # observed attenuation = |h|^2 / modelled LoS gain at estimated user positions
        g0 = los_gain(self.env, self.x[:, None], env.yw[:, None], env.meas[None])      # (N,K)
        obs = np.abs(self.last_H.T) ** 2 / g0
        att_db = 10 * np.log10(np.clip(obs, 1e-6, 10.0))
        los = (att_db > self.thr).astype(float)
        if (~los.astype(bool)).any():   # running estimate of NLoS attenuation (dB)
            self.kS = p['forget'] * self.kS + att_db[los == 0].sum()
            self.kC = p['forget'] * self.kC + (los == 0).sum()
        obs = los
        f = p['forget']
        ui = np.tile(self.ue_idx(env.meas), p['N'])
        az, db = self.dir_idx(np.repeat(self.x, p['K']), np.repeat(env.yw, p['K']), np.tile(env.meas, (p['N'], 1)))
        o = obs.ravel()
        self.S *= f; self.Cnt *= f
        np.add.at(self.S, (ui, az, db), o); np.add.at(self.Cnt, (ui, az, db), 1.0)
        self.gS = f * self.gS + o.mean(); self.gC = f * self.gC + 1.0


# ------------------------------------------------------------------ runner
SCHEMES = {
    'prism': lambda e: PRISM(e),
    'nopred': lambda e: PRISM(e, use_pred=False),
    'nomap': lambda e: PRISM(e, use_map=False),
    'hung': lambda e: PRISM(e, mode='hungarian'),
    'reactive': lambda e: Reactive(e),
    'fixed': lambda e: FixedPASS(e),
    'mimo': lambda e: ConvMIMO(e),
    'genie': lambda e: GeniePASS(e),
}


def run(prm_over=None, seed=0, schemes=('prism', 'reactive', 'fixed', 'mimo', 'genie'),
        trace=False):
    prm = dict(DEFAULT); prm.update(prm_over or {})
    env = Env(prm, seed)
    env.step()
    S = {k: SCHEMES[k](env) for k in schemes}
    T0, T = prm['warm'], prm['T']
    acc = {k: np.zeros(prm['K']) for k in schemes}
    tr = {k: [] for k in schemes}
    # save/restore RNG so every scheme sees identical fading (common random numbers)
    for t in range(T0 + T):
        env.step()
        for k, s in S.items():
            s.act()
            r = s.serve()
            if t >= T0:
                acc[k] += r
            if trace:
                tr[k].append(r.sum())
    out = {k: acc[k] / T for k in schemes}
    if trace:
        return out, tr, S
    return out


def metrics(per_user_list):
    """per_user_list: list over seeds of per-user average rates (K,)"""
    a = np.array(per_user_list)
    sr = a.sum(1)
    jain = a.sum(1) ** 2 / (a.shape[1] * (a ** 2).sum(1))
    return dict(sum=sr.mean(), sum_std=sr.std(), p5=np.percentile(a.ravel(), 5),
                jain=jain.mean(), all=a.ravel())


# ======================================================================
# PRISM v2: Predictive Rendezvous-and-Intercept Sweep with blockage-aware
# Micro-positioning  (the proposed algorithm used in the manuscript)
# ======================================================================
class PRISMSweep(PRISM):
    name = 'PRISM (proposed)'

    def __init__(self, env, use_pred=True, use_map=True, use_sweep=True):
        super().__init__(env, use_pred=True, use_map=use_map)
        p = env.p
        self.use_pred2, self.use_sweep = use_pred, use_sweep
        self.name = 'PRISM (proposed)'
        if not use_pred: self.name = 'PRISM w/o prediction'
        if not use_map: self.name = 'PRISM w/o blockage map'
        if not use_sweep: self.name = 'PRISM w/o sweep ordering'
        N = p['N']
        self.dir = np.where(np.arange(N) % 2 == 0, 1, -1)
        self.tgt_user = -np.ones(N, int)
        self.dwell = np.zeros(N, int)
        self.last_served = -np.ones(N, int)
        self.Hi = p.get('H_int', 40)            # intercept look-ahead (slots)
        self.D0 = p.get('D0', 6)                # nominal dwell (slots)
        self.Dmax = p.get('Dmax', 20)
        self.Ta = p.get('Ta', 10)               # clustering period (slots)
        self.offs = np.arange(-3.0, 3.01, 0.5)  # micro-positioning offsets (m)
        self.t = 0
        self.cluster = None

    # ---- slow timescale: load-balanced user-to-waveguide clustering
    def recluster(self, U):
        p = self.p; N, K = p['N'], p['K']
        cap = int(np.ceil(K / N))
        ymean = U[:, :, 1].mean(1)
        cost = np.abs(ymean[:, None] - np.repeat(self.env.yw, cap)[None])   # (K, N*cap)
        r, c = linear_sum_assignment(cost)
        cl = np.empty(K, int); cl[r] = c // cap
        self.cluster = cl

    def predict_traj(self, H):
        p = self.p
        if self.use_pred2:
            return self.kf.predict(H, p['L'], p['W'])
        return np.repeat(self.env.meas[:, None], H, 1)

    def intercept(self, n, k, U):
        """earliest slot h at which PA n can reach user k's predicted x."""
        step = self.p['v_pa'] * self.p['dt']
        hs = np.arange(1, U.shape[1] + 1)
        ok = np.abs(U[k, :, 0] - self.x[n]) <= step * hs
        h = int(np.argmax(ok)) if ok.any() else U.shape[1] - 1
        return h, U[k, h]

    def micro(self, n, ue):
        """blockage-aware service point near the rendezvous position."""
        p = self.p
        xs = np.clip(ue[0] + self.offs, 0, p['L'])
        g = los_gain(self.env, xs, self.env.yw[n], ue[None])
        if self.use_map:
            g = g * self.lookup(xs, np.full_like(xs, self.env.yw[n]), np.broadcast_to(ue, (len(xs), 2)))
        return xs[int(np.argmax(g))]

    def choose_next(self, n, U):
        members = np.where(self.cluster == n)[0]
        members = members[members != self.last_served[n]] if len(members) > 1 else members
        if len(members) == 0:
            return -1
        w = 1 / self.Rbar
        if not self.use_sweep:          # PF-greedy target selection, no ordering
            return members[int(np.argmax(w[members]))]
        for _ in range(2):
            ahead = [(self.intercept(n, k, U)[1][0] - self.x[n]) * self.dir[n] for k in members]
            ahead = np.array(ahead)
            m = ahead >= -0.5
            if m.any():
                # nearest user ahead, PF-weighted tie-breaking
                score = ahead[m] / (w[members[m]] / w[members].mean())
                return members[m][int(np.argmin(score))]
            self.dir[n] *= -1
        return members[0]

    def act(self):
        env, p = self.env, self.p
        N = p['N']
        self.kf.update(env.meas)
        U = self.predict_traj(self.Hi)
        if self.cluster is None or self.t % self.Ta == 0:
            self.recluster(U[:, :10])
        self.t += 1
        step = p['v_pa'] * p['dt']
        w = 1 / self.Rbar
        for n in range(N):
            k = self.tgt_user[n]
            if k < 0 or self.cluster[k] != n:
                k = self.tgt_user[n] = self.choose_next(n, U); self.dwell[n] = 0
                if k < 0:
                    continue
            h, ue = self.intercept(n, k, U)
            tgt = self.micro(n, ue if self.dwell[n] == 0 else U[k, 0])
            if self.dwell[n] > 0:                       # serving: track user
                self.dwell[n] -= 1
                if self.dwell[n] == 0:
                    self.last_served[n] = k
                    self.tgt_user[n] = self.choose_next(n, U)
            elif abs(tgt - self.x[n]) <= step and h <= 1:   # rendezvous reached
                cw = w[self.cluster == n]
                self.dwell[n] = int(np.clip(np.ceil(self.D0 * w[k] / cw.mean()), 1, self.Dmax))
            self.x[n] += np.clip(tgt - self.x[n], -step, step)


class BlindSweep(Scheme):
    """Periodic end-to-end roaming at v_max (blind antenna roaming)."""
    name = 'Blind roaming PASS'
    def __init__(self, env):
        super().__init__(env)
        p = env.p
        self.x = np.linspace(0, p['L'], p['N'] + 2)[1:-1].copy()
        self.dir = np.where(np.arange(p['N']) % 2 == 0, 1.0, -1.0)
    def act(self):
        p = self.p
        self.x += self.dir * p['v_pa'] * p['dt']
        lo, hi = self.x < 0, self.x > p['L']
        self.x = np.clip(self.x, 0, p['L']); self.dir[lo | hi] *= -1


SCHEMES.update({
    'prism': lambda e: PRISM(e),
    'nopred': lambda e: PRISM(e, use_pred=False),
    'nomap': lambda e: PRISM(e, use_map=False),
    'hung': lambda e: PRISM(e, mode='hungarian'),
    'sweep': lambda e: PRISMSweep(e),
    'blind': lambda e: BlindSweep(e),
})


class GenieTrack(GeniePASS):
    """Perfect position/blockage knowledge, one-to-one tracking, speed-limited."""
    name = 'Genie tracking (speed-limited)'
    def act(self):
        x0 = self.x.copy(); super().act()
        st = self.p['v_pa'] * self.p['dt']
        self.x = x0 + np.clip(self.x - x0, -st, st)


SCHEMES['gtrack'] = lambda e: GenieTrack(e)
