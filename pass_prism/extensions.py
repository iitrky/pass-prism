"""Extensions for the revised manuscript:
 - Warehouse-aisle scenario (AGV users on lanes, tall shelf rows)
 - Oracle-PRISM (perfect future trajectories and true future blockage)
 - DDPG baseline (NumPy implementation, trained offline on separate seeds)
"""
import copy, pickle, numpy as np
from . import simulator as P
from .simulator import *


# ============================================================ aisle scenario
class AisleMob:
    """AGV-like users: move along aisle lanes (x) and change lanes only at hall ends."""
    def __init__(self, n, L, W, lanes, v, rng, end=2.0):
        self.rng, self.L, self.lanes, self.end = rng, L, np.asarray(lanes), end
        self.lane = rng.integers(0, len(lanes), n)
        self.p = np.c_[rng.uniform(end, L - end, n), self.lanes[self.lane]]
        self.v = v * rng.uniform(0.7, 1.3, n)
        self.dirx = rng.choice([-1.0, 1.0], n)
        self.wp = self._along(np.arange(n))
        self.mode = np.zeros(n, int)          # 0: along aisle, 1: lane change at end

    def _along(self, idx):
        tx = np.where(self.dirx[idx] > 0, self.L - self.end, self.end)
        return np.c_[tx, self.lanes[self.lane[idx]]]

    def step(self, dt):
        d = self.wp - self.p
        dist = np.linalg.norm(d, axis=1) + 1e-9
        stepl = self.v * dt
        arr = dist <= stepl
        self.p = np.where(arr[:, None], self.wp, self.p + d / dist[:, None] * stepl[:, None])
        for i in np.where(arr)[0]:
            if self.mode[i] == 0:              # reached end of aisle: maybe change lane
                self.lane[i] = int(np.clip(self.lane[i] + self.rng.integers(-1, 2), 0, len(self.lanes) - 1))
                self.wp[i] = [self.p[i, 0], self.lanes[self.lane[i]]]; self.mode[i] = 1
            else:                              # lane change done: travel back along aisle
                self.dirx[i] *= -1
                self.wp[i] = self._along(np.array([i]))[0]; self.mode[i] = 0
        return self.p


_orig_env_init = P.Env.__init__


def _env_init(self, prm, seed):
    _orig_env_init(self, prm, seed)
    if prm.get('scenario') == 'aisle':
        rng = self.rng
        L, W, N = prm['L'], prm['W'], prm['N']
        lanes = self.yw
        rows = (lanes[:-1] + lanes[1:]) / 2
        xs = np.arange(prm.get('shelf_x0', 4.0), L - prm.get('shelf_x0', 4.0) + 1e-9, prm.get('shelf_dx', 1.0))
        self.shelf = np.array([[x, y] for y in rows for x in xs])
        self.p = dict(prm); self.p['n_shelf'] = len(self.shelf)
        self.users = AisleMob(prm['K'], L, W, lanes, prm['v_user'], rng)
        self.meas = self.users.p.copy()


P.Env.__init__ = _env_init

AISLE = dict(scenario='aisle', shelf_r=0.5, shelf_h=2.8, shelf_loss=30.0, n_human=5)


# ============================================================ shared BCA
def bca_place(self, rate, disc, step):
    N = self.p['N']
    w = 1 / self.Rbar
    cidx = np.argmin(np.abs(self.cand[None, :] - self.x[:, None]), 1)
    for _ in range(self.n_bca):
        changed = False
        for n in range(N):
            cur = rate[np.arange(N), :, cidx, :]
            other = np.max(np.delete(cur, n, 0), 0)
            val = np.maximum(rate[n], other[:, None, :]) @ disc
            obj = w @ val
            obj[cidx[n]] *= (1 + 1e-6)
            b = int(np.argmax(obj))
            if b != cidx[n]:
                cidx[n] = b; changed = True
        if not changed:
            break
    tgt = self.cand[cidx]
    self.x = self.x + np.clip(tgt - self.x, -step, step)


class OraclePRISM(PRISM):
    """PRISM placement fed with the TRUE future user positions and TRUE future
    geometric blockage over the horizon (clairvoyant information benchmark)."""
    name = 'Oracle-PRISM (clairvoyant)'

    def __init__(self, env):
        super().__init__(env, use_map=False)
        self.name = 'Oracle-PRISM (clairvoyant)'

    def act(self):
        env, p = self.env, self.p
        H, step = self.H, p['v_pa'] * p['dt']
        fut = copy.deepcopy(env)              # schemes never draw from env.rng -> exact future
        U = np.empty((p['K'], H, 2)); obs = []
        for h in range(H):
            fut.step(); U[:, h] = fut.users.p; obs.append(fut.obs)
        hs = self.hs
        xt = self.x[:, None, None] + np.clip(self.cand[None, :, None] - self.x[:, None, None], -step * hs, step * hs)
        g = los_gain(env, xt[:, None], env.yw[:, None, None, None], U[None, :, None])      # (N,K,C,H)
        bl = np.empty_like(g)
        for h in range(H):
            c, rr, hh, Lb = obs[h]
            for n in range(p['N']):
                ux, inv = np.unique(np.round(xt[n, :, h], 6), return_inverse=True)
                pa = np.c_[ux, np.full(len(ux), env.yw[n])]
                b = blockage_loss_dB(pa[None, :], U[:, h][:, None], c, rr, hh, Lb, p['d'], p['hu'])  # (K,u)
                bl[n, :, :, h] = b[:, inv]
        rate = np.log2(1 + env.P * g * 10 ** (-bl / 10) / env.sigma2)
        bca_place(self, rate, self.disc, step)


# ============================================================ NumPy DDPG
class MLP:
    def __init__(self, sizes, rng, out_act=None, scale_last=3e-3):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lim = scale_last if i == len(sizes) - 2 else 1 / np.sqrt(a)
            self.W.append(rng.uniform(-lim, lim, (a, b))); self.b.append(np.zeros(b))
        self.out_act = out_act
        self.m = [np.zeros_like(x) for x in self.W + self.b]
        self.v = [np.zeros_like(x) for x in self.W + self.b]
        self.t = 0

    def forward(self, x):
        self.cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            self.cache.append(x)
        if self.out_act == 'tanh':
            x = np.tanh(x); self.cache[-1] = x
        return x

    def backward(self, g):
        """g: dLoss/dOutput. Returns (param grads, dLoss/dInput)."""
        if self.out_act == 'tanh':
            g = g * (1 - self.cache[-1] ** 2)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_in = self.cache[i]
            gW[i] = a_in.T @ g; gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (self.cache[i] > 0)
        return gW + gb, g

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        params = self.W + self.b
        for i, (p_, g) in enumerate(zip(params, grads)):
            g = np.clip(g, -clip, clip)
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t); vh = self.v[i] / (1 - b2 ** self.t)
            p_ -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, o, tau=1.0):
        for a, b in zip(self.W + self.b, o.W + o.b):
            a *= (1 - tau); a += tau * b


def ddpg_state(env, x, Rbar):
    p = env.p
    w = 1 / Rbar; w = w / w.mean()
    z = env.meas
    return np.r_[x / p['L'] * 2 - 1, z[:, 0] / p['L'] * 2 - 1, z[:, 1] / p['W'] * 2 - 1, np.log(w)]


class DDPGScheme(Scheme):
    name = 'DDPG PASS (NumPy)'

    def __init__(self, env, actor=None, noise=0.0, rng=None):
        super().__init__(env)
        self.x = np.full(env.p['N'], env.p['L'] / 2)
        self.actor, self.noise = actor, noise
        self.rng = rng or np.random.default_rng(0)

    def act(self):
        p = self.p
        s = ddpg_state(self.env, self.x, self.Rbar)
        a = self.actor.forward(s[None])[0]
        if self.noise > 0:
            a = np.clip(a + self.rng.normal(0, self.noise, a.shape), -1, 1)
        self.s, self.a = s, a
        tgt = (a + 1) / 2 * p['L']
        st = p['v_pa'] * p['dt']
        self.x = self.x + np.clip(tgt - self.x, -st, st)


def train_ddpg(prm_over=None, episodes=60, T_ep=400, seed=0, log=print):
    prm = dict(DEFAULT); prm.update(prm_over or {})
    rng = np.random.default_rng(seed)
    N, K = prm['N'], prm['K']
    sd = N + 3 * K
    actor = MLP([sd, 128, 128, N], rng, 'tanh'); actor_t = MLP([sd, 128, 128, N], rng, 'tanh'); actor_t.copy_from(actor)
    critic = MLP([sd + N, 128, 128, 1], rng); critic_t = MLP([sd + N, 128, 128, 1], rng); critic_t.copy_from(critic)
    cap = 60000; S = np.zeros((cap, sd)); A = np.zeros((cap, N)); R = np.zeros(cap); S2 = np.zeros((cap, sd))
    ptr = full = 0
    gamma, tau, B = 0.95, 0.005, 128
    hist = []
    for ep in range(episodes):
        env = Env(prm, 5000 + seed * 1000 + ep); env.step()
        noise = max(0.05, 0.4 * (1 - ep / (0.6 * episodes)))
        ag = DDPGScheme(env, actor, noise, rng)
        ep_r = 0.0
        for t in range(T_ep):
            env.step()
            ag.act(); r = ag.serve()
            rew = float(r.sum()) / 10.0          # sum-rate reward (the reported metric)
            s2 = ddpg_state(env, ag.x, ag.Rbar)
            if t > 20:                                    # skip PF transient
                S[ptr], A[ptr], R[ptr], S2[ptr] = ag.s, ag.a, rew, s2
                ptr = (ptr + 1) % cap; full = min(full + 1, cap)
            ep_r += rew
            if full > 2000:
                idx = rng.integers(0, full, B)
                s, a, rr, ss = S[idx], A[idx], R[idx], S2[idx]
                a2 = actor_t.forward(ss)
                y = rr[:, None] + gamma * critic_t.forward(np.c_[ss, a2])
                q = critic.forward(np.c_[s, a])
                gq, _ = critic.backward((q - y) / B)
                critic.adam(gq, 1e-3)
                ap = actor.forward(s)
                critic.forward(np.c_[s, ap])
                _, gin = critic.backward(-np.ones((B, 1)) / B)
                ga, _ = actor.backward(gin[:, sd:])
                actor.adam(ga, 1e-4)
                actor_t.copy_from(actor, tau); critic_t.copy_from(critic, tau)
        hist.append(ep_r / T_ep)
        log(f'ddpg ep {ep} noise {noise:.2f} avg_reward {ep_r / T_ep:.3f}')
    return actor, hist


_ACTORS = {}
import os
MODEL_DIR = os.environ.get('PRISM_MODEL_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))


def save_actor(actor, path):
    d = {f'W{i}': w for i, w in enumerate(actor.W)}; d.update({f'b{i}': b for i, b in enumerate(actor.b)})
    np.savez(path, **d)


def load_actor(tag):
    """Load a trained DDPG actor from models/ddpg_<tag>.npz."""
    if tag not in _ACTORS:
        z = np.load(os.path.join(MODEL_DIR, f'{tag}.npz' if tag.startswith('td3') else f'ddpg_{tag}.npz'))
        nl = len([k for k in z.files if k.startswith('W')])
        sizes = [z['W0'].shape[0]] + [z[f'W{i}'].shape[1] for i in range(nl)]
        a = MLP(sizes, np.random.default_rng(0), 'tanh')
        a.W = [z[f'W{i}'] for i in range(nl)]; a.b = [z[f'b{i}'] for i in range(nl)]
        _ACTORS[tag] = a
    return _ACTORS[tag]


SCHEMES['oracle'] = lambda e: OraclePRISM(e)
SCHEMES['ddpg'] = lambda e: DDPGScheme(e, load_actor('default'))
SCHEMES['ddpg_aisle'] = lambda e: DDPGScheme(e, load_actor('aisle'))


def metrics_ci(per_user_list):
    a = np.array(per_user_list)
    sr = a.sum(1); n = len(sr)
    return dict(sum=float(sr.mean()), sum_std=float(sr.std(ddof=1)), ci=float(1.96 * sr.std(ddof=1) / np.sqrt(n)),
                p5=float(np.percentile(a.ravel(), 5)),
                jain=float((a.sum(1) ** 2 / (a.shape[1] * (a ** 2).sum(1))).mean()),
                seed_sums=sr.tolist(), all=a.ravel().tolist())


# ---- faster genie: candidates restricted to +/-6 m around each user's x
def _genie_act(self):
    env, p = self.env, self.p
    offs = np.arange(-6.0, 6.01, 0.5)
    ue = env.users.p
    cx = np.clip(ue[:, 0][None, :, None] + offs[None, None, :], 0, p['L'])          # (1,K,C)
    cx = np.broadcast_to(cx, (p['N'],) + cx.shape[1:])
    pa = np.stack([cx, np.broadcast_to(env.yw[:, None, None], cx.shape)], -1)       # (N,K,C,2)
    c, rr, h, Lb = env.obs
    bl = blockage_loss_dB(pa, ue[None, :, None], c, rr, h, Lb, p['d'], p['hu'])
    g = los_gain(env, cx, env.yw[:, None, None], ue[None, :, None]) * 10 ** (-bl / 10)
    u = np.log2(1 + env.P * g / env.sigma2)
    bc = np.argmax(u, -1)
    A = (1 / self.Rbar)[None] * np.max(u, -1)
    rows, cols = linear_sum_assignment(-A)
    self.x[rows] = cx[rows, cols, bc[rows, cols]]
P.GeniePASS.act = _genie_act
