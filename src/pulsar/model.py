"""Toy Markov target/draft LM pair with full analytic access.

Every quantity the theory predicts -- acceptance probability, TV distance to
the effective proposal, stationary distribution -- is *exactly computable*
here, so theorems are verified against ground truth rather than against
another simulation.

Target  p(. | t3,t2,t1) : order-3 Markov mixing latent 'topic' templates.
Draft   q(. | t2,t1)    : order-2 Markov, context-blind.
Regimes: half the order-2 contexts are 'easy' -- the last two tokens already
determine the target's next-token row, so an order-2 draft can nail them --
and half 'hard' -- the unseen third token matters, so no context-blind draft
can.  Regime is a deterministic function of the draft's own context and is
readily predicted from draft-side features (draft entropy), which is what
makes confidence-gated adaptive pulse size beat every fixed block length.
"""
from __future__ import annotations
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(x.sum(-1, keepdims=True), 1e-300)


class ToyLM:
    def __init__(self, V: int = 20, K: int = 6, seed: int = 0,
                 lam: float = 0.06, tmpl_conc: float = 0.06,
                 easy_frac: float = 0.5,
                 easy_sigma: float = 0.35, hard_sigma: float = 1.40,
                 hard_mix: float = 0.30):
        self.V, self.K = V, K
        rng = np.random.default_rng(seed)
        irng = np.random.default_rng(seed + 10_000)
        drng = np.random.default_rng(seed + 5_000)

        # latent topic templates (sharp) + uniform smoothing
        T = _normalize(rng.dirichlet(np.full(V, tmpl_conc), size=K))  # (K,V)
        T = (1 - lam) * T + lam / V

        # context -> template index: (b,c) map; easy iff last-2 determine topic
        h_e = (np.arange(V)[:, None] * 7 + np.arange(V)[None, :]) % K      # [b,c]
        is_easy2 = irng.random((V, V)) < easy_frac
        a_idx = np.arange(V)
        h_h = (a_idx[:, None, None] * 13 + (np.arange(V)[None, :, None] * 7
              + np.arange(V)[None, None, :])) % K                            # [a,b,c]
        E = np.broadcast_to(is_easy2[None], (V, V, V))
        h = np.where(E, np.broadcast_to(h_e[None], (V, V, V)), h_h)         # [a,b,c]
        self.P = _normalize(T[h]).reshape(V ** 3, V)  # [c3,d]
        self.is_easy2 = is_easy2

        # draft: order-2 rows = noise-smoothed marginal of target over a
        marg = self.P.reshape(V, V, V, V).mean(axis=0)                       # [b,c,d]
        Q = np.empty((V * V, V))
        for c2 in range(V * V):
            b, c = divmod(c2, V)
            base = marg[b, c]
            if is_easy2[b, c]:
                Q[c2] = _normalize(base * np.exp(drng.normal(0, easy_sigma, V)))
            else:
                Q[c2] = _normalize((0.7 * base + 0.3 / V) *
                                   np.exp(drng.normal(0, hard_sigma, V)))
        self.Q = Q
        self.is_easy = is_easy2.reshape(V, V)
        self._qent = -(Q * np.log(np.maximum(Q, 1e-300))).sum(1)
        self._qtop = Q.max(1)

    # ----------------------------------------------------------- lookups ----
    def ctx3_index(self, ctx) -> int:
        arr = np.asarray(ctx, dtype=np.int64)
        c = np.zeros(3, dtype=np.int64)
        c[3 - min(len(arr), 3):] = arr[-3:]
        return int((c[0] * self.V + c[1]) * self.V + c[2])

    def ctx2_index(self, ctx) -> int:
        arr = np.asarray(ctx, dtype=np.int64)
        c = np.zeros(2, dtype=np.int64)
        c[2 - min(len(arr), 2):] = arr[-2:]
        return int(c[0] * self.V + c[1])

    def target_dist(self, ctx) -> np.ndarray:
        return self.P[self.ctx3_index(ctx)]

    def draft_dist(self, ctx) -> np.ndarray:
        return self.Q[self.ctx2_index(ctx)]

    # ------------------------------------------------------------ sampler ---
    def sample(self, n: int, rng) -> np.ndarray:
        """Exact target sampling (the 'vanilla AR baseline')."""
        cum = np.cumsum(self.P, axis=1)
        out = np.empty(n, dtype=np.int64)
        c0 = c1 = c2 = 0
        r = rng.random(n)
        for t in range(n):
            x = int(np.searchsorted(cum[(c0 * self.V + c1) * self.V + c2], r[t]))
            out[t] = x
            c0, c1, c2 = c1, c2, x
        return out

    def stationary(self, iters: int = 800, tol: float = 1e-15) -> np.ndarray:
        """Exact stationary distribution over order-3 states (einsum fixed point)."""
        V = self.V
        T = self.P.reshape(V, V, V, V)
        pi = np.full((V, V, V), 1.0 / V ** 3)
        d = np.inf
        for it in range(iters):
            new = np.einsum('abc,abcd->bcd', pi, T, optimize=True)
            new /= new.sum()
            d = np.abs(new - pi).max()
            pi = new
            if d < tol:
                break
        self._pi3 = pi
        self._pi3_resid = d
        self._pi3_iters = it
        return pi

    def exact_depth0_stats(self) -> dict:
        """Exact context-averaged depth-0 acceptance / TV / KL under stationarity.
        alpha = E_ctx[ sum_x q(x) min(1,p/q) ] = E_ctx[ 1 - TV(p,q) ]."""
        V = self.V
        pi = self.stationary()
        P4 = self.P.reshape(V, V, V, V)                        # [a,b,c,d]
        Qb = np.broadcast_to(self.Q.reshape(V, V, V)[None], (V, V, V, V))
        acc = np.minimum(1.0, P4 / np.maximum(Qb, 1e-12))
        e_row = (acc * Qb).sum(-1)                             # [a,b,c]
        tv_row = 0.5 * np.abs(P4 - Qb).sum(-1)
        kl_row = (P4 * np.log(np.maximum(P4, 1e-300) /
                             np.maximum(Qb, 1e-300))).sum(-1)
        mask = np.broadcast_to(self.is_easy2[None], (V, V, V))
        wE, wH = pi * mask, pi * ~mask
        sE, sH = wE.sum(), wH.sum()
        return dict(alpha=float((pi * e_row).sum()),
                    tv=float((pi * tv_row).sum()),
                    kl=float((pi * kl_row).sum()),
                    alpha_easy=float((wE * e_row).sum() / sE),
                    alpha_hard=float((wH * e_row).sum() / sH),
                    easy_weight=float(sE),
                    tv_easy=float((wE * tv_row).sum() / sE),
                    tv_hard=float((wH * tv_row).sum() / sH))
