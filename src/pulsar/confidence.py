"""M4c -- Confidence Field: sub-symbolic trainable estimator of acceptance.

The field assigns to every candidate node u a predicted acceptance
probability  ahat(u) = sigma(w . phi(u)), phi = cheap draft-side features
(draft log-prob, draft/target-free entropies, top-1 margin, context regime
proxies).  Training signal: verifier outcomes logged by M3 (supervised
cross-entropy) -- no human labels, no target-model calls at inference.

Why this is the research core: ahat feeds *both* the pulse generator
(M4b: block = water-level set {A(u) >= theta}) and the collapse (M4d).
Calibration of ahat is not cosmetic -- the water-level rule is optimal
*iff* ahat equals true acceptance probability (see pulse.proposition).

Leaky accumulator / LIF variant: evidence e_t(u) per tick integrates as
    V_t = (1-lambda) V_{t-1} + lambda R e_t   (R = rotor mixing)
with pulse when V_t >= theta.  Closed form: time-to-fire t* =
ceil( ln(1 - theta/I (1-lambda)) / ln(1-lambda) ) for constant input I,
i.e. membrane threshold == pulse timing -- the SNN reading of the core is
*equivalent* to the analytic pulse generator, not merely analogous.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LogisticConfidence:
    """Tiny fully-trainable confidence net (pure numpy GD)."""

    def __init__(self, dim: int, lr: float = 0.5, l2: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=dim)
        self.b = 0.0
        self.lr, self.l2 = lr, l2
        self.iso = None                    # optional (knots, values) logit->prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        if self.iso is not None:
            return np.clip(np.interp(z, self.iso[0], self.iso[1]), 1e-6, 1.0)
        return sigmoid(z)

    def fit_isotonic(self, X: np.ndarray, y: np.ndarray, knots: int = 512):
        """PAVA isotonic regression on training logits -> monotone logit->prob
        map, stored as quantile knots (linear interp at predict time)."""
        z = X @ self.w + self.b
        o = np.argsort(z)
        z_s, y_s = z[o], y[o]
        n = len(z_s)
        vals = y_s.astype(float).copy()
        wts = np.ones(n)
        # pool adjacent violators
        i = 0
        idx = list(range(n))
        # array-based stack PAVA
        st_val, st_w, st_c = [], [], []
        for j in range(n):
            v, w = vals[j], 1.0
            st_val.append(v); st_w.append(w); st_c.append(1)
            while len(st_val) > 1 and st_val[-2] > st_val[-1]:
                v2, w2, c2 = st_val.pop(), st_w.pop(), st_c.pop()
                v1, w1, c1 = st_val.pop(), st_w.pop(), st_c.pop()
                W = w1 + w2
                st_val.append((v1 * w1 + v2 * w2) / W)
                st_w.append(W); st_c.append(c1 + c2)
        fit = np.empty(n)
        pos = 0
        for v, c in zip(st_val, st_c):
            fit[pos:pos + c] = v
            pos += c
        qs = np.quantile(z_s, np.linspace(0, 1, min(knots, n)))
        kfit = np.interp(qs, z_s, fit)
        self.iso = (qs, kfit)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 400, batch: int = 512):
        n = len(X)
        rng = np.random.default_rng(1)
        for _ in range(steps):
            idx = rng.integers(0, n, size=min(batch, n))
            Xb, yb = X[idx], y[idx]
            p = sigmoid(Xb @ self.w + self.b)
            g = p - yb
            m = len(Xb)
            self.w -= self.lr * (Xb.T @ g / m + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    # --- baseline for ablation: use only the draft logit rank ------------
    @staticmethod
    def baseline_from_features(X: np.ndarray) -> np.ndarray:
        """Ablation baseline used by the engine when conf is None:
        ahat = exp(log q(top)) = q(top token)."""
        return np.exp(np.clip(X[:, 1], -20, 0))

    def score_report(self, X, y):
        p = self.predict(X)
        brier = float(((p - y) ** 2).mean())
        return brier, p


def lif_accumulate(inputs: np.ndarray, leak: float, theta: float,
                   mixing=None):
    """Simulate a leaky integrate-and-fire confidence accumulator.
    inputs[t, i]: evidence tick. Returns (fire_time per unit, V trajectory).
    `mixing`: optional callable v->Mv applied each tick (rotor mixing)."""
    T, n = inputs.shape
    V = np.zeros(n)
    fire = np.full(n, np.inf)
    for t in range(T):
        V = (1.0 - leak) * V + leak * inputs[t]
        if mixing is not None:
            V = mixing(V)
        fired = (V >= theta) & np.isinf(fire)
        fire[fired] = t
    return fire, V


def lif_fire_time_closed_form(I: float, leak: float, theta: float) -> float:
    """Constant drive I with V <- (1-lambda)V + lambda*I, V_0=0, evaluated
    after tick t (0-based): V_t = I(1 - (1-lambda)^{t+1}).  Fires at the
    smallest t with V_t >= theta:  t* = ceil(ln(1-theta/I) / ln(1-lambda)) - 1.
    Returns inf when I <= theta (drive below threshold never fires)."""
    if I <= theta:
        return np.inf
    t = np.log(1.0 - theta / I) / np.log(1.0 - leak)
    return float(np.ceil(t) - 1)
