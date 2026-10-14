"""M7 -- Calibrator: learned thresholds + confidence-field training.

Three coupled learners (all sub-symbolic, all supervised by the verifier):

  A. Fixed-point threshold:  theta_t = d * L_ema_t  (see pulse.proposition).
     Implemented in the engine; here we expose the solver + validation.
  B. REINFORCE bandit over a theta grid (the 'RL lives here' module):
     reward = tokens/sec under the cost model.
  C. LogisticConfidence (M4c) trained on engine logs with soft BCE.

Also: calibration diagnostics -- ECE, Brier, AUC, reliability slope -- and
the *calibration identity*: verifier rejection at depth d given proposal x
satisfies E[accept | x ~ q] = 1 - TV(p,q) (Leviathan); measured vs exact.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------- diagnostics --
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUC for continuous labels (fraction accepted)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    hi = scores >= 0.5
    n1, n0 = int(hi.sum()), int((~hi).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[hi].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_report(pred: np.ndarray, label: np.ndarray,
                       nbins: int = 10) -> dict:
    pred = np.asarray(pred); label = np.asarray(label)
    bins = np.linspace(0, 1, nbins + 1)
    b = np.clip(np.digitize(pred, bins) - 1, 0, nbins - 1)
    ece, cnt = 0.0, len(pred)
    rel = []
    for k in range(nbins):
        m = b == k
        if m.sum() > 0:
            gap = abs(pred[m].mean() - label[m].mean())
            ece += gap * m.sum() / cnt
            rel.append((k, m.sum(), float(pred[m].mean()), float(label[m].mean())))
    brier = float(((pred - label) ** 2).mean())
    A = np.vstack([pred, np.ones_like(pred)]).T
    slope, intercept = np.linalg.lstsq(A, label, rcond=None)[0]
    return dict(ece=ece, brier=brier, auc=auc(pred, label),
                slope=float(slope), intercept=float(intercept),
                reliability=rel, n=cnt)


# ------------------------------------------------------------------ REINFORCE --
def reinforce_theta(sim, thetas: np.ndarray, iters: int = 600, lr: float = 0.08,
                    tau: float = 0.05, seed: int = 0):
    """softmax policy over theta grid; sim(theta, rng) -> tokens/sec (float)."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(thetas))
    baseline = 0.0
    hist = []
    for it in range(iters):
        p = np.exp(logits - logits.max()); p /= p.sum()
        a = rng.choice(len(thetas), p=p)
        r = sim(thetas[a], rng)
        baseline = 0.97 * baseline + 0.03 * r
        adv = r - baseline
        g = np.zeros_like(logits); g[a] = adv
        g -= p * adv                      # score-function with baseline
        logits += lr * g / tau
        hist.append(float(p @ thetas))
    p = np.exp(logits - logits.max()); p /= p.sum()
    return p, hist


# ------------------------------------------------------- identity simulation --
def tv_acceptance_identity(p: np.ndarray, q: np.ndarray, n: int, rng) -> dict:
    """Monte-Carlo check:  E_x~q[min(1,p/q)] = 1 - TV(p,q)."""
    x = rng.choice(len(p), size=n, p=q)
    a = np.minimum(1.0, p[x] / np.maximum(q[x], 1e-12))
    tv_exact = 0.5 * np.abs(p - q).sum()
    return dict(mc=float(a.mean()), mc_se=float(a.std(ddof=1) / np.sqrt(n)),
                exact=float(1 - tv_exact), tv=float(tv_exact))
