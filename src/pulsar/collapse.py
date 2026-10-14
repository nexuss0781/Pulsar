"""M4d -- Collapse Operator: the measurement of the superposition.

Born/temperature sampling over the block frontier:
    p(u) = psi(u)^{1/T} / sum psi^{1/T}     (psi = draft path amplitude)
Temperature interpolation: T->0 gives argmax path (greedy 'projective
measurement'), T=1 the Born rule, T->inf the uniform trace.  Because M3's
verifier is proposal-agnostic (Exactness Theorem), *any* collapse policy --
even adversarial -- preserves the target distribution; temperature therefore
buys throughput/quality trade-offs only through acceptance rate, never
through output bias.
"""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def born_sample(probs: np.ndarray, rng, temperature: float = 1.0) -> int:
    if temperature <= 1e-9:
        return int(np.argmax(probs))
    p = softmax(np.log(np.maximum(probs, 1e-300)) / temperature)
    return int(rng.choice(len(p), p=p))


def gumbel_collapse(logpsi: np.ndarray, rng, temperature: float = 1.0) -> int:
    """Equivalent formulation: argmax (logpsi/T + Gumbel) -- Gumbel-Max trick.
    Verified identical in law to born_sample (theory_checks T07)."""
    g = -np.log(-np.log(rng.random(len(logpsi)) + 1e-30) + 1e-30)
    return int(np.argmax(logpsi / temperature + g))
