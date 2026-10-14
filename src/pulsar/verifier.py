"""M3 -- Verifier: exact rejection-sampling scheme (safety net, quality floor).

THEOREM (Exactness, proposal-agnostic).
Let p be any target distribution on V, q any proposal with q(x)>0 wherever
used, and define  a(x) = min(1, p(x)/q(x)),  r = norm(max(p - q, 0)).
Draw x ~ q; emit x with prob a(x), else draw ~ r.  Then the emitted token is
distributed exactly as p, *for every q* -- even an adversarial one.  Hence
any candidate generator (M1), any superposition policy (M2) and any collapse
(M4d) can be trained freely without ever moving the output distribution away
from the base model.  Pulsar's speed claims therefore never trade quality.
"""
from __future__ import annotations
import numpy as np

TINY = 1e-12


def residual_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    if s <= TINY:
        return p.copy()
    return r / s


def accept_prob(p: np.ndarray, q: np.ndarray, x: int) -> float:
    return float(min(1.0, p[x] / max(q[x], TINY)))


def verify_block(p_seq, q_seq, x_seq, rng):
    """Chain verification of one drafted path.

    p_seq[d], q_seq[d]: target/draft distributions at depth d (target rows
    computed by the single verification pass over the block).
    x_seq[d]: drafted token at depth d.

    Returns (emitted_tokens, n_accepted_draft).  On the first rejected depth
    we resample from the residual and stop (the suffix of the branch is
    invalid).  If all draft tokens are accepted, the caller may append the
    free 'bonus' token from the last target row (p_seq covers it).
    """
    emitted = []
    k = len(x_seq)
    for d in range(k):
        p, q, x = p_seq[d], q_seq[d], x_seq[d]
        if rng.random() < accept_prob(p, q, x):
            emitted.append(int(x))
        else:
            emitted.append(int(rng.choice(len(p), p=residual_dist(p, q))))
            return emitted, d
    return emitted, k


def exact_sample(p: np.ndarray, q: np.ndarray, rng) -> int:
    """Single-position version (used to verify the exactness theorem)."""
    x = int(rng.choice(len(p), p=q))
    if rng.random() < accept_prob(p, q, x):
        return x
    return int(rng.choice(len(p), p=residual_dist(p, q)))
