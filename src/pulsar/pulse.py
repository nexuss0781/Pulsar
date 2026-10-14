"""M4b -- Pulse Generator: WHEN to collapse, and HOW BIG the pulse is.

Block selection is a WATER-LEVEL SET of cumulative confidence:
    emit block { u : A(u) >= theta },   A(u) = prod_{v<=u} ahat_v
prefix-closed automatically (A decreases along paths) -- i.e. only regions of
the superposition where 'mass density' exceeds the pulse threshold get
measured; everything below stays un-collapsed for the next tick.

PROPOSITION (optimality of the water level).  Unit = one target forward pass.
Verifying a block of S nodes costs C(S) = 1 + d*S (d = marginal draft+tree
cost per node, 2d << 1), and yields expected tokens
    T(S) = 1 + sum_{u in S, u != root} A_true(u)
per step, where A_true(u) = prod of TRUE acceptance probs along the path
(verifier exactness makes each accepted node contribute exactly 1 token and
rejection always pays the residual token).  Latency/token L = C/T.  Adding a
candidate node w with cumulative confidence A_w changes the sign of L by
    c*T - C*A_w   =>   include w  <=>  A_w >= d * T(S)/C(S) = d / L(S) = d * S(S).
So the optimal block is a water level with SELF-CONSISTENT threshold
    theta* = d * S* ,   block* = { A >= theta* }  (fixed point in speedup S=1/L).
(The discrete rule: sort A descending, add while A_w >= d*T/C evaluated on the
current set; T/C increases while additions are beneficial and A decreases, so
the sign of the marginal condition flips at most once -> global optimum.)
With A_w decreasing along paths and cost linear, L(S*) is the global
minimum -- verified exhaustively against brute force over all prefix-closed
subtrees (theory_checks T05) and against grid search for the chain case
(T03: closed-form optimal gamma from transcendental critical point).

Uniform alpha-chain special case:  tau(H) = (1-alpha^{H+1})/(1-alpha)
(Leviathan et al.), speedup S(H) = tau/(1+Hc), optimum solves
    ln(alpha) * (1 - alpha^H) + H (1-alpha) alpha^H * ... (see closed_form_optimum).
"""
from __future__ import annotations
import numpy as np


def tau_expected(alpha: float, gamma: int) -> float:
    """E[tokens per step] for chain with per-node acceptance alpha (iid)."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup_chain(alpha: float, gamma: int, c: float) -> float:
    """Tokens/sec relative to AR: cost per step = 1 + c*gamma target passes."""
    return tau_expected(alpha, gamma) / (1.0 + c * gamma)


def closed_form_optimum(alpha: float, c: float, tol: float = 1e-12):
    """Maximizer over reals of speedup_chain via bisection on the derivative.
    d/dH tau = alpha^H * ( -(1-alpha)?? ) derived numerically-safe:
    use golden-section on H in [0, 60] (tau closed form, exact)."""
    lo, hi = 0.0, 60.0
    f = lambda H: -speedup_chain(alpha, H, c)
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    x1, x2 = b - gr * (b - a), a + gr * (b - a)
    for _ in range(200):
        if f(x1) < f(x2):
            b, x2 = x2, x1
            x1 = b - gr * (b - a)
        else:
            a, x1 = x1, x2
            x2 = a + gr * (b - a)
        if b - a < tol:
            break
    H = (a + b) / 2
    return float(H)


def water_level_block(Acum: np.ndarray, theta: float,
                      depth: np.ndarray | None = None,
                      max_nodes: int | None = None) -> np.ndarray:
    """Node ids (excluding root 0) with A(u) >= theta, root always in."""
    keep = Acum >= theta
    keep[0] = False
    ids = np.nonzero(keep)[0]
    if max_nodes is not None and len(ids) > max_nodes:
        ids = ids[np.argsort(-Acum[ids])][:max_nodes]
    return ids


def theta_fixed_point(d: float, Acum_fn, max_iter: int = 60):
    """Solve theta* = d * S(block(theta)) by iteration (S = 1/L).
    Acum_fn(theta) -> (S_nodes, T_expected(S), C(S)) via supplied closure.
    Returns (theta*, L*, S*)."""
    L = 1.0  # start at AR latency
    for _ in range(max_iter):
        theta = d * L
        S, Te, Ce = Acum_fn(theta)
        L_new = Ce / Te
        if abs(L_new - L) < 1e-14:
            L = L_new
            break
        L = 0.5 * L + 0.5 * L_new
    return d * L, L, S
