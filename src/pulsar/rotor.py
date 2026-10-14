"""M4a -- Rotor: the rotating beam that sweeps the candidate field.

Two coupled mechanisms, both cheap and both *provably* useful:

1. STRATIFIED READOUT.  Each tick the engine can afford to 'look at' only a
   batch of B nodes (run the confidence MLP / update evidence).  The rotor
   visits nodes in fixed cyclic order, so over a full cycle every node is
   read exactly once: this is *sampling without replacement*, whose sample
   mean has variance sigma^2/n * (1 - f) (finite population correction,
   f = B/N per tick, n = total reads), strictly below the (1/n) of i.i.d.
   random readout.  Verified in theory_checks (T06).

2. CIRCULANT SIBLING MIXING.  Evidence carried between ticks is mixed by
   M = (1-eps) I + (eps/n) J over sibling groups: row-stochastic (Born
   mass preserved: sum of amplitudes is invariant), eigenvalues
   {1, (1-eps)^(n-1)} -- a contraction on the disagreement subspace, i.e.
   the rotor smooths noise in the confidence field without shrinking total
   mass.  The 'phase' of the rotor is the cycle position; nothing unitary
   is claimed beyond the permutation part (rotation is a permutation matrix,
   which *is* unitary).
"""
from __future__ import annotations
import numpy as np


class Rotor:
    def __init__(self, n_nodes: int, batch: int, eps: float = 0.25):
        self.N = max(int(n_nodes), 1)
        self.B = max(int(batch), 1)
        self.eps = float(eps)
        self.phase = 0

    # 1. stratified cyclic readout -------------------------------------------------
    def readout(self, tick: int) -> np.ndarray:
        """Indices visited at this tick under round-robin coverage."""
        start = (tick * self.B) % self.N
        idx = (start + np.arange(self.B)) % self.N
        return idx

    def coverage(self, ticks: int) -> np.ndarray:
        """Read count per node after `ticks` ticks (stratification certificate)."""
        cnt = np.zeros(self.N, dtype=np.int64)
        for t in range(ticks):
            np.add.at(cnt, self.readout(t), 1)
        return cnt

    # 2. circulant sibling mixing -----------------------------------------------------
    @staticmethod
    def mixing_matrix(n: int, eps: float) -> np.ndarray:
        return (1.0 - eps) * np.eye(n) + (eps / n) * np.ones((n, n))

    @staticmethod
    def mix(values: np.ndarray, groups: list[np.ndarray], eps: float) -> np.ndarray:
        out = values.copy()
        for g in groups:
            if len(g) > 1:
                m = Rotor.mixing_matrix(len(g), eps)
                out[g] = m @ values[g]
        return out


def stratified_vs_iid_variance(N: int, B: int, n_reads: int, sigma2: float,
                               rng, reps: int = 4000):
    """Analytic: Var_stratified = sigma2/n * (1-f_used), Var_iid = sigma2/n,
    where with per-tick batches of B without replacement across a cycle the
    effective reduction is the finite-population correction (1 - B/N).
    Returns (measured_var_strat, measured_var_iid, predicted ratio)."""
    n_total = n_reads
    f = B / N
    true_vals = rng.normal(0.0, np.sqrt(sigma2), size=(reps, N))
    # stratified: disjoint consecutive batches covering the cycle
    order = rng.permutation(N, size=N)
    strat_means = true_vals[:, order[:n_total]].reshape(reps, -1).mean(1) \
        if n_total <= N else None
    # i.i.d. reads with replacement
    sel = rng.integers(0, N, size=(reps, n_total))
    iid_means = np.array([true_vals[i, sel[i]].mean() for i in range(reps)])
    return strat_means, iid_means, (1.0 - f)
