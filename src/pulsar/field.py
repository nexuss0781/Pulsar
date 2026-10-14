"""M1 + M2 -- Candidate Generator and Superposition Field.

M1: K candidate continuations per position.  Default backend: greedy
top-B expansion of the draft distribution (a 'Medusa-style' / prompt-lookup
swappable generator; any generator that returns tokens + log-probs plugs in).

M2: the candidates live in an explicit *superposed* structure: a prefix tree
with path amplitudes

    psi(u) = prod_{v in path(u)} q(x_v | ctx(v))        (draft path prob)
    A(u)   = prod_{v in path(u)} ahat_v                 (cumulative accept)

with Born normalisation over any antichain (set of mutually exclusive
branches):  pi(u) = psi(u) / sum_{frontier} psi.  The amplitudes make the
'superposition' precise: no future is collapsed until M4d measures it, and
mutually exclusive futures share probability mass exactly like orthogonal
quantum branches (the density operator here is diagonal -- dephased -- so
the formalism is Born-rule-shaped without quantum hardware).
"""
from __future__ import annotations
import numpy as np


class CandidateField:
    def __init__(self, max_nodes: int = 4096):
        self.parent: list[int] = [-1]
        self.token: list[int] = [-1]
        self.depth: list[int] = [0]
        self.logq: list[float] = [0.0]        # log q(x_u | prefix(u))
        self.psi: list[float] = [1.0]         # path amplitude (product form)
        self.ahat: list[float] = [1.0]        # per-node *predicted* acceptance
        self.Acum: list[float] = [1.0]        # cumulative predicted acceptance
        self.ctxs: list[tuple] = [( )]        # context (tuple of tokens) at node
        self.n = 1                            # node 0 = virtual root

    # ------------------------------------------------------------- building --
    def add(self, parent: int, tok: int, logq: float, ctx) -> int:
        u = self.n
        self.parent.append(parent)
        self.token.append(int(tok))
        self.depth.append(self.depth[parent] + 1)
        self.logq.append(logq)
        self.psi.append(self.psi[parent] * np.exp(logq))
        self.ahat.append(1.0)
        self.Acum.append(self.Acum[parent])
        self.ctxs.append(ctx)
        self.n += 1
        return u

    def set_ahat(self, u: int, a: float):
        a = float(min(max(a, 1e-6), 1.0))
        self.ahat[u] = a
        self.Acum[u] = self.Acum[self.parent[u]] * a
        for c in range(self.n - 1, u, -1):
            pass  # children fixed later; Acum rebuilt by rebuild_acum()

    def rebuild_acum(self):
        for u in range(1, self.n):
            self.Acum[u] = self.Acum[self.parent[u]] * self.ahat[u]

    # ------------------------------------------------------------- queries --
    @property
    def depths(self):
        return np.asarray(self.depth)

    def frontier(self) -> np.ndarray:
        """Leaves of the current tree (an antichain: mutually exclusive)."""
        is_parent = np.zeros(self.n, dtype=bool)
        for p in self.parent[1:]:
            is_parent[p] = True
        return np.nonzero(~is_parent)[0]

    def born(self, ids: np.ndarray) -> np.ndarray:
        psi = np.array([self.psi[i] for i in ids])
        return psi / psi.sum()

    def path(self, u: int) -> list[int]:
        out = []
        while u != 0:
            out.append(u)
            u = self.parent[u]
        return out[::-1]

    def to_arrays(self):
        return dict(parent=np.asarray(self.parent),
                    token=np.asarray(self.token),
                    depth=np.asarray(self.depth),
                    logq=np.asarray(self.logq),
                    psi=np.asarray(self.psi),
                    ahat=np.asarray(self.ahat),
                    Acum=np.asarray(self.Acum))


def generate_candidates(lm, ctx: list[int], H: int, B: int,
                        field: CandidateField | None = None) -> CandidateField:
    """M1 default backend: expand top-B draft tokens depth by depth.

    Returns field with per-node draft log-probs; contexts tracked so the
    verifier can evaluate the target exactly.
    """
    f = field or CandidateField()
    V = lm.V
    layer = [0]
    layer_ctx = [tuple(ctx)]
    for _ in range(H):
        nxt, nxt_ctx = [], []
        for u, cu in zip(layer, layer_ctx):
            q = lm.draft_dist(np.asarray(list(cu)))
            topk = np.argsort(q)[::-1][:B]
            for x in topk:
                v = f.add(u, int(x), float(np.log(max(q[x], 1e-300))),
                          cu + (int(x),))
                nxt.append(v)
                nxt_ctx.append(cu + (int(x),))
        layer, layer_ctx = nxt, nxt_ctx
        if not layer:
            break
    return f
