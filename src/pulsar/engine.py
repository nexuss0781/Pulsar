"""Pulsar end-to-end engine -- M1..M7 wired through the interface contracts.

EXACTNESS CONTRACT (Theorem 1, verified): the verifier tests the EFFECTIVE
collapse policy, not the raw draft.  At a tree node u with kept children C(u)
the sequential Born-with-temperature policy is

    pi_u(x) = (q(x)/k_u)^{1/T} / Z_u,  k_u = sum_{x in C(u)} q(x),
    Z_u = sum_{x in C(u)} (q(x)/k_u)^{1/T},

(and for delta policies -- 'greedy'/'min' modes -- pi_u = point mass).  Chain
rejection sampling with acceptance min(1, p/pi) and residual
norm(max(p - pi, 0)) then emits tokens EXACTLY distributed as p for ANY
tree, ANY confidence model, ANY theta, ANY temperature, ANY policy -- speed
never trades quality.

Modes:
  'ar'          -- vanilla target sampling, cost 1/token.
  'chain'       -- Leviathan-style sampled-draft chain (independent draft LM).
  'pulsar'      -- water-level tree + confidence-gated pulse + Born policy.
  'pulsar_fixed'-- same, theta=0 (fixed full-depth blocks).
  'pulsar_max'  -- delta(argmax) policy at every node (greedy collapse).
  'pulsar_min'  -- delta(argmin-amplitude) policy (adversarial stress test).
"""
from __future__ import annotations
import math
import numpy as np
from .verifier import residual_dist


class PulsarConfig:
    def __init__(self, H_max: int = 8, B_branch: int = 4,
                 mode: str = "pulsar",
                 c_draft: float = 0.08, c_tick: float = 0.002,
                 c_v1: float = 0.02, c_chain: float = 0.25,
                 tick_batch: int = 64,
                 temperature: float = 1.0, theta: float | None = None,
                 ema: float = 0.05, theta_floor: float = 2e-3,
                 max_nodes: int = 96, log: bool = True,
                 adv_conf: bool = False, oracle_conf: bool = False,
                 per_node_draft: bool = False, theta_mult: float = 2.5):
        self.__dict__.update(locals()); del self.self


class PulseClock:
    """M6: fixed per-tick budget => deterministic worst-case step work."""

    def __init__(self, cfg):
        self.cfg = cfg

    def ticks_for(self, n_reads: int) -> int:
        return max(1, math.ceil(max(n_reads, 1) / self.cfg.tick_batch))

    def bound_step_units(self, n_nodes_max: int) -> float:
        c = self.cfg
        return (c.c_draft * n_nodes_max
                + c.c_tick * self.ticks_for(n_nodes_max)
                + 1.0 + c.c_v1 * c.H_max)


class PulsarEngine:
    CAP = 4096

    def __init__(self, lm, cfg: PulsarConfig, conf_net=None):
        self.lm, self.cfg, self.conf = lm, cfg, conf_net
        self.clock = PulseClock(cfg)
        self.L_ema = 1.0
        self.logs_feats: list[np.ndarray] | None = []
        self.logs_label: list[np.ndarray] | None = []
        self.trace: list | None = None          # set to [] to record (theta, cost, emit)
        V, C = lm.V, self.CAP
        self.tok = np.zeros(C, np.int64)
        self.par = np.zeros(C, np.int64)
        self.dep = np.zeros(C, np.int64)
        self.nq = np.zeros(C)
        self.keep = np.zeros(C)          # k_u at the node's parent
        self.zsum = np.ones(C)           # Z at the node's parent (pre-T)
        self.A = np.ones(C)
        self.ah = np.ones(C)
        self.c2 = np.zeros(C, np.int64)
        self.c3 = np.zeros(C, np.int64)
        self.X = np.zeros((C, 6))
        self.psi = np.ones(C)            # product of q/keep along path

    # ------------------------------------------------------- theta (M7-A/FxP)
    @property
    def theta(self):
        c = self.cfg
        if c.mode in ("pulsar_fixed",):
            return 0.0
        if c.theta is not None:
            return c.theta
        # M7 variant A (self-consistent fixed point): node worth building iff
        # A_u >= c_node * S  (S = achieved speedup = 1/latency).  theta = d*S.
        S = 1.0 / max(self.L_ema, 1e-6)
        return max(c.theta_floor, c.c_v1 * c.theta_mult * S)

    def predict_a(self, X: np.ndarray) -> np.ndarray:
        if self.conf is not None:
            a = self.conf.predict(X)
        else:
            a = np.exp(np.clip(X[:, 1], -20, 0))    # baseline: q(token) only
        if self.cfg.adv_conf:                        # sabotaged field: 1 - a
            a = 1.0 - a
        return a

    # ------------------------------------------------------ policy utilities --
    def collapse_dist(self, u: int):
        """Effective proposal law pi_u at node u.  FALL-THROUGH DESIGN: the
        collapse draws from pi_u (raw draft, temperature-shaped, or a delta
        for greedy modes) over the WHOLE simplex; the block (kept subtree)
        only decides whether VERIFICATION CONTINUATION is precomputed, never
        which token may be emitted.  Exactness: verifier tests pi_u itself.
        Renormalizing the proposal to the kept set (EAGLE/Medusa-style tree
        sampling) strictly lowers per-node acceptance by increasing
        TV(p, pi); fall-through avoids this (ablation E06 measures it)."""
        cfg, lm = self.cfg, self.lm
        if cfg.mode in ("pulsar_max", "pulsar_min"):
            out = np.zeros(lm.V)
            q = lm.Q[self.c2[u]]
            x = int(np.argmax(q) if cfg.mode == "pulsar_max" else np.argmin(q))
            out[x] = 1.0
            return out
        q = lm.Q[self.c2[u]]
        if abs(cfg.temperature - 1.0) < 1e-12:
            return q
        lg = np.log(np.maximum(q, 1e-300)) / max(cfg.temperature, 1e-9)
        lg -= lg.max()
        w = np.exp(lg)
        s = w.sum()
        return w / s if np.isfinite(s) and s > 0 else q

    # ------------------------------------------------------------------ step --
    def step(self, ctx: list[int], rng) -> dict:
        cfg, lm, V = self.cfg, self.lm, self.lm.V
        if cfg.mode == "ar":
            p = lm.target_dist(np.asarray(ctx))
            return dict(emitted=[int(rng.choice(V, p=p))], cost=1.0,
                        n_accepted=1, n_block=0)
        if cfg.mode == "chain":
            return self._step_chain(ctx, rng)

        theta = self.theta
        B, H = cfg.B_branch, cfg.H_max
        nN = 1
        self.par[0] = -1
        self.keep[0] = 1.0
        self.A[0] = 1.0
        self.psi[0] = 1.0
        self.dep[0] = 0
        self.c2[0] = lm.ctx2_index(ctx)
        self.c3[0] = lm.ctx3_index(ctx)
        front = np.array([0], dtype=np.int64)

        # ---- M1 + M4c + M4b: grow tree, water-level prune ------------------
        levels = 0
        for depth in range(1, H + 1):
            m = len(front)
            if m == 0 or nN + m * B > self.CAP - 8:
                break
            f_c2 = self.c2[front]
            q = lm.Q[f_c2]                                    # (m,V) draft rows
            ent = lm._qent[f_c2]
            top1 = np.log(np.maximum(lm._qtop[f_c2], 1e-300))
            part = np.argpartition(-q, B - 1, axis=1)[:, :B]   # top-B ids (m,B)
            child_q2 = np.take_along_axis(q, part, 1)
            ord2 = np.argsort(-child_q2, axis=1)               # sort children by q
            child_q2 = np.take_along_axis(child_q2, ord2, 1)
            part = np.take_along_axis(part, ord2, 1)
            cq = child_q2.ravel()
            ctok = part.ravel()
            pidx = np.repeat(np.arange(m), B)                  # local parent idx
            lq = np.log(np.maximum(cq, 1e-300))
            rank = np.tile(np.arange(B), m)                    # rank within siblings
            Xc = np.stack([np.ones_like(lq), lq, -np.repeat(ent, B),
                           lq - np.repeat(top1, B), np.repeat(top1, B),
                           rank / max(B - 1, 1)], 1)
            if cfg.oracle_conf:
                ac3o = np.repeat(self.c3[front], B)      # row at parent context
                ah = np.minimum(1.0, lm.P[ac3o, ctok] / np.maximum(cq, 1e-12))
            else:
                ah = self.predict_a(Xc)
            # cumulative expected marginal tokens (superposition mass density):
            # reach-and-accept at node u = prod_{v<=u} q_v * ahat_v  (= prod min(q,p) at oracle)
            Apk = np.repeat(self.A[front], B) * np.maximum(cq, 0.0) * ah
            keep = (cq > 0) & (Apk >= theta)
            if not keep.any():
                break
            sel = np.nonzero(keep)[0]
            cap = cfg.max_nodes - (nN - 1)
            if len(sel) > cap:
                sel = sel[np.argsort(-Apk[sel])][:cap]
            lp = pidx[sel]
            ksum = np.bincount(lp, weights=cq[sel], minlength=m)   # renorm mass
            new_ids = np.arange(nN, nN + len(sel), dtype=np.int64)
            self.tok[new_ids] = ctok[sel]
            self.par[new_ids] = front[lp]
            self.dep[new_ids] = depth
            self.nq[new_ids] = cq[sel]
            self.keep[new_ids] = ksum[lp]
            self.A[new_ids] = Apk[sel]
            self.ah[new_ids] = ah[sel]
            self.X[new_ids] = Xc[sel]
            self.psi[new_ids] = self.psi[front[lp]] * cq[sel] / np.maximum(ksum[lp], 1e-300)
            pc2, pc3 = self.c2[front[lp]], self.c3[front[lp]]
            self.c2[new_ids] = (pc2 % V) * V + ctok[sel]
            self.c3[new_ids] = (pc3 % (V * V)) * V + ctok[sel]
            nN += len(sel)
            levels += 1
            front = new_ids

        # ---- weak pulse: nothing survived the water level -> one probe ------
        if nN == 1:
            q = lm.draft_dist(np.asarray(ctx))
            p = lm.target_dist(np.asarray(ctx))
            delta = cfg.mode in ("pulsar_max", "pulsar_min")
            if delta:
                pi = np.zeros(V); x = int(np.argmax(q) if cfg.mode == "pulsar_max"
                                           else np.argmin(q)); pi[x] = 1.0
            else:
                pi = q; x = int(rng.choice(V, p=pi))
            if rng.random() < min(1.0, p[x] / max(pi[x], 1e-300)):
                emitted, n_acc = [x], 1
            else:
                emitted = [int(rng.choice(V, p=residual_dist(p, pi)))]
                n_acc = 0
            self.L_ema = (1 - cfg.ema) * self.L_ema + cfg.ema * 1.0
            return dict(emitted=emitted, cost=1.0, n_accepted=n_acc, n_block=0)

        # ---- M4d sequential collapse + M3 verification (one target pass) ---
        # At node u: draw x ~ pi_u; accept w.p. min(1, p/pi_u(x)); on reject
        # emit residual and STOP.  If accepted AND the continuation child x
        # lies inside the precomputed block, descend (free); otherwise the
        # token is still emitted exactly and we fall out ('fall-through').
        par_view = self.par[1:nN]
        tok_view = self.tok[1:nN]
        emitted, n_acc, ok = [], 0, True
        u, path_len, fall_throughs = 0, 0, 0
        renorm = cfg.mode == "pulsar_renorm"
        while True:
            if renorm:
                kids = np.nonzero(par_view == u)[0] + 1
                if len(kids) == 0 or path_len >= H:
                    break
                pi = np.zeros(V)
                pi[self.tok[kids]] = self.nq[kids] / np.maximum(self.keep[kids], 1e-300)
                pi /= pi.sum()
                p = lm.P[self.c3[u]]
                path_len += 1
                x = int(rng.choice(V, p=pi))
                if ok and rng.random() < min(1.0, p[x] / max(pi[x], 1e-300)):
                    emitted.append(x); n_acc += 1
                    u = int(kids[int(np.nonzero(self.tok[kids] == x)[0][0])])
                else:
                    ok = False
                    emitted.append(int(rng.choice(V, p=residual_dist(p, pi))))
                    break
                continue
            if u != 0 and path_len >= H:
                break
            pi = self.collapse_dist(u)
            p = lm.P[self.c3[u]]
            path_len += 1
            x = (int(np.argmax(pi)) if cfg.mode in ("pulsar_max", "pulsar_min")
                 else int(rng.choice(V, p=pi)))
            if ok and rng.random() < min(1.0, p[x] / max(pi[x], 1e-300)):
                emitted.append(x)
                n_acc += 1
                kids = np.nonzero(par_view == u)[0] + 1
                hit = np.nonzero(tok_view[kids - 1] == x)[0] if len(kids) else None
                if hit is not None and len(hit) > 0:
                    u = int(kids[int(hit[0])])            # continue in block
                else:
                    fall_throughs += 1
                    break                                  # accepted, no row
            else:
                ok = False
                r = residual_dist(p, pi)
                emitted.append(int(rng.choice(V, p=r)))
                break
        if ok and path_len == H:
            # reached full depth with all accepted -> bonus token (same pass)
            p_bonus = lm.target_dist(np.asarray(ctx + emitted))
            emitted.append(int(rng.choice(V, p=p_bonus)))

        # ---- M4c supervised log: realized acceptance per built node --------
        if cfg.log and self.logs_feats is not None:
            nodes = np.arange(1, nN)
            parents = self.par[nodes]
            p_at = lm.P[self.c3[parents], self.tok[nodes]]
            a_true = np.minimum(1.0, p_at / np.maximum(self.nq[nodes], 1e-12))
            self.logs_feats.append(self.X[nodes].copy())
            self.logs_label.append(a_true)

        per_node = cfg.c_draft * (nN - 1) if cfg.per_node_draft else cfg.c_draft * levels
        cost = (per_node
                + cfg.c_tick * self.clock.ticks_for(nN - 1)
                + 1.0 + cfg.c_v1 * (nN - 1))
        self.L_ema = (1 - cfg.ema) * self.L_ema + \
            cfg.ema * (cost / max(len(emitted), 1))
        if self.trace is not None:
            self.trace.append((self.theta, cost, len(emitted)))
        return dict(emitted=emitted, cost=cost, n_accepted=n_acc,
                    n_block=nN - 1, path_len=path_len)

    # ---- classical chain speculation baseline (independent draft LM) -------
    def _step_chain(self, ctx, rng) -> dict:
        """Classic Leviathan-style chain speculation.  The draft model runs
        gamma sequential passes BEFORE any target row is known; verification
        is a single batched pass over the drafted block (suffix discarded on
        rejection -- but the draft cost was already paid)."""
        cfg, lm, V = self.cfg, self.lm, self.lm.V
        g = cfg.H_max
        c2 = lm.ctx2_index(ctx)
        toks = np.empty(g, np.int64)
        qrows = np.empty((g, V))
        c2s = np.empty(g, np.int64)
        c3x = lm.ctx3_index(ctx)
        for d in range(g):                      # blind drafting
            q = lm.Q[c2]
            qrows[d] = q
            c2s[d] = c2
            x = int(rng.choice(V, p=q))
            toks[d] = x
            c2 = (c2 % V) * V + x
            c3x = (c3x % (V * V)) * V + x
        emitted, n_acc, ok = [], 0, True
        c3 = lm.ctx3_index(ctx)
        for d in range(g):                      # one batched verify pass
            q, x = qrows[d], int(toks[d])
            p = lm.P[c3]
            if rng.random() < min(1.0, p[x] / max(q[x], 1e-12)):
                emitted.append(x)
                n_acc += 1
            else:
                ok = False
                emitted.append(int(rng.choice(V, p=residual_dist(p, q))))
                break
            c3 = (c3 % (V * V)) * V + int(toks[d])
        if ok:
            p_bonus = lm.target_dist(np.asarray(ctx + emitted))
            emitted.append(int(rng.choice(V, p=p_bonus)))
        cost = 1.0 + (cfg.c_chain + cfg.c_v1) * g
        self.L_ema = (1 - cfg.ema) * self.L_ema + \
            cfg.ema * (cost / max(len(emitted), 1))
        return dict(emitted=emitted, cost=cost, n_accepted=n_acc,
                    n_block=len(toks))

    # ------------------------------------------------- train M4c from logs ----
    def train_confidence(self, steps=800, lr=0.6, seed=0):
        from .confidence import LogisticConfidence
        X = np.concatenate(self.logs_feats, 0)
        y = np.concatenate(self.logs_label, 0)
        net = LogisticConfidence(X.shape[1], lr=lr, seed=seed)
        net.fit(X, y, steps=steps)
        net.fit_isotonic(X, y)           # M4c calibration layer
        self.conf = net
        return net, X, y

    # ------------------------------------------------------------------ run ----
    def generate(self, prompt: list[int], n_tokens: int, rng,
                  record_steps: bool = False, reset_logs: bool = False) -> dict:
        if reset_logs and self.logs_feats is not None:
            self.logs_feats, self.logs_label = [], []
        ctx = list(prompt)
        out: list[int] = []
        tot_cost = 0.0
        per_step, blocks, accs, emits = [], [], [], []
        while len(out) < n_tokens:
            s = self.step(ctx, rng)
            out.extend(s["emitted"])
            tot_cost += s["cost"]
            per_step.append(s["cost"] / max(len(s["emitted"]), 1))
            blocks.append(s.get("n_block", 0))
            accs.append(s["n_accepted"])
            emits.append(len(s["emitted"]))
            ctx.extend(s["emitted"])
        out = np.asarray(out[:n_tokens])
        n = len(out)
        ps = np.asarray(per_step)
        return dict(tokens=out, n=n, cost=tot_cost,
                    speedup=n / tot_cost, latency=tot_cost / n,
                    mean_block=float(np.mean(blocks)),
                    mean_accept=float(np.mean(accs)),
                    mean_emit=float(np.mean(emits)),
                    steps=len(per_step),
                    lat_mean=float(ps.mean()), lat_sd=float(ps.std()),
                    per_step=ps if record_steps else None)
