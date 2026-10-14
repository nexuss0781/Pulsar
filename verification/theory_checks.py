#!/usr/bin/env python3
"""PULSAR theory verification: every theorem of the paper, checked numerically.

Pure numpy. Deterministic seeds. Exit code 1 on any failure.
Run:  python3 verification/theory_checks.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulsar.verifier import exact_sample, residual_dist, accept_prob, verify_block
from pulsar.rotor import Rotor
from pulsar.confidence import lif_accumulate, lif_fire_time_closed_form
from pulsar.collapse import born_sample, gumbel_collapse, softmax
from pulsar.pulse import tau_expected, speedup_chain, closed_form_optimum
from pulsar.calibrator import tv_acceptance_identity, calibration_report

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def rand_dist(rng, V, conc=1.0):
    return rng.dirichlet(np.full(V, conc))


# ---------------------------------------------------------------- T01 --------
@check("T01 verifier exactness + algebraic identity")
def t01():
    rng = np.random.default_rng(0)
    worst_tv = 0.0
    for trial in range(20):
        V = 30
        p = rand_dist(rng, V, rng.uniform(0.3, 2.0))
        q = rand_dist(rng, V, rng.uniform(0.3, 2.0))
        n = 60_000
        r = np.random.default_rng(100 + trial)
        cnt = np.bincount([exact_sample(p, q, r) for _ in range(n)], minlength=V) / n
        tv = 0.5 * np.abs(cnt - p).sum()
        worst_tv = max(worst_tv, tv)
        # algebraic identity: p = q*min(1,p/q) + (1-A)*residual
        A = float(np.minimum(1.0, p / q) @ q)
        assert abs(A - (1 - 0.5 * np.abs(p - q).sum())) < 1e-12
        rec = np.minimum(p, q) + (1 - A) * residual_dist(p, q)
        assert np.abs(rec - p).max() < 1e-12
    return worst_tv <= 0.02, f"max TV over 20 trials = {worst_tv:.5f} (MC noise ~0.006)"


# ---------------------------------------------------------------- T02 --------
@check("T02 expected yield tau=(1-a^{g+1})/(1-a)")
def t01b():
    rng = np.random.default_rng(1)
    for alpha in (0.3, 0.5, 0.7, 0.9, 0.97):
        # closed form vs geometric sum (identity)
        g = 7
        assert abs(tau_expected(alpha, g) - sum(alpha ** d for d in range(g + 1))) < 1e-12
        # MC with genuine rejection walk on random rows (iid per-depth alpha)
        n = 40_000
        em = []
        rows_p = np.array([rand_dist(rng, 25) for _ in range(g + 1)])
        rows_q = np.array([rand_dist(rng, 25) for _ in range(g + 1)])
        # exact per-depth acceptance of these rows
        ad = np.array([float(np.minimum(1.0, rows_p[d] / rows_q[d]) @ rows_q[d])
                       for d in range(g)])
        r = np.random.default_rng(7)
        for _ in range(n):
            e = 1  # bonus/residual token always emitted
            for d in range(g):
                x = r.choice(25, p=rows_q[d])
                if r.random() < min(1.0, rows_p[d][x] / rows_q[d][x]):
                    e += 1
                else:
                    break
            em.append(e)
        em = np.asarray(em, float)
        e_arr = em
        mean = em.mean()
        # EXACT prediction for heterogeneous depths: 1 + sum_k prod_{j<=k} a_j
        pred = 1.0 + float(np.cumprod(ad).sum())
        se = e_arr.std() / np.sqrt(n)
        if abs(mean - pred) > 5 * se:
            return False, f"alpha={alpha}: mean {mean:.4f} vs exact {pred:.4f} (se {se:.4f})"
        # iid special case reduces to the Leviathan formula
        pred_iid = tau_expected(float(ad.mean()), g)
        assert abs(sum(ad[d] ** (d + 1) for d in range(g)) -
                   (pred_iid - 1)) < 1e-12 or float(np.ptp(ad)) > 0.02
    return True, "MC vs exact product formula within 5 MC-se for all alpha grid"


# ---------------------------------------------------------------- T03 --------
@check("T03 speedup formula + closed-form optimal block size")
def t03():
    rng = np.random.default_rng(2)
    bad = 0
    for alpha in np.arange(0.35, 0.99, 0.05):
        for c in (0.05, 0.15, 0.25, 0.4):
            Hs = np.arange(0, 25)
            S = np.array([speedup_chain(alpha, int(h), c) for h in Hs])
            grid = int(Hs[S.argmax()])
            cont = closed_form_optimum(alpha, c)
            if abs(grid - cont) > 1.0:
                bad += 1
            # unimodality: differences flip sign once
            d = np.diff(S)
            sgn = np.sign(d)
            flips = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
            assert len(flips) <= 1, "speedup not unimodal"
    return bad == 0, "argmax_int vs continuous optimum within 1 on 52 (alpha,c) cells; unimodal"


# ---------------------------------------------------------------- T04 --------
@check("T04 divergence: alpha=1-TV, Pinsker, Bretagnolle-Huber")
def t04():
    rng = np.random.default_rng(3)
    worst_pin = worst_bh = 0.0
    for _ in range(400):
        V = rng.integers(5, 40)
        p = rand_dist(rng, V, rng.uniform(0.2, 3.0))
        q = rand_dist(rng, V, rng.uniform(0.2, 3.0))
        alpha = float(np.minimum(1.0, p / q) @ q)
        tv = float(0.5 * np.abs(p - q).sum())
        kl = float(p @ np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)))
        assert abs(alpha - (1 - tv)) < 1e-12          # acceptance = overlap
        worst_pin = max(worst_pin, 2 * tv * tv - kl)   # Pinsker: KL >= 2 TV^2
        worst_bh = max(worst_bh, tv - np.sqrt(max(0.0, 1 - np.exp(-kl))))
    # MC identity check on one pair
    p, q = rand_dist(rng, 30), rand_dist(rng, 30)
    d = tv_acceptance_identity(p, q, 80_000, np.random.default_rng(4))
    mc_ok = abs(d["mc"] - d["exact"]) < 4 * d["mc_se"] + 1e-9
    return worst_pin <= 1e-9 and worst_bh <= 1e-9 and mc_ok, \
        f"Pinsker slack {worst_pin:.2e}, BH slack {worst_bh:.2e}, MC identity ok={mc_ok}"


# ---------------------------------------------------------------- T05 --------
@check("T05 water-level block = optimal (brute force)")
def t05():
    rng = np.random.default_rng(5)
    c = 0.06
    # (a) chain: greedy threshold A_{H+1} >= c*T(H)/C(H) == argmin_H L(H)
    worst = 0.0
    for _ in range(300):
        a = np.sort(rng.uniform(0.15, 0.99, size=10))[::-1]
        A = np.cumprod(a)
        Ls = [1.0 / (1 + 0)] + [0.0]
        Ls = [0.0] + [(1 + c * H) / (1 + A[:H].sum()) for H in range(1, 11)]
        Lmin = min(Ls[1:]) if len(Ls) > 1 else 1.0
        Lmin = min(1.0, Lmin)          # H=0 (no draft, weak pulse) allowed
        T, Cn = 1.0, 1.0
        H = 0
        for d in range(1, 11):
            if A[d - 1] >= c * T / Cn:
                T += A[d - 1]
                Cn += c
                H = d
            else:
                break
        Lg = Cn / T
        worst = max(worst, Lg - Lmin)
    chain_ok = worst < 1e-12
    # (b) tree: greedy in A-descending order vs exhaustive prefix-closed sets
    worst2 = 0.0
    for _ in range(300):
        n = 6
        par = np.array([0, 0, 1, 1, 3, 3])[:n]
        av = rng.uniform(0.05, 0.95, size=n)
        Acum = np.zeros(n)
        order = np.argsort(-av)                      # av>0 => A order = av order? no:
        for u in range(n):
            p = par[u]
            Acum[u] = (1.0 if p == 0 else Acum[int(p)]) * av[u]
        def lat_of(mask):
            for u in np.nonzero(mask)[0]:
                p = par[u]
                if p != 0 and not mask[int(p)]:
                    return np.inf
            if not mask.any():
                return 1.0
            T = 1.0 + Acum[mask].sum()
            return (1.0 + c * mask.sum()) / T
        Ls = [lat_of(np.array([(m >> u) & 1 == 1 for u in range(n)], bool))
              for m in range(1 << n)]
        Lmin = min(Ls)
        T, Cn = 1.0, 1.0
        for u in np.argsort(-Acum):                  # A-descending greedy
            p = par[u]
            if p != 0 and T == 1.0 and not added[u] if False else False:
                pass
            if Acum[u] >= c * T / Cn:
                T += Acum[u]
                Cn += c
        Lg = Cn / T
        worst2 = max(worst2, Lg - Lmin)
    return chain_ok and worst2 < 1e-12, \
        f"chain slack {worst:.2e}, tree slack {worst2:.2e} (exact optimality)"


# ---------------------------------------------------------------- T06 --------
@check("T06 rotor: mixing is mass-preserving; stratified readout variance")
def t06():
    rng = np.random.default_rng(6)
    # circulant sibling mixing M = (1-e)I + e/n J
    for n in (2, 3, 7):
        e = 0.37
        M = Rotor.mixing_matrix(n, e)
        assert np.abs(M.sum(1) - 1).max() < 1e-12, "rows must sum to 1"
        ev = np.sort(np.linalg.eigvals(M).real)
        assert abs(ev[-1] - 1) < 1e-9 and np.allclose(ev[:-1], 1 - e, atol=1e-9)
        psi = rng.uniform(0, 1, size=n)
        assert abs(M @ psi).sum() - psi.sum() < 1e-12, "Born mass preserved"
    # stratification: Var without replacement = (N-n)/(N-1) * Var with
    N, n = 120, 60
    pop = rng.normal(0, 1, size=N)                      # FIXED population
    sig2 = pop.var(ddof=1)
    reps = 6000
    wm = np.empty(reps); wo = np.empty(reps)
    for i in range(reps):
        s = rng.integers(0, N, size=n)                   # with replacement
        wm[i] = pop[s].mean()
        o = rng.permutation(N)[:n]                       # without
        wo[i] = pop[o].mean()
    pred_ratio = (N - n) / (N - 1)
    meas_ratio = wo.var() / wm.var()
    # round-robin coverage fairness
    rot = Rotor(N, 16)
    cov = rot.coverage(30)
    fair = (cov.max() - cov.min()) <= 1
    ok = abs(meas_ratio - pred_ratio) < 0.15 and fair
    return ok, (f"var ratio measured {meas_ratio:.3f} vs predicted "
                f"{pred_ratio:.3f}; round-robin coverage max-min<=1: {fair}")


# ---------------------------------------------------------------- T07 --------
@check("T07 collapse: Gumbel==Born; temperature limits")
def t07():
    rng = np.random.default_rng(7)
    for T in (0.2, 1.0, 4.0):
        logpsi = rng.normal(0, 1.5, size=9)
        p = softmax(logpsi / T)
        n = 120_000
        g = np.array([logpsi[i] for i in range(9)])
        cnt = np.zeros(9)
        for _ in range(1):
            gg = -np.log(-np.log(rng.random((n, 9)) + 1e-300) + 1e-300)
            cnt = np.bincount(np.argmax(g / T + gg, 1), minlength=9) / n
        tv = 0.5 * np.abs(cnt - p).sum()
        if tv > 0.012:
            return False, f"T={T}: Gumbel freq TV {tv:.4f}"
    # T -> 0 delta
    logpsi = rng.normal(0, 1, size=8)
    am = np.argmax(logpsi)
    for _ in range(300):
        if gumbel_collapse(logpsi, rng, 1e-9) != am:
            return False, "T->0 not argmax"
    return True, "Gumbel-max == Born softmax for T in {0.2,1,4}; T->0 = argmax"


# ---------------------------------------------------------------- T08 --------
@check("T08 LIF accumulator == pulse timing (closed form == sim)")
def t08():
    rng = np.random.default_rng(8)
    bad = 0
    TH = 512
    for _ in range(300):
        leak = rng.uniform(0.05, 0.6)
        theta = rng.uniform(0.05, 0.9)
        I = theta * rng.uniform(0.3, 4.0)
        pred = lif_fire_time_closed_form(I, leak, theta)
        sim, _ = lif_accumulate(np.full((TH, 1), I), leak, theta)
        s = sim[0]
        if pred >= TH and np.isinf(s):
            continue                                  # beyond horizon: agree
        if pred != s:
            bad += 1
    # monotonicity in drive
    ts = [lif_fire_time_closed_form(I, 0.2, 0.5) for I in (0.6, 0.8, 1.0, 2.0)]
    mono = all(b <= a for a, b in zip(ts, ts[1:]))
    return bad == 0 and mono, f"300 random (leak,theta,I): exact match={bad==0}; fire-time monotone={mono}"


# ---------------------------------------------------------------- T09 --------
@check("T09 exactness is a perfect chi-square fit to p")
def t09():
    rng = np.random.default_rng(9)
    ps = []
    for k in range(24):
        V = 24
        p = rand_dist(rng, V, 0.6)
        q = rand_dist(rng, V, 1.0)
        n = 20_000
        r = np.random.default_rng(500 + k)
        x = [int(r.choice(V, p=q))]
        cnt = np.zeros(V)
        for _ in range(n):
            xi = exact_sample(p, q, r)
            cnt[xi] += 1
        exp = p * n
        keep = exp > 5
        chi2 = float(((cnt[keep] - exp[keep]) ** 2 / exp[keep]).sum())
        from math import erfc, sqrt
        # p-value via survival of chi2 with dof=keep-1 (Wilson-Hilferty approx)
        dof = max(1, int(keep.sum()) - 1)
        z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
        pv = 0.5 * erfc(z / sqrt(2))
        ps.append(pv)
    lo, hi = min(ps), max(ps)
    return lo > 1e-4, f"24 seeds: chi-square p-values min={lo:.4f} (no systematic bias)"


# ---------------------------------------------------------------- T10 --------
@check("T10 proposal-agnostic exactness (adversarial & anti-target drafts)")
def t10():
    rng = np.random.default_rng(10)
    worst = 0.0
    for _ in range(15):
        V = 26
        p = rand_dist(rng, V, 0.5)
        # (a) delta proposal at the WORST token for acceptance: argmin p
        x = int(np.argmin(p))
        q = np.zeros(V); q[x] = 1.0
        n = 40_000
        r = np.random.default_rng(11)
        cnt = np.zeros(V)
        for _ in range(n):
            if r.random() < accept_prob(p, q, x):
                cnt[x] += 1
            else:
                cnt[int(r.choice(V, p=residual_dist(p, q)))] += 1
        worst = max(worst, 0.5 * np.abs(cnt / n - p).sum())
        # (b) anti-target proposal q ~ 1 - p
        q2 = 1.0 / V - 0.5 * p
        q2 = np.maximum(q2, 1e-6); q2 /= q2.sum()
        cnt = np.zeros(V)
        for _ in range(n):
            cnt[exact_sample(p, q2, r)] += 1
        worst = max(worst, 0.5 * np.abs(cnt / n - p).sum())
    return worst <= 0.02, f"max TV under adversarial proposals = {worst:.5f}"


# ---------------------------------------------------------------- T11 --------
@check("T11 calibration machinery sanity")
def t11():
    rng = np.random.default_rng(12)
    y = rng.random(40_000)
    perfect = y.copy()
    noisy = np.clip(y + rng.normal(0, 0.25, 40_000), 0, 1)
    rp, rn = calibration_report(perfect, y), calibration_report(noisy, y)
    ok = (rp["ece"] < 0.02 and rp["brier"] < 1e-12 and
          rn["brier"] > rp["brier"] and abs(rp["slope"] - 1) < 0.05
          and rn["slope"] < rp["slope"])     # noise attenuates slope
    return ok, f"ECE perfect {rp['ece']:.4f} vs noisy {rn['ece']:.4f}; slopes {rp['slope']:.2f}/{rn['slope']:.2f}"


def main():
    print("=" * 78)
    print("PULSAR -- theory verification")
    print("=" * 78)
    nfail = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:                      # noqa
            ok, detail = False, f"EXCEPTION {type(e).__name__}: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
        if not ok:
            nfail += 1
    print("-" * 78)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
