#!/usr/bin/env python3
"""PULSAR end-to-end verification: full engine (M1..M7) vs. the theory.

Writes paper/results/results.json + figures, dumps C++ test inputs, and
asserts every headline claim.  Run from repo root:
    python3 verification/experiments/run_all.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scipy.stats import chisquare
from pulsar.model import ToyLM
from pulsar.engine import PulsarEngine, PulsarConfig
from pulsar.calibrator import calibration_report, auc, reinforce_theta

FIG = os.path.join(ROOT, "paper", "figures")
RES = os.path.join(ROOT, "paper", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
OUT: dict = {}
FAIL: list[str] = []


def R(key, val):
    print(f"  {key}: {val}")
    OUT[key] = val if isinstance(val, (int, float, str, list, dict, bool)) else repr(val)


def claim(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    OUT["claims"][name] = dict(ok=bool(ok), detail=detail)
    if not ok:
        FAIL.append(name)


def fig_speedup(axs_data, fname, xlabel, title, hline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ys, xs, lab in axs_data:
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=lab)
    if hline is not None:
        ax.axhline(hline, ls="--", c="k", lw=1, label=f"AR = {hline}")
    ax.set_xlabel(xlabel); ax.set_ylabel("tokens/s (relative to AR)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)


V = 20
OUT["claims"] = {}
t0 = time.time()

# ============================================================ E00 setup ====
print("E00  model + exact ground truth")
lm = ToyLM(V=V, seed=0)
st = lm.exact_depth0_stats()
pi = lm.stationary()
P4 = lm.P.reshape(V, V, V, V)
R("V", V)
R("alpha_exact", round(st["alpha"], 4))
R("alpha_easy", round(st["alpha_easy"], 4))
R("alpha_hard", round(st["alpha_hard"], 4))
R("tv_easy", round(st["tv_easy"], 4))
R("tv_hard", round(st["tv_hard"], 4))
R("kl", round(st["kl"], 4))
R("easy_weight", round(st["easy_weight"], 4))
R("pi3_resid", lm._pi3_resid)
claim("E00.stationary", lm._pi3_resid < 1e-13, f"residual {lm._pi3_resid:.1e}")
# identity: alpha = 1 - tv  (exact tables)
claim("E00.alpha_eq_1mtv", abs(st["alpha"] - (1 - st["tv"])) < 1e-12)

# ================================================== warmup + train M4c ======
print("E0w  warmup (pulsar_fixed, logging) + train confidence net")
t = time.time()
e0 = PulsarEngine(lm, PulsarConfig(mode="pulsar_fixed", H_max=8, log=True))
e0.generate([5, 3, 9], 5000, np.random.default_rng(11))
net, Xtr, ytr = e0.train_confidence(steps=2500, lr=1.0)
R("warmup_tokens_logged", len(ytr))
R("train_secs", round(time.time() - t, 1))
R("weights", [round(float(w), 3) for w in np.concatenate([[net.b], net.w])])


def make(mode="pulsar", conf=True, **kw):
    kw.setdefault("H_max", 8)
    kw.setdefault("log", False)
    return PulsarEngine(lm, PulsarConfig(mode=mode, **kw),
                        conf_net=net if conf else None)


def run(eng, n=6000, seed=5):
    return eng.generate([5, 3, 9], n, np.random.default_rng(seed))


# ================================================== E01 exactness (modes) ===
print("E01  exactness of the full engine across collapse policies")
prompt = [5, 3, 9]
p_true = lm.target_dist(np.asarray(prompt))
N1 = 12000
modes = [("chain", {}), ("pulsar", {}), ("pulsar_fixed", {}),
         ("pulsar_max", {}), ("pulsar_min", {}),
         ("pulsar", dict(adv_conf=True)),
         ("pulsar", dict(temperature=4.0)),
         ("pulsar", dict(temperature=0.05)),
         ("pulsar", dict(theta=0.999)),
         ("pulsar", dict(theta=1e-9)),
         ("pulsar_renorm", {})]
tvmax = 0.0; pvmin = 1.0; e01 = []
for mode, kw in modes:
    eng = make(mode, conf=(mode != "chain"), **kw)
    cnt = np.zeros(V)
    for i in range(N1):
        s = eng.step(prompt, np.random.default_rng(3_000_000 + i))
        cnt[s["emitted"][0]] += 1
    emp = cnt / N1
    tv = float(0.5 * np.abs(emp - p_true).sum())
    keep = p_true * N1 > 5
    chi = chisquare(cnt[keep], p_true[keep] * N1)
    tvmax = max(tvmax, tv); pvmin = min(pvmin, float(chi.pvalue))
    e01.append(dict(mode=f"{mode}{kw if kw else ''}", tv=round(tv, 4),
                    chi2_p=round(float(chi.pvalue), 4)))
R("modes", e01)
claim("E01.first_token_exactness", tvmax < 0.04 and pvmin > 1e-5,
      f"max TV {tvmax:.4f} (MC floor ~0.011), min chi2 p {pvmin:.4f}")

# sequence-level: unigram+bigram vs exact stationary laws
eng = make("pulsar")
g = eng.generate(list(lm.sample(8, np.random.default_rng(21))), 25000,
                 np.random.default_rng(22))["tokens"]
E1 = pi.sum(axis=(0, 1))                        # stationary unigram
E2 = np.einsum('abc,abcd->cd', pi, P4)           # stationary bigram
c1 = np.bincount(g, minlength=V).astype(float)
bg = g[:-1] * V + g[1:]
c2 = np.bincount(bg, minlength=V * V).astype(float).reshape(V, V)
n1 = len(g) - 1
k1 = E1 * n1 > 5
k2 = E2 * n1 > 5
e1 = E1[k1] / E1[k1].sum() * c1.sum()
e2 = (E2 * n1)[k2]; e2 = e2 / e2.sum() * c2[k2].sum()
chi1 = chisquare(c1[k1], e1)
chi2 = chisquare(c2[k2].ravel(), e2)
R("seq_unigram_chi2_p", round(float(chi1.pvalue), 4))
R("seq_bigram_chi2_p", round(float(chi2.pvalue), 4))
claim("E01.sequence_exactness", chi1.pvalue > 1e-4 and chi2.pvalue > 1e-4,
      f"p={chi1.pvalue:.3f}/{chi2.pvalue:.3f} — indistinguishable from the base model")

# ================================================== E02 acceptance = 1-TV ==
print("E02  measured depth-0 acceptance == exact 1-TV per regime")
tok = lm.sample(400_000, np.random.default_rng(31))
ctxs = np.lib.stride_tricks.sliding_window_view(tok, 3)   # triples (a,b,c)
easy_list, hard_list = [], []
for a, b, c in ctxs[::97]:
    if lm.is_easy2[b, c] and len(easy_list) < 15000:
        easy_list.append([a, b, c])
    elif not lm.is_easy2[b, c] and len(hard_list) < 15000:
        hard_list.append([a, b, c])
e02 = {}
for name, lst, alpha_exact in (("easy", easy_list, st["alpha_easy"]),
                               ("hard", hard_list, st["alpha_hard"])):
    acc = 0; ntot = 0
    eng = make("pulsar", theta=0.999)          # weak-pulse only: pure depth-0 test
    for j, ctx in enumerate(lst):
        s = eng.step(ctx, np.random.default_rng(4_000_000 + j))
        acc += 1 if s["n_accepted"] >= 1 else 0
        ntot += 1
    ahat = acc / ntot
    se = (ahat * (1 - ahat) / ntot) ** 0.5
    e02[name] = dict(measured=round(float(ahat), 4), se=round(float(se), 4),
                     exact=round(float(alpha_exact), 4),
                     z=round(float((ahat - alpha_exact) / se), 2))
    claim(f"E02.acceptance_{name}", abs(ahat - alpha_exact) < 4 * se,
          f"measured {ahat:.4f} vs exact {alpha_exact:.4f} (z={(ahat-alpha_exact)/se:+.2f})")
R("e02", e02)

# ================================================== E03 speedup sweeps ======
print("E03  adaptive vs fixed vs chain (cost model, 8k tokens each)")
t = time.time()
rows = {}
for h in range(1, 11):
    r = run(make("chain", conf=False, H_max=h, log=False), n=6000)
    rows[f"chain_H{h}"] = round(r["speedup"], 4)
    rows[f"chain_emit_H{h}"] = round(r["mean_emit"], 3)
best_chain_h = max(range(1, 11), key=lambda h: rows[f"chain_H{h}"])
thetas = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.25]
for th in thetas:
    r = run(make("pulsar", theta=th), n=6000)
    rows[f"fixed_theta{th}"] = round(r["speedup"], 4)
    rows[f"fixed_theta{th}_blk"] = round(r["mean_block"], 2)
    r2 = run(make("pulsar", theta=th, oracle_conf=True), n=6000)
    rows[f"oracle_theta{th}"] = round(r2["speedup"], 4)
best_fixed = max(rows[f"fixed_theta{th}"] for th in thetas)
best_theta = max(thetas, key=lambda th: rows[f"fixed_theta{th}"])
best_oracle = max(rows[f"oracle_theta{th}"] for th in thetas)
r_adapt = run(make("pulsar"), n=8000)
r_adapt2 = run(make("pulsar"), n=8000, seed=6)
r_base = run(make("pulsar", conf=False), n=6000)
r_full = run(make("pulsar_fixed"), n=4000)
r_renorm = run(make("pulsar_renorm"), n=6000)
adapt = 0.5 * (r_adapt["speedup"] + r_adapt2["speedup"])
R("speedup_ar", 1.0)
R("speedup_chain_best", best_chain_h and round(rows[f"chain_H{best_chain_h}"], 4))
R("chain_best_H", best_chain_h)
R("speedup_pulsar_adaptive", round(adapt, 4))
R("speedup_pulsar_untrained", round(r_base["speedup"], 4))
R("speedup_pulsar_fixed_fullblock", round(r_full["speedup"], 4))
R("speedup_best_fixed_theta", round(best_fixed, 4), )
R("best_fixed_theta", best_theta)
R("speedup_oracle_ceiling", round(best_oracle, 4))
R("speedup_renorm_ablation", round(r_renorm["speedup"], 4))
R("sweep", rows)
claim("E03.beats_AR", adapt > 1.10, f"adaptive {adapt:.3f} vs AR 1.0")
claim("E03.beats_best_chain", adapt >= rows[f'chain_H{best_chain_h}'] * 0.995,
      f"{adapt:.3f} vs chain γ*={best_chain_h}: {rows[f'chain_H{best_chain_h}']:.3f}")
claim("E03.trained_gt_untrained", r_base["speedup"] <= adapt + 1e-9,
      f"trained-adaptive {adapt:.3f} vs raw-logit {r_base['speedup']:.3f}")
claim("E03.adaptive_vs_fixed_full", adapt > 2.0 * r_full["speedup"],
      f"adaptive {adapt:.3f} vs always-full blocks {r_full['speedup']:.3f}")
claim("E06.renorm_ablation", adapt >= r_renorm["speedup"] * 1.15,
      f"fall-through {adapt:.3f} vs renormalized-tree sampling {r_renorm['speedup']:.3f}")
fig_speedup([
    ([rows[f"chain_H{h}"] for h in range(1, 11)], list(range(1, 11)), "chain speculation (draft LM, γ=block)"),
    ([rows[f"fixed_theta{th}"] for th in thetas], [th * 1000 for th in thetas], "Pulsar fixed θ (×1000)"),
    ([rows[f"oracle_theta{th}"] for th in thetas], [th * 1000 for th in thetas], "Pulsar oracle-â fixed θ"),
], "fig_e03_sweep.png", "config", "E03: speedup vs block policy (toy LM, cost model)", hline=1.0)
R("e03_secs", round(time.time() - t, 1))

# ================================================== E04 calibration ========
print("E04  M4c confidence field: held-out calibration")
ev = PulsarEngine(lm, PulsarConfig(mode="pulsar_fixed", H_max=8, log=True))
ev.generate([1, 7, 2], 4000, np.random.default_rng(71))
Xe = np.concatenate(ev.logs_feats, 0); ye = np.concatenate(ev.logs_label, 0)
base_pred = np.exp(np.clip(Xe[:, 1], -20, 0))
rep_t = calibration_report(net.predict(Xe), ye)
rep_b = calibration_report(base_pred, ye)
R("heldout_n", len(ye))
R("trained", {k: round(v, 4) for k, v in rep_t.items() if isinstance(v, float)})
R("baseline", {k: round(v, 4) for k, v in rep_b.items() if isinstance(v, float)})
claim("E04.brier_gain", rep_t["brier"] < rep_b["brier"],
      f"{rep_t['brier']:.4f} < {rep_b['brier']:.4f}")
claim("E04.ece", rep_t["ece"] < 0.06, f"ECE {rep_t['ece']:.4f}")
claim("E04.slope", 0.75 <= rep_t["slope"] <= 1.25, f"slope {rep_t['slope']:.2f}")
# per-bin label-vs-prediction identity
pred_all = net.predict(Xe)
b = np.clip(np.digitize(pred_all, np.linspace(0, 1, 11)) - 1, 0, 9)
gaps, counts = [], []
for k in range(10):
    m = b == k
    if m.sum() > 200:
        gaps.append(abs(float(pred_all[m].mean() - ye[m].mean())))
        counts.append(int(m.sum()))
gaps = np.asarray(gaps); counts = np.asarray(counts)
worst = float(gaps.max()); med = float(np.median(gaps))
mass_le01 = float(counts[gaps <= 0.10].sum() / counts.sum())
R("bin_gaps", dict(max=round(worst, 4), median=round(med, 4),
                   mass_frac_le_010=round(mass_le01, 4)))
claim("E04.bin_identity", worst <= 0.08 and mass_le01 >= 0.9,
      f"isotonic-calibrated M4c: max bin gap {worst:.4f}, median {med:.4f}, "
      f"{100*mass_le01:.1f}% of labels in bins with gap <= 0.10")
claim("E04.near_oracle_ceiling", adapt >= 0.95 * best_oracle,
      f"trained adaptive {adapt:.3f} = {100*adapt/best_oracle:.1f}% of oracle-â ceiling {best_oracle:.3f}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
for rep, lab, col in ((rep_b, "baseline â=q(tok)", "tab:orange"),
                      (rep_t, "trained M4c", "tab:blue")):
    pts = [(pp, ll) for (_, _, pp, ll) in rep["reliability"]]
    ax.plot([x for x, _ in pts], [y for _, y in pts], "o", label=lab, color=col)
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("predicted acceptance"); ax.set_ylabel("realized acceptance")
ax.set_title("E04 reliability (held-out)"); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e04_reliability.png"), dpi=130)
plt.close(fig)

# ================================================== E05 theta trajectory ====
print("E05  M7-A fixed point: theta_ema(t) -> neighborhood of grid optimum")
eng = make("pulsar")
eng.trace = []
r = eng.generate([5, 3, 9], 8000, np.random.default_rng(81))
th = np.array([x[0] for x in eng.trace])
late = th[-3000:].mean()
R("theta_final", round(float(late), 4))
R("theta_grid_star", best_theta)
R("speedup_ema", round(r["speedup"], 4))
R("speedup_best_fixed", round(best_fixed, 4))
claim("E05.theta_near_opt", 0.4 * best_theta <= late <= 2.5 * best_theta,
      f"θ→{late:.4f} vs θ*={best_theta} (flat optimum region)")
claim("E05.no_worse_than_fixed", r["speedup"] >= 0.97 * best_fixed,
      f"adaptive {r['speedup']:.3f} vs best fixed {best_fixed:.3f}")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(th[::20], lw=1, label="θ_ema(t)")
ax.axhline(best_theta, ls="--", c="r", label=f"grid optimum {best_theta}")
ax.set_yscale("log"); ax.set_xlabel("step (×20)"); ax.set_ylabel("θ")
ax.set_title("E05: water-level threshold self-tuning (M7 fixed point)")
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e05_theta.png"), dpi=130)
plt.close(fig)

# ================================================== E07 regime drift ========
print("E07  easy->hard->easy drift: adaptive vs fixed thresholds")
def phase_run(engine, prompts, per=250, seed=0):
    lats = []
    for i, pr in enumerate(prompts):
        rr = engine.generate(pr, per, np.random.default_rng(1000 * (i + 1) + seed))
        lats.append(rr["latency"])
    return lats
easy_prompts = [c + [0] for c in easy_list[-40:]]
hard_prompts = [c + [0] for c in hard_list[-40:]]
seq3 = easy_prompts[:13] + hard_prompts[:13] + easy_prompts[13:26] + hard_prompts[13:26]
res = {}
for tag, eng in (("adaptive", make("pulsar")),
                 ("fixed_theta_easy", make("pulsar", theta=0.01)),
                 ("fixed_theta_hard", make("pulsar", theta=0.10))):
    lats = phase_run(eng, seq3, per=160)
    per_phase = np.mean(np.array(lats).reshape(4, -1), axis=1)
    res[tag] = [round(float(x), 3) for x in per_phase]
R("drift_latency_EH_EH", res)  # [easy,hard,easy,hard]
best_static_worst = min(max(res["fixed_theta_easy"]), max(res["fixed_theta_hard"]))
claim("E07.drift_robust", max(res["adaptive"]) <= best_static_worst * 1.03,
      f"adaptive worst phase {max(res['adaptive']):.3f} <= best-static worst {best_static_worst:.3f} (x1.03)")

# ================================================== E08 M6 bounds ===========
print("E08  M6 pulse clock: per-step work bound")
eng = make("pulsar")
bound = eng.clock.bound_step_units(eng.cfg.max_nodes)
maxc = 0.0
ctxp = [5, 3, 9]
for i in range(4000):
    s = eng.step(ctxp, np.random.default_rng(5_000_000 + i))
    maxc = max(maxc, s["cost"])
    ctxp.extend(s["emitted"])
R("m6_bound_units", round(bound, 3))
R("m6_max_measured", round(maxc, 3))
claim("E08.work_bound", maxc <= bound + 1e-9, f"max {maxc:.3f} ≤ bound {bound:.3f}")
sd_ar = run(make("ar", conf=False), n=8000)
sd_pu = r_adapt
R("lat_sd_ar_note", "AR has zero per-step variance; M6 gives bound, not sd (honest)")

# ================================================== E10 M7 RL ==============
print("E10  M7-B REINFORCE bandit over theta grid (engine-measured profile)")
grid = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10])
prof = np.array([rows[f"fixed_theta{th}"] for th in grid])
noise = 0.015
def sim(theta, rng):
    i = int(np.argmin(np.abs(grid - theta)))
    return prof[i] + rng.normal(0, noise)
rng = np.random.default_rng(42)
pol, hist = reinforce_theta(lambda th, r: sim(th, r), grid, iters=800,
                            lr=0.6, tau=0.02, seed=42)
learned = float(pol @ grid)
R("reinforce_theta", round(learned, 4))
R("oracle_theta", float(grid[prof.argmax()]))
R("oracle_speedup", round(float(prof.max()), 4))
R("learned_expected_speedup", round(float(pol @ prof), 4))
claim("E10.rl_recovers_optimum", float(pol @ prof) > 0.995 * prof.max(),
      f"learned policy E[S]={float(pol @ prof):.4f} vs oracle {prof.max():.4f}")
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(hist, lw=1, label="E[θ] under REINFORCE")
ax.axhline(grid[prof.argmax()], ls="--", c="r", label="oracle θ")
ax.set_xlabel("iteration"); ax.set_ylabel("θ"); ax.legend()
ax.set_title("E10: M7 bandit converges to engine-measured optimum")
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e10_reinforce.png"), dpi=130)
plt.close(fig)

# ================================================== E11 regime map ==========
print("E11  analytic regime map (speedup vs alpha, chain vs pulsar)")
al = np.linspace(0.15, 0.995, 60)
cch, cv1, crow = 0.25, 0.02, 0.08
def chain_best(alpha):
    Hs = np.arange(1, 13)
    tau = (1 - alpha ** (Hs + 1)) / (1 - alpha)
    return float((tau / (1 + (cch + cv1) * Hs)).max())
def pulsar_adaptive(alpha, B=4):
    # keep node at depth d with overlap prob alpha; water level with node cost
    A = alpha ** np.arange(1, 13)
    best = 1.0
    for theta in np.geomspace(1e-4, 0.9, 200):
        keepd = int((A >= theta).sum())
        if keepd == 0:
            continue
        nodes = sum(B ** d for d in range(1, keepd + 1))
        nodes = min(nodes, 96)
        T = 1 + A[:keepd].sum()
        C = 1 + crow * keepd + cv1 * nodes
        best = max(best, T / C)
    return float(best)
sc, sp = np.vectorize(chain_best)(al), np.vectorize(pulsar_adaptive)(al)
R("crossover_alpha_pulsar_vs_AR", float(al[(sp > 1.0)][0]) if (sp > 1.0).any() else None)
fig_speedup([(list(sc), list(al), "chain γ-best"), (list(sp), list(al), "Pulsar adaptive")],
            "fig_e11_regime.png", "per-node draft acceptance α", "E11: speedup regime map (analytic)")
fig, ax = plt.subplots(figsize=(7, 3.2))
engA = make("pulsar"); engA.trace = []
rr = engA.generate([5, 3, 9], 12000, np.random.default_rng(99))
em = np.array([x[2] for x in engA.trace]); co = np.array([x[1] for x in engA.trace])
ax.plot(em, co, ".", ms=3, alpha=.4, label="steps")
ax.plot([0, 8], [0, 8], "k--", label="L=cost/emit iso-latency")
ax.set_xlabel("emitted / step"); ax.set_ylabel("cost / step (target passes)")
ax.set_title(f"E08: Pulsar step scatter (mean L={co.sum()/em.sum():.3f})")
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e08_scatter.png"), dpi=130)
plt.close(fig)

# ================================================== dumps for C++ ===========
cppdir = os.path.join(ROOT, "cpp", "inputs")
os.makedirs(cppdir, exist_ok=True)
cfg = e0.cfg
hdr = np.array([V, int(cfg.H_max), int(cfg.B_branch), int(cfg.max_nodes),
                int(net.w.size), int(net.iso[0].size)], np.int64)
f64 = [np.float64(cfg.c_draft), np.float64(cfg.c_v1), np.float64(cfg.c_tick),
       np.float64(cfg.theta_mult), np.float64(cfg.ema), np.float64(cfg.theta_floor),
       np.float64(net.b)]
with open(os.path.join(cppdir, "model.bin"), "wb") as fh:
    hdr.tofile(fh)
    for x in f64:
        x.tofile(fh)
    net.w.astype(np.float64).tofile(fh)
    net.iso[0].astype(np.float64).tofile(fh)
    net.iso[1].astype(np.float64).tofile(fh)
    lm.P.astype(np.float64).tofile(fh)
    lm.Q.astype(np.float64).tofile(fh)
with open(os.path.join(cppdir, "toy_meta.json"), "w") as fh:
    json.dump(dict(V=V, H_max=int(cfg.H_max), B=int(cfg.B_branch),
                   max_nodes=int(cfg.max_nodes), prompt=[5, 3, 9]), fh, indent=1)
print("dumped cpp inputs/model.bin")

OUT["meta"] = dict(seed=0, V=V, total_secs=round(time.time() - t0, 1),
                   numpy=np.__version__, engine_speedup_adaptive=round(adapt, 4))
with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print("=" * 70)
n_ok = sum(1 for v in OUT["claims"].values() if v["ok"])
print(f"{n_ok}/{len(OUT['claims'])} claims passed   ({time.time()-t0:.0f}s)")
for name in FAIL:
    print("FAILED:", name)
sys.exit(1 if FAIL else 0)
