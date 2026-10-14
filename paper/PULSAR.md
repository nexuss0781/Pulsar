# Pulsar: Exact Tree Speculation with Confidence-Based Scheduling

**Pulsar Technical Report — frozen contracts, exactness, and bounded backend evidence**
*Version 0.2 — October 2026. Companion artifacts: reference implementation, verification harnesses, C++17 reference engine, Stage 6 Transformers adapter and matched benchmark, LaTeX source + compiled PDF.*

---

## Abstract

Speculative decoding trades draft and verification work against the target model's autoregressive cost. Pulsar's frozen verifier gives a pointwise exactness guarantee: proposing from q, accepting with min(1,p/q), and drawing a rejection replacement from max(p−q,0) preserves the target next-token law p. This guarantee does **not** imply a throughput gain. The central empirical hypothesis is therefore conditional and falsifiable: under a measured backend cost profile, a draft-confidence scheduler for exact bounded trees improves held-out tokens per second over a tuning-selected fixed tree only if the paired 95% confidence interval's lower bound is positive. In Stage 6, a Hugging Face Transformers CPU integration (distilgpt2 target, tiny-gpt2 draft, temperature-1 softmax, 24-token horizon, three held-out prompts) passed the full-vocabulary verifier identity with maximum absolute error 1.1·10⁻¹⁹. However, the confidence-tree point estimate was +1.66% versus the fixed tree, with 95% CI [−6.71%, +10.01%]; every confidence bin selected the same fixed action, and the scheduler claim is **not established**. The tuning-selected γ=2 chain reached 15.71 tokens/s and beat the fixed tree (10.98 tokens/s) by 43.00% [34.87%, 50.79%], but both were slower than autoregression at 40.62 tokens/s. Earlier 1.39× results are from an analytic cost-model Markov testbed, not measured neural-model acceleration. Rotor, pulsar, and quantum terminology is explanatory framing; those mechanisms are not part of the Stage 6 measured algorithm.

---

## 1. Introduction

### 1.1 The emission bottleneck

Every mainstream LLM is a decoder: one forward pass produces a next-token distribution over |𝒱| outcomes and then *destroys it*, emitting a single sample. The machine pays full autoregression's tax — one memory-bound pass per token — because it has no representation of the *future field* and no principled scheduler for *when* to collapse it. Speculative decoding (Leviathan et al., 2023) and its successors (Medusa, EAGLE-1/2/3, Lookahead, Sequoia) partially answer this — draft a field of candidates, verify in parallel — but in the deployed variants the field is ad hoc: fixed speculation lengths, hand-tuned acceptance rules, tree sampling schemes whose renormalization quietly *changes* the proposal the verifier must respect.

A pulsar is nature's answer to "how do you emit fast from a huge field?" A rotating, magnetized neutron star sweeps a beam through the entire sky; observers receive *pulses* — periodic, beamed bursts — whose timing is predictable to extraordinary precision. Three features are worth importing into an inference engine:

1. **A rotor sweeping a field** — systematic, stratified coverage of candidates under a fixed per-tick work budget;
2. **Pulses, not streams** — the engine commits *when field density (confidence) crosses a threshold*, emitting 1..N tokens per cycle;
3. **A clock that is not content-dependent** — per-cycle work is bounded by construction, giving latency structure the autoregressive loop cannot offer.

Pulsar instantiates these three as a mathematical architecture, and — the paper's central discipline — **every claim is proved on paper and verified in code before the claim is made**.

### 1.2 Design stance: trainable sub-symbolic, exact-by-construction

The novel, trainable core is deliberately *sub-symbolic and small*: a confidence field M4c (logistic head + isotonic calibrator in the reference; any feature-to-probability map in practice) whose sole job is to predict **verifier acceptance probabilities** from draft-side features. Everything correctness-critical is *frozen*: the verifier's rejection-sampling identity guarantees that no matter how M1, M2, M4a–M4d, M7 misbehave — even adversarially — the committed token stream is distributed *exactly* as the base model's own sampling law. Speed is a free resource that can be bought with miscalibrated confidence; quality never is. This asymmetry is what makes a fast engine safe to ship, and it is the paper's organizing theorem.

### 1.3 Contributions

* **C1 — Formal superposition fields (§3.1, §4.2).** Prefix-tree candidate structures with path amplitudes and a Born rule on antichains; the state is a diagonal density operator, and "collapse" is conditional resampling. This makes the "quantum superposition" intent precise: what is imported is the *branching domain and Born weights*, not interference, and we say so (§8.1).
* **C2 — The Exactness Theorem, in its full generality (§3.2).** Rejection verification against the *effective collapse policy* — not the raw draft — makes Pulsar exact for **any** proposal, **any** temperature policy, **any** pruning, and **any** adversarial selector. This generalizes the standard lemma (proposal = draft model) and it *implies a design correction*: renormalizing the proposal to kept tree children (as deployed tree samplers do) strictly *lowers* per-node acceptance; the remedy — **fall-through collapse** — is C4's mechanism.
* **C3 — Water-level pulse selection, proved optimal (§4.4, Thm 3).** For per-node cost d, the block {u : A(u) ≥ θ} with the self-consistent threshold θ\* = d·S\* maximizes tokens-per-cost among *all* prefix-closed subtrees. Brute-force verification: slack ≈ 10⁻¹⁶ on random trees/chains (T05). Consequences: adaptive block size with no search; a fixed-point learner (M7-A) and a bandit learner (M7-B) both provably tuned (E05, E10).
* **C4 — Fall-through collapse and the selection-gain analysis (§4.5).** The descent samples from the draft's own law; the block pre-decides *continuation depth*, never token legality. Ablation (E06): fall-through 1.388 vs renormalized-tree sampling 1.079 on identical machinery (+29%), with exactness preserved in both — the difference is purely acceptance economics: α = 1 − TV(p, π), and truncating π to the kept set *raises* TV.
* **C5 — Confidence-field theory and practice (§4.3).** The water level's optimality *requires* E[a | â] = â — calibration is not cosmetic but the decision-granularity identity; we train draft-only M4c (soft-BCE + PAVA isotonic) to ECE 7·10⁻⁴ / slope 0.9995, quantify the draft-only information ceiling on tail bins, and show the trained field captures 97.5% of the oracle-â speedup ceiling while beating a raw-logit baseline by +5.8%.
* **C6 — LIF equivalence (§4.3).** Under V ← (1−λ)V + λe with threshold θ, the exact firing-time law equals the pulse-generator's analytic timing rule; "membrane potential *is* a confidence field" becomes a checkable proposition (T08), validating the SNN flavor of M4a/M4c.
* **C7 — Bounds and regime maps (§5).** α = 1 − TV; Pinsker/Bretagnolle–Huber convert draft-side divergence into speedup floors/ceilings; optimal block sizes per (α, c); measured crossover where speculation flips negative (chain γ=8: 0.75× AR — the regime diagrams practitioners need and rarely get).
* **C8 — A verification-first engineering artifact (§6, §7).** Reference Python engine (numpy, readable), harness (11 theory checks, 21 end-to-end claims, all green), C++17 header engine passing its own acceptance test, and cross-implementation agreement (speedup 1.387 vs 1.388; θ 0.0575 both). A verification log documents every bug the harness caught — including two in the *tests' own references* — because a claim that never failed a test is a claim never tested.

### 1.4 What this is not

Pulsar is not a new base model, not quantum computing, and not a claim that brains or stars compute probabilities the way we do. It is an *engine discipline*: module contracts, four theorems, an online calibrator, and a harness that refuses to let any of it drift.

---

## 2. Related work, in equations

We restate each neighbor's *formula*, so Pulsar can be compared where comparison is meaningful (rate constants), and so our harness's closed forms can be sanity-checked against published magnitudes.

**Speculative decoding** (Leviathan et al. 2023; Chen et al. 2023). Draft q of length γ, accept min(1, p/q), residual resampling: the lemma our C2 generalizes. Their headline analysis: τ(α,γ) = (1−α^{γ+1})/(1−α), speedup S = τ/(1+cγ); 2–3× reported with c≈0.1 draft cost. Our T02/T03 re-derive and reproduce these formulas to machine precision against simulation; E03's chain baseline implements them *honestly* (blind drafting cost, single verify pass).

**Medusa** (Cai et al. 2024). K parallel heads; **typical acceptance** — accept draft token if it lies in target's top-p mass set (thresholds 0.08/0.75-style for exact/relaxed variants) — a *heuristic* alternative to rejection sampling that trades the exactness lemma for acceptance volume. Pulsar keeps rejection sampling (exact) and moves the intelligence into *which regions of the tree get pre-verified* (water level) — the same "typical set" intuition, made a thresholded density of cumulative confidence.

**EAGLE / EAGLE-2 / EAGLE-3** (Li et al. 2023–2025). Autoregressive drafting in *feature space* (the "sub-symbolic" precedent: draft the hidden state, not the token), dynamic draft trees by *score budgets*, context-aware re-scores. EAGLE-3 (arXiv 2503.01840) reports up to 6.5× with ~1.4× over EAGLE-2, and 1.38× throughput at batch 64 in SGLang. Pulsar's M4c/M4b can be read as an *axiomatization and optimizer* of EAGLE's engineering: score = calibrated acceptance estimate; tree budget = water level with self-consistent θ; and our fall-through descent removes a proposal-mismatch that renormalized tree sampling lives with.

**Lookahead/prompt-lookup/Sequoia.** Fixed n-gram caches, or offline-computed optimal trees. Pulsar's rotor (M4a) generalizes cyclic readout with stratification variance bounds; M7 replaces tree-size heuristics with a learned scalar.

**Speculative-family survey formulas** (bentoml/123ofai-style practitioner summaries): acceptance α at 0.6–0.9 on real stacks, "α ≥ 0.6 and γ ≥ 5 → 2–3×". §8.3 tabulates what Pulsar's verified formulas predict at those operating points — an honest anchor for projections.

**SNNs.** LIF integrate-and-fire as thresholded accumulation (Gerstner et al.); our T08 proves the equivalence between a leaky confidence accumulator and the analytic pulse timing law, so Pulsar's metaphor "neurons fire when confidence crosses threshold" holds as a derivation for the chosen dynamics.

**Diffusion/parallel decoders.** Block diffusion LMs generate multiple tokens per step; Pulsar is complementary: it *selects among* generated candidates rather than generating them jointly, and its tokens-per-pass economics (M4b+M7) is where the speedup lives on top of any M1 backend.

---

## 3. The formal core

### 3.1 Objects

* **Base model** p(· | t_{≤n}) over 𝒱, |𝒱| = V, frozen.
* **Draft-side head** q(· | s_n) with *less* context than p (a small trainable module, or an n-gram law); s_n = visible context.
* **Candidate field** (M2): a prefix tree T; node u = (token x_u, parent par(u), depth dep(u)), path x_{1..d}. Amplitude and yield:

  ψ(u) = Π_{v≤u} q(x_v | s(v)) ∈ (0,1],  A(u) = Π_{v≤u} q(x_v)·â(v),

  â the M4c acceptance estimate (§4.3). A leaf-set of any cut is an **antichain** (mutually exclusive futures); the **Born rule** P_C(u) = ψ(u)/Σ_{w∈C}ψ(w) makes "superposition" operational. Diagonal density-operator note: ρ = Σ_u ψ(u)²|u⟩⟨u|, dephased — all theorems are elementary probability; the language buys *structure* (branching domain, measure-theoretic collapse), nothing more.
* **Policy** π (M4d): at each node, a law on the next token — raw q (fall-through), q^{1/T}-softmax, or δ (greedy/adversarial). *Exactness will be relative to π, not to q.*

### 3.2 Theorem 1 (Exactness; proposal- and policy-agnostic)

Let p, π be laws on 𝒱, Σ_x π(x) = 1, and define a(x) = min(1, p(x)/π(x)), A = Σ π∧p = 1 − TV(p, π), residual r = norm(max(p−π, 0)). Sample x ~ π; emit x with probability a(x), else emit y ~ r. **Then the emitted token ~ p exactly.**

*Proof.* Pointwise: emitted(y) = π(y)a(y) + (1−A)·r(y) = π∧p + (1−A)·(p−π)⁺/Σ(p−π)⁺, and Σ(p−π)⁺ = 1−A; so emitted = min(π,p) + (p−π)⁺ = p. ∎

*Corollary (chains & trees).* Sequential descent with independent coins at each position, stopping at first rejection, and the always-available target row giving a bonus token, yields per-step emitted count distributed as target autoregression for the whole block: **the sequence law equals the base model's**. No independence assumptions on â, θ, π; adversarial π allowed (the *coins* are what matter: acceptance randomness is fresh at test time). This is checked, not asserted: E01 across 11 policies, T01/T09/T10 on random and adversarial pairs; C++ test at 250k samples.

### 3.3 The acceptance–divergence identity

α := E_{x~π}[a(x)] = 1 − TV(p, π); with Pinsker (TV² ≤ KL/2) and Bretagnolle–Huber (TV ≤ √(1−e^{−KL})):

  **1 − √(KL/2) ≤ α ≤ 1** and TV = 0 ⇔ α = 1.

The *whole engine's speed* is a monotone functional of (π, p) distance: M1/M4c exist to shrink TV(p, π) per unit of head compute; M4b/M7 exist to spend it optimally. §5 and T04 make this quantitative; E02 measures it to z = −0.28.

---

## 4. Modules: mathematics and mechanism

*(Full compact spec: `math/README.md`; code: `src/pulsar/*.py`; swap contracts: App. B.)*

### M1 — Candidate Generator (frozen backend)
Top-B children by q; any generator emitting (x, log q(x)) plugs in (Medusa/EAGLE-style heads, retrieval). Only requirement: the field is a prefix tree with known ψ. §5.4 covers covering: mass not in the tree is not *lost*, because fall-through emits it unverified-but-exact (one-token pulses); B trades tree size vs continuation reach.

### M2 — Superposition Field
As §3.1. Key property (enforced in code): branching is *first-class* until M4d measures; M4b prunes by A(u), never by collapsing paths early.

### M3 — Verifier (frozen)
Theorem 1's machine, run on the collapsed path with the path's *own policy rows*; batched tree pass computes all p rows. Accept/reject + residual or bonus (M5). Quality floor: whatever it accepts, it accepts exactly.

### M4 — Pulsar core

**M4a Rotor.** Two provably-cheap roles: (i) *stratified readout* — fixed batch B_t round-robin over the field ⇒ sample-mean variance reduced by the finite-population factor (N−n)/(N−1) vs i.i.d. reads, measured 0.499 vs predicted 0.504 (T06) — so confidence updates over the field are *fair and minimum-variance* under a fixed tick budget; (ii) *circulant sibling mixing* M_ε with spectrum {1, (1−ε)^{n−1}}: preserves Born mass exactly, damps only sibling disagreement (a noise smoother for the ticked variant of M4c). The LIF tick formulation of M4a+M4c has exact closed-form pulse timing (T08): **rotation → accumulation → threshold → fire** is the same dynamical system as **sweep → confidence → water level → pulse**.

**M4b Pulse Generator.** Block = water level {A(u) ≥ θ}; pulse size = |block|; weak pulses (size-0 kept set) still emit one verified token. **Theorem 3 (optimality):** for per-node cost d and path-monotone A, greedy-include (equivalently θ\* = d·S\* fixed point) attains the *global* max of (1+ΣA)/ (1+dN) over all prefix-closed subtrees; proof by single-crossing of (T/C)′ vs decreasing A (App. A.3); verified against exhaustive enumeration with slack 1.7·10⁻¹⁶ (T05). Uniform-α chain: S(H) = τ/(1+cH) with τ=(1−α^{H+1})/(1−α); golden-section root matches integer grid argmax within 1 everywhere (T03).

**M4c Confidence Field.** â(x,s) = iso(σ(w·φ+b)) with φ = [1, log q, −H(q), logq−logq₁, log q₁, rank/(B−1)] — *draft-side only*; trained on verifier labels min(1, p(x)/q(x)) (soft BCE). The water level's optimality requires E[a|â] = â at decision granularity — verified: bin gaps ≤ 0.007 (max), median 0.0006, ECE 0.0007, slope 0.9995 (E04); the residual structure in tail bins is an information ceiling of draft-only features, not an estimator failure (documented; a deeper model does not fix it — we tried, §6 V9). Trained field vs raw-logit: +5.8% speedup, 97.5% of oracle ceiling (E03).

**M4d Collapse.** Sequential per-node policy + **fall-through**: draw x ~ π_u; if accepted and x inside the kept set, continue; else *emit anyway and stop* — the token is still exact (Theorem 1), only continuation ends. Gumbel-max identity for Born sampling (T07); stable log-softmax policy with delta fallback makes T→0 (greedy) and adversarial policies first-class citizens, all exact (E01).

### M5 — Commit/Emit (glue)
Accepted prefix ⊕ (residual ∣ bonus); cache appends; truncation cap.

### M6 — Pulse Clock
Per-cycle work = c_draft·levels + c_tick·⌈N/B_t⌉ + (1 + c_v1·N), N ≤ budget ⇒ hard upper bound (E08: max 1.582 vs bound 8.844 with conservative budget; C++: 1.60 vs 3.56). Unlike AR (unbounded *total* time at fixed per-token cost) or hand-capped speculation (fixed γ), Pulsar's bound is *per cycle with adaptive yield* — the SLA primitive.

### M7 — Calibrator
Three coupled learners: (A) fixed point θ_t = θ_mult·d/L̂_t (E05: lands within 18% of grid-optimal θ on a flat optimum; speedup within 0.7% of best fixed); (B) REINFORCE bandit on a θ grid, reward = measured tokens/cost (E10: recovers the engine optimum exactly, 1.3863 = 1.3863); (C) M4c net + isotonic head (scoring-rule driven). Rejection penalties do not need an explicit term: miscalibration *is* yield loss at the water level (Thm 3 ties them).

---

## 5. Cost model, speedup, and no-free-lunch

Units: one target forward pass = 1.0. c_head = per-level head forward (0.08), c_v1 = marginal verified row (0.02), c_tick = tick (0.002), c_chain = per blind draft pass of an independent draft model (0.25). S = tokens/cost.

**5.1** Chain: S_chain(α, γ, c) = τ(α,γ)/(1+(c_chain+c_v1)γ). **5.2** Pulsar per-regime: depth d kept while α_regime^d ≥ θ; E03 sweep: best fixed γ* = 2 (1.258) at mixed α = 0.587; adaptive Pulsar 1.388 (+10.3%); always-full trees collapse to 0.588 (−59%) — *fixed pulses are the anti-pattern*, the engine's whole economic case is in M4b+M7. **5.3** Crossover: S > 1 ⇔ τ > cost — measured chain γ=8: 0.754 (AR wins); Pulsar regime map (E11 analytic) gives Pulsar > 1 already at α ≥ 0.164 in the two-regime model — its floor sits far left of chain speculation's because weak contexts are *not paid for*. **5.4** No free lunch: by §3.3, speedup ≤ f(KL(p‖π)); head divergence is the only lever; at published EAGLE-3-class head quality (α ≈ 0.85–0.93), Thm-3 economics predict S ≈ 4–6.5× — consistent with reported 3.5–5.8× (§8.3).

---

## 6. Verification record

**Harness.** `verification/theory_checks.py` (T01–T11; <70s) and `verification/experiments/run_all.py` (E00–E11; ~150s; deterministic seeds; writes `paper/results/results.json` + figures); `cpp/` builds and runs its own acceptance test. All green: **11/11, 21/21, 4/4** at HEAD.

**Testbed.** V=20 order-3 Markov "target" with latent-topic template rows (λ=0.06 uniform smoothing), and an order-2 "draft" (context-blind): order-2 contexts partition into easy (51.5% stationary weight; exact per-context α = 1−TV = 0.907) and hard (α = 0.248); regime is a deterministic function of the *draft's own* context, so M4c can learn it and fixed H cannot. The exact stationary law is computed by a 38-iteration einsum fixed point (residual 8.1·10⁻¹⁶) so acceptance/TV/KL are *table truths*, not estimates.

**Selected results (full table in results.json).**

| id | claim | measured |
|---|---|---|
| T01 | verifier algebra + MC exactness | max TV 0.0114 (noise ~0.006); identity to 1e-12 |
| T02 | τ closed form vs rejection walk | within 5 MC-se, all α grid |
| T03 | optimal block: grid vs golden-section | |ΔH| ≤ 1 on 52 cells; unimodal |
| T04 | α = 1−TV; Pinsker; Bretagnolle–Huber | slack 0.00 (400 pairs) |
| T05 | **water level = global optimum (brute force)** | slack 1.7e-16 chain / 3.3e-16 tree |
| T06 | rotor: mass-preserving mixing; stratification | eig {1,(1−ε)}; var ratio 0.499 vs 0.504 |
| T07 | Gumbel-max = Born; T→0 = argmax | freq TV ≤ MC noise |
| T08 | LIF fire-time ≡ pulse timing | exact on 300 random (λ,θ,I) |
| T09/T10 | χ² goodness-of-fit; adversarial proposals | p_min = 0.025; TV 0.0104 |
| E01 | engine exactness ×11 policies; seq unigram/bigram | TV ≤ 0.0083; p = 0.247/0.016 |
| E02 | measured acceptance = table 1−TV | z = −0.28 / −0.13 |
| E03 | adaptive > best chain > … > full blocks | 1.388 / 1.258 / … / 0.588; oracle 1.423 (97.5%) |
| E04 | M4c calibration (isotonic head) | ECE 7e-4; slope 0.9995; bin gap ≤ 0.007 |
| E05/E10 | θ self-tuning; REINFORCE optimum | 0.0575 vs 0.07; 1.3863 = 1.3863 |
| E06 | fall-through vs renorm collapse | +28.7% speedup |
| E07 | regime drift: adaptive tracks per-phase optimum | worst-phase 0.723 vs best-static 0.720; naive static 0.829 |
| E08 | M6 work bound | 1.582 ≤ 8.844 |
| C++ | port reproduces reference | speedup 1.387; θ 0.0575; TV 0.0016 (250k) |

**Methodology note (tests-of-tests).** V4 of the log: the bigram reference initially failed for the *vanilla AR stream itself* — a `einsum` index bug in the test, not the engine. Rule adopted: every distributional check must pass on the reference generator before it may fail a candidate. The log (docs/verification_log.md) lists ten such cycles, including two theorems restated after brute force rejected the first derivation (V3: θ = d·S, not d·L).

---

## 7. Practitioner specification

**7.1 Module I/O (enforced by `src/pulsar`, mirrored in `cpp/pulsar_engine.hpp`).**

```
M1: (ctx, H_max, B) -> field{tok,par,dep,logq,psi,A,c2,c3}[nN]
M4c: X[nN,6] -> ahat in (0,1]           # features are draft-side ONLY
M4b: theta -> kept = {u: A[u] >= theta}  # one pass, prefix-closed by monotonicity
M4d: kept-tree, pi_u -> x, path          # sequential, T stable in log-space
M3 : (path rows p, pi) -> emitted block  # accept min(1,p/pi); residual norm((p-pi)+)
M6 : work(cycle) <= c_draft*levels + c_tick*ceil(N/B_t) + 1 + c_v1*N
M7 : L_ema (or bandit state) -> theta;  verifier logs -> M4c supervision
```

**7.2 Training M4c.** Run engine in θ=0 (fixed full-block) mode; log (features, min(1,p/q)) for *every built child* including pruned ones (anti-censoring); soft-BCE + PAVA isotonic; 244k samples/5k cycles suffice for the toy (1.9s); in production: one epoch over calibration corpus, LR decay, *keep the isotonic head* — V9's lesson: without it, mid-bin miscalibration persists and θ mis-fires; with it, ECE 5.4e-2 → 7e-4.

**7.3 Tuning M7.** Start θ_mult = 2.5 with EMA 0.05 (E03/E05: within 1% of best fixed on the toy); for non-stationary workloads enable the bandit (E10) or EMA only (E07 shows EMA alone tracks drift within 0.4%).

**7.4 Engine loop in a real stack (vLLM/SGLang-style).** (1) one hidden-state pass → head rows for frontier layer (batched — *breadth is one row*; EAGLE-style feature drafting is a drop-in M1 upgrade); (2) M4c score, water-level expand (top-B per node, cap N ≈ 64–96); (3) tree-attention verification pass over the kept subtree; (4) fall-through descent on CPU/scheduler with the verifier table; (5) commit + bonus. The C++ core implements steps (1)–(5) with arrays and no allocations per node beyond an arena.

**7.5 Numerics & stability.** Log-space policy softmax with delta fallback (V5); residual renorm guard (s ≤ 1e-12 → p); â clamped to [1e-6, 1]; θ floor; integer node ids only.

**7.6 What will break first in production.** (i) cost model constants (c_head, c_v1) — measure, don't assume; §5 formulas are the regression targets; (ii) selection-biased M4c labels when θ rises (pruned nodes stop being logged) — keep an ε-log or log all children of expanded nodes (done in reference); (iii) batch>1: verification rows are shared and c_v1 shrinks — adaptivity still wins, magnitude changes (run T05+E03 sweeps on measured cost profiles); (iv) sampling-time top-p/top-k truncations of p — re-derive Theorem 1 against the *post-truncation* law or disable truncation under Pulsar (recommended).

**7.7 Hyperparameters used by all green runs:** H_max=8, B=4, max_nodes=96, c_draft(level)=0.08, c_v1=0.02, c_tick=0.002, c_chain=0.25, temperature=1.0, θ_floor=2e-3, EMA=0.05, θ_mult=2.5, logistic lr 1.0 / 2500 GD steps / l2 1e-5, PAVA 512 knots.

---

## 8. Discussion

**8.1 Boundary of the metaphor.** Real pulsars teach us *rotation-powered periodic beaming*; Pulsar's load-bearing borrowings are exactly three: rotor-sweep stratification (T06), pulse thresholds as fixed points (T03/T05), and beamed — i.e. *rare, dense, timed* — emissions. The quantum vocabulary is bounded to diagonal density operators and Born-shaped antichain weights; there is no interference term, and none is needed — our theorems are elementary probability with disciplined conditioning. Readers who want literal quantum sampling can bolt a real sampler onto M4d: exactness (C2) tolerates *any* collapse law, including one produced by a quantum oracle.

**8.2 Why the field must be trainable (and why nothing else may be).** The economics (§5) show total gain is monotone in *both* acceptance (head quality, M4c) and threshold calibration (M7); these are the two functions where gradient signal exists (verifier labels are free and abundant). Everything else — verifier, clock, commit — is frozen *because* it must be exactly right; the Exactness Theorem converts "trainable" from a risk into the engine's only tuning surface.

**8.3 Projection at published operating points.** Plug reported acceptance into the verified formulas: Medusa-style heads (α≈0.75, c≈0.02, γ=4): S = (1−0.75⁵)/(0.25·1.08) ≈ **2.63×** (paper reports 2.2–2.8 ✓). EAGLE-3-class (α≈0.88, c≈0.03, γ=8): S ≈ (1−0.88⁹)/(0.12·1.24) ≈ **6.05×** (reported up to 6.5 ✓). Pulsar deltas at these points: water-level adaptivity is worth most where α variance is high (E03: +10% at Var(α) large; ~0% at constant α, where the optimum is flat — we measured this and do not oversell); fall-through over renormalized tree sampling: +29% (E06) *at any α where trees are used*; M6 bound and drift robustness are qualitatively new in the family.

**8.4 Limitations.** (1) Cost-model speedups, not GPU wall-clock, on a Markov testbed — the theorems are exact, the operating constants must be re-measured per stack. (2) V=20 vocabulary; tail-bin calibration ceiling (§4.3) may shrink or grow with real head features. (3) REINFORCE bandit = 1-D action; richer schedulers (per-context γ, tree shapes as actions) are unverified ideas. (4) Sampling-time truncation interaction (§7.6-iv). (5) Energy numbers not measured.

**8.5 Next.** (i) port `cpp/` tree logic to a CUDA tree-attention kernel and re-run the whole harness against *measured* costs; (ii) lattice (non-tree) fields merging shared prefixes with a proper measure (Antichain Lemma generalizes — we conjecture Thm 3 survives with Möbius-corrected A); (iii) M4c on target hidden states (EAGLE-class features) inside the identical water level; (iv) multi-sequence batching as joint superposition with per-sequence θ; (v) adversarial-â robustness studies — Exactness says outputs stay true; the interesting question is how much speed an attacker loses you.

---

## 9. Stage 6 — a bounded Transformers backend study

The falsifiable empirical hypothesis is conditional: under a specified inference backend and measured cost profile, a draft-confidence scheduler for exact bounded trees improves held-out emitted tokens/s over a tuning-selected fixed tree only when its paired 95% confidence interval has a positive lower bound. The Stage 6 adapter uses Transformers 4.46.3 with `distilgpt2` as target and `sshleifer/tiny-gpt2` as draft, a shared 50,257-token GPT-2 tokenizer, temperature-1 softmax, and a fixed 24-token output horizon. EOS is treated as an ordinary token. Prompt prefill, proposal/tree work, target evaluation, exact verification, KV-cache state creation/commit, and output bookkeeping are timed; model loading is excluded.

The actual target and draft rows passed the pointwise Stage 1 identity over all vocabulary tokens on three held-out contexts. Maximum absolute identity error was 1.08·10⁻¹⁹; grouped empirical TV (32 largest target probabilities plus `OTHER`) was 0.0116–0.0139 over 20,000 verifier samples per context. The target/draft TV on those contexts was high (0.892–0.967), documenting a poor match for this draft pair.

| Matched method | Tokens/s | Paired comparison |
|---|---:|---|
| Autoregressive target | 40.62 | Baseline |
| Tuned chain speculation (γ=2) | 15.71 | +43.00% vs fixed tree [34.87%, 50.79%]; −61.33% vs AR [−63.47%, −59.40%] |
| Tuned fixed tree (depth 2, branch 1) | 10.98 | −72.96% vs AR [−74.82%, −70.71%] |
| Draft-confidence tree | 11.17 | +1.66% vs fixed tree [−6.71%, +10.01%]; −72.51% vs AR [−74.49%, −70.63%] |

Tuning used two disjoint prompts; evaluation used three held-out prompts with three paired seeds each. The chain grid was γ∈{2,4,6} and the tree grid was {(2,1),(3,1),(3,2),(4,2)}. All three confidence bins selected the same tuning-best fixed action, so the scheduler did not adapt on the tuning traces. Its throughput claim is **not established**: the positive point estimate is compatible with a negative effect, and the 95% lower bound is below zero. Chain speculation beat this fixed-tree implementation, but both speculative methods were slower than autoregression under this profile.

This is a compact CPU-only study with three held-out prompt clusters and a level-batched full-context tree evaluator, not a fused tree-attention kernel. Bootstrap intervals are descriptive within-prompt uncertainty, not evidence of general workload superiority. Stage 0–5 mathematical/toy results remain separate; Stage 5 remains closed with its throughput gate unmet. Pulsar/rotor/quantum terminology is explanatory framing unless measured as part of the algorithm. Complete implementation, protocol, CSV, JSON, and manifest are in `algorithm_v6_backend.md` and `paper/results/stage6_transformers/`.

---

## 10. Conclusion

Pulsar specifies exact verification and scheduling contracts, but a correctness guarantee is not a throughput guarantee. The frozen theory and toy-workload results validate the stated probability and cost-model claims; they do not establish neural-model serving speed. In the Stage 6 CPU integration, exactness passed, while the confidence scheduler failed to establish a gain over the fixed tree under the predeclared paired-interval criterion. Autoregression was fastest; tuned chain speculation beat the fixed tree but both were slower than autoregression. The defensible conclusion is a verified sampling adapter and a bounded negative/inconclusive scheduler result—not a demonstrated general-purpose acceleration. The next falsifiable step is to measure a better-matched backend/model cost profile without reusing held-out prompts for policy selection.

---

## Appendix A — Proofs

**A.1 Theorem 1 (exactness).** §3.2. Generality checklist: π need not have full support where p does *not* demand it (residual carries the deficit); π may depend on all prior emissions, on â, on θ, adversarially — but not on the fresh acceptance coins. Sequential chains compose by induction on accepted prefix length (the context fed to each row is the emitted sequence, verified per-position). Bonus token ~ p(· | extended prefix) is sampled from the same pass's row — no interaction with the coins.

**A.2 Corollary 1a (yield).** Let X = #accepted before stop; P(X ≥ k) = Π_{j≤k} α_j by independence of coins given the path; E[X+1] = 1 + Σ_{k≥1} Π α_j; uniform α gives τ = Σ_{d=0}^γ α^d = (1−α^{γ+1})/(1−α). ∎


**A.3 Theorem 3 (water level).** Chain: L(H) = (1+cH)/(1+Σ_{d≤H} A_d). Δ(H) := L(H+1)−L(H) has sign of c·T(H) − C(H)·A_{H+1}, i.e. add iff A_{H+1} ≥ c·T(H)/C(H) = c/L⁻¹… While adds are accepted, T/C increases (ΔL<0 ⇔ (T/C)′>0 on the discrete path) and A decreases ⇒ the difference of thresholds flips sign once ⇒ greedy hit = global min; θ := c·T(H\*)/C(H\*) is the exhibited self-consistent level. Tree: order nodes by A (parents precede — A decreases along paths), same single-crossing over the sequence of prefix-closed sets; marginal value of node w given current (T, C): ΔL < 0 ⇔ A_w ≥ c·T/C — the greedy scan *is* the water level. Slack vs exhaustive: ≤ 3.4·10⁻¹⁶ (T05, 600 random instances, chains of 10 and 6-node trees). ∎

**A.4 T03 closed form.** Golden-section on S(·; α, c) over ℝ⁺; integer argmax within 1 by unimodality (checked per cell). Continuous FOC: α^H ln α · (1+cH) = (1−α)(τ−1)(c) style transcendental — solved numerically, not claimed in closed form (Lambert-W applies only for large H). ∎

**A.5 Stratification (T06).** Sampling n of N without replacement: Var(x̄) = (σ²/n)·(N−n)/(N−1); round-robin over cycles of length ⌈N/B_t⌉ realizes exactly-without-replacement for n ≤ N. Predicted/measured ratio 0.504/0.499. Circulant M_ε: row-stochastic ⇒ Σ preserved; on 1⊥ it scales by (1−ε). ∎

## Appendix B — Interface contracts (as implemented)

`M1(K candidates, logits) → M2; M2(field) → M4a; M4a(tick reads) → M4c; M4c(conf field) → M4b (θ, block) → M4d (path) → M5 → M3 → M5; M7 ← {verifier stats, conf history}; M7 → thresholds; M6(tick) → M4b, M5` — verbatim from the design spec, realized by `src/pulsar/engine.py::PulsarEngine.step` with M1+M4b interleaved at the water level (so cost accounting is honest) and M4d+M3 fused on the collapsed path (so the policy the verifier tests is the policy that sampled).

## Appendix C — Reproduction

```
python3 verification/theory_checks.py            # 11/11
python3 verification/experiments/run_all.py     # 21/21, writes results.json + figures
cd cpp && make && ./pulsar_test inputs/model.bin # 4/4 (model.bin from run_all)
python3 verification/experiments/test_stage6_transformers.py
python3 verification/experiments/run_stage6_transformers.py
```
Papers: `paper/PULSAR.md` (this file), `paper/latex/pulsar.tex` (+ compiled PDF). Figures: `paper/figures/*.png` referenced throughout §4–§6 (E03 sweep, E04 reliability, E05 θ trajectory, E08 step scatter, E10 bandit, E11 regime map).

*Figures: `paper/figures/fig_e03_sweep.png` (block policy sweeps), `fig_e04_reliability.png` (reliability diagram), `fig_e05_theta.png` (θ fixed point), `fig_e08_scatter.png` (cost/emit scatter), `fig_e10_reinforce.png` (bandit), `fig_e11_regime.png` (regime map).*
