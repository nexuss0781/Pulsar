# Pulsar — compact mathematics per module

Notation: vocabulary V = |𝒱|; target (frozen) next-token law
p(· | t_{≤n}); draft-side candidate law q(· | s_n) from a tiny trainable head
(s_n = the context the draft can see). One decode cycle = one *pulse*.

---

## M1 — Candidate Generator (frozen backend, trainable heads allowed)

Top-B expansion: children(u) = argtop_B q(· | s(u)), each child carrying
(x_u, log q(x_u | s_u)). Alternatives (any of): Medusa heads, EAGLE-style
feature-drafting head, n-gram/Prompt-Lookup retrieval. Requirement: emits
(x, log q(x)) pairs. Swappable by contract.

Covering guarantee (heuristic): mass coverage κ_u = Σ_{kept} q(x) → tail
fall-through (below) covers the rest exactly.

## M2 — Superposition Field

Prefix tree T rooted at u=∅. Two products per node:

    amplitude    ψ(u) = Π_{v ≤ u} q(x_v | s(v))          (draft path law)
    yield score  A(u) = Π_{v ≤ u} q(x_v) · â(v)          (M4b water level)

where â(v) estimates the acceptance probability min(1, p(x_v)/q(x_v)).
Antichain = mutually exclusive futures (leaves of a cut). Born rule on any
antichain C:  P(u) = ψ(u)/Σ_{w∈C} ψ(w).

**Quantum language, precisely bounded:** the state is a *diagonal* density
operator ρ = Σ_u ψ(u)² |u⟩⟨u| over branch basis; "collapse" = conditional
resampling; there is NO interference or entanglement — the structure that
matters is the branching (non-autoregressive) domain and the Born-shaped
weights. All theorems below are elementary probability; nothing needs
quantum mechanics.

## M3 — Verifier (frozen; safety net & quality floor)

Position-wise rejection-sampling step, tested against the EFFECTIVE collapse
policy π (raw q for sampled descent, per-node temperature-shaped q, or a
delta for greedy/adversarial modes):

    accept x ~ π  w.p.  a(x) = min(1, p(x)/π(x))
    on reject: y ~ r = norm( max(p − π, 0) )

**Theorem 1 (Exactness, proposal-agnostic).** For ANY π, the emitted token
law is exactly p. Proof: p(x) = min(π,p)(x) + (1−Σπ∧p)·r(x) pointwise;
accepted branch emits π∧p, rejected branch emits the residual; sum = p. ∎

**Corollary 1a.** Block yield: E[tokens | block] = 1 + Σ_{u ∈ block path}
Π_v a_v(π), the Leviathan τ formula for the uniform case:
τ(α, γ) = (1 − α^{γ+1})/(1 − α).

**Corollary 1b.** Per-node acceptance α(π) = Σ π min(1, p/π) = 1 − TV(p, π).
Verified vs exact table: z = −0.28 / −0.13 (E02).

## M4a — Rotor

Deterministic round-robin readout of candidates under a fixed per-tick batch
budget B_t. Over one cycle every node read exactly once ⇒ the readout is
*stratified sampling without replacement*:

    Var(mean of n reads) = σ²_pop/n · (N − n)/(N − 1)  <  σ²_pop/n (i.i.d.)

(measured ratio 0.499 vs predicted 0.504, T06). Circulant sibling mixing
M_ε = (1−ε)I + (ε/n)11ᵀ is row-stochastic: Σ_u ψ(u) preserved; spectrum
{1, (1−ε)^{n−1}}: contraction on the disagreement subspace only.

## M4c — Confidence Field (trainable; the sub-symbolic core)

â(x | s) = iso_θ(σ(w·φ(x,s) + b)), φ = draft-only features
[1, log q(x), −H(q), log q(x)−log q(top1), log q(top1), rank/(B−1)].

Labels = verifier-observed min(1, p(x)/q(x)); soft-BCE training; PAVA
isotonic head. **Decision-granular calibration identity** (what optimality
of M4b requires): E[a | â] ≈ â (bins: max gap 0.007, ECE 0.0007, E04).
Bayes-optimal limit with draft-only features documented (tail bins).

LIF equivalence (SNN reading): leaky accumulator
V_t = (1−λ)V_{t−1} + λ·e_t fires at t* = ceil(ln(1 − θ/I)/ln(1−λ)) − 1 for
constant drive I > θ — verified exact vs simulation, T08. Membrane threshold
≡ pulse threshold; "neurons as confidence integrators" is a theorem, not a
metaphor, under this mapping.

## M4b — Pulse Generator: water-level blocks with self-consistent θ

Block = { u : A(u) ≥ θ }. With per-node cost d (target-pass units), step
cost C = 1 + d|block|, yield T = 1 + Σ_{u∈block} A_true(u), latency L = C/T.

**Theorem 2 (Water-level optimality).** If A decreases along paths, the
greedy "add node while A_u ≥ d·T/C evaluated on the current set" maximizes
T/C globally (the marginal condition flips sign at most once since T/C is
increasing during accepted additions and A decreases). Equivalent
self-consistent threshold: θ* = d·S*, S* = 1/L*. Verified vs exhaustive
prefix-closed enumeration, slack 1.7e-16 / 3.3e-16 (T05, chain + tree).

Uniform-α chain special case: speedup S(H) = τ(α,H)/(1 + cH); optimum is
the integer nearest the golden-section root of
τ′(H)(1+cH) = c·τ(H) (T03: argmax_int within 1 of continuous root, all
(alpha,c) cells; unimodality asserted per cell).

## M4d — Collapse Operator

Sequential per-node policy at node u over ALL v: π_u = q(·|s(u)) (or
softmax(log q / T), or δ_top1 / δ_topB for greedy/adversarial). Continue
into precomputed block only if x ∈ kept children, else FALL THROUGH: token
emitted exactly; depth stops. Temperature therefore modulates *reach and
throughput*, never output law (Theorem 1). Gumbel-max identity: argmax
(log ψ/T + G) ≡ Born(ψ^{1/T}) (T07, freq TV ≤ MC noise).

## M5 — Commit / Emit

tokens ← (accepted prefix) ⊕ (residual | bonus), where bonus ~ p(· | extended
context) is FREE (the verification pass computed that row) whenever the
full-depth path accepted. KV/hidden-state cache append per token.

## M6 — Pulse Clock

Per-cycle work = c_draft·levels + c_tick·⌈N/B_t⌉ + (1 + c_v1·N) target-pass
units with N ≤ budget ⇒ deterministic per-step upper bound (measured max
1.582 ≤ 8.844, E08). Gives admission control/SLAs the AR loop cannot offer.

## M7 — Calibrator (trainable; RL lives here)

Learns θ (and the M4c net). Three coupled estimators:
1. EMA fixed point θ_t = θ_mult · d / L̂_t — self-tunes to optimum (E05:
   0.0575 vs grid 0.07, speedup within 0.7% of best fixed).
2. REINFORCE bandit on discrete θ grid, reward = tokens/cost — recovers the
   engine-measured optimum exactly (E10: 1.3863 = 1.3863).
3. Isotonic/PAVA calibration head for M4c (scoring-rule driven).
Reward = speedup; rejection penalties enter through the verifier-labeled
calibration (no explicit penalty needed: miscalibration shows up as yield
loss at the water level).

## Global bounds (no free lunch)

α = 1 − TV(p, q_head) ⇒ via Pinsker + Bretagnolle–Huber:

    α ≥ 1 − √(KL/2),      α ≥ 1 − √(1 − e^{−KL})

so expected speedup is an increasing function of *draft-side divergence*:
the whole engine's job is to buy down TV(p, π) with the cheapest possible
trainable head — every module optimizes one side of this inequality.
(Verified: T04, 400 random pairs, zero slack violations.)
