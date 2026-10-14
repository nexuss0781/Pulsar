# Pulsar — verification log (what was wrong, what caught it, what shipped)

Kept because a design claim you cannot reproduce is a rumor. Each entry:
**failure → detector → fix**. Reproduce with
`python3 verification/theory_checks.py && python3 verification/experiments/run_all.py`.

### V1 — Verifier row off-by-one (correctness bug, severity: critical)
* Symptom: engine first-token TV vs target row = **0.71** (must be ≈0.01).
* Cause: verification evaluated `p(.|ctx incl. token)` instead of the
  parent-context row: `P[c3[u]]` should be `P[c3[parent[u]]]`.
* Detector: fixed-prompt exactness microtest (25–30k steps, compare histogram
  to the *exactly computable* target row).
* Fix: parent-row semantics documented in `engine.step`; test now passes at
  0.0083 (MC noise floor).

### V2 — "Exactness for any collapse" was false for *renormalized* tree sampling
* Symptom: adversarial argmin-leaf collapse TV = 0.22; T=1 leaf-Born was fine
  only because verifier tested renormalized q on kept children — but the
  *raw* q test (as used by classic tree methods) breaks under pruning, and
  renormalized proposals lose acceptance (TV increases).
* Fix (design change, now a headline result): **fall-through collapse** —
  the descent always draws from the raw draft row; the block only decides
  where verification continuation is precomputed. Per-node acceptance equals
  the chain optimum 1−TV(p,q), and *any* policy (delta, adversarial, hot/cold
  temperature) stays exact because the verifier tests the policy itself.
  Ablation E03/E06: fall-through 1.388 vs renormalized 1.079 (+29%).

### V3 — θ used latency where the algebra demands speedup
* Symptom: water-level optimality failed brute force by 3–4e-2 (T05); engine
  over-built blocks in hard contexts.
* Cause: marginal analysis of L(H)=C(H)/T(H) gives include-while
  A_d ≥ c·T/C = c·**S** (speedup), not c·L.
* Fix: `theta = c_v1 * theta_mult / L_ema` (M7-A). T05 now passes with slack
  1.7e-16; engine adaptive speedup within 0.7% of best fixed θ.

### V4 — Test-of-test: the reference bigram law was wrong, not the engine
* Symptom: sequence-level bigram χ² p = 0 for BOTH pulsar and vanilla AR.
* Cause: `einsum('abc,abcd->bd')` indexed the pair by the middle slot;
  correct is `->cd` (last state slot = t).
* Lesson kept in harness: every distributional test must first pass on the
  *reference sampler* itself.

### V5 — Temperature underflow killed the cold limit
* Symptom: crash (rng.choice with zero-sum p) at T = 0.05; earlier leaf-level
  temperature mismatch.
* Fix: per-node log-space softmax with delta fallback (exact, since the
  verifier tests the realized policy); collapse semantics unified: policy is
  *per node*, never leaf-level reweighting.

### V6 — LIF firing-time discretization off-by-one
* Symptom: T08 mismatch on random (leak, θ, I).
* Cause: post-update indexing V_t = I(1−(1−λ)^{t+1}).
* Fix: `t* = ceil(ln(1−θ/I)/ln(1−λ)) − 1`; 300 random instances exact.

### V7 — The chain baseline was illegally optimistic
* Symptom: "chain" beat Pulsar 1.65 vs 1.17.
* Cause: baseline drafted sequentially *with early stop at first rejection*,
  avoiding the blind-γ draft cost real chain speculation must pay before the
  single verification pass.
* Fix: `_step_chain` drafts γ tokens blind, verifies in one pass. Baselines
  are only baselines when they pay their real costs.

### V8 — Draft logit ≠ acceptance probability (motivation for M4c)
* Symptom: raw-â water level over-pruned everything (mean block 2.1); oracle
  mis-served due to a second row-shift bug in oracle_conf.
* Understanding: q(top-1) underestimates E[min(1,p/q)] = 1−TV; the trainable
  field closes it. Trained adaptive beats raw-logit baseline 1.312 → 1.388
  and reaches 97.5% of the oracle-â ceiling (1.423).

### V9 — Calibration capacity vs information ceiling
* Symptom: logistic M4c bin gaps up to 0.30 in mid-bins; interactions
  destabilized GD (ECE 0.054 → 0.104).
* Fix: **isotonic (PAVA) head on the logits** — max bin gap 0.0070, median
  0.0006, ECE 0.0007, slope 0.9995; residual tail-bin error documented as the
  draft-only-features ceiling (asserted at that granularity in E04).

### V10 — C++ port: TV units
* Symptom: C++ test TV=410.19 with sane z-scores — histogram counts vs
  probabilities. Fixed; TV 0.00164 at 250k samples, PASS.

### Cross-implementation agreement (a verification in its own right)
The independently written C++ engine reproduces the Python engine's
cost-model speedup (1.387 vs 1.388), θ fixed point (0.0575 both), and
mean emit/step (1.654). Two implementations, one mathematics.

### V11 — frozen v1 probability contract
* Normative definitions for target `p`, draft `q`, effective collapse law `π`,
  residual `max(p−π,0)`, candidate-tree fall-through, and probability path
  weights are frozen in `algorithm_v1.md`; optional modules are deferred.
* The C++17 core samples and verifies with the same explicit `π`. With
  `π != q`, the deterministic 300k-sample test measured output TV **0.00018**
  (limit 0.008); accepted off-tree tokens were emitted unchanged and stopped
  pulse extension. The `p=π` case never rejected; zero residual is explicit.
* `make clean && make CXXFLAGS='-O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror'
  test` passed both core and legacy engine suites. The v1 core test also passed
  AddressSanitizer and UndefinedBehaviorSanitizer.


### V12 — Stage 1 generic one-token verifier verification
* API: `sample_verified_token(target, effective_pi, rng)` now accepts any standard C++ UniformRandomBitGenerator by reference; it remains an isolated one-token component and is not connected to a language model. `algorithm_v1.md` documents array lifetime, weight/vocabulary assumptions, zero support, equality behavior, and floating-point limits.
* Algebra detector: every fixture checks, token by token, `π(x)·a(x) + Z·r(x) = p(x)` with `a(x)=min(1,p(x)/π(x))` where `π(x)>0`; the deterministic fixtures include proportional unnormalized weights, `p=π`, `π(x)=0<p(x)`, concentrated and nearly disjoint support, and a positive residual of approximately `1e-12`. A further 2,000 seeded randomized finite laws pass the same identity.
* Sampling detector: 20 fixtures (including every row of a 3-state Markov transition model) × 40,000 draws each passed a simultaneous 99% Hoeffding/union-bound envelope of ±0.0111206 per token. This is secondary evidence; the algebra identity remains the correctness gate. The existing `π != q`/tree fall-through test measured output TV 0.00018 and confirms unchanged accepted tokens and stop-on-rejection behavior.
* Edge contract: exact zero residual throws if requested and is never requested by the equal-law sampling path; no epsilon cutoff or fallback law is used. Small positive residuals are retained within working floating-point precision.
* Validation: strict `-O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror make test` passed both core and existing C++ engine suites; expanded core tests also passed AddressSanitizer and UndefinedBehaviorSanitizer. No model integration was added.

### V13 — Stage 2 speculative chain
* Added an isolated C++17 block sampler layered on v1. It drafts all `γ` proposals sequentially from context-dependent `q` rows, fetches aligned target rows for the proposed contexts, and verifies left-to-right using the same `q` law in the acceptance ratio and residual. A rejection appends one residual replacement and ends verification; a cached target row may provide the bonus only after all proposals pass.
* Kept v1's general `p`/`π` API unchanged and added a plain fixed-token-count autoregressive reference sampler using the same target rows, token IDs, and categorical helper.
* Stage 2 tests force rejection at every block position, cover missing/present bonus rows and context-aware row lookup, and compare conditional transition laws for speculative and AR sequences generated with independent RNG seeds.
* Validation: strict `-O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror make test` passed v1 core, Stage 2, and legacy engine tests. For the independent-seed Markov test, conditional TV(spec, p) was 0.0019–0.0035, TV(AR, p) was 0.0008–0.0035, and TV(spec, AR) was 0.0015–0.0036 across the three contexts. The legacy engine's 300k-token independent-run unigram TV was 0.0054; its former 0.004 cutoff reflected the old shared-stream comparison, so it is now a 0.007 Monte Carlo gate, while Stage 2 is primarily gated by context-conditional tests.
* Foundation regression check: `python3 verification/theory_checks.py` passed all 11/11 existing T01–T11 checks.


### V14 — Stage 3 bounded candidate tree with fall-through
* Added `cpp/pulsar_tree_v3.hpp` and an independent Stage 3 test target. The builder stores each complete context-specific `q` row and retains at most fixed `B` positive-q children at each nonterminal position, with deterministic token-ID tie breaking. `max_depth` is the maximum number of verifier positions in one pulse, including the root.
* Sampling defaults to the full stored `q` as `π`, with an explicit effective-`π` callback available when the collapse law differs. The existing v1 verifier samples and tests that same `π`; a rejection emits the residual and stops, while an accepted token is emitted unchanged and descends only through its exact retained child. Accepted off-tree tokens fall through unchanged and stop. Root, child index, parent, depth, retained-set and context checks prevent stale/non-child continuation. v1 and Stage 2 files/contracts remain unchanged.
* Tests cover the retained-tree bound and full q rows, direct full-`q` sampling with `B=1` (measured root `P(0)=0.5982` vs `0.6`), accepted off-tree eligibility, residual rejection/stop, stale roots and children, and deterministic `πa + Zr = p` identities (including zero support and equality).
* Independent-seed 3-state conditional-transition checks use `B=1`, depth 4, and `π != q`: tree-vs-target conditional TV was 0.00013, 0.00345, and 0.00104; tree-vs-AR TV was 0.00182, 0.00528, and 0.00134 across states 0–2. These are statistical checks supporting—not replacing—the deterministic verifier identities.
* Validation: strict `-O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror make test` passed v1, Stage 2, Stage 3, and legacy C++ engine suites; the Stage 3 test passed AddressSanitizer and UndefinedBehaviorSanitizer; `python3 verification/theory_checks.py` passed all 11/11 checks (T01–T11).
* Scope note: this is a standalone C++ tree sampler and verification layer; no model/runtime integration or performance claim was added.


### V15 — Stage 4 measured cost before pulse-rule optimization
* Added a benchmark-only fixed policy around the unchanged Stage 3 tree sampler. The grid sweeps max depth {1,2,4}, branch factor {1,2,4}, and minimum retained level mass {0,0.20,0.40}. The threshold retains complete levels based on cumulative normalized q-path mass; selection/build work is timed, not hidden.
* The C++ harness uses the existing `inputs/model.bin` synthetic Markov fixture and paired AR/tree runs for three fixed prompts, batch size 1, temperature-1 categorical setting, same token limit and per-prompt/repetition seed. Raw rows include work counts, callback and wall times, cycle percentiles, emitted/accepted counts, tree shape/payload estimates, and output checksums. A Python aggregator validates the paired-run contract and creates cost/yield and throughput plots with raw run summaries and metadata.
* **Scope boundary:** draft/target callbacks are static table lookups and copies; these are actual CPU wall-clock measurements of the fixture, not real model forward time. Tree memory is vector-payload capacity only. KV-cache measurement is marked unsupported/not applicable. Sampler time minus target callback is an inclusive verification/sampling proxy, not an isolated verifier-kernel measurement. No scheduler training or global-optimality claim is introduced.
* Reproduce via `algorithm_v4_measurements.md`. Stage 0–3 implementations and contracts remain unchanged; the new threshold selector is benchmark-only and tested independently.

### V16 — Stage 6 Transformers adapter and bounded exactness/throughput test
* Added a concrete Hugging Face Transformers 4.46.3 adapter for `distilgpt2` target and `sshleifer/tiny-gpt2` draft. It exposes actual target/draft rows, cached chain verification, level-batched tree target rows, draft tree expansion, per-prefix KV cache reuse, and committed-token cache updates. The target law is the temperature-1 model softmax, without top-k/top-p or other logits processors; EOS is ordinary under the fixed horizon.
* Adapter checks passed 5/5: cache append parity, batched branch parity, chain prefix-row/cache crop parity, tree row/cache parity (7-node fixture), and the actual-row pointwise identity. The final three-context audit passed over all 50,257 vocabulary items; max identity error `1.08e-19`. Grouped empirical TV was `0.0116–0.0139` across 20,000 draws/context.
* Matched held-out CPU runs (4 threads, temperature 1, 24 tokens, 3 prompts × 3 repeats) measured AR 40.62 tokens/s, tuned chain (`γ=2`) 15.71, fixed tree (`2,1`) 10.98, and draft-confidence tree 11.17. Chain beat fixed tree by `+43.00%` (95% CI `[+34.87%, +50.79%]`) but lost to AR. The confidence scheduler selected the same fixed action in all bins; its `+1.66%` point estimate versus fixed tree had 95% CI `[-6.71%, +10.01%]`, so its throughput claim is **NOT ESTABLISHED**.
* Scope: one CPU/model pair, short context, three test prompts, and a non-fused level-batched tree implementation; no general GPU/serving claim. Stage 0–5 contracts and Stage 5 closeout/results remain frozen. Full protocol and raw evidence: `algorithm_v6_backend.md` and `paper/results/stage6_transformers/`.


### V17 — frozen regression after Stage 6 integration
* In a temporary project copy (because `run_all.py` writes result/figure outputs), the theory harness passed **11/11**, the end-to-end harness passed **21/21**, the Stage 5 Python tests passed **12/12**, and Stage 5 native exact-path parity passed **1,728/1,728**. The strict C++ `make test` suite completed with `-Wall -Wextra -Wpedantic -Werror`; v1 core, v2 chain, v3 tree, Stage 4 selector, and model-fixture checks passed.
* SciPy 1.17 was supplied only through a temporary test dependency directory; no project or system dependency configuration was changed for the regression.
* SHA-256 checks confirmed the original Stage 0–4 output files and Stage 5 report/results/manifest remained byte-identical. This validates frozen-stage regressions only; Stage 5 remains **completed with throughput gate unmet**, and its pooled calibration result remains separate.
