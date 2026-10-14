# Pulsar Stage 6 — one real backend, exact tree verification, and matched timing

Stage 6 integrates **Hugging Face Transformers 4.46.x** through the small GPT-2-family target/draft pair `distilgpt2` / `sshleifer/tiny-gpt2`. It adds no changes to the frozen Stage 0–4 contracts and does not reopen Stage 5. Stage 5 remains completed with its throughput gate unmet; pooled calibration improvement remains a separate finding.

## Narrow claim and falsification rule

The claim under test is deliberately conditional:

> On the reported CPU, Transformers/runtime versions, target/draft checkpoints, temperature-1 decoding law, prompt distribution, and fixed token horizon, a scheduler using the draft root's maximum-token probability to choose among bounded exact trees increases held-out emitted tokens/s over the tuning-selected fixed tree policy while preserving the target categorical law.

The claim is **supported only** if (a) the deterministic pointwise verifier identity passes on the actual model rows, and (b) the scheduler-vs-fixed-tree paired 95% confidence interval has a strictly positive lower bound on held-out episodes. Otherwise throughput is **not established**. A result on this setup is not a claim about GPU serving, batching across requests, long contexts, or other model pairs. Confidence is `max_x q(x|context)`, not a calibrated estimate of verifier acceptance.

## Backend adapter contract

Implementation: `src/pulsar/transformers_adapter.py`.

| Adapter operation | Contract |
|---|---|
| `row(logits)` | Returns the full normalized law `softmax(logits / temperature)` as float64. No top-k/top-p, penalties, EOS filtering, or other logits processors are used. |
| `prefill(context)` / `prefill_many(contexts)` | Consumes actual token IDs, returns the model KV cache after the context and the logits predicting its next token. `prefill_many` batches equal-length full contexts and splits per-context caches. |
| `append(state, token)` | Consumes one committed token, returning a new context, KV cache, and next-token logits. |
| `append_many(states, tokens)` | Batches one-token extensions of same-length cache states; used for draft-tree expansion. |
| `verify_chain(root, proposals)` | Uses one target forward over the proposal block, exposes the root/prefix target rows and crops cached target KV states to every proposal prefix for exact rejection/commit. |
| `verified_sample(p,q,rng)` | Draws from `q`, then uses the frozen Stage 1 acceptance and positive residual. |
| `verify_proposal(p,q,proposal,rng)` | Verifies the already-drawn chain proposal; draft sampling and verifier use the same `q`. |

A tree node stores the complete actual draft row `q`, context, depth, and draft cache. It retains at most `B` positive-q children by descending probability then token-ID order, exactly as Stage 3. Target rows/cache states for each node are obtained by full-context target forwards batched by tree level. Sampling draws from the full `q`; acceptance uses that same `q`; rejected proposals use `max(p-q,0)`. An accepted off-tree token remains unchanged and stops pulse continuation. For the next pulse, accepted retained branches reuse their per-node target/draft caches; residual and accepted off-tree tokens append to the current cache. The harness reports target/draft forward-call and input-token counts so the cost of level-batched tree evaluation remains visible.

This is a Transformers cache adapter, **not** a fused tree-attention kernel. Tree rows are produced by actual target-model forwards over full candidate contexts; the benchmark times that work and cache production/commit. No theoretical KV-cache savings are assumed.

## Decoding law and exactness validation

For context `c`, the measured target law is the actual target model's processed next-token distribution

`p(x|c) = softmax(logits_target(c) / T)[x]`, with `T=1.0` in the reported benchmark.

The draft law is defined analogously from the actual draft-model logits. The chain/tree verifier is the frozen `π=q` case of Stage 1. EOS is treated as an ordinary token; all paired runs use the same fixed number of new tokens. This avoids method-specific early-stop horizons and fully specifies the sampling law.

The experiment checks the pointwise identity over the **entire 50,257-token vocabulary** on three held-out prompt contexts:

`q(x)·min(1,p(x)/q(x)) + Z·r(x) = p(x)`, where `Z=Σ max(p-q,0)` and `r=max(p-q,0)/Z` when `Z>0`.

It also draws verifier samples and compares grouped empirical frequencies (32 highest-p tokens plus `OTHER`) against `p`. The statistical check supplements, but does not replace, the deterministic identity. `test_stage6_transformers.py` separately checks cache-append parity, batched branch parity, speculative prefix rows/cache crops, tree target rows/caches, and the actual-row identity.

## Matched baselines and selection protocol

The target/draft pair, tokenizer, temperature, device/thread count, prompt/seed episode, and output-token horizon are shared by all methods. Model loading/download is excluded; prompt prefill, draft work, target verification, cache updates, exact sampling, and output bookkeeping are included. Batch size is one request; target tree rows are batched by level internally.

1. **Autoregressive target:** one target categorical draw per output token, using the same cached target backend.
2. **Tuned chain speculation:** conventional left-to-right exact speculative decoding with block lengths `γ ∈ {2,4,6}`. `γ` is selected on tuning prompts only; target block verification uses one forward over the proposal tokens, cached target rows for each prefix, standard rejection/residual replacement, and an available target bonus row after an all-accepted block.
3. **Tuned fixed tree:** Stage 3 fall-through tree policies on the grid `(depth, branch) ∈ {(2,1),(3,1),(3,2),(4,2)}`. The best aggregate tuning tokens/s selects the held-out fixed comparator.
4. **Draft-confidence tree:** on tuning-only fixed-action pulse traces, split root max-q confidence into three quantile bins and select the action with the highest measured emitted-tokens/time utility in each bin. Empty bins fall back to the tuned fixed action. No held-out labels or test timings choose thresholds/actions.

Tuning prompts (`The history of science is`; `In a small village,`) are disjoint from three held-out test prompts. Test methods are run on paired prompts and seeds with randomized method order. Throughput intervals use paired bootstrap resampling within prompt clusters; prompt is the independent cluster, not each generated token. Per-episode evidence and backend work counters are preserved in CSV.

This is a compact exploratory test (3 held-out prompt clusters), so bootstrap intervals are a descriptive within-prompt uncertainty measure and are not population-wide evidence.

## Reproduction

Requirements used for this run: Python 3.12, PyTorch 2.7.1 CPU, Transformers 4.46.3, NumPy, and access to the two public Hugging Face checkpoints. Install Transformers if needed:

```bash
python3 -m pip install 'transformers==4.46.3' 'safetensors>=0.4.1'
```

Run adapter checks and the full benchmark from the project root:

```bash
python3 verification/experiments/test_stage6_transformers.py
python3 verification/experiments/run_stage6_transformers.py
```

Default protocol: 24 output tokens, 2 tuning repeats per prompt, 3 held-out repeats per prompt, 20,000 actual-row verifier draws per context, and 2,000 paired bootstrap replicates. `--tokens`, `--tune-repeats`, `--eval-repeats`, `--audit-draws`, `--bootstrap-samples`, `--threads`, `--device`, and `--output-dir` are configurable. Outputs are isolated in `paper/results/stage6_transformers/`.

## Interpretation boundaries

- A positive scheduler result would establish only this measured CPU/configuration result, not universal tree superiority.
- A negative/inconclusive result is a valid outcome; do not tune on the held-out set or report a positive point estimate as a pass.
- The benchmark is not matched to Stage 4 or Stage 5 wall times; those use different fixtures/runners.
- No inference-quality or language-model capability claim is made for the deliberately small draft checkpoint.
- “Pulsar,” rotor/beam/pulse, Born-rule, and quantum terms are explanatory framing unless independently represented in and measured as part of the implementation. The Stage 6 algorithm is exact probability sampling plus bounded tree precomputation.

## Measured closeout

Run `stage6-20261013T142914Z-seed60613` (PyTorch 2.7.1 CPU, Transformers 4.46.3, 4 CPU threads) completed the declared protocol. The actual-row identity passed on all three held-out contexts, with maximum full-vocabulary pointwise error `1.08e-19`; grouped top-32-plus-OTHER empirical TV was `0.0116–0.0139` over 20,000 draws per context.

| Method | Held-out tokens/s | Paired speedup evidence |
|---|---:|---|
| Autoregressive target | 40.62 | — |
| Tuned chain speculation (`γ=2`) | 15.71 | +43.00% vs fixed tree [34.87%, 50.79%]; −61.33% vs AR [−63.47%, −59.40%] |
| Tuned fixed tree (`depth=2, branch=1`) | 10.98 | −72.96% vs AR [−74.82%, −70.71%] |
| Draft-confidence tree | 11.17 | +1.66% vs fixed tree [−6.71%, +10.01%]; −72.51% vs AR [−74.49%, −70.63%] |

The confidence map chose the tuned fixed-tree action in all three bins. Thus the scheduler did not adapt on these tuning traces, and its held-out point estimate did not meet the predeclared positive-lower-bound criterion. The scheduler throughput claim is **NOT ESTABLISHED**. Chain speculation was faster than the fixed-tree implementation in this matched harness, but both were slower than autoregression. The measured target–draft TV was `0.892–0.967` on the audited contexts; this documents a poor match for this particular draft, but the benchmark does not identify it as the sole cause of the observed speed ordering.

Scope is limited to three held-out prompt clusters, nine paired episodes per method, one CPU backend and short fixed horizons. The bootstrap intervals are descriptive within-prompt uncertainty, not evidence for broad workload generality. See [`paper/results/stage6_transformers/stage6_report.md`](paper/results/stage6_transformers/stage6_report.md), the per-episode CSV, results JSON, tuning JSON, and SHA-256 manifest for complete evidence.
