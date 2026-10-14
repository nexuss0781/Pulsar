# Stage 6 — Transformers backend integration and matched benchmark

**Run:** `stage6-20261013T142914Z-seed60613`. **Exactness audit:** PASS

## Falsifiable claim under test

On the reported CPU, Transformers version, `distilgpt2` target / `sshleifer/tiny-gpt2` draft, temperature-1 softmax law, prompt set, and fixed output horizon, the draft-top-probability scheduler selects a depth/branch tree action that improves held-out tokens/s over the tuning-selected fixed tree policy, while the exact verifier preserves the target categorical law. The claim is supported only if the paired 95% lower confidence bound for scheduler-vs-fixed-tree speedup is > 0; otherwise it is not established for this operating condition.

## Exactness against actual backend rows

The target row is obtained from the loaded Transformers target model's logits after its declared temperature processor. The proposal row is obtained from the loaded draft model. The deterministic check applies the Stage 1 `q·a + Z·r = p` identity to every vocabulary token; the separate Monte Carlo check groups the 32 highest target-probability tokens and the remaining vocabulary into one `OTHER` bin.

| Context | p–q TV | Pointwise max error | Grouped empirical TV |

| When the lights went out, | 0.892239 | 1.08e-19 | 0.011620 |
| A recipe for bread begins | 0.966536 | 2.71e-20 | 0.012726 |
| The most important lesson I learned was | 0.957472 | 1.08e-19 | 0.013850 |

## Tuning and held-out benchmark

- Best chain block length: **γ=2** (candidates [2, 4, 6]).
- Best fixed tree: **depth=2, branch=1** (grid [[2, 1], [3, 1], [3, 2], [4, 2]]).
- Confidence map (thresholds on max draft probability): [2.2283282688970282e-05, 2.2573851481720864e-05]; bin actions [0, 0, 0].

All methods use the same target model, tokenizer, temperature, prompt/seed episodes, and generated-token budget. Tuning prompts and test prompts are disjoint. The fixed tree and speculative block length are selected on tuning runs only. Benchmark timing includes prompt prefill, draft expansion, target rows, exact verification, KV-cache state creation/commit, and output updates; model downloads/loading are excluded. EOS is treated as an ordinary token so every paired run has the same fixed horizon.

| Method | Tokens/s | Paired change vs best fixed tree (95% CI) | Paired change vs AR (95% CI) |
|---|---:|---:|---:|
| autoregressive | 40.62 | +269.77% [+241.07%, +296.19%] | — |
| tuned_chain | 15.71 | +43.00% [+34.87%, +50.79%] | -61.33% [-63.47%, -59.40%] |
| fixed_tree | 10.98 | — | -72.96% [-74.82%, -70.71%] |
| confidence_tree | 11.17 | +1.66% [-6.71%, +10.01%] | -72.51% [-74.49%, -70.63%] |

**Scheduler claim:** NOT ESTABLISHED; the paired 95% lower bound is not positive.

### Per-episode results

See `stage6_policy_runs.csv` for paired timings, seeds, output IDs, acceptance counts, and backend forward-call/input-token counters.

## Backend adapter and cache behavior

- `TransformersCausalLM.row`: actual model logits -> declared processed law.
- `verify_chain`: one target call over the speculative block, with prefix KV caches cropped to the accepted/replacement point.
- `append_many` + full-row draft distributions: batched branch expansion with complete q rows and top-q child retention.
- `prefill_many`: level-batched target rows and per-node KV states for tree verification; accepted branches commit the matching target/draft cache, while residual/off-tree tokens append to the current state.
- No custom tree-attention kernel or GPU execution is claimed. The tree adapter evaluates full candidate contexts batched by level; the call and input-token counters expose this cost.

## Limitations and scope

This is one CPU, batch-size-1, short-context study on GPT-2-family checkpoints. `tiny-gpt2` is a deliberately small draft, not a separately trained distillation of `distilgpt2`; the evaluation has only three held-out prompts and does not establish workload-wide statistical generality. No GPU, server batching, quantization, top-p/top-k, repetition penalties, long-context, or production serving claim is made. Confidence means max draft-token probability, not a calibrated estimate of verifier acceptance. Pulsar's rotor/quantum language is explanatory framing here; it is not part of the tested algorithm.

Stage 5 remains **completed with throughput gate not met**; its separate pooled calibration result is unchanged. Stage 6 does not retroactively pass Stage 5.

Reproduce with `python3 verification/experiments/run_stage6_transformers.py`. See `algorithm_v6_backend.md` for the adapter contract and experiment protocol.
