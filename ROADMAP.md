# Pulsar Production Implementation Roadmap

Status date: 2026-10-16

This engineering roadmap translates the owner-approved working vision in [`VISION.md`](VISION.md) into gated phases. It is separate from research Stages 0–6: those stages preserve theoretical and experimental results, while these phases move the runnable C++ implementation toward the vision. A phase is not considered complete merely because its code exists; its exit evidence must pass. The current performance evidence does not establish an acceleration, and this roadmap does not promise one.

## Agreed product direction

Pulsar targets offline, batched general text generation, with throughput and latency treated as co-equal goals. The first production implementation targets CPU-only Linux on x86_64 and ARM64, using C++ and llama.cpp. The working vision calls for exact preservation of the target model's configured sampling law; any speedup is a hypothesis to measure. Quantum terminology remains analogy unless a quantum mechanism is independently implemented and evaluated.

## Phase 0 — Vision and objective

Status: complete.

The project owner approved the working vision in `VISION.md`: offline/batch general text generation; balanced throughput and latency; exactness as a hard constraint; quality as a separately measured objective; and staged validation before any broad performance claim.

## Phase 1 — Native CPU runtime foundation

Status: implemented; broader release gates remain open.

The current C++ runtime uses a pinned llama.cpp C API and GGUF models. It provides a batch CLI with per-request context/KV state, controlled sampling, reproducible request RNGs, and JSONL results. Autoregressive decoding is the default. An opt-in exact speculative chain reuses Pulsar's verifier, applies residual replacement and bonus sampling, and repairs rejected KV suffixes.

Evidence recorded so far includes a strict x86_64 build, seven runtime test groups, staged install and downstream CMake-consumer checks, ARM64 cross-build and QEMU tests, and real tiny-GGUF smoke runs exercising rejected-block rollback. A single tiny-model smoke comparison was slower with speculation, so it is not performance evidence or a speedup claim. Native ARM testing and broader model/statistical validation remain open. See [`cpp/production/STATUS.md`](cpp/production/STATUS.md).

## Phase 2 — Integrate the exact bounded candidate tree

Status: implemented and validated locally; publication is tracked separately.

Add a fixed-branching, fixed-depth tree to the production C++ runtime before adding a learned scheduler. Preserve each node's full proposal law. The tree limits continuation work, not token eligibility: draw from the specified effective proposal, verify against the target law, replace rejections from the exact residual, and emit an accepted off-tree token before stopping that pulse. Each branch must retain or correctly reconstruct its own context and KV state. If llama.cpp cannot safely represent or restore the required branch state, stop and redesign rather than silently approximating it.

Exit checks: deterministic pointwise verifier identities over every token in tested rows; tests for rejection at every tree depth, residual edge cases, accepted off-tree fall-through, stale/incorrect contexts, KV rollback and commit, batch isolation, and stopping behavior; statistical sequence-law checks as a secondary test. The autoregressive path remains the correctness reference, and fixed-tree mode remains opt-in until validated.

## Phase 3 — Measure fixed policies and the real cost profile

Status: complete; benchmarked on the declared CPU fixture with paired held-out evidence.

On matched target/draft models, CPU, prompts, seeds, batch sizes, output horizons, and sampling processors, compare autoregression, llama.cpp's built-in chain speculation, and fixed Pulsar trees. Tune only on designated tuning prompts; report held-out paired results. Include drafting, target forwards, verifier work, tree management, cache operations, and output overhead in wall time. Record aggregate tokens/second, total batch completion time, per-prompt p50/p95 latency, peak memory/KV use, and work counters. Set acceptance margins and workload definitions before the confirmatory run. Do not train an adaptive policy against an unmeasured or mismatched cost model.

Exit check: reproducible measurements with raw run data and matched baselines; fixed-tree settings and regimes where they win or lose are stated. This phase need not pass a speedup gate to be complete, but it must not overstate results.

Evidence: [`cpp/production/benchmark/PHASE3_PROTOCOL.md`](cpp/production/benchmark/PHASE3_PROTOCOL.md), [`paper/results/phase3/20261015T193708Z/PHASE3_REPORT.md`](paper/results/phase3/20261015T193708Z/PHASE3_REPORT.md), the retained per-batch raw records, outputs, logs, and hashes in that run directory, plus [`analysis_correction.json`](paper/results/phase3/20261015T193708Z/analysis_correction.json). On the TinyStories 15M CPU fixture, neither llama.cpp draft-simple nor the fixed Pulsar tree beat autoregression at batch sizes 1 or 2; these are fixture-specific losses, not a universal performance claim. The initial bootstrap keying error and its preserved report are documented; final estimates use all matched seed/prompt groups, and no raw observations were altered.

## Phase 4 — Calibrate confidence and test adaptive scheduling

Status: completed 2026-10-15; no overall acceleration was established.

Collect verifier-derived acceptance labels and candidate-side features. Split calibration, tuning, and evaluation by prompt/context groups. Report Brier score and calibration by depth, workload, and the scheduler-selected distribution. Compare the learned confidence scheduler with the best tuning-selected fixed tree and an oracle upper bound. Keep calibration, EMA/bandit updates, and scheduling policies separately testable before combining them.

Performance claim gate: exactness passes, calibration is evaluated on held-out groups, and the adaptive policy improves the declared end-to-end objective over the best fixed policy on predeclared workloads, with an appropriate positive paired confidence bound. A calibration gain alone is not a throughput pass. If it does not beat the fixed policy, report that honestly and keep the simpler policy.

Outcome on the CPU-only TinyStories 15M fixture: adaptive versus the tuning-selected fixed tree passed the throughput/p95 gate at batch 1 (throughput ratio 1.049, 95% CI 1.031–1.066; p95 latency ratio 0.944, CI 0.930–0.981). Batch 2 was inconclusive (throughput ratio 1.007, CI 0.989–1.024). The overall adaptive-versus-autoregressive acceleration gate failed at both batch sizes (throughput ratios 0.322 and 0.274); **no overall acceleration was established**. The scheduler chose (1,2) for every batch-1 pulse and for 92.1% of batch-2 pulses, so the batch-1 difference does not demonstrate a benefit from action switching. Keep the adaptive policy experimental.

Evidence: [`PHASE4_PROTOCOL.md`](cpp/production/benchmark/PHASE4_PROTOCOL.md), [`PHASE4_REPORT.md`](paper/results/phase4/20261015T202100Z/PHASE4_REPORT.md), and the selected run artifacts with a relative-path SHA-256 manifest in [`paper/results/phase4/20261015T202100Z/`](paper/results/phase4/20261015T202100Z/).

## Phase 5 — Production hardening and native architecture validation

Status: not started.

Run on native Linux x86_64 and ARM64 hardware, not only emulation. Expand tests across model families, longer contexts, malformed inputs/models, resource exhaustion and memory pressure, many-request batches, cancellation/error recovery, repeated warm runs, sanitizer builds, and installation/deployment paths. Document supported llama.cpp revision, GGUF/tokenizer compatibility, sampling processors, CPU feature behavior, thread policy, memory limits, and known exclusions. Add a serving interface only after the batch runtime and measurements justify it.

Exit check: reproducible, architecture-specific evidence; bounded and documented failure behavior; no unresolved correctness, cache, memory-safety, or packaging issues for the declared support matrix.

## Phase 6 — Separate model training or fine-tuning track

Status: deferred.

Only after the inference path has a stable contract, select a specific model-quality objective, dataset/evaluation protocol, and acceptable resource budget. Evaluate model fine-tuning separately from inference-scheduler improvements so changes in answer quality are not mistaken for decoding speedups. “Perfect model” is not a measurable endpoint; define use-case quality targets and report trade-offs.

## Immediate proposal

Phase 4 is complete, but no overall acceleration was established; keep the adaptive policy experimental and the measured autoregressive path as the baseline. Phase 5 remains the next candidate, but this publication contains no Phase 5 implementation. Do not make a general speedup claim; model training/fine-tuning and quantum mechanisms remain out of scope.
