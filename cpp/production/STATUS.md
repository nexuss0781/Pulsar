# Production runtime status

**As of 2026-10-16.** This is a runnable native C++ CPU inference implementation; the evidence below is not a production-readiness declaration.

## Implemented

The CLI loads local GGUF models through the pinned llama.cpp C API, reuses model/context state across bounded microbatches, isolates request KV state, and emits JSONL results. It supports temperature, top-k, top-p, repetition penalty, and independent reproducible request RNGs. Optional draft/target chain speculation uses Pulsar's existing exact v2 verifier with full processed vocabulary distributions, EOS handling, residual replacement, bonus sampling, and rejected-suffix KV rollback. JSONL reports draft proposal, acceptance, rejection, and throughput counters; autoregressive generation remains the default.

## Phase 2 candidate-tree path

The opt-in `--tree-branching N --tree-depth D` CLI mode uses a fixed bounded prefix tree, while autoregressive generation remains the default and exact chain speculation remains available through `--speculative-tokens`. Each tree node stores the full processed draft proposal row; top-N pruning controls only which accepted proposals can continue in the current pulse. An accepted off-tree token is emitted and ends that pulse. Rejection emits the exact verifier residual and ends the pulse; an accepted end token terminates generation. Branch KV prefixes are cloned into isolated sequence slots, the verified final prefix is copied back to the request sequence, and scratch branch slots are cleared.

This Phase 2 implementation is local and uncommitted; it has not been published. Tree mode requires a compatible draft model and backend sequence capacity for the configured batch/tree bound.

## Verification

A Release build against the pinned llama.cpp source passes with `-Werror` on Linux x86_64, with GPU backends disabled. The production runtime suite passes all 12 deterministic groups, including tree rejection at every depth, terminal stopping, exact residual replacement, batch isolation, branch-KV commit/cleanup, and capacity rejection. CTest also runs the existing tree verifier suite, including the pointwise identity, off-tree fall-through, stale-context rejection, and statistical conditional-law checks. The full llama.cpp-linked build and both CTest targets pass.

The AArch64 version was cross-compiled with GNU 13.3 for Linux ARM64. All seven test groups passed under QEMU user-mode emulation. The AArch64 staged install was identified as an AArch64 ELF executable, its installed binary ran outside the build tree, and it completed real TinyStories GGUF inference in both autoregressive and exact speculative modes. This is emulated architecture coverage, not a native ARM hardware test or performance measurement.

The x86_64 CMake install was also staged into a fresh prefix. Its shared-library dependencies resolved without missing libraries, the installed CLI printed help, and the installed binary completed real GGUF inference without relying on the build-tree RPATH. A separate downstream CMake consumer compiled and ran against the installed `Pulsar::pulsar_runtime` target and headers.

Real-model tests used the public TinyStories 15M model repository (Q2_K target and Q3_K_M draft), two prompts, batch size 2, context 128, and two CPU threads. With identical target/draft weights and greedy sampling, speculative and autoregressive completions matched exactly. With Q2_K target and Q3_K_M draft, gamma 4, temperature 0.8, top-k 40, top-p 0.95, and 40 output tokens per prompt, JSONL parsing passed and the runtime recorded **10 rejected blocks** across 89 proposals (58 accepted); the installed x86_64 binary completed the run without a KV-rollback error. This proves that the real rollback branch was exercised, not that a broader statistical exactness study has been completed.

Phase 2 real-GGUF smoke tests used the same Q2_K model as target and draft, branch factor 2, depth 3, and greedy sampling: its 16-token tree completion matched autoregressive output exactly, with 16/16 accepted proposals. A distinct Q2_K/Q3_K_M target/draft run at temperature 0.8, top-k 40, top-p 0.95 emitted 20 tokens and exercised **one rejection across 20 proposals (19 accepted)** without a KV-copy/rollback error. The staged install includes `tree_generation.hpp`; an external translation unit compiled and ran against the installed header, and the installed CLI exposed both tree flags.

In one matched tiny-model run, autoregressive generation produced 80 tokens in 267.808 ms (about **298.7 aggregate tokens/s**) while Q2-target/Q3-draft speculation produced 80 in 740.499 ms (about **108.0 aggregate tokens/s**). These are single-run smoke observations, not benchmark results; they show no speedup and are not generalized to other models or workloads.


## Phase 4 confidence calibration and adaptive scheduling

The Phase 4 implementation and experiment were completed in an isolated workspace before publication. The preregistered CPU-only TinyStories 15M fixture used 24 disjoint topic/context groups (8 calibration, 8 tuning, 8 locked evaluation), two threads, batch sizes 1 and 2, and ten paired evaluation seed roots. The frozen report records that runtime and Python correctness tests passed.

Adaptive versus the tuning-selected fixed tree passed the throughput/p95 gate at batch 1 (throughput ratio 1.049, paired 95% CI 1.031–1.066; p95 latency ratio 0.944, CI 0.930–0.981). Batch 2 was inconclusive (throughput ratio 1.007, CI 0.989–1.024). Adaptive remained substantially slower than autoregressive decoding: 0.322× throughput at batch 1 and 0.274× at batch 2. Thus, **no overall acceleration was established**. The batch-1 scheduler selected only (branching=1, depth=2); at batch 2 it selected that action 92.1% of the time and (2,2) 7.9%. Calibration metrics are descriptive; no uncalibrated Brier comparator was included.

See [`PHASE4_PROTOCOL.md`](benchmark/PHASE4_PROTOCOL.md), [`PHASE4_REPORT.md`](../../paper/results/phase4/20261015T202100Z/PHASE4_REPORT.md), and the selected evidence/hash manifest in `../../paper/results/phase4/20261015T202100Z/`.

## Phase 5 production hardening

The Phase 5 hardening pass is documented separately in [`paper/results/phase5/20261016/PHASE5_REPORT.md`](../../paper/results/phase5/20261016/PHASE5_REPORT.md). On native x86_64, the Release build and CTest passed (3/3), including 16 deterministic runtime groups and a staged `find_package(PulsarRuntime)` consumer. AddressSanitizer+UndefinedBehaviorSanitizer passed the CPU-only runtime/verifier/consumer tests (3/3). Real-GGUF smoke tests passed on TinyLlama/Llama and TinyStories/GPT-2 fixtures, including chain/tree modes, a 961-token GPT-2 prompt, 16 requests across four microbatches, four repeatable warm outputs, malformed-file/path handling, cooperative SIGINT cancellation, and a 384 MiB address-space failure that returned CLI status 2 instead of a signal.

An AArch64 `armv8-a` cross-build passed both CTest targets under QEMU 8.2.2; the installed AArch64 CLI, downstream consumer and TinyLlama inference also ran under QEMU. This is **cross/emulated** evidence, not native ARM64 validation. Native ARM64 hardware and performance validation remain open. The tested x86_64 build explicitly requires SSE4.2, AVX, AVX2, F16C, FMA and BMI2, has no runtime CPU-variant dispatch, and is not a generic x86-64 binary. No Phase 5 performance or speedup claim is made.

## Source provenance

The backend uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), pinned to commit [`cd0a5b6a180dff0048e5fd993098b21c0b2c91b8`](https://github.com/ggml-org/llama.cpp/tree/cd0a5b6a180dff0048e5fd993098b21c0b2c91b8); the exact dependency record is in [`UPSTREAM.md`](UPSTREAM.md).

The smoke model is [tensorblock/tinyllama-15M-stories-GGUF](https://huggingface.co/tensorblock/tinyllama-15M-stories-GGUF). Its [repository metadata](https://huggingface.co/api/models/tensorblock/tinyllama-15M-stories-GGUF) marks it public and MIT-licensed and reports Llama architecture with context length 256. Tested files were `tinyllama-15M-stories-Q2_K.gguf` and `tinyllama-15M-stories-Q3_K_M.gguf`; they were kept under `/tmp`, not added to Pulsar.

## Remaining release gates

Native ARM64 hardware build, runtime, and performance validation remain. The two real fixtures are tiny Llama-family and GPT-2-family samples; broader GGUF architectures, tokenizer variants and a real-model statistical exactness study remain unverified. The statistical sequence-law tests cover synthetic verifier contexts, not a broad real-model study. Draft models must share vocabulary semantics and token IDs: the runtime checks each input's tokenization and end-token alignment across the full vocabulary, but it does not fingerprint every tokenizer token mapping. Exact speculation depends on the backend supporting partial KV suffix removal; autoregressive mode does not. Candidate-tree branch rows are evaluated sequentially, and no tree throughput benefit has been established. Phase 4's adaptive scheduler was measured experimentally but did not establish overall acceleration. The deliverable remains a local batch CLI and reusable runtime core, not an HTTP service or multi-user queue; comprehensive production deployment and failure-injection coverage remain open.
