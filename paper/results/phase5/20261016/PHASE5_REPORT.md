# Phase 5 Production Hardening Report

**Date:** 2026-10-16
**Disposition:** Hardening and validation pass completed in the isolated Phase 5 working copy. **The phase exit gate remains open and this is not a production-readiness sign-off.** Native ARM64 hardware was unavailable, and real-model coverage is intentionally narrow. No new speedup or performance claim is made.

## Summary

The runtime now supports cooperative cancellation at safe inference boundaries, reports unfinished requests with `finish_reason: "cancelled"`, and maps CLI SIGINT/SIGTERM to conventional exit statuses. Regression coverage now includes 16 deterministic runtime groups, larger request batches, a context boundary, repeated fixed-seed runs, partial KV failure recovery, and cancellation across autoregressive, speculative-chain, and candidate-tree paths. CMake installation now exports a versioned `PulsarRuntime` package usable through `find_package`; native and emulated downstream-consumer checks passed.

Native x86_64 Release tests, real-GGUF CLI tests on Llama and GPT-2 architectures, AddressSanitizer/UndefinedBehaviorSanitizer tests, and a full AArch64 cross-build with QEMU user-mode tests all passed. The latter is **cross/emulated** validation, not native ARM64 validation.

## Changes

- Added a cancellation callback to `GenerationOptions`, a `cancelled` finish reason, and safe-boundary cancellation checks in autoregressive, exact chain-speculative and candidate-tree generation. Candidate-tree cancellation is observed only after a pulse's selected prefix is committed and scratch branch KV slots have been cleared.
- Added CLI SIGINT/SIGTERM handling. A cancelled in-flight request produces a partial JSONL result and the CLI returns `128 + signal` (the integration test exercised SIGINT/status 130).
- Added deterministic tests for a 128-request fake-backend batch, five repeated warm runs, a 2,048-token context boundary, cancellation in all three generation modes, candidate-tree scratch cleanup, and recovery after an injected decode error that mutates KV state before throwing.
- Added `PulsarRuntimeConfig.cmake` and version metadata, plus an installed-header downstream consumer using `find_package(PulsarRuntime 0.1.0 CONFIG REQUIRED)`. The install-consumer CTest is native-only; cross builds test installed ARM binaries separately under QEMU.
- Added the support contract and repeatable smoke harness. Existing Phase 3/4 report files and quantitative conclusions were left unchanged.

## Validation results

| Validation | Result | Scope and interpretation |
|---|---|---|
| Native x86_64 Release build and CTest | **PASS — 3/3 CTest targets; 16 runtime groups** | Full CPU-only llama.cpp-linked build; unit suite, tree verifier suite, staged install and external CMake consumer. Host: Linux x86_64, GCC 13.3.0, CMake 3.28.3. |
| Real-GGUF CLI matrix | **PASS — 9 checks** | Llama-family autoregressive, chain and tree paths; GPT-2-family generation with a 961-token prompt; malformed path/file and oversized context; 16 prompts in four microbatches with JSONL file output; four fixed-seed GPT-2 outputs were identical; cancellation and memory-pressure tests. Repeatability is not a speed measurement. |
| Failure and resource pressure | **PASS — bounded CLI error** | Under a child-process `RLIMIT_AS` of 384 MiB, context initialization attempted a 27,648 MB allocation (llama.cpp diagnostic), returned an error and CLI status 2, and did not terminate by signal. This is a constrained allocation test, not a process-wide memory-cap API. |
| AddressSanitizer + UndefinedBehaviorSanitizer | **PASS — 3/3 CTest targets** | Separate CPU-only build with `-fsanitize=address,undefined`, leak detection and halt-on-UB. The sanitizer build did not link llama.cpp; real-model backend tests were run in the Release build. |
| AArch64 cross-build and QEMU | **PASS — 2/2 CTest targets** | GNU 13.3.0 cross compiler, `-march=armv8-a`, QEMU user-mode 8.2.2. The installed AArch64 CLI and independent downstream consumer ran under QEMU, and the installed CLI completed a TinyLlama GGUF inference. **Not native ARM64 evidence.** |
| Native installed-package consumer | **PASS** | Fresh install prefix, versioned `find_package(PulsarRuntime)`, compile/run of consumer including batch, speculative and tree public headers; installed CLI `--help` check. |

The complete real-model smoke output, machine-readable summary ([`phase5_evidence.json`](phase5_evidence.json)) and concise CTest/build evidence are in [`paper/results/phase5/20261016/`](.). Reproduction steps and fixture sources are in [`cpp/production/PHASE5_VALIDATION.md`](../../../../cpp/production/PHASE5_VALIDATION.md); the support contract is [`cpp/production/SUPPORT.md`](../../../../cpp/production/SUPPORT.md).

## CPU and model support statements

The inspected x86_64 build compiles with `-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2`; AVX512 and AVX-VNNI are disabled. It uses `GGML_NATIVE=OFF` and `GGML_CPU_ALL_VARIANTS=OFF`, so it avoids `-march=native` but does **not** have a generic x86-64 feature floor or runtime CPU-variant dispatch. The host exposes those instructions; deploy only after validating the target CPU. The AArch64 build used a separate `armv8-a` baseline.

The pinned backend is llama.cpp commit `cd0a5b6a180dff0048e5fd993098b21c0b2c91b8`. The tested public model fixtures were:

| Fixture | Architecture/context | Local file size | SHA-256 | Notes |
|---|---|---:|---|---|
| [TinyLlama 15M Stories](https://huggingface.co/tensorblock/tinyllama-15M-stories-GGUF), Q2_K target | Llama / 256 | 13,717,344 bytes | `f7e39dc9f26f3d39bf59e885349c6eec65880f685322d591f53e6cdb46ceb2e9` | Model metadata reports MIT; local fixture `/tmp/pulsar-smoke.gguf`. |
| [TinyLlama 15M Stories](https://huggingface.co/tensorblock/tinyllama-15M-stories-GGUF), Q3_K_M draft | Llama / 256 | 14,028,384 bytes | `58e4a0c44851897f4007f9c9018822348f7d74851790fa0ea1370ed82c36e003` | Same model/tokenizer family; local fixture `/tmp/pulsar-smoke-draft.gguf`. |
| [TinyStories GPT-2 3M](https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF), Q2_K | GPT-2 / 2,048 | 7,754,464 bytes | `0cfb3ef83cd05fba1daa40a8a0207edb7f4e9f9992bcf43b0bc6e679d6e8e58b` | Metadata reports `gpt2`; no license was declared in the fetched card/API. Local fixture `/tmp/pulsar-smoke-gpt2-q2.gguf`; not redistributed. |

These small fixtures show that two tested architectures load and execute under the pinned backend; they do **not** establish broad GGUF compatibility. Exact chain/tree correctness remains bounded by the documented tokenizer compatibility checks; there is no full tokenizer fingerprint.

## Remaining gates and claim limits

1. **Native ARM64 hardware build/runtime validation remains open.** Cross-compilation and QEMU user mode cannot substitute for native CPU, kernel, memory-pressure or performance evidence.
2. The real-model smoke matrix covers only tiny Llama and GPT-2 fixtures. Additional GGUF architectures, tokenizer variants, quantizations, longer real contexts and a broad real-model statistical sequence-law study remain unverified.
3. The memory check used a subprocess address-space limit and exercised a returned context-allocation failure. It does not prove recovery from every llama.cpp allocator/backend failure, establish a global RSS cap, or validate a production cgroup policy.
4. The current deliverable remains a local batch CLI and C++ runtime core. Multi-user serving, a broader deployment matrix, operational monitoring and end-to-end production rollout are out of scope.
5. Four identical fixed-seed warm outputs were used only as a determinism/repeatability check. **No latency, throughput or acceleration claim is made.** Phase 4's overall acceleration gate remains failed/inconclusive as recorded in its unchanged report.

## Publication and evidence preservation

The Phase 5 implementation and evidence were produced in isolated working copy `/workspace/pulsar-phase4-isolated`; the reviewed 24-file transfer was applied to `/workspace/pulsar-public` after confirming the checkout and live `origin/main` were both at the Phase 4 base commit `b08b5986ecec2c109457ef0acaa4022c719ff2ea`. This report is included in the public Phase 5 change. The Phase 3 and Phase 4 report hashes were verified unchanged, and prior evidence folders were not modified by this work. This report does not close the native ARM64 release gate.
