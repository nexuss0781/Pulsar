# Production Runtime Support and Limits

This document describes the CPU-only batch runtime built from this source tree. It is a support contract and evidence index, **not a production-readiness declaration**. Results and unverified gates are tracked in [`STATUS.md`](STATUS.md) and the separate Phase 5 report.

## Platform and build behavior

- Intended platforms: Linux on x86_64 and AArch64, using C++17 and the pinned llama.cpp revision in [`UPSTREAM.md`](UPSTREAM.md).
- The CMake integration disables GPU backends, sets `GGML_NATIVE=OFF`, and disables `GGML_CPU_ALL_VARIANTS`. Model GPU layers and context offload flags are also disabled.
- The inspected x86_64 build uses `-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2`; `GGML_AVX512=OFF` and `GGML_AVX_VNNI=OFF`. The binary therefore requires SSE4.2, AVX, AVX2, F16C, FMA and BMI2, and is **not** a generic x86-64 baseline build. `GGML_NATIVE=OFF` means it does not use `-march=native`; it does not turn off explicit feature flags.
- This configuration does **not** build a set of CPU variants for runtime dispatch. Build separate binaries per architecture and verify the actual compiler flags/cache before deployment. Do not copy an x86 binary to ARM64 or assume optional ARM dot-product/SVE instructions are present. A user-supplied compiler target flag can raise the minimum CPU requirement.
- `--threads 0` leaves thread counts to llama.cpp defaults. A positive `--threads N` sets both evaluation and batch-evaluation thread counts to `N` on each model context. Target and draft contexts are separate; this is not a global process-wide thread budget, so dual-model modes may use more than `N` runnable threads.

## Model and tokenizer compatibility

- Inputs are local GGUF files loaded by the pinned llama.cpp C API. This adapter is for decoder-only text generation; encoder-decoder models are rejected. Multimodal/vision/audio inputs and model-specific adapters outside the pinned backend's supported GGUF loaders are not claimed.
- The supported compatibility baseline is the exact llama.cpp revision recorded in [`UPSTREAM.md`](UPSTREAM.md). Upgrades require a fresh build, regression suite, install check, and real-GGUF tests. The Phase 5 sample fixtures include a Llama-family [TinyLlama GGUF repository](https://huggingface.co/tensorblock/tinyllama-15M-stories-GGUF) and a separate GPT-2-family [TinyStories GGUF repository](https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF); they validate only these files and architectures, not every model supported by llama.cpp.
- Autoregressive mode uses one model. Exact chain/tree modes require target and draft token IDs to mean the same thing. The runtime checks vocabulary size, end-token alignment, and equality of token IDs for the supplied prompt, but it does **not** fingerprint every tokenizer token mapping. Use target/draft files from the same tokenizer/base model; equal vocabulary sizes alone are insufficient.
- Candidate-tree and chain modes are opt-in. Autoregressive decoding remains the default and does not require draft-model compatibility.

## Sampling contract

The supported processor order and controls are temperature, top-k, top-p, and a repetition penalty over a bounded recent-token window. `temperature=0` is deterministic greedy selection with a lower-token-ID tie break. The runtime does not expose arbitrary logits processors, token biases, grammar-constrained sampling, min-p, typical sampling, or model-specific custom samplers. Exact speculative verification applies to the target distribution produced by the supported processors; it does not cover processors that are not exposed here.

## Context, batch and memory bounds

- `--context-size` is a per-sequence token limit; prompts that exceed it fail before generation. At the limit, the runtime may emit the final token from cached next-token logits but will not insert another token into KV state.
- `--batch-size` bounds concurrent requests per microbatch. The CLI reads input incrementally and emits one JSONL result per request; this is not an unbounded in-memory queue.
- `--batch-tokens` bounds prefill work per llama.cpp batch. It does not cap total process memory.
- The backend allocates one llama.cpp context sized to `context-size × configured-sequence-capacity`. Chain mode increases token-batch capacity. Tree mode reserves `batch-size × candidate_tree_node_bound(branching, depth)` sequence slots in both target and draft contexts. Large context, batch, branch or depth settings can therefore require substantial KV memory before any request is served.
- `kv_cache_bytes_per_token_estimate()` is a planning estimate for the pinned default F16 K/V cache and compatible metadata. It does not include model weights, computation buffers, allocator overhead, OS page cache, or other process memory, and it is not a hard memory cap. There is currently no global RSS limit or memory reservation API. Use operating-system cgroups/ulimits for an enforced process budget; on allocation failure the CLI reports a nonzero error when the backend returns control normally.
- Real deployments should set context, batch, branch/depth and thread values explicitly for the host budget. A lower memory limit can prevent startup or cause a bounded allocation error; this code does not promise that all llama.cpp/upstream allocation failures are recoverable.

## Cancellation and error recovery

- The C++ API accepts an optional cancellation callback. The CLI maps SIGINT and SIGTERM to it, emits completed partial JSONL results with `finish_reason: "cancelled"` for unfinished requests, and exits with the conventional `128 + signal` status.
- Cancellation is cooperative: it is checked between autoregressive decode rounds, after an exact speculative block is verified/repaired, or after a candidate-tree pulse has committed and scratch branch KV slots have been cleared. It does not interrupt a llama.cpp load, prefill, or active decode call, and it does not create resumable checkpoints.
- Backend errors propagate to the caller. The CLI prints a diagnostic to stderr and exits with status 2; a failed CLI batch is not retried automatically. A new generation call resets model KV state before prefill. This reset-on-next-call behavior is tested with an injected partial backend failure; it is not a guarantee that a process can continue after every upstream failure.

## Exclusions and claims

No HTTP/server interface, multi-user request scheduler, GPU backend, native ARM64 performance result, broad statistical equivalence study across real model families, universal GGUF compatibility guarantee, or general speedup claim is made. Phase 5 may close only the gates supported by recorded evidence; QEMU and cross-build results are never labeled native hardware results.
