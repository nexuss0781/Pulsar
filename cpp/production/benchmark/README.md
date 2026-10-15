# Phase 3 benchmark

This directory contains the frozen workload/protocol, prompt manifest, and benchmark runner for Phase 3 of `ROADMAP.md`. The benchmark compares **Pulsar autoregressive**, llama.cpp's built-in **`draft-simple` chain**, and **fixed Pulsar candidate trees**. It introduces no adaptive scheduler.

## Build

Use the pinned llama.cpp revision recorded in `../UPSTREAM.md` and the same GGUF pair used by Phase 2 smoke tests. The helper below builds the Pulsar executable plus the upstream server in separate build directories:

```bash
cmake -S cpp/production -B /tmp/pulsar-phase3-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSAR_LLAMA_CPP_SOURCE_DIR=/tmp/llama.cpp \
  -DPULSAR_FETCH_LLAMA_CPP=OFF -DBUILD_TESTING=ON
cmake --build /tmp/pulsar-phase3-build \
  --target pulsar-phase3-bench pulsar_runtime_tests pulsar_tree_v3_tests -j2
ctest --test-dir /tmp/pulsar-phase3-build --output-on-failure

cmake -S /tmp/llama.cpp -B /tmp/llama-phase3-upstream \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_HIP=OFF
cmake --build /tmp/llama-phase3-upstream --target llama-server -j2
```

The selected model and server paths can be passed explicitly. Defaults in the runner point to `/tmp/pulsar-smoke.gguf`, `/tmp/pulsar-smoke-draft.gguf`, and the two build paths above.

## Run

Run the complete, preregistered tuning-then-confirmatory sequence:

```bash
python3 cpp/production/benchmark/run_phase3.py --phase all
```

Or run tuning first, inspect the generated frozen selection, then run held-out confirmation without retuning:

```bash
python3 cpp/production/benchmark/run_phase3.py --phase tuning
python3 cpp/production/benchmark/run_phase3.py \
  --phase confirmatory --run-dir /absolute/path/to/the/tuning/run

# Recompute the report from completed raw files without rerunning models
python3 cpp/production/benchmark/run_phase3.py \
  --phase analyze --run-dir /absolute/path/to/the/completed/run
```

Each new tuning/all run receives a unique UTC directory below `paper/results/phase3/`; existing raw results are never overwritten. `frozen_config.json` stores the chosen setting separately for each batch size, the tuning candidate scores, and SHA-256 hashes of all tuning raw files. Confirmatory mode accepts only a configuration marked frozen before held-out runs.

## Evidence

Each run folder retains the exact protocol and prompts, hashes and host metadata, commands and server logs, warm-up records, all generated outputs, unaggregated per-batch JSONL, process metadata, bootstrap analysis, and `PHASE3_REPORT.md`. Raw batch files are authoritative; summaries do not replace them. Server-only counters that the pinned llama.cpp metrics do not expose are `null` and explicitly explained rather than guessed. Native KV bytes are an F16/equal-head-dimension estimate; the llama.cpp `n_tokens_max` gauge is a maximum single-sequence length, not total live KV occupancy.

All benchmark timing and decision rules are frozen in [`PHASE3_PROTOCOL.md`](PHASE3_PROTOCOL.md). Results are descriptive and workload-specific; no speedup gate is imposed.
