# Pulsar Stage 4 — measure cost before optimizing the pulse rule

Stage 4 adds an instrumented CPU benchmark for the **existing synthetic Markov fixture** at `cpp/inputs/model.bin`. It does not add a real language-model runtime and does not change Stages 0–3.

## Fixed-policy sweep

The benchmark compares a plain autoregressive sampler against the Stage 3 bounded-tree sampler on the same loaded fixture, prompt, output-token limit, batch size (1), temperature-1 categorical sampling setting, and deterministic seed for each `(prompt, repetition)` pair.

The fixed grid is:

| Parameter | Values |
|---|---|
| Maximum pulse depth (verifier positions, root included) | 1, 2, 4 |
| Branch factor | 1, 2, 4 |
| Minimum retained level mass threshold | 0.00, 0.20, 0.40 |

The threshold is a **benchmark-only fixed policy**, not the learned confidence/pulse rule: after building the configured Stage 3 tree, the selector keeps a whole next level only when the total normalized `q` probability mass of its retained prefixes is at least the threshold. Selection truncates at a level boundary, preserving the Stage 3 sampler's tree-integrity expectations. Threshold 0 disables this cutoff. Full configured-tree construction occurs before threshold selection; that work is included in the measured cost rather than hidden.

Prompts are fixed in the C++ source and recorded in every raw row: `[5,3,9]` (the fixture's canonical prompt), `[1,4,7]`, and `[0,0,0]`. Each prompt/repetition baseline is paired with every tree-grid candidate using the same seed, output-token limit, and batch size.

## Measurements

- **`C(T)`**: observed `steady_clock` wall time per candidate pulse, including full configured-tree construction, threshold selection/truncation, Stage 3 sampling, and output/context updates. The AR reference's per-token cycle time is reported as its one-token baseline cost.
- **`Y(T)`**: observed emitted tokens divided by pulse count. Candidate accepted and emitted token counts are both retained; the baseline has no speculative acceptance count.
- **Draft/proposal work**: draft table callback count, returned probability values, callback time, and inclusive tree-build time. For this fixture, these are table lookups/context conversion/row copies—not neural forward passes or FLOP counts.
- **Target verification work/time**: target callback count, returned values, callback time, and time per callback, reported alongside tree node count, nodes by depth, and selected depth. Sampler time minus target callback time is an **inclusive verifier/sampling/integrity-check proxy**, not an isolated verifier-kernel timer.
- **Construction and selection overhead**: separately recorded inclusive tree-build and threshold-selection wall time.
- **Memory**: maximum pre-threshold and selected-tree payload estimated from the capacities of the tree's vectors (node structs, contexts, complete `q` rows, child-index arrays). This excludes allocator metadata, transient pruning peak, and process RSS.
- **Latency and throughput**: per-cycle p50/p90/p95/p99 and end-to-end emitted tokens/second for each run.
- **KV cache**: not measurable or applicable. This Markov fixture has no transformer runtime or KV cache; raw CSV leaves `kv_cache_bytes` blank and marks the status explicitly.

## Reproduce

From `/workspace/pulsar`:

```bash
mkdir -p paper/results/stage4
cd cpp
make pulsar_stage4_bench CXXFLAGS='-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror'
./pulsar_stage4_bench --model inputs/model.bin --tokens 512 --repeats 3 --raw ../paper/results/stage4/stage4_raw.csv
cd ..
python3 verification/experiments/summarize_stage4.py \
  --raw paper/results/stage4/stage4_raw.csv \
  --out-dir paper/results/stage4
```

The aggregator verifies paired prompts/seeds/batch sizes/output limits/sampling settings, then writes `stage4_summary.csv`, JSON metadata, and cost/yield, throughput, and verifier/sampling-cost-versus-tree-size plots. Raw per-run timing records contain seeds, configuration, workload, output checksums, timing percentiles, work counters, tree shape, and memory estimates. Measured wall times include the steady-clock probes used to instrument callbacks; no instrumentation-overhead subtraction is applied. Timings will vary across runs and machines; the configuration and generated outputs are reproducible, but wall-clock values are observations rather than deterministic constants.

The currently recorded run and its interpretation are summarized in [`paper/results/stage4/stage4_report.md`](paper/results/stage4/stage4_report.md).

## Interpretation and scope

These are **actual CPU wall-clock measurements of this C++ implementation on this toy fixture**. They are not model-forward timings, GPU timings, transformer serving results, or KV-cache estimates. The target/draft callbacks return rows from static arrays. The benchmark defines `C(T)` to include tree selection/construction and sample work; any future scheduler theorem or optimality claim must use a matching cost definition. No scheduler is trained and no global-optimality claim is made in Stage 4.
