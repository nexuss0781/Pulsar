# Stage 4 measurement results

**Scope:** actual C++ CPU wall-clock measurements on the existing synthetic order-3 target/order-2 draft Markov model in `cpp/inputs/model.bin`; not a real language model. The run used 3 prompts × 3 repetitions, 512 emitted tokens per run, batch size 1, temperature-1 categorical sampling, and matched seeds for AR/tree runs. The 27-point grid produced 243 candidate runs plus 9 baselines (252 raw rows total).

## Illustrative aggregate results

| Measurement | AR baseline | Highest-yield fixed tree point |
|---|---:|---:|
| Configuration | one target token/cycle | depth 4, branch 4, threshold 0.00 |
| `C(T)` mean wall time/cycle | 1.523 µs/token | 163.634 µs/pulse |
| `Y(T)` observed emitted tokens/cycle | 1.000 | 1.899 |
| End-to-end throughput | 586,779 tokens/s | 10,468 tokens/s |
| Mean of per-run p95 cycle latencies | 1.982 µs | 266.175 µs |
| Mean selected nodes per pulse | — | 84.59 |
| Mean selected-tree payload estimate | — | 372,089 bytes/pulse |
| Maximum selected-tree payload estimate | — | 719,736 bytes |

Across this measured grid, the highest mean throughput among tree candidates was **249,307 tokens/s**, below the AR fixture baseline. The highest-yield point emits more tokens per pulse, but incurs much more measured CPU work on this tiny table-lookup fixture. This is a measurement of the current C++ implementation and fixed grid, not evidence about real-model serving speed or a globally optimal policy.

## Reproducibility and interpretation

The raw CSV contains every prompt/configuration/repetition/seed, output checksum, work count, tree shape, timing percentile, and memory estimate. A second full execution reproduced all 252 runs' seeds, output hashes, counters, configurations, and memory/work fields; wall-time fields varied as expected. The summary's latency-percentile columns are means of per-run percentiles, while `C(T)`, `Y(T)`, work, and throughput are pooled from run totals.

Draft/target callback timings are table lookup, context conversion, and row-copy timings. The verifier/sampling diagnostic excludes the target callback but includes random sampling and tree-integrity checks; it is not an isolated verifier-kernel timer. Wall times include per-row clock probes. The tree memory figure uses vector capacities and excludes transient pruning peak, allocator metadata, and process RSS. **KV-cache bytes are unsupported/not applicable:** this toy model has no transformer runtime or KV cache.

No scheduler was trained and no global-optimality claim is made. Any future theorem for `C(T)` must use the measured definition documented in [`algorithm_v4_measurements.md`](../../../algorithm_v4_measurements.md).
