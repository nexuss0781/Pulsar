# Phase 4: confidence calibration and adaptive tree scheduling

Run: `20261015T202100Z`
Frozen at: 2026-10-15T20:35:27+00:00
Evaluation completed: 2026-10-15T20:53:15+00:00
Protocol SHA-256: `dcede8db550d134b0fde0ee1a847a7ea5d52fb6fb730ebec504cc4032f903492`

## Scope and exactness

CPU-only TinyStories 15M GGUF fixture, target/draft model hashes frozen before evaluation, two CPU threads, batch sizes 1 and 2, 24 disjoint topic/context groups split 8/8/8, 64-token maximum, and the preregistered sampling law. These results apply only to this small-model x86_64 fixture.

Correctness tests before measurement: **PASS** (both CTest targets and the Python fitter/analysis tests). The adaptive C++ test covers calibration prediction, EMA update/reset, legal action choice, trace capture, and greedy-output equality on deterministic fake models. Inherited verifier tests cover pointwise residual identities, accepted off-tree behavior, and statistical conditional next-token laws on synthetic contexts. These are implementation-level checks, not a broad real-GGUF statistical sequence-law study. Exactness means preserving the target sampling law and verifier/residual semantics; different methods may consume random numbers differently, so sampled text is not required to match token-for-token.

## Tuning-only selections

| Batch | Built-in chain n_max | Selected fixed tree (B, depth) | Median tuning tok/s |
|---:|---:|---:|---:|
| 1 | 2 | (1, 2) | 52.882 |
| 2 | 2 | (1, 2) | 57.306 |

## Locked evaluation throughput and latency

| Batch | Method | Aggregate tok/s | Total batch seconds | Request p50 / p95 ms | Peak RSS MiB |
|---:|---|---:|---:|---:|---:|
| 1 | autoregressive | 173.880 | 29.446 | 374.10 / 436.55 | 36.5 |
| 1 | builtin | 124.057 | 41.271 | 516.80 / 589.13 | 82.9 |
| 1 | adaptive | 55.979 | 91.462 | 1136.15 / 1319.05 | 84.1 |
| 1 | fixed_selected | 53.376 | 95.924 | 1184.04 / 1397.12 | 65.4 |
| 1 | fixed_B1_D2 | 53.376 | 95.924 | 1184.04 / 1397.12 | 65.4 |
| 1 | fixed_B2_D2 | 43.239 | 118.411 | 1410.49 / 1792.65 | 69.2 |
| 1 | fixed_B2_D3 | 30.483 | 167.961 | 2080.07 / 2548.88 | 85.5 |
| 2 | autoregressive | 216.381 | 23.662 | 580.72 / 698.64 | 38.8 |
| 2 | builtin | 209.400 | 24.451 | 594.42 / 677.27 | 91.1 |
| 2 | adaptive | 59.361 | 86.251 | 2129.95 / 2393.72 | 111.2 |
| 2 | fixed_selected | 58.966 | 86.829 | 2120.52 / 2427.06 | 74.3 |
| 2 | fixed_B1_D2 | 58.966 | 86.829 | 2120.52 / 2427.06 | 74.3 |
| 2 | fixed_B2_D2 | 47.927 | 106.828 | 2600.64 / 3105.49 | 82.1 |
| 2 | fixed_B2_D3 | 32.696 | 156.596 | 3834.36 / 4679.64 | 112.1 |

### Paired comparisons (10,000 cluster-bootstrap resamples)

| Batch | Adaptive vs | Throughput ratio (95% CI) | p95 latency ratio (95% CI) |
|---:|---|---:|---:|
| 1 | fixed_selected | 1.049 (1.031–1.066) | 0.944 (0.930–0.981) |
| 1 | autoregressive | 0.322 (0.317–0.327) | 3.022 (2.974–3.145) |
| 1 | builtin | 0.451 (0.440–0.463) | 2.239 (2.045–2.333) |
| 2 | fixed_selected | 1.007 (0.989–1.024) | 0.986 (0.935–1.039) |
| 2 | autoregressive | 0.274 (0.269–0.284) | 3.426 (3.290–3.548) |
| 2 | builtin | 0.283 (0.278–0.289) | 3.534 (3.230–3.705) |

## Preregistered performance gate

- Batch 1: adaptive vs selected fixed tree **PASS**; adaptive vs autoregressive overall-acceleration condition **FAIL / INCONCLUSIVE**.
- Batch 2: adaptive vs selected fixed tree **FAIL / INCONCLUSIVE**; adaptive vs autoregressive overall-acceleration condition **FAIL / INCONCLUSIVE**.

A fixed-tree-only win is not an overall acceleration claim. Results are reported per batch size; no cross-batch pooling is used for pass/fail.

## Calibration and adaptive selection

Brier score and 10-bin reliability summaries are in `calibration_metrics.json`, grouped by method (fixed-action or scheduler-selected), batch size, branching, verifier depth, and outcome. The calibrator was fitted only on C01–C08 and was not recalibrated on locked evaluation groups.

Adaptive action pulse counts (not per-depth decision-row counts) are included in `calibration_metrics.json`. The offline oracle is descriptive only.

Across the scheduler-selected configuration/depth cells, row-weighted adaptive Brier scores for `accepted` / `continued` were 0.1560 / 0.1995 at batch 1 and 0.1562 / 0.2010 at batch 2. These scores are not compared with an uncalibrated baseline and do not establish an incremental calibration gain. The scheduler chose (1,2) for all 3,358 batch-1 pulses; at batch 2 it chose (1,2) for 3,074 pulses (92.1%) and (2,2) for 265 (7.9%), with no (2,3) choices. Therefore, the batch-1 adaptive/fixed throughput difference is not evidence that changing actions improved performance: the adaptive policy selected the same action throughout that workload.

## Offline fixed-action oracle upper bound (descriptive)

| Batch | Oracle tok/s | Selected action counts |
|---:|---:|---|
| 1 | 54.105 | `{"fixed_B1_D2": 71, "fixed_B2_D2": 9}` |
| 2 | 59.125 | `{"fixed_B1_D2": 37, "fixed_B2_D2": 3}` |

## Limitations and retained evidence

The oracle uses locked evaluation timings to select the fastest fixed action for each observed batch and is not an implementable policy. This is one CPU-only tiny-model fixture, with eight evaluation context groups and ten paired seed roots; uncertainty is cluster-resampled over contexts. llama.cpp server telemetry does not expose all draft KV/cache-copy counters; unavailable values remain null. Native KV bytes are estimates, not allocator telemetry. Component timers are inclusive/overlapping and are not additive. The scheduler's limited action diversity and the absence of an uncalibrated Brier comparator constrain any claim about learned adaptation or calibration quality.

Raw JSONL, generated outputs, verifier observations, commands, logs, model/source hashes, calibrator coefficients, action costs, and freeze metadata are retained under this run directory.
