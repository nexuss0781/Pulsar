# Phase 3: fixed-policy and cost-profile benchmark

Run ID: `20261015T193708Z`  
Confirmatory data completed: 2026-10-15T19:51:56+00:00  
Protocol: [PHASE3_PROTOCOL.md](../../../../cpp/production/benchmark/PHASE3_PROTOCOL.md)  
Raw evidence: `./confirmatory/raw/`, `./confirmatory/outputs/`, and `./confirmatory/logs/`.

## Design and frozen settings

This is a CPU-only TinyStories 15M GGUF fixture run, with target/draft Q2_K/Q3_K_M, 2 threads, batch sizes 1 and 2, eight fixed prompts split 4 tuning/4 held out, 64-token cap, 256-token context, common target sampling settings, and paired seed-root/prompt-ID inputs. Tuning used only T01–T04. No adaptive scheduling was implemented or evaluated. There is no speedup pass/fail gate.

| Batch | Method | Frozen setting | Tuning median aggregate tok/s |
|---:|---|---|---:|
| 1 | Autoregressive | target only | 211.01 |
| 1 | llama.cpp built-in chain | draft-simple n_max=2 | 157.48 |
| 1 | Pulsar fixed tree | branching=1, depth=2, node bound=3 | 70.33 |
| 2 | Autoregressive | target only | 249.58 |
| 2 | llama.cpp built-in chain | draft-simple n_max=2 | 228.65 |
| 2 | Pulsar fixed tree | branching=1, depth=2, node bound=3 | 62.84 |

## Held-out paired results

### Batch size 1

| Method | Aggregate tok/s | Total batch time (s) | Request latency p50 / p95 (ms) | Peak RSS (MiB) | Peak KV tokens / estimated bytes | Paired throughput ratio (95% CI) | Throughput assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| Pulsar autoregressive reference | 204.14 | 12.540 | 305.29 / 370.86 | 36.3 | target 97 / 670464 B | 1.000 (reference) | reference |
| llama.cpp draft-simple | 142.76 | 17.932 | 445.26 / 522.80 | 76.6 | max seq 97; live total n/a | 0.699 (0.669–0.732) | practical loss |
| Pulsar fixed tree | 64.61 | 39.620 | 964.62 / 1156.15 | 65.4 | target 193 / 1334016 B; draft 194 / 1340928 B | 0.317 (0.305–0.329) | practical loss |

| Method | Total batch-time ratio vs AR (95% CI) | Assessment | Request p50 latency ratio (95% CI) | Assessment | Request p95 latency ratio (95% CI) | Assessment |
|---|---:|---|---:|---|---:|---|
| llama.cpp draft-simple | 1.430 (1.367–1.495) | practical loss | 1.458 (1.385–1.536) | practical loss | 1.410 (1.321–1.594) | practical loss |
| Pulsar fixed tree | 3.159 (3.043–3.278) | practical loss | 3.160 (2.984–3.342) | practical loss | 3.117 (2.877–3.606) | practical loss |

Per-prompt request latency summaries (milliseconds; 40 held-out prompt/seed observations per method):

| Prompt | Autoregressive p50 / p95 | llama.cpp p50 / p95 | Pulsar tree p50 / p95 |
|---|---:|---:|---:|
| H01 | 294.94 / 369.52 | 411.94 / 487.53 | 948.90 / 1099.90 |
| H02 | 312.47 / 346.96 | 443.82 / 558.11 | 973.21 / 1172.79 |
| H03 | 316.50 / 370.29 | 457.76 / 521.56 | 985.72 / 1234.08 |
| H04 | 307.64 / 375.93 | 457.93 / 509.52 | 958.31 / 1053.27 |

Aggregate work-counter totals:

| Method | Target prefill / decode tokens | Draft prefill / decode tokens | Drafted / accepted tokens | Target verification tokens | Tree nodes / pulses | Tree build / verify inclusive ms | Cache copies / rollbacks (target + draft) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pulsar autoregressive | 1280 / 2520 | n/a / n/a | n/a / n/a | n/a | 0 / 0 | 0.0 / 0.0 | 0 / 0 |
| llama.cpp draft-simple | 1280 / n/a | n/a / n/a | 2465 / 1260 | n/a | 0 / 0 | 0.0 / 0.0 | n/a / n/a |
| Pulsar fixed tree | 1280 / 2520 | 1280 / 3205 | 2560 / 2121 | 2520 | 3245 / 1637 | 15935.5 / 11410.6 | 5725 / 3274 |

### Batch size 2

| Method | Aggregate tok/s | Total batch time (s) | Request latency p50 / p95 (ms) | Peak RSS (MiB) | Peak KV tokens / estimated bytes | Paired throughput ratio (95% CI) | Throughput assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| Pulsar autoregressive reference | 229.50 | 11.155 | 545.72 / 668.36 | 78.6 | target 192 / 1327104 B | 1.000 (reference) | reference |
| llama.cpp draft-simple | 182.42 | 14.033 | 692.50 / 781.78 | 83.3 | max seq 97; live total n/a | 0.795 (0.756–0.839) | practical loss |
| Pulsar fixed tree | 34.68 | 73.823 | 2443.78 / 8194.47 | 78.6 | target 288 / 1990656 B; draft 289 / 1997568 B | 0.151 (0.119–0.198) | practical loss |

| Method | Total batch-time ratio vs AR (95% CI) | Assessment | Request p50 latency ratio (95% CI) | Assessment | Request p95 latency ratio (95% CI) | Assessment |
|---|---:|---|---:|---|---:|---|
| llama.cpp draft-simple | 1.258 (1.192–1.323) | practical loss | 1.269 (1.157–1.346) | practical loss | 1.170 (1.115–1.346) | practical loss |
| Pulsar fixed tree | 6.618 (5.042–8.389) | practical loss | 4.478 (3.753–6.817) | practical loss | 12.261 (7.270–13.354) | practical loss |

Per-prompt request latency summaries (milliseconds; 40 held-out prompt/seed observations per method):

| Prompt | Autoregressive p50 / p95 | llama.cpp p50 / p95 | Pulsar tree p50 / p95 |
|---|---:|---:|---:|
| H01 | 547.92 / 667.94 | 646.17 / 762.72 | 2652.98 / 8029.81 |
| H02 | 550.82 / 670.96 | 654.46 / 794.13 | 2839.04 / 8261.75 |
| H03 | 539.53 / 637.33 | 708.64 / 766.71 | 2073.87 / 6468.29 |
| H04 | 541.20 / 639.05 | 714.25 / 793.52 | 2138.65 / 6571.75 |

Aggregate work-counter totals:

| Method | Target prefill / decode tokens | Draft prefill / decode tokens | Drafted / accepted tokens | Target verification tokens | Tree nodes / pulses | Tree build / verify inclusive ms | Cache copies / rollbacks (target + draft) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pulsar autoregressive | 1280 / 2520 | n/a / n/a | n/a / n/a | n/a | 0 / 0 | 0.0 / 0.0 | 0 / 0 |
| llama.cpp draft-simple | 1280 / n/a | n/a / n/a | 2459 / 1263 | n/a | 0 / 0 | 0.0 / 0.0 | n/a / n/a |
| Pulsar fixed tree | 1280 / 2520 | 1280 / 3205 | 2560 / 2121 | 2520 | 3245 / 1637 | 26061.8 / 18217.7 | 5725 / 3274 |

## Analysis correction

A post-run audit found that llama.cpp's raw `batch_index` is a run-global ordinal across seed roots, unlike the native per-seed ordinal. The initial paired analysis/report that keyed on `batch_index` is preserved under `attempt_failures/analysis_batch_index_pairing_bug/`. The corrected paired bootstrap uses the preregistered key `(seed_root, tuple(batch_prompt_ids))`, includes all 40 matched groups at batch size 1 and all 20 at batch size 2, and leaves every raw observation unchanged. The server's original `batch_index` remains in raw JSONL for traceability but is not used for pairing.

## Work counters and caveats

The raw JSONL is authoritative for batch observations. Native records expose target/draft prefill/decode work, tree node/pulse counts, candidate construction and verification time, sequence-cache copies/rollbacks, prompt/completion tokens, and inclusive batch timing. The llama.cpp server's /metrics endpoint was disabled for this run; exact proposed and accepted draft-token totals and maximum per-sequence lengths were recovered from retained per-task server timing logs, with the excluded warm-up task removed and log hashes recorded in the analysis JSON. Exact draft-context prefill/KV use, target verifier-step count, and cache copy/rollback operations are unavailable and remain null rather than inferred. Native KV bytes are an F16/equal-head-dimension estimate from live sequence lengths, not allocator telemetry; the server maximum sequence length is not total live KV occupancy. End-to-end batch wall time includes generation, verification, drafting, tree/cache work, response handling, and output record writes. Component timers overlap where documented in the protocol and must not be summed as disjoint stages.

## Reproducibility and retained data

`run_manifest.json` records model SHA-256s, source/upstream revisions, host/CPU/OS/build details and timestamps. `frozen_config.json` records tuning scores and raw-data hashes before any held-out run. Commands, process logs, warm-up records, prompt copies, generated outputs, per-batch raw JSONL, paired analysis JSON, and this report remain in the run directory. Bootstrap seed is 5555 with 10,000 paired cluster resamples. See [PHASE3_PROTOCOL.md](../../../../cpp/production/benchmark/PHASE3_PROTOCOL.md) for the predeclared margins, workloads, and timing contracts.

## Interpretation

Read results by workload, balancing throughput, total completion time, p50/p95 latency, memory, and exposed work counters. A practical win is not required for Phase 3 completion; practical ties, trade-offs, losses, or inconclusive estimates are all valid findings for this fixture.
