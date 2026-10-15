# Phase 3 benchmark protocol (predeclared)

Status: **finalized before the first measurement** (protocol revision 1.1; no tuning or confirmatory runs have started). No adaptive scheduler, learned policy, or Phase 4 feature is included.

## Objective and scope

Measure the cost profile of autoregressive decoding, llama.cpp's built-in `draft-simple` chain speculation, and fixed Pulsar candidate trees on one matched CPU/GGUF setup. This is a benchmark of the declared TinyStories 15M GGUF pair, not a claim about larger models or other CPU architectures. The AR path is the target-model reference. llama.cpp is pinned to the production runtime's recorded upstream revision.

## Frozen workload matrix

| Setting | Frozen value |
|---|---|
| Target / draft | `/tmp/pulsar-smoke.gguf` / `/tmp/pulsar-smoke-draft.gguf` (the same TinyStories 15M Q2_K / Q3_K_M pair used by Phase 2 smoke tests) |
| Device | CPU-only Linux; no GPU offload |
| Threads | 2 target threads; 2 draft threads |
| Batch sizes | 1 and 2 independent prompts; fixed grouping/order by prompt ID |
| Prompt groups | 4 tuning prompts (`T01`–`T04`), 4 held-out prompts (`H01`–`H04`) |
| Output horizon | 64 generated tokens maximum/request; EOS/context stops retained and recorded |
| Context capacity | 256 tokens/request |
| Sampling | temperature 0.8, top-k 40, top-p 0.95, repetition penalty 1.0 (disabled), repetition window 64 |
| Seeds | tuning roots 1701, 2801, 3907; confirmatory roots 5101, 6203, 7307, 8407, 9509, 10613, 11717, 12821, 13901, 15013 |
| Warm-up | one unrecorded batch using tuning prompt(s), before measured batches; model-load time excluded from per-batch latency and separately captured in process peak RSS |
| Prompt cache | disabled for the built-in server; each request starts with its full prompt and no prior request KV reuse |

The external per-request seed input is `root_seed + global_prompt_id` (all predeclared values fit the server's 32-bit seed field). The server receives that value directly; Pulsar deterministically applies its existing `splitmix64` seed derivation to the same root/ID pair before initializing its `mt19937_64`. The RNG implementations therefore do not produce identical streams, but every method receives the same external seed input. The built-in `draft-simple` proposal sampler is an upstream method-internal top-k-10 distribution (retuned to the request temperature/seed); the **final target sampling processor order and settings** are matched across methods. The server is launched with sampler order `penalties;temperature;top_k;top_p`, matching Pulsar's target order. Every method receives the same prompt IDs, external seed inputs, generation cap, target sampling settings, batch membership, and order.

## Tuning-only choices

Tune only using `T01`–`T04` and the tuning roots above. For each batch size independently:

- llama.cpp built-in `draft-simple`: `n_max` in `{2, 4, 6}`;
- Pulsar fixed tree: `(branching, depth)` in `{(1,2), (2,2), (2,3)}`;
- AR has no parameter to tune.

Select one configuration per method and batch size by the highest **median aggregate generated tokens/second across tuning batches**. Tie-break toward the smaller draft limit / smaller tree node bound. Freeze the selected configuration and its tuning raw data before evaluating `H01`–`H04`. Do not use held-out prompt timings, output text, or counters to change settings.

## Confirmatory comparison and predeclared margins

For each held-out batch, compute paired ratios against AR using the same prompt IDs, seeds, batch membership, output horizon, sampling processors, and model files. Report aggregate throughput, total batch completion time, per-prompt latency p50/p95, peak RSS, peak live KV tokens/estimated KV bytes, and work counters.

A **practical throughput win** is at least 5% higher aggregate tokens/s and a paired 95% bootstrap confidence interval for the throughput ratio wholly above 1.00. Aggregate throughput is total output tokens divided by summed batch wall time; bootstrap resamples matched batch groups (same seed root and prompt group) with replacement and recomputes the ratio of the two aggregate rates (10,000 draws, fixed analysis seed 5555, percentile interval). A **practical latency win** is at least 5% lower batch-completion or per-prompt latency and a paired 95% bootstrap interval for the latency ratio wholly below 1.00; latency intervals resample matched batch groups and retain all contained prompt/seed observations, preserving within-batch pairing (the same method reports p50 and p95 intervals). Ratios inside 0.95–1.05 are a practical tie for that metric; opposite-sign throughput and latency outcomes are reported as a trade-off, not collapsed into one score. Results outside these rules but with intervals crossing 1.00 are inconclusive. These are interpretation margins only; Phase 3 has **no speedup pass/fail gate**.

## Timing and instrumentation contract

- Native Pulsar batch wall time starts immediately before generation and ends after result JSON serialization, file write, and flush. Model load is outside this per-batch timer. Per-prompt latency is the runtime's completion timestamp relative to batch generation start.
- Built-in llama.cpp latency is client-observed request time from dispatch through complete JSON response receipt. For a batch, requests are dispatched concurrently and batch completion is the time until every response arrives. This includes server generation, response serialization, and loopback HTTP transport.
- Peak process RSS includes model load and the full benchmark process/server lifetime. Peak live KV is measured from the runtime's sequence-position bookkeeping; KV bytes are reported as an explicitly labeled F16/equal-head-dimension estimate for this LLaMA-family fixture. This estimate is not a direct allocator measurement.
- Native Pulsar instrumentation records target/draft prefill and decode call/token counts, proposal/accept/reject counts, target verification tokens, sequence-cache copy/rollback/reset counts and their inclusive durations, and peak live KV tokens. Candidate-tree proposal/node counts cover tree management work; complete generation wall time includes verifier, tree management, sampling, string construction, and output writing.
- llama.cpp built-in `draft-simple` records the pinned server's prompt/generation timing and exported draft-token, accepted-token, and verification-step counters. Its public metrics do not expose exact per-request target decode-call count, cache copy/rollback count, or draft-context KV occupancy; those raw fields are `null` with a reason, never inferred as exact. The total wall time still includes those operations. The report separates exposed counters from unavailable ones.
- Run output includes raw per-batch observations, server metrics snapshots/deltas, commands/configuration, model SHA-256 hashes, upstream revision, compiler/build identity, and host CPU/OS metadata. Do not delete or replace raw observations after analysis.

## Prompt manifest

See `prompts.jsonl`. It is fixed before measurement. Prompts are short, general story-writing contexts chosen to match the TinyStories fixture domain; held-out prompts are not used for configuration selection.
