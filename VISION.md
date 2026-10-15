# Pulsar — Working Vision v0.2

Locked by the project owner on 2026-10-15.

## Vision

Build Pulsar as a next-generation inference engine for offline, batched general text generation. It explores multiple candidate continuations, decides where to spend computation, and verifies emitted tokens against the target model’s sampling law. The goal is not to replace the target model’s intelligence, but to help it produce the same intended outputs with better use of compute.

## Core loop

A draft model proposes candidate paths; a scheduler allocates a bounded amount of work across them; an exact verifier preserves the target output distribution; the engine records speed, latency, memory, and calibration evidence.

## Components

1. **Candidate generator:** creates candidate paths.
2. **Pulsar scheduler:** turns “rotate and pulse” into a concrete, testable selection rule.
3. **Exact verifier:** protects correctness.
4. **Calibration monitor:** compares confidence with actual verification outcomes.
5. **Backend adapter:** connects real target/draft models and their caches.
6. **Training or fine-tuning:** a later, separate track for the base model.

## Success criteria

Evaluate matched batches on the same models, prompts, token budget, and hardware against autoregression, tuned chain speculation, and fixed-tree policies. Require exactness tests to pass. Report aggregate tokens per second, total batch completion time, per-prompt p50/p95 latency, and memory. Set acceptable throughput/latency trade-off margins before each experiment; do not hide one metric behind another. Track language quality separately on a held-out evaluation set.

## Research discipline

The scheduler’s speedup is a hypothesis, not an assumption. Current Stage 6 preserved the tested target sampling law but did not establish scheduler acceleration; its small CPU experiment does not prove performance for this future batch workload. Quantum language remains an analogy for multiple probability-weighted paths unless an actual quantum method is implemented and measured. “Listening to its own words” means calibration and monitoring, not consciousness. There is no absolute “perfect model”; define quality targets for a use case and report where the system succeeds or fails.
