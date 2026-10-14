# Stage 5 — multi-workload grouped calibration and adaptivity

**Run ID: `stage5-20261013T131903Z-seed20261013`.** Artifacts are SHA-256 listed in `stage5_manifest.json`.

**Work status: COMPLETED. Formal exit gate: NOT_PASSED.** Stage 5 research work is complete; completion does not imply that every exit criterion passed. This suite covers three independently initialized synthetic order-3 target/order-2 draft Markov workloads; native C++17 policy timings are not transformer/KV-cache evidence.

## Closeout findings

- **Separate calibration finding: improved.** Pooled learned-minus-constant Brier was -0.050975 (95% CI [-0.084073, -0.013241]); learned-minus-draft-top was -0.083765 (95% CI [-0.126323, -0.043833]). Negative Brier differences favor the learned predictor. This is a separate finding, not a substitute for throughput.
- **Throughput exit criterion: NOT MET.** Paired speedup was -0.01% (95% CI [-0.58%, +0.56%]); the lower bound is not positive.

## Held-out grouped calibration

Pooled evaluation includes 64218 rows across 295 workload-context clusters.

| Predictor/baseline | Brier | ECE |
|---|---:|---:|
| Learned draft-only predictor | 0.157196 | 0.084108 |
| Per-workload constant train rate | 0.208171 | 0.091010 |
| Draft top-probability baseline | 0.240962 | 0.215297 |

Paired 95% context-cluster bootstrap intervals for learned-minus-baseline Brier (negative favors learned):
- Versus constant: -0.050975 [-0.084073, -0.013241].
- Versus draft-top: -0.083765 [-0.126323, -0.043833].

### Workload-level calibration (shown to expose heterogeneity)

| Workload | Learned Brier | Constant Brier | Draft-top Brier | Learned−constant 95% CI | Learned−draft-top 95% CI |
|---|---:|---:|---:|---:|---:|
| markov_seed_7 | 0.190717 | 0.251091 | 0.208114 | [-0.149316, 0.057150] | [-0.077970, 0.040126] |
| markov_seed_19_easy70 | 0.105474 | 0.140887 | 0.228376 | [-0.057921, -0.014910] | [-0.164986, -0.077318] |
| markov_seed_31_hard65 | 0.191321 | 0.253198 | 0.287097 | [-0.108400, -0.013659] | [-0.184775, -0.008674] |
Pooled calibration beats both declared baselines at the 95% cluster-confidence gate.
Per-workload intervals are not uniformly below zero; use the table above as a limitation, not as evidence that each workload individually passes calibration.

## Paired held-out throughput

| Workload | Scheduler mode | Tuned fixed policy | Fixed tokens/s | Confidence tokens/s | Depth decisions (1/2/4) | Speedup vs fixed (95% CI) |
|---|---|---|---:|---:|---:|---:|
| markov_seed_7 | fixed fallback | D4_B2_T0.00 | 1180216.5 | 1180216.5 | 990/1012/522940 | +0.00% [+0.00%, +0.00%] |
| markov_seed_19_easy70 | fixed fallback | D4_B4_T0.00 | 1558926.6 | 1558926.6 | 618/653/330409 | +0.00% [+0.00%, +0.00%] |
| markov_seed_31_hard65 | utility map | D2_B4_T0.00 | 998129.1 | 997840.9 | 6997/488681/75997 | -0.03% [-1.41%, +1.38%] |
| All paired test episodes | utility map or exact fixed fallback | workload-specific tuned actions | — | — | 8605/490346/929346 | -0.01% [-0.58%, +0.56%] |

The suite-wide paired interval uses workload/prompt strata and the same test seed episode for confidence and its workload's tuning-selected fixed comparator (4608 pairs; 9 strata). The per-workload intervals are shown separately to prevent pooled results from hiding a reversal.

At least one workload executed its tuning-selected confidence utility map on held-out test episodes.
At least one active utility map selected a deeper action than its tuned fixed baseline.

## Exit gate

- Calibration beats constant baseline: **True**.
- Calibration beats draft-top baseline: **True**.
- Calibration beats both baselines in every workload (diagnostic): **False**.
- Confidence scheduler beats tuned fixed with positive paired 95% lower bound: **False**.
- Confidence utility map executed on held-out episodes (diagnostic): **True**.
- Deeper action activation (diagnostic): **True**.
- Multiple independent workloads verified: **True** (3 distinct ToyLM seeds/configurations).

Blocking reasons:
- confidence-scheduler throughput lacks a positive 95% paired-bootstrap lower bound.

## Scope and reproduction

This suite tests independently initialized synthetic Markov workloads only. Fixed and confidence policies use an isolated native C++17 lazy path that calls the frozen v1 exact verifier; zero-threshold lazy/eager output parity is checked against the Stage 3 tree sampler. Calibration and audit traces remain Python reference work and are excluded from timing. It does not establish performance on transformer models, GPU execution, model-forward batching, or KV-cache costs. The compiled context-action lookup remains timed.

Run `python3 verification/experiments/test_stage5.py && python3 verification/experiments/run_stage5_suite.py`. Per-workload raw reports, JSON, node CSVs, and policy runs are preserved under `workloads/`; generated fixtures are under `fixtures/`. Combined CSV/JSON/report artifacts are in this directory.
