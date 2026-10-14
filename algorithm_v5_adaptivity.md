# Pulsar Stage 5 — grouped acceptance calibration and adaptivity

Stage 5 is an experiment layered alongside Stages 0–4. It does not change the exact verifier, Stage 3 tree semantics, or Stage 4 artifacts. Python collects calibration labels and audit traces; fixed/confidence policy timing uses an isolated C++17 Stage 3-semantic runner that calls the frozen v1 verifier. This remains a synthetic research harness, not a transformer runtime or a replacement for Stage 4's eager-tree C++ benchmark.

## Experimental contract

- **Labels:** each example is a node actually visited by the Stage 3-semantic sampler. Its binary label is the exact verifier outcome for the sampled draft proposal (`accepted` or `rejected`), before residual replacement. It is a realized verifier label, not analytically computed overlap.
- **Features:** `pulsar.stage5.draft_features` records six base draft/context-only features (bias, normalized draft entropy, draft top probability, top-two margin, a context-token proxy, and verifier depth) plus 13 deterministic polynomial interactions among entropy, top probability, margin, context proxy, and depth. It never reads the target row. Context, pulse, prompt, and depth identifiers are retained for audit.
- **Calibration split:** rows are partitioned deterministically by last-two-token draft-context group. Repeated observations from one context cannot cross train/evaluation partitions. Report Brier score and expected calibration error (ECE) overall, by verifier depth, by low/high target–draft overlap regime, and on actual nodes visited in a held-out confidence-scheduler rollout.
- **Calibration baselines and uncertainty:** compare the learned predictor to both the constant train acceptance rate and the draft-top-probability baseline. Report paired 95% cluster-bootstrap intervals for learned-minus-baseline Brier differences, resampling whole held-out context groups; negative favors the learned predictor. The multi-workload report includes pooled intervals for the gate and each workload's Brier scores/intervals so a pooled gain cannot hide heterogeneous models.
- **Policy split:** all 27 Stage 4 grid settings (`depth ∈ {1,2,4}`, `branch ∈ {1,2,4}`, retained-level threshold `∈ {0,.2,.4}`) are measured on tuning episodes. The first tuning half fits score-conditioned action utility from per-pulse emitted-token/time traces; the disjoint second half selects the fixed comparator and checks whether the fitted adaptive map beats it. Fresh test episodes determine the gate. The test-set grid winner is descriptive only and cannot replace the frozen comparator.
- **Matched policy test:** policies share each test prompt/seed episode and token limit, with deterministic randomized run order. Report paired 95% throughput intervals stratified by workload and prompt. Prompts are strata within a workload, not independent workloads.
- **Confidence policy:** draft-only calibrated root scores are divided into up to four quantile bins, with bin edges computed without target labels. To limit policy flexibility, the confidence map varies **depth only** and holds branch factor and retained-level threshold at the workload's tuning-selected fixed action. Each bin selects among those three depth choices using measured tokens-per-nanosecond utility. Sparse bin rates shrink toward each action's global tuning rate; a context-group/bin refinement can replace the bin action only with at least 20 pulse observations and a shrunk utility gain of at least 0.5%. Utility fitting uses the first tuning half, while the map and tuned fixed baseline are compared on the disjoint second half. If the adaptive map is not faster there, the policy runs the fixed action unchanged and reuses those exact paired timings; otherwise the compiled map is applied on held-out calibration contexts and falls back to fixed outside them. The native runner can emit per-pulse context/action/emission/time traces after completing each timed pulse; trace I/O and utility fitting are outside the timed interval.
- **Oracle comparator:** feed exact `sum_x min(p(x),q(x))` into the same tuning-fitted action-utility bins in place of the learned estimate. This is a privileged diagnostic, not a theorem that total scheduler performance is upper-bounded.
- **Timing objective:** compare aggregate emitted tokens per second in one native C++17 path. Timed episodes include compiled action-table lookup, threshold selection/tree work, exact verification, sampling, and output/context updates; they exclude model loading, top-child/action-table compilation, separate audit replay, and neural model forward/KV-cache costs. For threshold-zero actions, lazy execution follows only the verifier-eligible path; regression tests compare it pulse-by-pulse with eager Stage 3 tree sampling under identical C++ RNG seeds. Nonzero retained-level thresholds use the Stage 3 tree builder and Stage 4 selector. These Stage 5 native timings are not comparable to Stage 4's eager-tree measurements.

## Independent online policies

`src/pulsar/stage5.py` specifies each controller independently; do not combine their state with the confidence predictor. The native runner times confidence, oracle, EMA, and UCB1 through the same verifier/tree path:

1. **Confidence scheduler:** stateless action selection from a fitted draft-only probability bin and a cross-fitted measured-utility action map.
2. **EMA threshold policy:** state `m` starts at a configured value and updates after each verifier label as `m ← (1−α)m + αy`. A fixed threshold map converts `m` to a tree action.
3. **UCB1 bandit:** chooses among fixed actions by `mean_reward + sqrt(c log(total_updates)/count)`. It updates from an explicit measured scalar reward; the runner uses clipped per-pulse throughput. Its state is separate from the EMA and predictor.

The runner times confidence, exact-overlap oracle, EMA, and UCB1 independently. EMA and UCB1 are online controls, not calibrated predictors.

## Reproduce

From the project root:

```bash
python3 verification/experiments/test_stage5.py
make -C cpp stage5_native_test CXXFLAGS='-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror'
python3 verification/experiments/run_stage5.py
python3 verification/experiments/run_stage5_suite.py
```

Defaults are 128 emitted tokens/run, 20 calibration repeats, 4 policy-tuning repeats, 8 held-out test repeats per prompt, and 2,000 bootstrap samples. Override them with `--tokens`, `--calibration-repeats`, `--tuning-repeats`, `--evaluation-repeats`, `--bootstrap-samples`, and `--seed`. The single-fixture runner defaults to `--backend native`; `--backend python` reproduces the former Python timing path for diagnosis only. The single-fixture runner requires at least two test repeats per prompt; use `run_stage5_suite.py` for the multi-workload exit gate.

Outputs are isolated under `paper/results/stage5/`:

- `stage5_nodes.csv`: combined draft-only features, exact verifier labels, workload/context groups, split, and predicted acceptance probabilities.
- `stage5_scheduler_nodes.csv`: actual held-out-context nodes from separate confidence-scheduler audit replays (not timed).
- `stage5_policy_runs.csv`: per-episode tuning and test measurements, seeds, timings, and backend labels; the gated fixed/confidence comparison uses the same native backend.
- `stage5_results.json`: pooled and per-workload calibration/policy evidence and fail-closed gate status.
- `stage5_report.md`: concise interpretation and limitations; per-workload reports and raw artifacts remain under `workloads/`, generated fixtures under `fixtures/`.

## Exit gate and scope

A Stage 5 research pass requires all of the following: (1) pooled held-out learned Brier improvement over **both** declared draft-only baselines with 95% workload-context-cluster intervals below zero; (2) suite-wide confidence-scheduler throughput improvement over the workload-specific tuning-selected fixed comparators with a positive 95% paired test interval; and (3) verification across multiple independent models or workloads. Missing any condition is **NOT PASSED**; point estimates alone do not pass the gate. Per-workload calibration and throughput intervals remain visible even though these pooled comparisons determine the two statistical gate fields.

The suite creates three independently seeded/configured synthetic order-3 target/order-2 draft Markov workloads, which can satisfy the multi-workload condition only within this synthetic scope. It does not validate a transformer runtime, GPU batching, or KV-cache costs. The current run (`stage5-20261013T131903Z-seed20261013`) is **completed as research work, but its formal exit gate is NOT PASSED**: the pooled calibration criteria pass and multiple synthetic workloads are verified, but the suite-wide confidence-scheduler throughput estimate is −0.01% (95% CI [−0.58%, +0.56%]), so its lower bound is not positive. Record the calibration improvement as a separate finding, not as a substitute for the throughput criterion. The gate definition remains unchanged: a research pass requires calibration to beat both baselines, a positive paired 95% throughput lower bound against workload-specific tuning-selected fixed comparators, and multiple workloads verified. Never turn a point estimate into a pass. Stage 0–4 contracts and Stage 4 artifacts remain untouched.
