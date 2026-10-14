# PULSAR — Superposition-Decode Inference Engine

> **Normative v1 core:** see [`algorithm_v1.md`](algorithm_v1.md). The
> integrated research engine and paper describe optional experiments; they do
> not override the frozen v1 contract.
>
> **Stage 2 speculative chain:** see [`algorithm_v2_speculative.md`](algorithm_v2_speculative.md).
>
> **Stage 3 bounded candidate tree:** see [`algorithm_v3_tree.md`](algorithm_v3_tree.md).

A verified mathematical design + reference implementation + verification
harness for **Pulsar**: a fast inference engine that holds multiple candidate
continuations in an explicit superposed field, sweeps it with a trainable
confidence rotor, and *collapses* (measures) only the mass-dense regions via
a calibrated, self-consistent pulse threshold — with a **provably exact**
verifier, so speed never moves the output distribution away from the base
model.

## Repository map

```
src/pulsar/            reference Python implementation (numpy only)
  model.py             ToyLM: Markov target/draft pair with EXACT ground truth
  field.py             M1 M2  candidate generator + superposition field
  verifier.py          M3     exact rejection-sampling verifier (Theorem 1)
  rotor.py             M4a    stratified rotating readout + circulant mixing
  confidence.py        M4c    trainable logistic + isotonic calibrator; LIF
  pulse.py             M4b    water-level pulse selection; theta fixed point
  collapse.py          M4d    Born/Gumbel/temperature collapse
  calibrator.py        M7     threshold learners (EMA fixed point, REINFORCE),
                               calibration diagnostics, TV identity
  engine.py            M5 M6  end-to-end engine + pulse clock (all modules)
verification/
  theory_checks.py             11 analytic/brute-force theorem checks (T01-T11)
  experiments/run_all.py       21 end-to-end claims (E00-E11) + figures +
                               results.json + C++ model dump
cpp/                   C++17 header-only engine + v1 verifier and acceptance tests
  pulsar_engine.hpp            port of the verified Python engine
  pulsar_core_v1.hpp           generic-URBG exact verifier (p, effective pi)
  pulsar_core_v1_test.cpp      algebra, randomized laws, Markov + MC bounds
  pulsar_speculative_v2.hpp    context-aware q-drafted speculative blocks + AR reference
  pulsar_speculative_v2_test.cpp rejection positions, bonus, conditional and sequence tests
  pulsar_tree_v3.hpp           bounded prefix tree + exact fall-through verifier
  pulsar_tree_v3_test.cpp      bounds, off-tree eligibility, stale-context and law tests
  pulsar_stage5_native.cpp     Stage 5-only lazy/eager policy timer; reuses frozen v1 verifier
  pulsar_test.cpp              exactness / speedup / bound / theta checks
  inputs/model.bin             dumped by run_all.py (toy tables + M4c weights)
math/README.md         module-by-module math spec (compact)
docs/                  design + verification log
paper/PULSAR.md        the paper (markdown)
paper/latex/           LaTeX source
paper/figures/         generated figures
paper/results/         results.json + logs
```

## Reproduce everything

```bash
python3 verification/theory_checks.py           # 11/11 expected, exit 0
python3 verification/experiments/run_all.py     # 21/21 claims, ~2.5 min
cd cpp && make test                             # v1 core + Stage 2 + engine acceptance
```

## Headline verified results (toy LM, analytic cost model)

| claim | number | where |
|---|---|---|
| Output exactness, 11 collapse policies incl. adversarial | max TV 0.0083 (MC floor ~0.011) | E01 |
| Sequence-level indistinguishability (unigram/bigram χ² vs exact stationary) | p = 0.25 / 0.016 | E01 |
| Per-node acceptance == 1 − TV (draft-vs-target divergence) | z = −0.28 / −0.13 | E02 |
| Speedup vs AR / vs best chain speculation | 1.388× / 1.258× | E03 |
| Trained M4c reaches oracle-ceiling fraction | 97.5% | E03/E04 |
| M4c calibration after isotonic head | ECE 0.0007, slope 0.9995 | E04 |
| Water-level block == provably optimal (brute-force, slack ~1e-16) | T05 | theory |
| θ fixed point self-tunes into optimum | θ→0.0575 vs θ*=0.07 | E05 |
| Fall-through vs renormalized-tree collapse ablation | 1.388 vs 1.079 | E03/E06 |
| REINFORCE recovers engine-measured optimum exactly | 1.3863 = 1.3863 | E10 |
| M6 deterministic work bound holds | max 1.58 ≤ 8.84 | E08 |
| C++ engine parity with Python verification | speedup 1.387, TV 0.0016 | cpp |

## Status & honesty notes

* The speedup numbers use an *analytic cost model* on a purpose-built exact
  Markov testbed — they verify the **mathematics and system logic**, not GPU
  wall-clock. Porting `cpp/pulsar_engine.hpp` tree logic into a vLLM/SGLang
  style engine with real tree attention is the intended next step.
* "Quantum" language (superposition, Born rule, collapse) denotes the exact
  probability-simplex structure proven in `math/README.md` — diagonal density
  operators only; no quantum hardware or interference is claimed.
* Related-work formulas reproduce published numbers (see paper §8 table), so
  Pulsar's projections are anchored to real-model acceptance rates.


## Stage 4 — measured fixed-policy costs (synthetic fixture only)

Stage 4 adds an instrumented CPU benchmark of the existing `cpp/inputs/model.bin` Markov toy fixture. It compares the plain AR baseline with a fixed tree grid over depth, branching, and a benchmark-only minimum retained-level-mass threshold. The definitions, paired-workload contract, metrics, and limitations are in [`algorithm_v4_measurements.md`](algorithm_v4_measurements.md).

Reproduce the full grid and plots:

```bash
mkdir -p paper/results/stage4
cd cpp
make pulsar_stage4_bench CXXFLAGS='-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror'
./pulsar_stage4_bench --model inputs/model.bin --tokens 512 --repeats 3 --raw ../paper/results/stage4/stage4_raw.csv
cd ..
python3 verification/experiments/summarize_stage4.py --raw paper/results/stage4/stage4_raw.csv --out-dir paper/results/stage4
```

The reported wall time is actual CPU time for the C++ toy-fixture path. It does **not** measure a real neural model's draft/target forward pass or KV-cache cost. The outputs include raw runs, aggregate metrics, cost/yield and throughput charts, and a tree-size/verifier-cost proxy chart. No scheduler is trained, and no global-optimality claim is made from a cost definition that differs from these measurements.

The measured run summary is [`paper/results/stage4/stage4_report.md`](paper/results/stage4/stage4_report.md); raw runs, aggregate CSV, metadata, and plots are in the same directory.


## Stage 5 — grouped confidence and independent adaptivity experiments

Stage 5 logs draft/context-side features with exact verifier labels, calibrates by held-out last-two-token context groups, and reports Brier/ECE against both a constant and draft-top baseline. Fixed policies and the confidence action map are selected on tuning seeds; frozen policies are compared on fresh paired test episodes. Python collects calibration/audit traces; the gated fixed/confidence policy measurements use an isolated native C++17 runner that reuses the frozen v1 verifier and is checked against eager Stage 3 pulse outputs. The multi-workload suite creates three independent synthetic Markov models and aggregates held-out calibration and throughput evidence. Confidence, oracle, EMA, and UCB1 remain independent policies. See [`algorithm_v5_adaptivity.md`](algorithm_v5_adaptivity.md) for the protocol and gate.

**Current closeout — run `stage5-20261013T131903Z-seed20261013`: completed; throughput exit gate not met.** The paired confidence-scheduler speedup is −0.01% (95% CI [−0.58%, +0.56%]), so its lower confidence bound is not positive. As a separate finding, pooled learned calibration improves Brier versus the constant baseline by −0.050975 (95% CI [−0.084073, −0.013241]) and versus draft-top by −0.083765 (95% CI [−0.126323, −0.043833]); these calibration gains do not satisfy or substitute for the throughput criterion. The run is limited to synthetic Markov workloads and native C++17 policy timings, not transformer runtime, GPU batching, model-forward cost, or KV-cache workload. Stage 0–4 are unchanged.

Run the Stage 5 checks and experiment:

```bash
python3 verification/experiments/test_stage5.py
make -C cpp stage5_native_test CXXFLAGS='-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror'
python3 verification/experiments/run_stage5.py
python3 verification/experiments/run_stage5_suite.py
```

Outputs go to `paper/results/stage5/`; these include combined node-level CSVs, paired per-run policy timings, JSON metrics, a report, and per-workload raw evidence. Defaults are 20 calibration, 4 tuning, and 8 held-out test repeats per prompt. The pre-upgrade Python-only results are preserved in `paper/results/stage5_python_baseline/`. The formal gate definition is unchanged: only a positive paired 95% throughput lower bound against tuning-selected workload-specific fixed policies can pass the throughput criterion. The Stage 5 C++ runner remains a synthetic Markov reference; its numbers are not comparable to Stage 4's eager-tree measurements.


## Stage 6 — one real inference backend and falsifiable tree-scheduler claim

Stage 6 adds a Hugging Face Transformers 4.46.x adapter (`src/pulsar/transformers_adapter.py`) for actual target/draft logits, Stage 3-style exact tree verification, draft expansion, and KV-cache updates. It uses `distilgpt2` / `sshleifer/tiny-gpt2`; the target law is the model's temperature-1 softmax with no additional logits processors. The matched harness compares autoregression, tuning-selected chain speculative decoding, a tuning-selected fixed tree, and a draft-max-probability tree scheduler. Model-loading, runtime, exactness, cache, tuning, statistical gate, and limitations are specified in [`algorithm_v6_backend.md`](algorithm_v6_backend.md).

Reproduce from this directory:

```bash
python3 verification/experiments/test_stage6_transformers.py
python3 verification/experiments/run_stage6_transformers.py
```

The Stage 6 claim passes only if the actual-row exactness checks pass and the held-out paired 95% CI lower bound for confidence-tree versus fixed-tree throughput is positive. This experiment is CPU-only, batch size 1, three held-out prompt clusters, and uses level-batched full-context target rows (no fused tree-attention kernel); it is not production-serving or GPU evidence. Stage 5 remains closed with throughput gate unmet; its calibration result is separate and unchanged.


**Stage 6 closeout — run `stage6-20261013T142914Z-seed60613`: exactness passed; scheduler throughput claim not established.** Full-vocabulary pointwise error was `1.08e-19`. Held-out throughput: AR 40.62 tokens/s; tuned chain (γ=2) 15.71; best fixed tree 10.98; confidence tree 11.17. The confidence map selected the fixed action in every bin; its +1.66% point estimate vs fixed tree had 95% CI [−6.71%, +10.01%]. Chain speculation beat the fixed tree by +43.00% [34.87%, 50.79%], but was slower than AR. This is a CPU-only result over three held-out prompts, not evidence of general serving speedup. Detailed report and raw evidence: [`paper/results/stage6_transformers/stage6_report.md`](paper/results/stage6_transformers/stage6_report.md).
## License

Pulsar is distributed under the GNU Affero General Public License, version 3 (AGPL-3.0). See [`LICENSE`](LICENSE) for the full license text.
