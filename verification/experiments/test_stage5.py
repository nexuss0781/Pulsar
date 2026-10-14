#!/usr/bin/env python3
"""Focused tests for the independent Stage 5 calibration and policy components."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_stage5 import (ToyModel, fit_bin_actions, precompute_action_lookup,
                        precompute_binned_action_lookup, precompute_root_confidence,
                        run_pulse, verifier_step)  # noqa: E402
from pulsar.stage5 import (  # noqa: E402
    AcceptancePredictor, ConfidenceScheduler, EMAThresholdPolicy, NodeExample,
    TreeAction, UCB1BanditPolicy, brier_score, calibration_report,
    clustered_brier_delta, draft_features, paired_bootstrap_speedup,
    selected_mask, split_by_groups,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class SequenceRng:
    def __init__(self, values: list[float]):
        self._values = iter(values)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        try:
            return next(self._values)
        except StopIteration as error:
            raise AssertionError("verifier requested an unexpected random draw") from error


def test_verifier_equal_laws_never_enters_residual_branch():
    p = np.asarray([0.25, 0.75])
    q = p.copy()
    rng = SequenceRng([0.0, 0.5])
    result = verifier_step(p, q, rng)
    require(result == (0, True, 0), "equal laws must accept the sampled proposal")
    require(rng.draws == 2, "equal laws must not draw from the zero-mass residual")


def test_verifier_samples_representable_small_positive_residual():
    p = np.asarray([0.5 - 1e-16, 0.5 + 1e-16])
    q = np.asarray([0.5, 0.5])
    residual_mass = float(np.maximum(p - q, 0.0).sum())
    require(0.0 < residual_mass < 1e-15,
            "fixture must have a representable positive residual below the old cutoff")
    # Propose token 0, reject at the representable acceptance boundary, then
    # sample the residual. Its only positive entry is token 1.
    rng = SequenceRng([0.0, 0.9999999999999999, 0.0])
    result = verifier_step(p, q, rng)
    require(result == (1, False, 0),
            "small positive residual must be normalized and sampled, not replaced by p")
    require(rng.draws == 3, "rejection must consume exactly one residual-sampling draw")


def test_lazy_zero_threshold_pulse_matches_eager_tree():
    model = ToyModel(ROOT / "cpp/inputs/model.bin")
    for depth, branch in ((1, 1), (2, 1), (4, 1), (2, 2), (4, 4)):
        action = TreeAction(depth, branch, 0.0, f"D{depth}_B{branch}")
        for seed in range(5):
            eager_examples, lazy_examples = [], []
            eager = run_pulse(model, (5, 3, 9), action, np.random.default_rng(seed),
                              "prompt", f"pulse-{seed}", eager_examples, eager_tree=True)
            lazy = run_pulse(model, (5, 3, 9), action, np.random.default_rng(seed),
                             "prompt", f"pulse-{seed}", lazy_examples)
            require(eager == lazy,
                    f"lazy/eager pulse mismatch for depth={depth}, branch={branch}, seed={seed}")
            require(eager_examples == lazy_examples,
                    "lazy traversal changed verifier labels or node audit features")


def test_compiled_confidence_action_lookup_matches_live_policy():
    model = ToyModel(ROOT / "cpp/inputs/model.bin")
    train_x = np.asarray([draft_features([0.8, 0.2], [0, 1, 2], 0, 20),
                          draft_features([0.2, 0.8], [3, 4, 5], 0, 20)])
    predictor = AcceptancePredictor(epochs=200).fit(train_x, [0, 1])
    confidence = precompute_root_confidence(model, predictor)
    actions = [TreeAction(1, 2, 0.0, "D1"), TreeAction(2, 1, 0.0, "D2"),
               TreeAction(4, 1, 0.0, "D4")]
    scheduler = ConfidenceScheduler(actions, [0.4, 0.7])
    held = {model.context_group((5, 3, 9))}
    fallback = TreeAction(1, 4, 0.0, "fallback")
    table = precompute_action_lookup(model, confidence, held, scheduler, fallback)
    for context in ((5, 3, 9), (0, 0, 0), (2, 4, 7)):
        expected = (scheduler.select(float(confidence[model.ctx3(context)]))
                    if model.context_group(context) in held else fallback)
        require(table[model.ctx3(context)] == expected,
                "compiled action table differs from the live confidence/fallback rule")


def test_measured_utility_bins_select_only_supported_faster_actions():
    model = ToyModel(ROOT / "cpp/inputs/model.bin")
    context = (5, 3, 9)
    held = {model.context_group(context)}
    state = model.ctx3(context)
    actions = (TreeAction(1, 1, 0.0, "D1_B1_T0"),
               TreeAction(2, 1, 0.0, "D2_B1_T0"))
    scores = np.full(model.v ** 3, 0.25)
    edges = np.asarray([0.5])
    fallback = actions[0]
    with tempfile.TemporaryDirectory() as temporary:
        trace = Path(temporary) / "pulse_trace.csv"
        rows = ["context3,depth,branch,threshold,emitted_tokens,elapsed_ns"]
        rows.extend(f"{state},1,1,0,1,100" for _ in range(20))
        rows.extend(f"{state},2,1,0,2,100" for _ in range(20))
        trace.write_text("\n".join(rows) + "\n")
        actions_by_bin, diagnostics, group_actions, group_diagnostics = fit_bin_actions(
            [trace], model, scores, held, edges, actions, fallback)
    require(actions_by_bin[0] == actions[1],
            "the supported action with higher measured utility should win its score bin")
    require(actions_by_bin[1] == fallback,
            "an unsupported score bin must use the fixed fallback")
    lookup = precompute_binned_action_lookup(model, scores, held, edges,
                                             actions_by_bin, fallback, group_actions)
    require(lookup[state] == actions[1], "compiled utility bin must select its fitted action")
    other_state = model.ctx3((0, 0, 0))
    require(lookup[other_state] == fallback,
            "contexts outside held-out calibration groups must use the fixed fallback")
    require(diagnostics[0]["selected_by_local_utility"],
            "utility diagnostics must record local action selection")
    require(isinstance(group_diagnostics, list),
            "context-group utility diagnostics must be serializable records")


def test_group_split_no_context_leakage():
    rows = [NodeExample((1.0, float(i % 2)), i % 2, f"p{i % 2}", f"ctx{i % 4}",
                        f"pulse{i // 2}", i % 3, "low") for i in range(40)]
    train, test = split_by_groups(rows, 0.25, seed=19)
    train_groups = {rows[i].context_group for i in train}
    test_groups = {rows[i].context_group for i in test}
    require(train_groups.isdisjoint(test_groups), "context groups crossed calibration split")
    train2, test2 = split_by_groups(rows, 0.25, seed=19)
    require(np.array_equal(train, train2) and np.array_equal(test, test2),
            "group split must be reproducible")


def test_predictor_and_weighted_calibration():
    x = np.asarray([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.asarray([0, 0, 1, 1])
    model = AcceptancePredictor(learning_rate=0.1, epochs=500).fit(x, y)
    pred = model.predict(x)
    require(pred[0] < pred[-1], "confidence should follow separable labels")
    require(brier_score(pred, y) < 0.1, "expected low Brier on separable fixture")
    weighted = calibration_report([0.2, 0.8], [0, 1], weights=[1.0, 3.0])
    require(weighted["brier"] < 0.04, "weighted Brier should use scheduler-node weights")


def test_draft_features_ignore_target_and_are_finite():
    features = draft_features([0.6, 0.3, 0.1], [4, 5, 6], depth=2, vocab_size=10)
    require(len(features) == 19 and all(np.isfinite(features)), "feature vector malformed")
    require(features[0] == 1.0 and features[5] == 2.0, "bias/depth base features missing")
    require(abs(features[6] - features[1] ** 2) < 1e-12,
            "draft-only polynomial interaction is incorrect")


def test_confidence_and_selected_distribution():
    actions = [TreeAction(1, 2, 0.0), TreeAction(2, 2, 0.0), TreeAction(4, 2, 0.0)]
    scheduler = ConfidenceScheduler(actions, [0.5, 0.8])
    require(scheduler.select(0.4).depth == 1 and scheduler.select(0.6).depth == 2 and
            scheduler.select(0.9).depth == 4, "confidence threshold mapping is wrong")
    nodes = [
        NodeExample((1.0,), 1, "p", "ctx1", "pulse1", 0, "high"),
        NodeExample((1.0,), 1, "p", "ctx2", "pulse1", 1, "high"),
        NodeExample((1.0,), 0, "p", "ctx3", "pulse1", 2, "low"),
        NodeExample((1.0,), 1, "p", "ctx4", "pulse2", 0, "high"),
        NodeExample((1.0,), 0, "p", "ctx5", "pulse2", 1, "low"),
    ]
    mask = selected_mask(nodes, [0.9, 0.9, 0.9, 0.6, 0.6], scheduler)
    require(mask.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0],
            "selected mask must retain action depth per pulse")


def test_ema_and_bandit_have_separate_state_updates():
    actions = [TreeAction(1, 1, 0.0), TreeAction(4, 1, 0.0)]
    ema = EMAThresholdPolicy(actions, [0.6], alpha=0.5, initial=0.5)
    require(ema.select().depth == 1, "EMA starts with configured state")
    ema.update(1)
    require(abs(ema.ema - 0.75) < 1e-12 and ema.select().depth == 4,
            "EMA state/update is incorrect")
    bandit = UCB1BanditPolicy(actions)
    require(bandit.select_index() == 0, "UCB1 must initialize arms in stable order")
    bandit.update(0, 0.1)
    require(bandit.select_index() == 1, "UCB1 must try an unobserved arm")
    require(ema.updates == 1 and bandit.total_updates == 1,
            "adaptive policies must maintain independent update state")


def test_cluster_bootstrap_detects_brier_improvement_and_is_reproducible():
    labels = np.asarray([0, 1] * 20)
    learned = np.where(labels == 1, 0.9, 0.1)
    baseline = np.full(len(labels), 0.5)
    groups = [f"ctx{i // 2}" for i in range(len(labels))]
    first = clustered_brier_delta(learned, baseline, labels, groups, samples=500, seed=7)
    second = clustered_brier_delta(learned, baseline, labels, groups, samples=500, seed=7)
    require(first["estimate"] < 0 and first["ci95_high"] < 0,
            "cluster bootstrap should recognize an unambiguous Brier improvement")
    require(first == second, "cluster bootstrap must be deterministic for a fixed seed")
    rejected_invalid = False
    try:
        clustered_brier_delta(learned, baseline, np.full(len(labels), 0.5), groups,
                             samples=100, seed=7)
    except ValueError:
        rejected_invalid = True
    require(rejected_invalid, "bootstrap helper must reject non-binary labels")


def test_paired_bootstrap_compares_untouched_policy_episodes():
    baseline_tokens = [100] * 8
    candidate_tokens = [100] * 8
    baseline_wall = [1000, 1010, 990, 1005, 1002, 998, 1008, 997]
    candidate_wall = [750, 760, 745, 755, 748, 752, 758, 747]
    groups = ["prompt-a"] * 4 + ["prompt-b"] * 4
    result = paired_bootstrap_speedup(candidate_tokens, candidate_wall,
                                      baseline_tokens, baseline_wall, groups,
                                      samples=500, seed=9)
    require(result["estimate_percent"] > 25 and result["ci95_low_percent"] > 0,
            "paired cluster interval should show a reliable throughput improvement")


def main():
    tests = [test_verifier_equal_laws_never_enters_residual_branch,
             test_verifier_samples_representable_small_positive_residual,
             test_lazy_zero_threshold_pulse_matches_eager_tree,
             test_compiled_confidence_action_lookup_matches_live_policy,
             test_measured_utility_bins_select_only_supported_faster_actions,
             test_group_split_no_context_leakage, test_predictor_and_weighted_calibration,
             test_draft_features_ignore_target_and_are_finite,
             test_confidence_and_selected_distribution,
             test_ema_and_bandit_have_separate_state_updates,
             test_cluster_bootstrap_detects_brier_improvement_and_is_reproducible,
             test_paired_bootstrap_compares_untouched_policy_episodes]
    for test in tests:
        test()
    print(f"Stage 5 tests: {len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
