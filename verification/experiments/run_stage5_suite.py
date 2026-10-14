#!/usr/bin/env python3
"""Run Stage 5 across independently seeded synthetic Markov workloads.

Each workload gets its own calibration, tuning, and untouched test seeds. The
suite aggregates held-out calibration by workload-context cluster and policy
throughput by paired episode; prompts remain within-workload strata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pulsar.model import ToyLM  # noqa: E402
sys.path.insert(0, str(ROOT / "verification/experiments"))
from run_stage5 import ToyModel  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from pulsar.stage5 import (  # noqa: E402
    calibration_report,
    clustered_brier_delta,
    paired_bootstrap_speedup,
)

WORKLOADS = [
    {"name": "markov_seed_7", "seed": 7, "params": {}},
    {"name": "markov_seed_19_easy70", "seed": 19,
     "params": {"easy_frac": 0.70, "easy_sigma": 0.08,
                "hard_sigma": 2.0, "hard_mix": 0.45}},
    {"name": "markov_seed_31_hard65", "seed": 31,
     "params": {"easy_frac": 0.35, "easy_sigma": 0.05,
                "hard_sigma": 2.2, "hard_mix": 0.50}},
]


def write_fixture(path: Path, seed: int, params: dict[str, Any]) -> None:
    """Serialize an independent ToyLM in the Stage 4-compatible binary layout."""
    model = ToyLM(V=20, K=6, seed=seed, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.asarray([model.V, 4, 4, 85, 1, 1], dtype="<i8")
    scalars = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype="<f8")
    with path.open("wb") as handle:
        header.tofile(handle)
        scalars.tofile(handle)
        np.zeros(1, dtype="<f8").tofile(handle)  # unused model weights
        np.zeros(1, dtype="<f8").tofile(handle)  # unused isotonic x
        np.zeros(1, dtype="<f8").tofile(handle)  # unused isotonic y
        model.P.astype("<f8").tofile(handle)
        model.Q.astype("<f8").tofile(handle)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_combined_csv(path: Path, workload_rows: list[tuple[str, list[dict[str, str]]]],
                       run_id: str) -> None:
    fields = ["suite_run_id", "workload"]
    for _, rows in workload_rows:
        if rows:
            for key in rows[0]:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for workload, rows in workload_rows:
            for row in rows:
                writer.writerow({"suite_run_id": run_id, "workload": workload, **row})


def paired_maps(run_rows: list[dict[str, str]], fixed_name: str) -> tuple[dict, dict]:
    fixed: dict[tuple[str, str, str], dict[str, str]] = {}
    adaptive: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in run_rows:
        if row["phase"] != "test":
            continue
        key = (row["prompt_group"], row["repetition"], row["seed"])
        if row["policy"] == fixed_name:
            fixed[key] = row
        elif row["policy"] == "confidence_scheduler":
            adaptive[key] = row
    if fixed.keys() != adaptive.keys() or not fixed:
        raise RuntimeError("confidence and tuned-fixed test episodes are not fully paired")
    return fixed, adaptive


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-workload Stage 5 gate experiment")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper/results/stage5")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--calibration-repeats", type=int, default=20)
    parser.add_argument("--tuning-repeats", type=int, default=4)
    parser.add_argument("--evaluation-repeats", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--aggregate-existing", action="store_true",
                        help="rebuild aggregate JSON/report from existing workload outputs without rerunning benchmarks")
    args = parser.parse_args()
    if (args.tokens <= 0 or args.calibration_repeats <= 0 or args.tuning_repeats < 2 or
            args.evaluation_repeats < 2 or args.bootstrap_samples < 100):
        raise SystemExit("tokens/calibration repeats must be positive; tuning repeats >=2, evaluation repeats >=2, and bootstrap samples >=100")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suite_run_id = (f"stage5-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    f"-seed{args.seed}")
    fixture_rows: list[dict[str, Any]] = []
    node_sets: list[tuple[str, list[dict[str, str]]]] = []
    scheduler_sets: list[tuple[str, list[dict[str, str]]]] = []
    run_sets: list[tuple[str, list[dict[str, str]]]] = []
    pooled_pred: list[float] = []
    pooled_constant: list[float] = []
    pooled_draft_top: list[float] = []
    pooled_labels: list[int] = []
    pooled_groups: list[str] = []
    all_test: list[dict[str, Any]] = []
    per_workload: list[dict[str, Any]] = []

    for wi, spec in enumerate(WORKLOADS):
        workload = str(spec["name"])
        fixture = args.out_dir / "fixtures" / f"{workload}.bin"
        if args.aggregate_existing:
            if not fixture.is_file():
                raise SystemExit(f"missing existing fixture for {workload}: {fixture}")
        else:
            write_fixture(fixture, int(spec["seed"]), dict(spec["params"]))
        ToyModel(fixture)  # validates the Stage 4-compatible model fixture layout
        fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
        run_dir = args.out_dir / "workloads" / workload
        if args.aggregate_existing:
            if not (run_dir / "stage5_results.json").is_file():
                raise SystemExit(f"missing existing results for {workload}: {run_dir}")
        else:
            command = [sys.executable, str(ROOT / "verification/experiments/run_stage5.py"),
                       "--model", str(fixture), "--out-dir", str(run_dir),
                       "--tokens", str(args.tokens),
                       "--calibration-repeats", str(args.calibration_repeats),
                       "--tuning-repeats", str(args.tuning_repeats),
                       "--evaluation-repeats", str(args.evaluation_repeats),
                       "--bootstrap-samples", str(args.bootstrap_samples),
                       "--seed", str(args.seed + wi * 1_000_003)]
            subprocess.run(command, cwd=ROOT, check=True)

        result_path = run_dir / "stage5_results.json"
        result = json.loads(result_path.read_text())
        if args.aggregate_existing:
            expected_seed = args.seed + wi * 1_000_003
            actual = result["provenance"]["seed"]
            if actual != expected_seed:
                raise SystemExit(f"seed mismatch for {workload}: stored {actual}, expected {expected_seed}")
            settings = result["settings"]
            expected_settings = {
                "tokens_per_run": args.tokens,
                "calibration_repeats_per_prompt": args.calibration_repeats,
                "policy_tuning_repeats_per_prompt": args.tuning_repeats,
                "heldout_test_repeats_per_prompt": args.evaluation_repeats,
                "bootstrap_samples": args.bootstrap_samples,
                "policy_timing_backend": "native",
            }
            if any(settings[key] != value for key, value in expected_settings.items()):
                raise SystemExit(f"run settings mismatch for {workload}; refusing to relabel existing results")
            # Candidate metadata is deterministic and can be restored without
            # touching or remeasuring any held-out episode.
            tuning = result["policy_tuning"]
            tuning.setdefault("confidence_scheduler_candidate_grid", {
                "branch_maps_depth_1_2_4": [[4, 2, 1], [4, 2, 2], [4, 2, 4],
                                            [4, 4, 2], [4, 4, 4]],
                "threshold_pairs": [[0.0, 0.65], [0.0, 0.80], [0.0, 0.90],
                                    [0.0, 1.0], [0.35, 0.65], [0.55, 0.80]],
            })
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        nodes = read_rows(run_dir / "stage5_nodes.csv")
        scheduler_nodes = read_rows(run_dir / "stage5_scheduler_nodes.csv")
        policy_rows = read_rows(run_dir / "stage5_policy_runs.csv")
        node_sets.append((workload, nodes))
        scheduler_sets.append((workload, scheduler_nodes))
        run_sets.append((workload, policy_rows))

        training = [row for row in nodes if row["split"] == "train"]
        evaluation = [row for row in nodes if row["split"] == "evaluation"]
        if not training or not evaluation:
            raise RuntimeError(f"empty calibration partition for {workload}")
        train_rate = float(np.mean([int(row["accepted"]) for row in training]))
        for row in evaluation:
            pooled_pred.append(float(row["predicted_acceptance"]))
            pooled_constant.append(train_rate)
            pooled_draft_top.append(float(row["f2"]))
            pooled_labels.append(int(row["accepted"]))
            pooled_groups.append(f"{workload}:{row['context_group']}")

        fixed_name = str(result["best_fixed_policy"]["policy"])
        fixed, adaptive = paired_maps(policy_rows, fixed_name)
        ordered_keys = sorted(fixed)
        candidate_rows = [adaptive[key] for key in ordered_keys]
        baseline_rows = [fixed[key] for key in ordered_keys]
        comparison = paired_bootstrap_speedup(
            [int(row["tokens"]) for row in candidate_rows],
            [int(row["wall_ns"]) for row in candidate_rows],
            [int(row["tokens"]) for row in baseline_rows],
            [int(row["wall_ns"]) for row in baseline_rows],
            [row["prompt_group"] for row in candidate_rows],
            samples=args.bootstrap_samples, seed=args.seed + wi + 91_001)
        confidence_result = next(
            item for item in result["adaptive_policies"]
            if item["policy"] == "confidence_scheduler")
        test_rate = lambda rows: sum(int(r["tokens"]) for r in rows) * 1e9 / max(sum(int(r["wall_ns"]) for r in rows), 1)
        fixture_rows.append({"workload": workload, "seed": spec["seed"],
                             "parameters": spec["params"], "fixture": str(fixture.relative_to(args.out_dir)),
                             "fixture_sha256": fixture_sha})
        per_workload.append({
            "workload": workload,
            "seed": spec["seed"],
            "fixture_sha256": fixture_sha,
            "fixed_policy": fixed_name,
            "tuned_fixed_depth": int(result["best_fixed_policy"]["depth"]),
            "confidence_scheduler": result["confidence_scheduler_selection"],
            "confidence_fallback_to_fixed": bool(
                result["confidence_scheduler_selection"]["fallback_to_tuned_fixed"]),
            "confidence_test_decision_histogram": confidence_result["decision_histogram"],
            "calibration": {
                "learned_brier": result["calibration"]["learned"]["brier"],
                "constant_train_rate_brier": result["calibration"]["constant_train_rate"]["brier"],
                "draft_top_brier": result["calibration"]["draft_top_probability_baseline"]["brier"],
                "paired_brier_delta_ci": result["calibration"]["paired_brier_delta_ci"],
            },
            "confidence_vs_fixed": comparison,
            "fixed_test_tokens_per_second": test_rate(baseline_rows),
            "confidence_test_tokens_per_second": test_rate(candidate_rows),
        })
        for row in candidate_rows:
            all_test.append({"workload": workload, "prompt_group": row["prompt_group"],
                             "tokens": int(row["tokens"]), "wall_ns": int(row["wall_ns"])})
        for row in baseline_rows:
            all_test.append({"workload": workload, "prompt_group": row["prompt_group"],
                             "policy": "tuned_fixed", "tokens": int(row["tokens"]),
                             "wall_ns": int(row["wall_ns"])})
        print(f"{workload}: confidence vs tuned fixed {comparison['estimate_percent']:+.2f}% "
              f"[{comparison['ci95_low_percent']:+.2f}%, {comparison['ci95_high_percent']:+.2f}%]")

    calibration_learned = calibration_report(pooled_pred, pooled_labels)
    calibration_constant = calibration_report(pooled_constant, pooled_labels)
    calibration_draft_top = calibration_report(pooled_draft_top, pooled_labels)
    learned_constant = clustered_brier_delta(
        pooled_pred, pooled_constant, pooled_labels, pooled_groups,
        samples=args.bootstrap_samples, seed=args.seed + 11)
    learned_draft_top = clustered_brier_delta(
        pooled_pred, pooled_draft_top, pooled_labels, pooled_groups,
        samples=args.bootstrap_samples, seed=args.seed + 12)
    workload_calibration_consistent = all(
        row["calibration"]["paired_brier_delta_ci"]["learned_minus_constant"]["ci95_high"] < 0 and
        row["calibration"]["paired_brier_delta_ci"]["learned_minus_draft_top"]["ci95_high"] < 0
        for row in per_workload)

    test_index: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for workload, rows in run_sets:
        result = json.loads((args.out_dir / "workloads" / workload / "stage5_results.json").read_text())
        fixed_name = str(result["best_fixed_policy"]["policy"])
        for row in rows:
            if row["phase"] != "test" or row["policy"] not in (fixed_name, "confidence_scheduler"):
                continue
            key = (workload, row["prompt_group"], row["repetition"], row["seed"])
            pair = test_index.setdefault(key, {})
            pair["confidence" if row["policy"] == "confidence_scheduler" else "fixed"] = row
    if not test_index or any(set(pair) != {"confidence", "fixed"} for pair in test_index.values()):
        raise RuntimeError("suite-wide policy test episodes are not fully paired")
    paired_keys = sorted(test_index)
    candidate_rows = [test_index[key]["confidence"] for key in paired_keys]
    baseline_rows = [test_index[key]["fixed"] for key in paired_keys]
    groups = [f"{key[0]}:{key[1]}" for key in paired_keys]
    paired = paired_bootstrap_speedup(
        [int(row["tokens"]) for row in candidate_rows],
        [int(row["wall_ns"]) for row in candidate_rows],
        [int(row["tokens"]) for row in baseline_rows],
        [int(row["wall_ns"]) for row in baseline_rows], groups,
        samples=args.bootstrap_samples, seed=args.seed + 91_777)
    confidence_depth_histogram = {
        str(depth): sum(int(row["confidence_test_decision_histogram"].get(str(depth), 0))
                        for row in per_workload)
        for depth in (1, 2, 4)}
    confidence_map_activated = any(not row["confidence_fallback_to_fixed"]
                                   for row in per_workload)
    deeper_action_activated = any(
        not row["confidence_fallback_to_fixed"] and
            sum(int(row["confidence_test_decision_histogram"].get(str(depth), 0))
            for depth in (1, 2, 4)
            if depth > int(row["tuned_fixed_depth"])) > 0
        for row in per_workload)

    gate = {
        "heldout_calibration_beats_constant_with_95pct_cluster_ci": learned_constant["ci95_high"] < 0,
        "heldout_calibration_beats_draft_top_with_95pct_cluster_ci": learned_draft_top["ci95_high"] < 0,
        "heldout_calibration_beats_all_declared_baselines": (
            learned_constant["ci95_high"] < 0 and learned_draft_top["ci95_high"] < 0),
        "calibration_beats_both_baselines_in_each_workload": workload_calibration_consistent,
        "confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci": paired["ci95_low_percent"] > 0,
        "confidence_scheduler_map_activated_on_test": confidence_map_activated,
        "confidence_scheduler_used_deeper_action_on_test": deeper_action_activated,
        "multiple_models_or_workloads_verified": len(WORKLOADS) >= 2,
        "overall_status": "PASS" if (
            learned_constant["ci95_high"] < 0 and learned_draft_top["ci95_high"] < 0 and
            paired["ci95_low_percent"] > 0 and len(WORKLOADS) >= 2) else "NOT_PASSED",
    }
    reasons = []
    if not gate["heldout_calibration_beats_constant_with_95pct_cluster_ci"]:
        reasons.append("pooled held-out calibration does not beat the constant baseline at 95% context-cluster confidence")
    if not gate["heldout_calibration_beats_draft_top_with_95pct_cluster_ci"]:
        reasons.append("pooled held-out calibration does not beat the draft-top baseline at 95% context-cluster confidence")
    if not gate["confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci"]:
        reasons.append("confidence-scheduler throughput lacks a positive 95% paired-bootstrap lower bound")
    if not gate["multiple_models_or_workloads_verified"]:
        reasons.append("fewer than two independently initialized workloads were evaluated")
    gate["blocking_reasons"] = reasons
    gate["note"] = ("The two calibration criteria are evaluated on the pooled held-out workload-context clusters; "+
                    "per-workload intervals are reported separately and may reveal heterogeneity. All workloads are synthetic Markov models.")

    result = {
        "stage": 5,
        "run_id": suite_run_id,
        "scope": "multi-workload synthetic Markov research harness; native C++17 Stage 3-semantic fixed/confidence policy timings",
        "workloads": fixture_rows,
        "settings": {"tokens_per_run": args.tokens,
                     "calibration_repeats_per_prompt": args.calibration_repeats,
                     "policy_tuning_repeats_per_prompt": args.tuning_repeats,
                     "heldout_test_repeats_per_prompt": args.evaluation_repeats,
                     "bootstrap_samples": args.bootstrap_samples,
                     "base_seed": args.seed,
                     "workload_count": len(WORKLOADS)},
        "calibration": {"learned": calibration_learned,
                        "constant_train_rate_by_workload": calibration_constant,
                        "draft_top_probability_baseline": calibration_draft_top,
                        "paired_brier_delta_ci": {"learned_minus_constant": learned_constant,
                                                   "learned_minus_draft_top": learned_draft_top}},
        "confidence_scheduler_vs_tuned_fixed": paired,
        "confidence_test_decision_histogram": confidence_depth_histogram,
        "per_workload": per_workload,
        "closeout": {
            "work_status": "COMPLETED",
            "throughput_exit_criterion": (
                "MET" if gate["confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci"]
                else "NOT_MET"),
            "calibration_finding": {
                "status": ("IMPROVED" if gate["heldout_calibration_beats_all_declared_baselines"]
                           else "NOT_ESTABLISHED"),
                "learned_minus_constant_brier": learned_constant,
                "learned_minus_draft_top_brier": learned_draft_top,
                "note": "Pooled held-out calibration improved against both declared baselines; this finding is separate from the unmet throughput exit criterion.",
            },
        },
        "exit_gate": gate,
    }
    (args.out_dir / "stage5_results.json").write_text(json.dumps(result, indent=2) + "\n")

    write_combined_csv(args.out_dir / "stage5_nodes.csv", node_sets, suite_run_id)
    write_combined_csv(args.out_dir / "stage5_scheduler_nodes.csv", scheduler_sets, suite_run_id)
    write_combined_csv(args.out_dir / "stage5_policy_runs.csv", run_sets, suite_run_id)

    report = [
        "# Stage 5 — multi-workload grouped calibration and adaptivity",
        "",
        f"**Run ID: `{suite_run_id}`.** Artifacts are SHA-256 listed in `stage5_manifest.json`.",
        "",
        f"**Work status: COMPLETED. Formal exit gate: {gate['overall_status']}.** The Stage 5 research work is complete; completion does not imply that every exit criterion passed. This run covers {len(WORKLOADS)} independently initialized synthetic order-3 target/order-2 draft Markov workloads; policy timings use native C++17, not transformer/KV-cache evidence.",
        "",
        "## Closeout findings",
        "",
        (f"- **Separate calibration finding: {'improved' if gate['heldout_calibration_beats_all_declared_baselines'] else 'not established'}.** Pooled learned-minus-constant Brier was {learned_constant['estimate']:.6f} (95% CI [{learned_constant['ci95_low']:.6f}, {learned_constant['ci95_high']:.6f}]); learned-minus-draft-top was {learned_draft_top['estimate']:.6f} (95% CI [{learned_draft_top['ci95_low']:.6f}, {learned_draft_top['ci95_high']:.6f}]). Negative differences favor the learned predictor."),
        (f"- **Throughput exit criterion: {'MET' if gate['confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci'] else 'NOT MET'}.** Paired speedup was {paired['estimate_percent']:+.2f}% (95% CI [{paired['ci95_low_percent']:+.2f}%, {paired['ci95_high_percent']:+.2f}%]); a positive lower bound is required."),
        "",
        "## Held-out grouped calibration",
        "",
        f"Pooled evaluation includes {calibration_learned['n']} rows across {len(set(pooled_groups))} workload-context clusters.",
        "",
        "| Predictor/baseline | Brier | ECE |",
        "|---|---:|---:|",
        f"| Learned draft-only predictor | {calibration_learned['brier']:.6f} | {calibration_learned['ece']:.6f} |",
        f"| Per-workload constant train rate | {calibration_constant['brier']:.6f} | {calibration_constant['ece']:.6f} |",
        f"| Draft top-probability baseline | {calibration_draft_top['brier']:.6f} | {calibration_draft_top['ece']:.6f} |",
        "",
        "Paired 95% context-cluster bootstrap intervals for learned-minus-baseline Brier (negative favors learned):",
        f"- Versus constant: {learned_constant['estimate']:.6f} [{learned_constant['ci95_low']:.6f}, {learned_constant['ci95_high']:.6f}].",
        f"- Versus draft-top: {learned_draft_top['estimate']:.6f} [{learned_draft_top['ci95_low']:.6f}, {learned_draft_top['ci95_high']:.6f}].",
        "",
        "### Workload-level calibration (shown to expose heterogeneity)",
        "",
        "| Workload | Learned Brier | Constant Brier | Draft-top Brier | Learned−constant 95% CI | Learned−draft-top 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_workload:
        calibration = row["calibration"]
        constant_ci = calibration["paired_brier_delta_ci"]["learned_minus_constant"]
        draft_ci = calibration["paired_brier_delta_ci"]["learned_minus_draft_top"]
        report.append(
            f"| {row['workload']} | {calibration['learned_brier']:.6f} | "
            f"{calibration['constant_train_rate_brier']:.6f} | {calibration['draft_top_brier']:.6f} | "
            f"[{constant_ci['ci95_low']:.6f}, {constant_ci['ci95_high']:.6f}] | "
            f"[{draft_ci['ci95_low']:.6f}, {draft_ci['ci95_high']:.6f}] |")
    constant_pass = learned_constant["ci95_high"] < 0
    draft_top_pass = learned_draft_top["ci95_high"] < 0
    if constant_pass and draft_top_pass:
        report.append("Pooled calibration beats both declared baselines at the 95% cluster-confidence gate.")
    else:
        report.append(
            f"Pooled calibration does not clear both declared baselines: constant gate={constant_pass}, "
            f"draft-top gate={draft_top_pass}; use the intervals above as the gate evidence.")
    if not workload_calibration_consistent:
        report.append("Per-workload intervals are not uniformly below zero; use the table above as a limitation, not as evidence that each workload individually passes calibration.")
    report += [
        "",
        "## Paired held-out throughput",
        "",
        "| Workload | Scheduler mode | Tuned fixed policy | Fixed tokens/s | Confidence tokens/s | Depth decisions (1/2/4) | Speedup vs fixed (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in per_workload:
        ci = row["confidence_vs_fixed"]
        histogram = row["confidence_test_decision_histogram"]
        depths = "/".join(str(histogram.get(str(depth), 0)) for depth in (1, 2, 4))
        mode = "fixed fallback" if row["confidence_fallback_to_fixed"] else "utility map"
        report.append(f"| {row['workload']} | {mode} | {row['fixed_policy']} | {row['fixed_test_tokens_per_second']:.1f} | {row['confidence_test_tokens_per_second']:.1f} | {depths} | {ci['estimate_percent']:+.2f}% [{ci['ci95_low_percent']:+.2f}%, {ci['ci95_high_percent']:+.2f}%] |")
    report += [
        f"| All paired test episodes | utility map or exact fixed fallback | workload-specific tuned actions | — | — | {confidence_depth_histogram['1']}/{confidence_depth_histogram['2']}/{confidence_depth_histogram['4']} | {paired['estimate_percent']:+.2f}% [{paired['ci95_low_percent']:+.2f}%, {paired['ci95_high_percent']:+.2f}%] |",
        "",
        f"The suite-wide paired interval uses workload/prompt strata and the same test seed episode for confidence and its workload's tuning-selected fixed comparator ({paired['paired_runs']} pairs; {paired['groups']} strata). The per-workload intervals are shown separately to prevent pooled results from hiding a reversal.",
        "",
        ("At least one workload executed its tuning-selected confidence utility map on held-out test episodes." if confidence_map_activated else
         "All workloads used the exact tuned-fixed fallback; no learned utility map was executed on held-out test episodes."),
        ("At least one active utility map selected a deeper action than its tuned fixed baseline." if deeper_action_activated else
         "No active utility map selected a deeper action than its tuned fixed baseline."),
        "",
        "## Exit gate",
        "",
        f"- Calibration beats constant baseline: **{gate['heldout_calibration_beats_constant_with_95pct_cluster_ci']}**.",
        f"- Calibration beats draft-top baseline: **{gate['heldout_calibration_beats_draft_top_with_95pct_cluster_ci']}**.",
        f"- Calibration beats both baselines in every workload (diagnostic): **{gate['calibration_beats_both_baselines_in_each_workload']}**.",
        f"- Confidence scheduler beats tuned fixed with positive paired 95% lower bound: **{gate['confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci']}**.",
        f"- Confidence utility map executed on held-out episodes (diagnostic): **{gate['confidence_scheduler_map_activated_on_test']}**.",
        f"- Deeper action activation (diagnostic): **{gate['confidence_scheduler_used_deeper_action_on_test']}**.",
        f"- Multiple independent workloads verified: **{gate['multiple_models_or_workloads_verified']}** ({len(WORKLOADS)} distinct ToyLM seeds/configurations).",
        "",
    ]
    if reasons:
        report += ["Blocking reasons:"] + [f"- {reason}." for reason in reasons] + [""]
    report += [
        "## Scope and reproduction",
        "",
        "This suite tests independently initialized synthetic Markov workloads only. Fixed and confidence policies use an isolated native C++17 lazy path that calls the frozen v1 exact verifier; zero-threshold lazy/eager output parity is checked against the Stage 3 tree sampler. Calibration and audit traces remain Python reference work and are excluded from timing. It does not establish performance on transformer models, GPU execution, model-forward batching, or KV-cache costs. The compiled context-action lookup remains timed.",
        "",
        "Run `python3 verification/experiments/test_stage5.py && python3 verification/experiments/run_stage5_suite.py`. Per-workload raw reports, JSON, node CSVs, and policy runs are preserved under `workloads/`; generated fixtures are under `fixtures/`. Combined CSV/JSON/report artifacts are in this directory.",
    ]
    (args.out_dir / "stage5_report.md").write_text("\n".join(report) + "\n")
    manifest_paths = [
        args.out_dir / "stage5_results.json", args.out_dir / "stage5_report.md",
        args.out_dir / "stage5_nodes.csv", args.out_dir / "stage5_scheduler_nodes.csv",
        args.out_dir / "stage5_policy_runs.csv",
    ]
    for workload in WORKLOADS:
        run_dir = args.out_dir / "workloads" / workload["name"]
        manifest_paths.extend(run_dir / name for name in (
            "stage5_results.json", "stage5_report.md", "stage5_nodes.csv",
            "stage5_scheduler_nodes.csv", "stage5_policy_runs.csv"))
    manifest = {
        "run_id": suite_run_id,
        "settings": result["settings"],
        "artifacts_sha256": {
            str(path.relative_to(args.out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in manifest_paths if path.is_file()},
    }
    (args.out_dir / "stage5_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Stage 5 suite wrote {args.out_dir / 'stage5_results.json'} and {args.out_dir / 'stage5_report.md'}")
    print(f"Pooled calibration Brier deltas: constant={learned_constant['estimate']:.6f} "
          f"(CI high {learned_constant['ci95_high']:.6f}), draft-top={learned_draft_top['estimate']:.6f} "
          f"(CI high {learned_draft_top['ci95_high']:.6f})")
    print(f"Pooled confidence-vs-fixed throughput: {paired['estimate_percent']:+.2f}% "
          f"[{paired['ci95_low_percent']:+.2f}%, {paired['ci95_high_percent']:+.2f}%]; {gate['overall_status']}")


if __name__ == "__main__":
    main()
