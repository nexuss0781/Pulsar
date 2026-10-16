#!/usr/bin/env python3
"""Open Phase 4 evaluation data only after the runner's all-methods-complete marker."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 5555
BATCH_SIZES = (1, 2)
TREE_ACTIONS = ((1, 2), (2, 2), (2, 3))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"required benchmark output is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def interval(values: list[float]) -> tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


def raw_paths(run_dir: Path, batch_size: int, freeze: dict[str, Any]) -> dict[str, Path]:
    evaluation = run_dir / "evaluation" / "raw"
    selected = freeze["selected_baselines"][str(batch_size)]
    rows = {
        "autoregressive": evaluation / f"evaluation_autoregressive_b{batch_size}.jsonl",
        "builtin": evaluation / f"evaluation_llama_draft_simple_n{selected['llama_cpp_n_max']}_b{batch_size}.jsonl",
        "adaptive": evaluation / f"evaluation_pulsar-adaptive_b{batch_size}.jsonl",
    }
    fixed = selected["fixed_tree"]
    rows["fixed_selected"] = evaluation / (
        f"evaluation_pulsar-tree_B{fixed['branching']}_D{fixed['depth']}_b{batch_size}.jsonl")
    for branching, depth in TREE_ACTIONS:
        rows[f"fixed_B{branching}_D{depth}"] = evaluation / (
            f"evaluation_pulsar-tree_B{branching}_D{depth}_b{batch_size}.jsonl")
    return rows


def pair_key(row: dict[str, Any]) -> tuple[int, tuple[str, ...]]:
    return int(row["seed_root"]), tuple(row["batch_prompt_ids"])


def validate_pairs(data: dict[str, dict[tuple[int, tuple[str, ...]], dict[str, Any]]]) -> list[tuple[str, ...]]:
    reference = set(data["autoregressive"])
    if not reference:
        raise RuntimeError("evaluation raw data is empty")
    for name, rows in data.items():
        keys = set(rows)
        if keys != reference:
            missing = sorted(reference - keys)[:3]
            extra = sorted(keys - reference)[:3]
            raise RuntimeError(f"paired-key mismatch for {name}: missing={missing}, extra={extra}")
    clusters = sorted({key[1] for key in reference})
    expected = 8 if len(clusters[0]) == 1 else 4
    if len(clusters) != expected:
        raise RuntimeError(f"expected {expected} independent group clusters, found {len(clusters)}")
    return clusters


def latency_values(row: dict[str, Any]) -> list[float]:
    return [float(value) for value in row.get("request_latency_ms", [])]


def throughput(row: dict[str, Any]) -> float:
    return float(row["total_completion_tokens"]) * 1000.0 / float(row["batch_completion_ms"])


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(row["total_completion_tokens"]) for row in rows)
    total_ms = sum(float(row["batch_completion_ms"]) for row in rows)
    latencies = [latency for row in rows for latency in latency_values(row)]
    summary: dict[str, Any] = {
        "batches": len(rows), "total_completion_tokens": total_tokens,
        "total_batch_wall_ms": total_ms,
        "aggregate_tokens_per_second": total_tokens * 1000.0 / total_ms if total_ms else None,
        "request_latency_p50_ms": percentile(latencies, 0.50),
        "request_latency_p95_ms": percentile(latencies, 0.95),
        "peak_process_rss_bytes": max((int(row["peak_process_rss_bytes"])
                                       for row in rows if row.get("peak_process_rss_bytes") is not None),
                                      default=None),
    }
    for field in ("proposed_tokens", "accepted_proposals", "rejected_blocks",
                  "target_verifier_decode_tokens", "candidate_tree_pulses",
                  "candidate_tree_nodes_built", "candidate_tree_build_inclusive_ms",
                  "candidate_tree_verification_inclusive_ms"):
        values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float))]
        summary[field] = sum(values) if values else None
    for side in ("target_model_work", "draft_model_work"):
        work_rows = [row.get(side) for row in rows if isinstance(row.get(side), dict)]
        totals: dict[str, float] = defaultdict(float)
        for work in work_rows:
            for key, value in work.items():
                if isinstance(value, (int, float)) and not str(key).startswith("peak_"):
                    totals[key] += float(value)
        peaks: dict[str, float] = {}
        for work in work_rows:
            for key, value in work.items():
                if isinstance(value, (int, float)) and str(key).startswith("peak_"):
                    peaks[key] = max(peaks.get(key, 0.0), float(value))
        summary[side] = {**dict(totals), **peaks} if work_rows else None
    summary["peak_kv_bytes_estimate"] = max(
        (float(row.get("target_model_work", {}).get("peak_kv_bytes_estimate", 0.0))
         for row in rows if isinstance(row.get("target_model_work"), dict)), default=None)
    return summary


def bootstrap_comparison(data: dict[str, dict[tuple[int, tuple[str, ...]], dict[str, Any]]],
                         clusters: list[tuple[str, ...]],
                         draws: list[list[tuple[str, ...]]], method: str,
                         reference: str) -> dict[str, Any]:
    reference_keys = list(data[reference])
    def point_rate(name: str) -> float:
        rows = list(data[name].values())
        return sum(int(row["total_completion_tokens"]) for row in rows) * 1000.0 / sum(
            float(row["batch_completion_ms"]) for row in rows)
    point_ratio = point_rate(method) / point_rate(reference)
    point_p95_ratio = percentile([x for row in data[method].values() for x in latency_values(row)], 0.95) / \
        percentile([x for row in data[reference].values() for x in latency_values(row)], 0.95)
    ratios: list[float] = []
    p95_ratios: list[float] = []
    seed_roots = sorted({key[0] for key in reference_keys})
    for draw in draws:
        numerator_tokens = numerator_ms = denominator_tokens = denominator_ms = 0.0
        numerator_latency: list[float] = []
        denominator_latency: list[float] = []
        for cluster in draw:
            for seed in seed_roots:
                key = (seed, cluster)
                lhs = data[method][key]
                rhs = data[reference][key]
                numerator_tokens += int(lhs["total_completion_tokens"])
                numerator_ms += float(lhs["batch_completion_ms"])
                denominator_tokens += int(rhs["total_completion_tokens"])
                denominator_ms += float(rhs["batch_completion_ms"])
                numerator_latency.extend(latency_values(lhs))
                denominator_latency.extend(latency_values(rhs))
        lhs_rate = numerator_tokens * 1000.0 / numerator_ms
        rhs_rate = denominator_tokens * 1000.0 / denominator_ms
        ratios.append(lhs_rate / rhs_rate)
        p95_ratios.append(percentile(numerator_latency, 0.95) / percentile(denominator_latency, 0.95))
    ratio_ci = interval(ratios)
    latency_ci = interval(p95_ratios)
    return {"method": method, "reference": reference,
            "throughput_ratio": point_ratio,
            "throughput_ratio_95_ci": list(ratio_ci),
            "p95_latency_ratio": point_p95_ratio,
            "p95_latency_ratio_95_ci": list(latency_ci),
            "bootstrap_draws": len(draws), "bootstrap_seed": BOOTSTRAP_SEED}


def load_calibrator(path: Path) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    models: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) != 15:
            raise RuntimeError(f"invalid calibrator TSV row: {line}")
        key = (int(values[0]), int(values[1]), int(values[2]), values[3])
        models[key] = {"means": [float(values[5]), float(values[7]), float(values[9])],
                       "scales": [float(values[6]), float(values[8]), float(values[10])],
                       "coefficients": [float(value) for value in values[11:15]]}
    return models


def predict(model: dict[str, Any], features: tuple[float, float, float]) -> float:
    score = model["coefficients"][0] + sum(
        model["coefficients"][index + 1] *
        (features[index] - model["means"][index]) / model["scales"][index]
        for index in range(3))
    score = max(-35.0, min(35.0, score))
    return 1.0 / (1.0 + math.exp(-score))


def reliability_group(rows: list[tuple[float, int]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "brier_score": None, "bins": []}
    brier = sum((probability - label) ** 2 for probability, label in rows) / len(rows)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in rows:
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, label))
    reliability = []
    for index, bucket in enumerate(buckets):
        if bucket:
            reliability.append({"bin": index, "lower": index / bins,
                "upper": (index + 1) / bins, "n": len(bucket),
                "mean_predicted": statistics.fmean(row[0] for row in bucket),
                "observed_frequency": statistics.fmean(row[1] for row in bucket)})
    return {"n": len(rows), "brier_score": brier, "bins": reliability}


def calibration_analysis(run_dir: Path, calibrator_path: Path) -> dict[str, Any]:
    models = load_calibrator(calibrator_path)
    all_events: list[dict[str, Any]] = []
    for path in sorted((run_dir / "evaluation" / "observations").glob("*.jsonl")):
        all_events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                          if line.strip())
    if not all_events:
        raise RuntimeError("no locked evaluation verifier observations were found")
    groups: dict[tuple[str, int, int, int, int, str], list[tuple[float, int]]] = defaultdict(list)
    for event in all_events:
        mode = str(event["mode"])
        batch = int(event["batch_size"])
        branching = int(event["branching"])
        configured_depth = int(event["configured_depth"])
        depth = int(event["depth_position"])
        features = (float(event["root_q_max"]), float(event["root_q_margin"]),
                    float(event["root_q_entropy"]))
        labels = [("accepted", int(bool(event["accepted"])))]
        if event["continuation_label_available"]:
            labels.append(("continued", int(bool(event["continued"]))))
        for outcome, label in labels:
            model = models.get((batch, branching, depth, outcome))
            if model is None:
                raise RuntimeError(f"missing locked calibration cell {(batch, branching, depth, outcome)}")
            groups[(mode, batch, branching, configured_depth, depth, outcome)].append((predict(model, features), label))
    result = {"feature_set": ["root_q_max", "root_q_margin", "root_q_entropy"],
              "scope": "locked evaluation rows; no recalibration on E groups",
              "by_mode_batch_branch_configured_depth_verifier_depth_outcome": []}
    for key in sorted(groups):
        mode, batch, branching, configured_depth, depth, outcome = key
        result["by_mode_batch_branch_configured_depth_verifier_depth_outcome"].append({
            "mode": mode, "batch_size": batch, "branching": branching,
            "configured_depth": configured_depth, "depth_position": depth, "outcome": outcome,
            **reliability_group(groups[key])})
    adaptive = [event for event in all_events if event["mode"] == "pulsar-adaptive"]
    pulse_actions: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    for event in adaptive:
        key = (event["batch_id"], event["prompt_numeric_id"], event["seed_root"],
               event["pulse_index"])
        action = (int(event["batch_size"]), int(event["branching"]),
                  int(event["configured_depth"]))
        if key in pulse_actions and pulse_actions[key] != action:
            raise RuntimeError(f"adaptive pulse selected multiple actions: {key}")
        pulse_actions[key] = action
    counts = Counter(pulse_actions.values())
    result["adaptive_action_frequencies"] = [
        {"batch_size": batch, "branching": branch, "depth": depth,
         "pulses": count,
         "fraction_within_batch_size": count / sum(n for (b, _, _), n in counts.items() if b == batch)}
        for (batch, branch, depth), count in sorted(counts.items())]
    result["observation_rows"] = len(all_events)
    result["adaptive_pulses"] = len(pulse_actions)
    return result


def offline_oracle(data_by_action: dict[str, dict[tuple[int, tuple[str, ...]], dict[str, Any]]]) -> dict[str, Any]:
    keys = set(next(iter(data_by_action.values())))
    chosen_rows = []
    chosen_actions = Counter()
    for key in sorted(keys):
        candidates = [(throughput(rows[key]), name, rows[key]) for name, rows in data_by_action.items()]
        def oracle_tiebreak(item: tuple[float, str, dict[str, Any]]) -> tuple[float, int, int, int]:
            row = item[2]
            branching = int(row["tree_branching"])
            depth = int(row["tree_depth"])
            nodes = sum(branching ** level for level in range(depth))
            return item[0], -nodes, -depth, -branching
        _, name, row = max(candidates, key=oracle_tiebreak)
        chosen_rows.append(row)
        chosen_actions[name] += 1
    tokens = sum(int(row["total_completion_tokens"]) for row in chosen_rows)
    elapsed = sum(float(row["batch_completion_ms"]) for row in chosen_rows)
    return {"aggregate_tokens_per_second": tokens * 1000.0 / elapsed if elapsed else None,
            "batch_rows": len(chosen_rows), "selected_action_counts": dict(chosen_actions),
            "interpretation": "descriptive offline upper bound: selects the fastest observed fixed action per locked seed/group batch; not deployable and not a causal online-scheduler estimate"}


def markdown_report(run_dir: Path, analyses: dict[str, Any]) -> str:
    freeze = analyses["freeze"]
    lines = [
        "# Phase 4: confidence calibration and adaptive tree scheduling",
        "",
        f"Run: `{run_dir.name}`  ",
        f"Frozen at: {freeze['frozen_at_utc']}  ",
        f"Evaluation completed: {analyses['evaluation_completed_at_utc']}  ",
        f"Protocol SHA-256: `{freeze['protocol_sha256']}`",
        "",
        "## Scope and exactness",
        "",
        "CPU-only TinyStories 15M GGUF fixture, target/draft model hashes frozen before evaluation, two CPU threads, batch sizes 1 and 2, 24 disjoint topic/context groups split 8/8/8, 64-token maximum, and the preregistered sampling law. These results apply only to this small-model x86_64 fixture.",
        "",
        f"Correctness tests before measurement: **{analyses['test_status']}**. Exactness means preserving the target sampling law and verifier/residual semantics; different methods may consume random numbers differently, so sampled text is not required to match token-for-token.",
        "",
        "## Tuning-only selections",
        "",
        "| Batch | Built-in chain n_max | Selected fixed tree (B, depth) | Median tuning tok/s |",
        "|---:|---:|---:|---:|",
    ]
    tuning = read_json(run_dir / "tuning" / "tuning_selection.json")
    for batch, record in sorted(tuning["by_batch_size"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {batch} | {record['llama_cpp']['n_max']} | ({record['pulsar_fixed_tree']['branching']}, {record['pulsar_fixed_tree']['depth']}) | {record['pulsar_fixed_tree']['median_tokens_per_second']:.3f} |")
    lines += ["", "## Locked evaluation throughput and latency", "",
              "| Batch | Method | Aggregate tok/s | Total batch seconds | Request p50 / p95 ms | Peak RSS MiB |", "|---:|---|---:|---:|---:|---:|"]
    for batch, methods in analyses["metrics_by_batch_size"].items():
        for method, metric in methods.items():
            rss = metric.get("peak_process_rss_bytes")
            rss_mib = "n/a" if rss is None else f"{rss / (1024**2):.1f}"
            lines.append(f"| {batch} | {method} | {metric['aggregate_tokens_per_second']:.3f} | {metric['total_batch_wall_ms']/1000:.3f} | {metric['request_latency_p50_ms']:.2f} / {metric['request_latency_p95_ms']:.2f} | {rss_mib} |")
    lines += ["", "### Paired comparisons (10,000 cluster-bootstrap resamples)", "",
              "| Batch | Adaptive vs | Throughput ratio (95% CI) | p95 latency ratio (95% CI) |", "|---:|---|---:|---:|"]
    for batch, comparisons in analyses["comparisons"].items():
        for comp in comparisons:
            lo, hi = comp["throughput_ratio_95_ci"]
            llo, lhi = comp["p95_latency_ratio_95_ci"]
            lines.append(f"| {batch} | {comp['reference']} | {comp['throughput_ratio']:.3f} ({lo:.3f}–{hi:.3f}) | {comp['p95_latency_ratio']:.3f} ({llo:.3f}–{lhi:.3f}) |")
    lines += ["", "## Preregistered performance gate", ""]
    for batch, gates in analyses["performance_gates"].items():
        lines.append(f"- Batch {batch}: adaptive vs selected fixed tree **{gates['vs_fixed_tree']}**; adaptive vs autoregressive overall-acceleration condition **{gates['vs_autoregressive']}**.")
    lines += ["", "A fixed-tree-only win is not an overall acceleration claim. Results are reported per batch size; no cross-batch pooling is used for pass/fail.",
              "", "## Calibration and adaptive selection", "",
              "Brier score and 10-bin reliability summaries are in `calibration_metrics.json`, grouped by method (fixed-action or scheduler-selected), batch size, branching, verifier depth, and outcome. The calibrator was fitted only on C01–C08 and was not recalibrated on locked evaluation groups.", "",
              "Adaptive action pulse counts (not per-depth decision-row counts) are included in `calibration_metrics.json`. The offline oracle is descriptive only.", "",
              "## Offline fixed-action oracle upper bound (descriptive)", "",
              "| Batch | Oracle tok/s | Selected action counts |", "|---:|---:|---|"]
    for batch, oracle in analyses["offline_oracle"].items():
        lines.append(f"| {batch} | {oracle['aggregate_tokens_per_second']:.3f} | `{json.dumps(oracle['selected_action_counts'], sort_keys=True)}` |")
    lines += ["", "## Limitations and retained evidence", "",
              "The oracle uses locked evaluation timings to select the fastest fixed action for each observed batch and is not an implementable policy. This is one CPU-only tiny-model fixture, with eight evaluation context groups and ten paired seed roots; uncertainty is cluster-resampled over contexts. llama.cpp server telemetry does not expose all draft KV/cache-copy counters; unavailable values remain null. Native KV bytes are estimates, not allocator telemetry. Component timers are inclusive/overlapping and are not additive.",
              "", "Raw JSONL, generated outputs, verifier observations, commands, logs, model/source hashes, calibrator coefficients, action costs, and freeze metadata are retained under this run directory.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    complete_path = run_dir / "evaluation" / "evaluation_complete.json"
    if not complete_path.exists():
        raise RuntimeError("refusing to read locked evaluation files before all methods complete")
    complete = read_json(complete_path)
    if complete.get("methods_complete") is not True:
        raise RuntimeError("evaluation completion marker is invalid")
    freeze = read_json(run_dir / "freeze.json")
    test_path = run_dir / "tests_status.json"
    test_status = read_json(test_path).get("status", "not recorded") if test_path.exists() else "not recorded"

    metrics_by_batch: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    oracle_results: dict[str, Any] = {}
    calibration_path = run_dir / "calibration" / "calibrator.tsv"
    for batch_size in BATCH_SIZES:
        paths = raw_paths(run_dir, batch_size, freeze)
        data: dict[str, dict[tuple[int, tuple[str, ...]], dict[str, Any]]] = {}
        metrics_by_batch[str(batch_size)] = {}
        for method, path in paths.items():
            rows = read_jsonl(path)
            indexed = {pair_key(row): row for row in rows}
            if len(indexed) != len(rows):
                raise RuntimeError(f"duplicate pair key in {path}")
            data[method] = indexed
            metrics_by_batch[str(batch_size)][method] = metric_summary(rows)
        clusters = validate_pairs(data)
        rng = random.Random(BOOTSTRAP_SEED)
        draws = [rng.choices(clusters, k=len(clusters)) for _ in range(BOOTSTRAPS)]
        comparison_rows = []
        for reference in ("fixed_selected", "autoregressive", "builtin"):
            comparison_rows.append(bootstrap_comparison(data, clusters, draws, "adaptive", reference))
        comparisons[str(batch_size)] = comparison_rows
        fixed_data = {name: data[name] for name in
                      ("fixed_B1_D2", "fixed_B2_D2", "fixed_B2_D3")}
        oracle_results[str(batch_size)] = offline_oracle(fixed_data)

    calibration = calibration_analysis(run_dir, calibration_path)
    performance_gates: dict[str, Any] = {}
    for batch, records in comparisons.items():
        fixed = next(row for row in records if row["reference"] == "fixed_selected")
        ar = next(row for row in records if row["reference"] == "autoregressive")
        performance_gates[batch] = {
            "vs_fixed_tree": "PASS" if fixed["throughput_ratio_95_ci"][0] > 1.0 and
                fixed["p95_latency_ratio_95_ci"][1] <= 1.10 else "FAIL / INCONCLUSIVE",
            "vs_autoregressive": "PASS" if ar["throughput_ratio_95_ci"][0] > 1.0 and
                ar["p95_latency_ratio_95_ci"][1] <= 1.10 else "FAIL / INCONCLUSIVE",
        }
    analyses = {
        "schema_version": 1, "evaluation_completed_at_utc": complete["completed_at_utc"],
        "freeze": freeze, "test_status": test_status,
        "metrics_by_batch_size": metrics_by_batch, "comparisons": comparisons,
        "performance_gates": performance_gates, "calibration": calibration,
        "offline_oracle": oracle_results,
        "bootstrap": {"draws": BOOTSTRAPS, "seed": BOOTSTRAP_SEED,
                      "cluster_unit": "prompt group at batch 1; predeclared adjacent prompt-group pair at batch 2"},
    }
    output_json = run_dir / "analysis" / "phase4_analysis.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analyses, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    (run_dir / "analysis" / "calibration_metrics.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = markdown_report(run_dir, analyses)
    (run_dir / "PHASE4_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(run_dir / "PHASE4_REPORT.md"),
                      "analysis": str(output_json), "performance_gates": performance_gates},
                     sort_keys=True))


if __name__ == "__main__":
    main()
