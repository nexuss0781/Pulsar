#!/usr/bin/env python3
"""Aggregate Stage 4 toy-fixture timing runs and write reproducible summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def optional_int(value: str) -> int | None:
    return int(value) if value and value != "NA" else None


def read_lscpu_model() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def verify_matched_runs(rows: list[dict[str, str]]) -> None:
    baselines: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    for row in rows:
        if row["kind"] == "baseline_ar":
            key = (row["prompt_id"], row["repetition"])
            if key in baselines:
                raise ValueError(f"duplicate AR baseline for {key}")
            baselines[key] = (row["seed"], row["token_limit"], row["batch_size"],
                              row["sampling_settings"], row["prompt_context"])
    if not baselines:
        raise ValueError("raw file has no AR baseline rows")
    for row in rows:
        key = (row["prompt_id"], row["repetition"])
        if key not in baselines:
            raise ValueError(f"missing paired AR baseline for {key}")
        expected = baselines[key]
        actual = (row["seed"], row["token_limit"], row["batch_size"],
                  row["sampling_settings"], row["prompt_context"])
        if actual != expected:
            raise ValueError(f"baseline/candidate run mismatch for {key}: {actual} != {expected}")
        if as_int(row, "output_tokens") != as_int(row, "token_limit"):
            raise ValueError(f"run did not meet its output-token limit: {key}")
        if row["sampling_settings"] != "categorical_temp1":
            raise ValueError(f"sampling setting mismatch for {key}: {row['sampling_settings']}")
    expected_grid = {(depth, branch, threshold)
                     for depth in (1, 2, 4)
                     for branch in (1, 2, 4)
                     for threshold in (0.0, 0.2, 0.4)}
    actual_grid = {(int(row["depth"]), int(row["branch_factor"]), float(row["threshold"]))
                   for row in rows if row["kind"] == "candidate_tree"}
    if actual_grid != expected_grid:
        raise ValueError(f"candidate grid is incomplete or unexpected: {actual_grid}")
    for config in expected_grid:
        config_rows = [row for row in rows if row["kind"] == "candidate_tree" and
                       (int(row["depth"]), int(row["branch_factor"]),
                        float(row["threshold"])) == config]
        config_keys = {(row["prompt_id"], row["repetition"]) for row in config_rows}
        if config_keys != set(baselines) or len(config_rows) != len(baselines):
            raise ValueError(f"candidate grid point {config} is not paired with all baselines")


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["kind"], row["depth"], row["branch_factor"], row["threshold"])
        groups[key].append(row)

    summaries: list[dict[str, object]] = []
    for (kind, depth, branch, threshold), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0] != "baseline_ar",
            int(item[0][1]) if item[0][1].isdigit() else 0,
            int(item[0][2]) if item[0][2].isdigit() else 0,
            float(item[0][3]) if item[0][3] not in ("NA", "") else -1.0,
        ),
    ):
        cycles = sum(as_int(row, "cycles") for row in group)
        emitted = sum(as_int(row, "output_tokens") for row in group)
        total_wall_ns = sum(as_int(row, "total_wall_ns") for row in group)
        cycle_ns = sum(as_int(row, "cycle_ns_total") for row in group)
        target_rows = sum(as_int(row, "target_rows") for row in group)
        draft_rows = sum(as_int(row, "draft_rows") for row in group)
        sample_ns = sum(as_int(row, "sample_ns") for row in group)
        target_callback_ns = sum(as_int(row, "target_callback_ns") for row in group)
        tree_nodes = sum(as_int(row, "selected_tree_nodes_total") for row in group)
        accepted_values = [optional_int(row["accepted_tokens"]) for row in group]
        has_acceptance = all(value is not None for value in accepted_values)
        accepted = sum(value or 0 for value in accepted_values)

        max_levels = max((len(row["tree_nodes_by_depth"].split(";"))
                          for row in group if row["tree_nodes_by_depth"]), default=0)
        level_totals = [0.0] * max_levels
        for row in group:
            encoded = row["tree_nodes_by_depth"].strip('"')
            if not encoded:
                continue
            for index, value in enumerate(encoded.split(";")):
                if index >= len(level_totals):
                    level_totals.extend([0.0] * (index + 1 - len(level_totals)))
                level_totals[index] += int(value)
        mean_nodes_by_depth = ";".join(
            f"{(value / cycles):.6f}" for value in level_totals
        ) if cycles and any(level_totals) else "NA"

        def total(field: str) -> int:
            return sum(as_int(row, field) for row in group)

        p50_runs = [as_int(row, "cycle_p50_ns") / 1000.0 for row in group]
        p90_runs = [as_int(row, "cycle_p90_ns") / 1000.0 for row in group]
        p95_runs = [as_int(row, "cycle_p95_ns") / 1000.0 for row in group]
        p99_runs = [as_int(row, "cycle_p99_ns") / 1000.0 for row in group]
        tree_bytes = [float(row["mean_selected_tree_payload_bytes"])
                      for row in group if row["mean_selected_tree_payload_bytes"] not in ("", "NA")]
        model_bytes = sorted({as_int(row, "model_storage_bytes") for row in group})
        seeds = sorted({int(row["seed"]) for row in group})
        checksums = [row["output_hash"] for row in group]

        summaries.append({
            "kind": kind,
            "depth": depth,
            "branch_factor": branch,
            "threshold": threshold,
            "run_count": len(group),
            "prompts": ";".join(sorted({row["prompt_id"] for row in group})),
            "batch_size": group[0]["batch_size"],
            "tokens_per_run": group[0]["token_limit"],
            "seed_set": ";".join(str(seed) for seed in seeds),
            "C_T_mean_us_per_cycle": cycle_ns / cycles / 1000.0 if cycles else math.nan,
            "Y_T_emitted_tokens_per_cycle": emitted / cycles if cycles else math.nan,
            "mean_selected_depth": (
                sum(float(row["mean_selected_depth"]) * as_int(row, "cycles") for row in group) / cycles
                if cycles and kind == "candidate_tree" else "NA"
            ),
            "accepted_tokens_per_cycle": accepted / cycles if cycles and has_acceptance else "NA",
            "accepted_fraction_of_emitted": accepted / emitted if emitted and has_acceptance else "NA",
            "end_to_end_tokens_per_second": emitted * 1e9 / total_wall_ns if total_wall_ns else math.nan,
            "mean_cycle_p50_us_across_runs": statistics.mean(p50_runs),
            "mean_cycle_p90_us_across_runs": statistics.mean(p90_runs),
            "mean_cycle_p95_us_across_runs": statistics.mean(p95_runs),
            "mean_cycle_p99_us_across_runs": statistics.mean(p99_runs),
            "draft_rows_per_cycle": draft_rows / cycles if cycles else math.nan,
            "draft_values_per_cycle": total("draft_values") / cycles if cycles else math.nan,
            "draft_callback_us_per_cycle": total("draft_callback_ns") / cycles / 1000.0 if cycles else math.nan,
            "tree_build_us_per_cycle": total("tree_build_ns") / cycles / 1000.0 if cycles else math.nan,
            "threshold_select_us_per_cycle": total("threshold_select_ns") / cycles / 1000.0 if cycles else math.nan,
            "target_rows_per_cycle": target_rows / cycles if cycles else math.nan,
            "target_values_per_cycle": total("target_values") / cycles if cycles else math.nan,
            "target_callback_us_per_cycle": target_callback_ns / cycles / 1000.0 if cycles else math.nan,
            "target_callback_us_per_row": target_callback_ns / target_rows / 1000.0 if target_rows else "NA",
            "sampler_excluding_target_callback_us_per_cycle": max(0, sample_ns - target_callback_ns) / cycles / 1000.0 if cycles else math.nan,
            "mean_selected_tree_nodes_per_cycle": tree_nodes / cycles if cycles else 0.0,
            "mean_selected_tree_nodes_by_depth": mean_nodes_by_depth,
            "max_selected_tree_nodes": max(as_int(row, "max_selected_tree_nodes") for row in group),
            "max_full_tree_payload_bytes": max(as_int(row, "max_full_tree_payload_bytes") for row in group),
            "max_selected_tree_payload_bytes": max(as_int(row, "max_selected_tree_payload_bytes") for row in group),
            "mean_selected_tree_payload_bytes_per_cycle": statistics.mean(tree_bytes) if tree_bytes else "NA",
            "model_storage_bytes": model_bytes[0],
            "kv_cache_bytes": "NA",
            "kv_cache_status": "unsupported: synthetic Markov fixture has no transformer KV cache",
            "mean_output_hash_across_runs": ";".join(checksums),
            "total_output_tokens": emitted,
            "total_cycle_ns": cycle_ns,
            "total_wall_ns": total_wall_ns,
        })
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        raise ValueError("no summary groups were produced")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def make_plots(summaries: list[dict[str, object]], output_dir: Path) -> None:
    baseline = next(item for item in summaries if item["kind"] == "baseline_ar")
    candidates = [item for item in summaries if item["kind"] == "candidate_tree"]
    colors = [float(item["threshold"]) for item in candidates]
    markers = {"1": "o", "2": "s", "4": "^"}

    fig, ax = plt.subplots(figsize=(10.6, 6.8), constrained_layout=True)
    for branch, marker in markers.items():
        subset = [item for item in candidates if item["branch_factor"] == branch]
        points = ax.scatter(
            [float(item["C_T_mean_us_per_cycle"]) for item in subset],
            [float(item["Y_T_emitted_tokens_per_cycle"]) for item in subset],
            c=[float(item["threshold"]) for item in subset], cmap="viridis",
            vmin=0.0, vmax=0.4, marker=marker, s=72, edgecolors="black",
            linewidths=0.35, label=f"branch B={branch}",
        )
    ax.scatter(float(baseline["C_T_mean_us_per_cycle"]),
               float(baseline["Y_T_emitted_tokens_per_cycle"]),
               marker="*", s=240, color="#c23b22", edgecolors="black",
               linewidths=0.5, label="AR baseline")
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Minimum retained level mass threshold")
    ax.set_xlabel("Measured C(T): mean wall time per pulse (µs)")
    ax.set_ylabel("Measured Y(T): emitted tokens per pulse")
    ax.set_title("Pulsar Stage 4 — measured cost/yield on the synthetic Markov fixture")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_dir / "stage4_cost_yield.png", dpi=180)
    fig.savefig(output_dir / "stage4_cost_yield.svg")
    plt.close(fig)

    candidates = sorted(candidates, key=lambda item: (
        float(item["threshold"]), int(item["depth"]), int(item["branch_factor"])))
    labels = [f"D{item['depth']}/B{item['branch_factor']}/θ{float(item['threshold']):.2f}"
              for item in candidates]
    values = [float(item["end_to_end_tokens_per_second"]) for item in candidates]
    fig, ax = plt.subplots(figsize=(13.5, 6.2), constrained_layout=True)
    xs = list(range(len(candidates)))
    ax.bar(xs, values, color=[plt.cm.viridis(float(item["threshold"]) / 0.4)
                              for item in candidates], edgecolor="#333333", linewidth=0.25)
    ax.axhline(float(baseline["end_to_end_tokens_per_second"]), color="#c23b22",
               linestyle="--", linewidth=1.6, label="AR baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Measured end-to-end output tokens/second")
    ax.set_title("Measured throughput by fixed tree configuration (toy fixture only)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "stage4_throughput.png", dpi=180)
    fig.savefig(output_dir / "stage4_throughput.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.6, 6.8), constrained_layout=True)
    for branch, marker in markers.items():
        subset = [item for item in candidates if item["branch_factor"] == branch]
        points = ax.scatter(
            [float(item["mean_selected_tree_nodes_per_cycle"]) for item in subset],
            [float(item["sampler_excluding_target_callback_us_per_cycle"]) for item in subset],
            c=[float(item["mean_selected_depth"]) for item in subset], cmap="plasma",
            vmin=1.0, vmax=4.0, marker=marker, s=72, edgecolors="black",
            linewidths=0.35, label=f"branch B={branch}",
        )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Mean selected depth")
    ax.set_xlabel("Mean selected tree nodes per pulse")
    ax.set_ylabel("Sampler wall time excluding target callback (µs/pulse)")
    ax.set_title("Verifier/sampling CPU proxy versus tree size (toy fixture)")
    ax.text(0.01, 0.99,
            "Includes verifier, RNG and tree-integrity work; excludes static target-row callback.",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_dir / "stage4_verifier_proxy.png", dpi=180)
    fig.savefig(output_dir / "stage4_verifier_proxy.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("cpp/stage4_raw.csv"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("cpp/inputs/model.bin"))
    args = parser.parse_args()
    raw_path = args.raw.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else raw_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("raw CSV has no rows")
    verify_matched_runs(rows)
    summaries = aggregate(rows)
    summary_path = out_dir / "stage4_summary.csv"
    write_summary_csv(summary_path, summaries)
    make_plots(summaries, out_dir)

    model_path = args.model.resolve()
    manifest = {
        "stage": 4,
        "fixture": "existing C++ inputs/model.bin; synthetic order-3 target/order-2 draft Markov toy model",
        "fixture_is_real_language_model": False,
        "batch_size": 1,
        "prompts": {
            "canonical_5_3_9": [5, 3, 9],
            "alternate_1_4_7": [1, 4, 7],
            "zero_0_0_0": [0, 0, 0],
        },
        "sampling": "Categorical sampling, temperature 1.0; AR samples p; tree uses full q as effective pi and Stage 3 exact rejection/fall-through sampler.",
        "fixed_grid": {"depth_positions_including_root": [1, 2, 4], "branch_factor": [1, 2, 4], "minimum_retained_level_mass": [0.0, 0.2, 0.4]},
        "threshold_definition": "After constructing the configured Stage 3 tree, retain complete levels while the cumulative normalized q mass over retained prefixes at the next level is at least the threshold; truncate the remaining levels. This threshold is benchmark-only, not a trained pulse rule.",
        "C_T_definition": "Observed steady-clock wall nanoseconds per pulse, including full configured-tree construction, threshold selection/truncation, sampling, and context/output updates.",
        "Y_T_definition": "Observed emitted tokens divided by pulse count.",
        "target_timing_note": "Target callback time measures toy-table lookup, context conversion, and row copy; sampler time minus that callback is an inclusive verifier/sampling/integrity-check proxy, not isolated verifier kernel time.",
        "draft_work_note": "Draft work is counted as toy-table row callbacks and returned probability values; no neural forward pass or FLOP count is represented.",
        "memory_note": "Tree payload bytes sum vector capacities (node structs, contexts, full q rows, and child indices); allocator metadata and process RSS are excluded.",
        "instrumentation_note": "Measured wall times include the steady-clock probes used for per-row callback timing; no instrumentation-overhead subtraction is applied.",
        "kv_cache_bytes": None,
        "kv_cache_status": "not supported or measured: this fixture has no transformer/KV cache",
        "model_storage_bytes": int(rows[0]["model_storage_bytes"]),
        "model_sha256": sha256(model_path),
        "raw_csv": str(raw_path),
        "raw_csv_sha256": sha256(raw_path),
        "summary_csv": str(summary_path.resolve()),
        "results_report": str((out_dir / "stage4_report.md").resolve()),
        "raw_run_count": len(rows),
        "summary_point_count": len(summaries),
        "summary_aggregation": "C(T), Y(T), work, and throughput are pooled from per-run totals; each latency percentile in summary is the mean of that per-run percentile across prompts/repetitions.",
        "reproduction_commands": [
            "cd /workspace/pulsar && mkdir -p paper/results/stage4 && cd cpp && make pulsar_stage4_bench CXXFLAGS='-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror'",
            "cd /workspace/pulsar/cpp && ./pulsar_stage4_bench --model inputs/model.bin --tokens 512 --repeats 3 --raw ../paper/results/stage4/stage4_raw.csv",
            "cd /workspace/pulsar && python3 verification/experiments/summarize_stage4.py --raw paper/results/stage4/stage4_raw.csv --out-dir paper/results/stage4",
        ],
        "compiler_flags": "-O2 -std=c++17 -march=native -Wall -Wextra -Wpedantic -Werror",
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "plots": ["stage4_cost_yield.png", "stage4_throughput.png", "stage4_verifier_proxy.png"],
        "hardware": {
            "platform": platform.platform(),
            "cpu_model": read_lscpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "compiler": subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        },
        "source_sha256": {
            "benchmark_cpp": sha256(ROOT / "cpp/pulsar_stage4_bench.cpp"),
            "threshold_header": sha256(ROOT / "cpp/pulsar_stage4_bench.hpp"),
            "stage3_tree_sampler": sha256(ROOT / "cpp/pulsar_tree_v3.hpp"),
            "v1_verifier": sha256(ROOT / "cpp/pulsar_core_v1.hpp"),
            "summarizer": sha256(Path(__file__).resolve()),
        },
    }
    metadata_path = out_dir / "stage4_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {len(rows)} paired raw runs; wrote {summary_path}")
    print(f"Plots: {out_dir / 'stage4_cost_yield.png'}, {out_dir / 'stage4_throughput.png'}, and {out_dir / 'stage4_verifier_proxy.png'}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
