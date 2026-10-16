#!/usr/bin/env python3
"""Fit frozen, deterministic logistic confidence models from Phase 4 calibration traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FEATURES = ("root_q_max", "root_q_margin", "root_q_entropy")
L2_STRENGTH = 1.0
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-10
SCALE_FLOOR = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with deterministic partial-pivot Gauss-Jordan."""
    size = len(vector)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: (abs(augmented[row][column]), -row))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ArithmeticError("singular logistic Hessian")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, size + 1):
            augmented[column][j] /= divisor
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            for j in range(column, size + 1):
                augmented[row][j] -= multiplier * augmented[column][j]
    return [augmented[row][size] for row in range(size)]


def fit_logistic(features: list[tuple[float, float, float]], labels: list[int]) -> dict[str, Any]:
    if not features or len(features) != len(labels):
        raise ValueError("logistic cell has no matched features and labels")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("logistic labels must be Boolean")
    means = [statistics.fmean(row[index] for row in features) for index in range(3)]
    scales = []
    for index in range(3):
        variance = statistics.fmean((row[index] - means[index]) ** 2 for row in features)
        scales.append(max(math.sqrt(max(0.0, variance)), SCALE_FLOOR))
    design = [[1.0] + [(row[index] - means[index]) / scales[index] for index in range(3)]
              for row in features]
    coefficients = [0.0, 0.0, 0.0, 0.0]
    n = len(labels)
    for _ in range(MAX_ITERATIONS):
        gradient = [0.0] * 4
        hessian = [[0.0] * 4 for _ in range(4)]
        for vector, label in zip(design, labels):
            score = sum(coefficients[j] * vector[j] for j in range(4))
            probability = sigmoid(score)
            residual = probability - label
            weight = max(probability * (1.0 - probability), 1e-12)
            for j in range(4):
                gradient[j] += residual * vector[j] / n
                for k in range(4):
                    hessian[j][k] += weight * vector[j] * vector[k] / n
        for j in range(1, 4):
            gradient[j] += L2_STRENGTH * coefficients[j]
            hessian[j][j] += L2_STRENGTH
        step = solve_linear(hessian, gradient)
        updated = [coefficients[j] - step[j] for j in range(4)]
        delta = max(abs(updated[j] - coefficients[j]) for j in range(4))
        coefficients = updated
        if delta <= CONVERGENCE_TOLERANCE:
            break
    else:
        raise RuntimeError("logistic model did not converge within 100 Newton steps")
    return {"observations": n, "means": means, "scales": scales,
            "coefficients": coefficients}


def load_observations(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"no calibration observation files found in {directory}")
    rows: list[dict[str, Any]] = []
    input_hashes: list[dict[str, str]] = []
    for path in paths:
        input_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "calibration" or row.get("mode") != "pulsar-tree":
                raise RuntimeError(f"non-calibration trace at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError("calibration observation files contain no rows")
    return rows, input_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations-dir", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()

    all_rows, input_hashes = load_observations(args.observations_dir)
    max_depth_by_branch: dict[int, int] = {}
    for row in all_rows:
        branch = int(row["branching"])
        max_depth_by_branch[branch] = max(max_depth_by_branch.get(branch, 0),
                                          int(row["configured_depth"]))
    selected = [row for row in all_rows
                if int(row["configured_depth"]) == max_depth_by_branch[int(row["branching"])]]
    cells: dict[tuple[int, int, int, str], list[tuple[tuple[float, float, float], int]]] = defaultdict(list)
    for row in selected:
        key_prefix = (int(row["batch_size"]), int(row["branching"]), int(row["depth_position"]))
        feature = tuple(float(row[name]) for name in FEATURES)
        cells[(*key_prefix, "accepted")].append((feature, int(bool(row["accepted"]))))
        if row["continuation_label_available"]:
            cells[(*key_prefix, "continued")].append((feature, int(bool(row["continued"]))))

    if not cells:
        raise RuntimeError("no calibration cells were produced")
    models: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for key in sorted(cells):
        samples = cells[key]
        models[key] = fit_logistic([row[0] for row in samples], [row[1] for row in samples])

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# Phase 4 calibrator: 15 tab-separated fields" + chr(10))
        for key in sorted(models):
            batch_size, branching, depth, outcome = key
            model = models[key]
            values: list[Any] = [batch_size, branching, depth, outcome, model["observations"]]
            for mean, scale in zip(model["means"], model["scales"]):
                values.extend([format(mean, ".17g"), format(scale, ".17g")])
            values.extend(format(value, ".17g") for value in model["coefficients"])
            stream.write(chr(9).join(str(value) for value in values) + chr(10))

    metadata = {
        "schema_version": 1,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_used": "calibration",
        "feature_order": list(FEATURES),
        "outcomes": ["accepted", "continued"],
        "l2_strength": L2_STRENGTH,
        "intercept_penalty": 0.0,
        "feature_scale_floor": SCALE_FLOOR,
        "optimizer": "deterministic Newton-Raphson with partial-pivot Gauss-Jordan; at most 100 steps; max coefficient delta <= 1e-10",
        "configured_depth_selection": "for each branching factor, use only its largest predeclared tree depth to avoid duplicate calibration observations from overlapping action configs",
        "training_observation_count_all_actions": len(all_rows),
        "training_observation_count_selected": len(selected),
        "input_hashes": input_hashes,
        "calibrator_tsv_sha256": sha256_file(args.output_tsv),
        "cells": [
            {"batch_size": key[0], "branching": key[1], "depth_position": key[2],
             "outcome": key[3], **models[key]}
            for key in sorted(models)
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(json.dumps({"cells": len(models), "observations": len(selected),
                      "calibrator": str(args.output_tsv),
                      "metadata": str(args.metadata)}, sort_keys=True))


if __name__ == "__main__":
    main()
