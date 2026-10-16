#!/usr/bin/env python3
"""Small deterministic tests for Phase 4 calibration fitting and locked analysis helpers."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = import_module("phase4_fit_test", HERE / "fit_calibration.py")
analysis = import_module("phase4_analysis_test", HERE / "analyze_phase4.py")


features = [(float(i), float((i * 7) % 11), float((i * 3) % 5)) for i in range(40)]
labels = [int(row[0] + 0.3 * row[1] - row[2] > 19.0) for row in features]
model_a = fit.fit_logistic(features, labels)
model_b = fit.fit_logistic(features, labels)
assert model_a == model_b, "logistic fitter must be deterministic"
probabilities = [analysis.predict(model_a, row) for row in features]
assert all(0.0 < probability < 1.0 for probability in probabilities)
assert probabilities[-1] > probabilities[0], "the synthetic signal should be learned in the expected direction"

seed_roots = [11, 23, 37]
clusters = [("E01",), ("E02",), ("E03",), ("E04",)]
methods = {"adaptive": {}, "fixed": {}}
for seed in seed_roots:
    for index, cluster in enumerate(clusters):
        row = {"seed_root": seed, "batch_prompt_ids": list(cluster),
               "total_completion_tokens": 64,
               "batch_completion_ms": 100.0 + index,
               "request_latency_ms": [100.0 + index]}
        methods["adaptive"][(seed, cluster)] = dict(row)
        methods["fixed"][(seed, cluster)] = dict(row)
rng = random.Random(5555)
draws = [rng.choices(clusters, k=len(clusters)) for _ in range(100)]
comparison = analysis.bootstrap_comparison(methods, clusters, draws, "adaptive", "fixed")
assert abs(comparison["throughput_ratio"] - 1.0) < 1e-12
assert abs(comparison["p95_latency_ratio"] - 1.0) < 1e-12
assert comparison["throughput_ratio_95_ci"] == [1.0, 1.0]
assert comparison["p95_latency_ratio_95_ci"] == [1.0, 1.0]
print("Phase 4 Python fitter/analysis tests: PASS")
