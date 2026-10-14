"""Stage 5: grouped acceptance calibration and independent scheduling policies.

This module is experimental and does not modify the frozen Stage 1–4
samplers. Features are draft/context-side only; labels are binary outcomes
returned by the exact verifier. Group splitting is explicit to prevent
near-duplicate nodes from one context leaking across calibration folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class NodeExample:
    features: tuple[float, ...]
    accepted: int
    prompt_group: str
    context_group: str
    pulse_group: str
    depth: int
    regime: str
    selected: bool = False

    def __post_init__(self):
        if self.accepted not in (0, 1):
            raise ValueError("accepted label must be binary")
        if not self.features or not all(math.isfinite(x) for x in self.features):
            raise ValueError("features must be a non-empty finite vector")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")


def draft_features(q: Sequence[float], context: Sequence[int], depth: int,
                   vocab_size: int | None = None) -> tuple[float, ...]:
    """Cheap inference-time features; never reads the target distribution.

    The base six features are bias, normalized draft entropy, top probability,
    top-two margin, normalized oldest-token ID (a context proxy), and depth.
    Thirteen deterministic polynomial interactions let a linear calibrator
    model workload/context-dependent effects without reading target rows.
    """
    row = np.asarray(q, dtype=np.float64)
    if row.ndim != 1 or row.size == 0 or np.any(~np.isfinite(row)) or np.any(row < 0):
        raise ValueError("q must be a finite non-negative vector")
    total = float(row.sum())
    if total <= 0:
        raise ValueError("q must have positive mass")
    row = row / total
    ranked = np.sort(row)
    entropy = float(-(row[row > 0] * np.log(row[row > 0])).sum())
    entropy /= math.log(row.size) if row.size > 1 else 1.0
    top = float(ranked[-1])
    second = float(ranked[-2]) if row.size > 1 else 0.0
    v = int(vocab_size or row.size)
    oldest = int(context[-3]) if len(context) >= 3 else (int(context[0]) if context else 0)
    oldest_proxy = oldest / max(v - 1, 1)
    margin = top - second
    d = float(depth)
    return (1.0, entropy, top, margin, oldest_proxy, d,
            entropy * entropy, top * top, margin * margin,
            entropy * top, entropy * margin, top * margin,
            entropy * oldest_proxy, top * oldest_proxy, margin * oldest_proxy,
            d * entropy, d * top, d * margin, d * oldest_proxy)


def split_by_groups(examples: Sequence[NodeExample], holdout_fraction: float = 0.25,
                    seed: int = 0, group_by: str = "context_group") -> tuple[np.ndarray, np.ndarray]:
    """Return train/eval row indices with no group crossing the split."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if group_by not in ("context_group", "prompt_group"):
        raise ValueError("group_by must be context_group or prompt_group")
    groups = sorted({getattr(row, group_by) for row in examples})
    if len(groups) < 2:
        raise ValueError("at least two distinct groups are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_eval = min(len(groups) - 1, max(1, int(round(len(groups) * holdout_fraction))))
    eval_groups = {groups[int(i)] for i in order[:n_eval]}
    train_idx = np.asarray([i for i, row in enumerate(examples)
                            if getattr(row, group_by) not in eval_groups], dtype=np.int64)
    eval_idx = np.asarray([i for i, row in enumerate(examples)
                           if getattr(row, group_by) in eval_groups], dtype=np.int64)
    if not len(train_idx) or not len(eval_idx):
        raise ValueError("group split produced an empty partition")
    return train_idx, eval_idx


class AcceptancePredictor:
    """L2-regularized logistic acceptance predictor with train-only scaling."""

    def __init__(self, learning_rate: float = 0.15, l2: float = 1e-3,
                 epochs: int = 1200):
        if learning_rate <= 0 or l2 < 0 or epochs <= 0:
            raise ValueError("invalid optimizer settings")
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weight: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, features: np.ndarray, labels: Sequence[int]) -> "AcceptancePredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or len(y) < 2:
            raise ValueError("fit expects a 2-D feature matrix and at least two labels")
        if np.any(~np.isfinite(x)) or np.any((y != 0) & (y != 1)):
            raise ValueError("features must be finite and labels must be binary")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        z = (x - self.mean) / self.scale
        self.weight = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = math.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6))
        # Full-batch deterministic gradient descent keeps the fitted model reproducible.
        for _ in range(self.epochs):
            p = self._sigmoid(z @ self.weight + self.bias)
            residual = p - y
            self.weight -= self.learning_rate * (z.T @ residual / len(y) + self.l2 * self.weight)
            self.bias -= self.learning_rate * float(residual.mean())
        return self

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weight is None:
            raise RuntimeError("predictor has not been fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.weight):
            raise ValueError("feature shape does not match the fitted predictor")
        return self._sigmoid(((x - self.mean) / self.scale) @ self.weight + self.bias)


def brier_score(probabilities: Sequence[float], labels: Sequence[int],
                weights: Sequence[float] | None = None) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities and labels must be non-empty matching vectors")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be binary")
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != p.shape or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and have positive mass")
    return float(np.average((p - y) ** 2, weights=w))


def calibration_bins(probabilities: Sequence[float], labels: Sequence[int],
                     bins: int = 10, weights: Sequence[float] | None = None) -> list[dict[str, float | int]]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0 or bins <= 0:
        raise ValueError("invalid probability/label vectors or bin count")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("probabilities must be in [0, 1] and labels binary")
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != p.shape or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid calibration weights")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, float | int]] = []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1])) & (w > 0)
        if np.any(mask):
            mass = float(w[mask].sum())
            result.append({"bin": i, "count": int(mask.sum()), "weight": mass,
                           "mean_confidence": float(np.average(p[mask], weights=w[mask])),
                           "observed_acceptance": float(np.average(y[mask], weights=w[mask]))})
    return result


def calibration_report(probabilities: Sequence[float], labels: Sequence[int],
                       bins: int = 10, weights: Sequence[float] | None = None) -> dict[str, object]:
    table = calibration_bins(probabilities, labels, bins=bins, weights=weights)
    total = sum(float(item["weight"]) for item in table)
    ece = sum(float(item["weight"]) * abs(float(item["mean_confidence"]) -
              float(item["observed_acceptance"])) for item in table) / total
    return {"n": len(probabilities), "brier": brier_score(probabilities, labels, weights),
            "ece": ece, "bins": table}


def clustered_brier_delta(probabilities: Sequence[float], baseline: Sequence[float],
                          labels: Sequence[int], groups: Sequence[str],
                          samples: int = 2000, seed: int = 0) -> dict[str, float | int]:
    """Cluster-bootstrap the paired Brier difference (model minus baseline).

    Resampling whole context groups preserves within-context dependence. A
    negative estimate/interval favors ``probabilities``. The baseline must be
    declared independently of the held-out labels; this helper does not select
    a baseline for the caller.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    y_raw = np.asarray(labels, dtype=np.float64)
    group = np.asarray(groups, dtype=str)
    if (p.ndim != 1 or p.shape != base.shape or p.shape != y_raw.shape or
            p.shape != group.shape or len(p) == 0 or samples < 100):
        raise ValueError("paired Brier bootstrap requires matching vectors and at least 100 samples")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.any(~np.isfinite(base)) or np.any((base < 0) | (base > 1)):
        raise ValueError("baseline probabilities must be finite and in [0, 1]")
    if np.any(~np.isfinite(y_raw)) or np.any((y_raw != 0) & (y_raw != 1)):
        raise ValueError("labels must be binary")
    y = y_raw.astype(np.int64)
    unique, inverse = np.unique(group, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two distinct groups")
    observed = brier_score(p, y) - brier_score(base, y)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        multiplicity = np.bincount(rng.integers(0, len(unique), size=len(unique)),
                                   minlength=len(unique))
        weights = multiplicity[inverse].astype(np.float64)
        deltas[i] = brier_score(p, y, weights) - brier_score(base, y, weights)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"estimate": float(observed), "ci95_low": float(low),
            "ci95_high": float(high), "n": int(len(p)),
            "groups": int(len(unique)), "bootstrap_samples": int(samples)}


def paired_bootstrap_speedup(candidate_tokens: Sequence[int], candidate_wall_ns: Sequence[int],
                             baseline_tokens: Sequence[int], baseline_wall_ns: Sequence[int],
                             groups: Sequence[str], samples: int = 2000,
                             seed: int = 0) -> dict[str, float | int]:
    """Cluster-stratified paired bootstrap for candidate throughput improvement.

    The reported effect is percent change in aggregate tokens/second. Episodes
    are paired by their caller-provided order and resampled within each group
    (normally prompt/workload), reducing sensitivity to run-order noise.
    """
    ct = np.asarray(candidate_tokens, dtype=np.float64)
    cw = np.asarray(candidate_wall_ns, dtype=np.float64)
    bt = np.asarray(baseline_tokens, dtype=np.float64)
    bw = np.asarray(baseline_wall_ns, dtype=np.float64)
    group = np.asarray(groups, dtype=str)
    n = len(ct)
    if (n == 0 or any(x.shape != (n,) for x in (cw, bt, bw, group)) or samples < 100):
        raise ValueError("paired speedup bootstrap requires matching vectors and at least 100 samples")
    if (np.any(~np.isfinite(ct)) or np.any(~np.isfinite(bt)) or
            np.any(~np.isfinite(cw)) or np.any(~np.isfinite(bw)) or
            np.any(ct < 0) or np.any(bt < 0) or np.any(cw <= 0) or np.any(bw <= 0)):
        raise ValueError("tokens must be non-negative and wall times finite and positive")
    unique = np.unique(group)
    if len(unique) < 2:
        raise ValueError("paired bootstrap requires at least two distinct groups")
    strata = [np.flatnonzero(group == item) for item in unique]

    def effect(indices: np.ndarray) -> float:
        candidate_rate = ct[indices].sum() / cw[indices].sum()
        baseline_rate = bt[indices].sum() / bw[indices].sum()
        return 100.0 * (candidate_rate / baseline_rate - 1.0)

    all_indices = np.arange(n)
    observed = effect(all_indices)
    rng = np.random.default_rng(seed)
    effects = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draws = [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        effects[i] = effect(np.concatenate(draws))
    low, high = np.quantile(effects, [0.025, 0.975])
    return {"estimate_percent": float(observed), "ci95_low_percent": float(low),
            "ci95_high_percent": float(high), "paired_runs": int(n),
            "groups": int(len(unique)), "bootstrap_samples": int(samples)}


@dataclass(frozen=True)
class TreeAction:
    depth: int
    branch_factor: int
    threshold: float
    name: str = ""

    def __post_init__(self):
        if self.depth < 1 or self.branch_factor < 0 or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("invalid fixed-tree action")


class ConfidenceScheduler:
    """Static confidence thresholds mapping a calibrated score to a tree action."""
    def __init__(self, actions: Sequence[TreeAction], thresholds: Sequence[float]):
        if not actions or len(thresholds) != len(actions) - 1:
            raise ValueError("need one fewer confidence threshold than actions")
        if any(not 0 <= t <= 1 for t in thresholds) or any(
                thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("confidence thresholds must be ordered in [0, 1]")
        if any(actions[i].depth > actions[i + 1].depth for i in range(len(actions) - 1)):
            raise ValueError("actions must be ordered by non-decreasing depth")
        self.actions = tuple(actions)
        self.thresholds = tuple(float(t) for t in thresholds)

    def select(self, confidence: float) -> TreeAction:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        index = sum(confidence >= threshold for threshold in self.thresholds)
        return self.actions[index]


class EMAThresholdPolicy:
    """Independent EMA policy; update only with observed verifier outcomes."""
    def __init__(self, actions: Sequence[TreeAction], thresholds: Sequence[float],
                 alpha: float = 0.2, initial: float = 0.5):
        if (not math.isfinite(alpha) or not 0 < alpha <= 1 or
                not math.isfinite(initial) or not 0 <= initial <= 1 or
                not actions or len(thresholds) != len(actions) - 1):
            raise ValueError("invalid EMA policy settings")
        if any(not 0 <= t <= 1 for t in thresholds) or any(
                thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("EMA thresholds must be in [0, 1]")
        if any(actions[i].depth > actions[i + 1].depth for i in range(len(actions) - 1)):
            raise ValueError("actions must be ordered by non-decreasing depth")
        self.actions = tuple(actions)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.alpha = alpha
        self.ema = float(initial)
        self.updates = 0

    def select(self) -> TreeAction:
        return self.actions[sum(self.ema >= t for t in self.thresholds)]

    def update(self, accepted: int) -> None:
        if accepted not in (0, 1):
            raise ValueError("EMA outcome must be binary")
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * accepted
        self.updates += 1


class UCB1BanditPolicy:
    """Independent UCB1 policy over fixed actions, using measured scalar rewards."""
    def __init__(self, actions: Sequence[TreeAction], exploration: float = 2.0):
        if not actions or not math.isfinite(exploration) or exploration < 0:
            raise ValueError("bandit needs actions and non-negative exploration")
        self.actions = tuple(actions)
        self.exploration = exploration
        self.counts = np.zeros(len(actions), dtype=np.int64)
        self.reward_sums = np.zeros(len(actions), dtype=np.float64)
        self.total_updates = 0

    def select_index(self) -> int:
        for i, count in enumerate(self.counts):
            if count == 0:
                return i
        means = self.reward_sums / self.counts
        bonus = np.sqrt(self.exploration * math.log(max(self.total_updates, 1)) / self.counts)
        return int(np.argmax(means + bonus))

    def select(self) -> TreeAction:
        return self.actions[self.select_index()]

    def update(self, action_index: int, measured_reward: float) -> None:
        if not 0 <= action_index < len(self.actions):
            raise ValueError("unknown bandit action")
        if not math.isfinite(measured_reward):
            raise ValueError("bandit reward must be finite")
        self.counts[action_index] += 1
        self.reward_sums[action_index] += measured_reward
        self.total_updates += 1


def selected_mask(examples: Sequence[NodeExample], root_confidence: Sequence[float],
                  scheduler: ConfidenceScheduler) -> np.ndarray:
    """Mark nodes retained by a root-confidence depth decision within each pulse.

    `examples` must be ordered in contiguous pulse blocks and each block must
    start at depth zero. The selected-node distribution can then be scored
    with calibration_report(weights=selected_mask(...)).
    """
    conf = np.asarray(root_confidence, dtype=np.float64)
    if conf.shape != (len(examples),):
        raise ValueError("one root-confidence value is required per example")
    mask = np.zeros(len(examples), dtype=np.float64)
    pulse_start = 0
    while pulse_start < len(examples):
        if examples[pulse_start].depth != 0:
            raise ValueError("each pulse block must start at depth zero")
        pulse_id = examples[pulse_start].pulse_group
        end = pulse_start + 1
        while end < len(examples) and examples[end].pulse_group == pulse_id:
            end += 1
        # A unique pulse_group id is expected for every pulse block.
        limit = scheduler.select(float(conf[pulse_start])).depth
        for i in range(pulse_start, end):
            if examples[i].depth < limit:
                mask[i] = 1.0
        pulse_start = end
    return mask
