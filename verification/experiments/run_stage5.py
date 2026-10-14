#!/usr/bin/env python3
"""Stage 5 grouped calibration and adaptive-policy experiment on model.bin.

Python collects calibration/audit evidence; fixed and adaptive policy timing
uses the isolated native runner by default and reports its distinction from
Stage 4's C++ eager-tree benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pulsar.stage5 import (  # noqa: E402
    AcceptancePredictor, ConfidenceScheduler, EMAThresholdPolicy, NodeExample,
    TreeAction, UCB1BanditPolicy,
    calibration_report, clustered_brier_delta, draft_features,
    paired_bootstrap_speedup, split_by_groups,
)


class ToyModel:
    def __init__(self, path: Path):
        data = path.read_bytes()
        header_size = 6 * 8
        float_header_size = 7 * 8
        if len(data) < header_size + float_header_size:
            raise ValueError("model fixture is truncated")
        dims = struct.unpack_from("<6q", data, 0)
        self.v, _, _, _, wdim, nis = map(int, dims)
        if self.v <= 0 or wdim <= 0 or nis <= 0:
            raise ValueError("model fixture header has invalid dimensions")
        offset = header_size + float_header_size
        counts = (wdim, nis, nis, self.v ** 4, self.v ** 3)
        arrays = []
        for count in counts:
            byte_count = count * 8
            if offset + byte_count > len(data):
                raise ValueError("model fixture arrays are truncated")
            arrays.append(np.frombuffer(data, dtype="<f8", count=count, offset=offset).copy())
            offset += byte_count
        if offset != len(data):
            raise ValueError("model fixture contains unexpected trailing bytes")
        self.p = arrays[3].reshape(self.v ** 3, self.v)
        self.q = arrays[4].reshape(self.v ** 2, self.v)
        self._q_order = np.argsort(-self.q, axis=1, kind="stable")

    def ctx2(self, context: tuple[int, ...]) -> int:
        tail = (0, 0) + tuple(context[-2:])
        return tail[-2] * self.v + tail[-1]

    def ctx2(self, context: tuple[int, ...]) -> int:
        tail = (0, 0) + tuple(context[-2:])
        b, c = tail[-2:]
        return b * self.v + c

    def ctx3(self, context: tuple[int, ...]) -> int:
        tail = context[-3:]
        if len(tail) < 3:
            tail = (0,) * (3 - len(tail)) + tail
        a, b, c = tail
        return (a * self.v + b) * self.v + c

    def qrow(self, context: tuple[int, ...]) -> np.ndarray:
        return self.q[self.ctx2(context)]

    def retained_tokens(self, context: tuple[int, ...], branch: int) -> list[int]:
        row_index = self.ctx2(context)
        return [int(i) for i in self._q_order[row_index]
                if self.q[row_index, i] > 0][:branch]

    def prow(self, context: tuple[int, ...]) -> np.ndarray:
        return self.p[self.ctx3(context)]

    def context_group(self, context: tuple[int, ...]) -> str:
        tail = (0, 0) + tuple(context[-2:])
        return f"{tail[-2]}:{tail[-1]}"


def sample_cat(row: np.ndarray, rng: np.random.Generator) -> int:
    return min(int(np.searchsorted(np.cumsum(row), rng.random(), side="right")), len(row) - 1)


def retained_tokens(q: np.ndarray, branch: int) -> list[int]:
    ids = np.argsort(-q, kind="stable")
    return [int(i) for i in ids if q[i] > 0][:branch]


def precompute_root_confidence(model: ToyModel,
                               predictor: AcceptancePredictor) -> np.ndarray:
    """Compile draft-only root scores for every order-3 context.

    The feature vector depends only on the last three context tokens and the
    order-2 draft row, so this table is an exact memoization of predictor
    inference for this model. Construction is outside measured policy time.
    """
    first_context = (0, 0, 0)
    feature_count = len(draft_features(model.qrow(first_context), first_context, 0, model.v))
    features = np.empty((model.v ** 3, feature_count), dtype=np.float64)
    for index in range(model.v ** 3):
        a, rem = divmod(index, model.v ** 2)
        b, c = divmod(rem, model.v)
        context = (a, b, c)
        features[index] = draft_features(model.qrow(context), context, 0, model.v)
    return predictor.predict(features)


def precompute_oracle_confidence(model: ToyModel) -> np.ndarray:
    """Exact overlap by context; privileged oracle comparator, never a learned feature."""
    result = np.empty(model.v ** 3, dtype=np.float64)
    for index in range(model.v ** 3):
        a, rem = divmod(index, model.v ** 2)
        b, c = divmod(rem, model.v)
        context = (a, b, c)
        p = model.prow(context)
        q = model.qrow(context)
        result[index] = float(np.minimum(p / p.sum(), q / q.sum()).sum())
    return result


def precompute_action_lookup(model: ToyModel, root_confidence: np.ndarray,
                             heldout_groups: set[str], scheduler: ConfidenceScheduler,
                             fallback_action: TreeAction) -> tuple[TreeAction, ...]:
    """Compile the confidence policy and group fallback into a context LUT."""
    if root_confidence.shape != (model.v ** 3,):
        raise ValueError("root-confidence table has the wrong shape")
    actions: list[TreeAction] = []
    for index in range(model.v ** 3):
        a, rem = divmod(index, model.v ** 2)
        b, c = divmod(rem, model.v)
        if model.context_group((a, b, c)) in heldout_groups:
            actions.append(scheduler.select(float(root_confidence[index])))
        else:
            actions.append(fallback_action)
    return tuple(actions)


def precompute_binned_action_lookup(model: ToyModel, scores: np.ndarray,
                                    heldout_groups: set[str], edges: np.ndarray,
                                    actions_by_bin: tuple[TreeAction, ...],
                                    fallback_action: TreeAction,
                                    group_actions: dict[tuple[str, int], TreeAction] | None = None
                                    ) -> tuple[TreeAction, ...]:
    """Compile a draft-only confidence-bin utility policy into a context LUT."""
    if scores.shape != (model.v ** 3,) or len(actions_by_bin) != len(edges) + 1:
        raise ValueError("score-bin policy dimensions do not match the model")
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("score-bin edges must be finite and strictly increasing")
    actions: list[TreeAction] = []
    for index in range(model.v ** 3):
        a, rem = divmod(index, model.v ** 2)
        b, c = divmod(rem, model.v)
        group = model.context_group((a, b, c))
        if group in heldout_groups:
            bin_index = int(np.searchsorted(edges, float(scores[index]), side="right"))
            actions.append((group_actions or {}).get((group, bin_index),
                            actions_by_bin[bin_index]))
        else:
            actions.append(fallback_action)
    return tuple(actions)


def fit_bin_actions(trace_paths: list[Path], model: ToyModel, scores: np.ndarray,
                    heldout_groups: set[str], edges: np.ndarray,
                    actions: tuple[TreeAction, ...], fallback_action: TreeAction
                    ) -> tuple[tuple[TreeAction, ...], list[dict[str, object]],
                               dict[tuple[str, int], TreeAction], list[dict[str, object]]]:
    """Fit confidence-bin actions from independent tuning pulse throughput.

    Per-bin rates shrink toward each action's global rate to reduce noisy
    context-bin choices. An action replaces the fixed fallback only when its
    shrunk local rate is at least 0.5% better and has support in the traces.
    """
    action_keys = {(action.depth, action.branch_factor, action.threshold): i
                   for i, action in enumerate(actions)}
    bins = len(edges) + 1
    token_sums = np.zeros((bins, len(actions)), dtype=np.float64)
    wall_sums = np.zeros_like(token_sums)
    pulse_counts = np.zeros_like(token_sums)
    global_tokens = np.zeros(len(actions), dtype=np.float64)
    global_wall = np.zeros(len(actions), dtype=np.float64)
    group_cells: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for path in trace_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                state = int(row["context3"])
                a, rem = divmod(state, model.v ** 2)
                b, c = divmod(rem, model.v)
                context_group = model.context_group((a, b, c))
                if context_group not in heldout_groups:
                    continue
                action_index = action_keys.get((int(row["depth"]), int(row["branch"]),
                                                float(row["threshold"])))
                if action_index is None:
                    continue
                tokens = float(row["emitted_tokens"])
                wall_ns = float(row["elapsed_ns"])
                if tokens <= 0 or wall_ns <= 0:
                    continue
                bin_index = int(np.searchsorted(edges, float(scores[state]), side="right"))
                key = (context_group, bin_index)
                if key not in group_cells:
                    group_cells[key] = (np.zeros(len(actions), dtype=np.float64),
                                        np.zeros(len(actions), dtype=np.float64),
                                        np.zeros(len(actions), dtype=np.float64))
                group_token_sums, group_wall_sums, group_counts = group_cells[key]
                token_sums[bin_index, action_index] += tokens
                wall_sums[bin_index, action_index] += wall_ns
                pulse_counts[bin_index, action_index] += 1
                group_token_sums[action_index] += tokens
                group_wall_sums[action_index] += wall_ns
                group_counts[action_index] += 1
                global_tokens[action_index] += tokens
                global_wall[action_index] += wall_ns
    global_rates = global_tokens / np.maximum(global_wall, 1.0)
    fallback_index = action_keys.get((fallback_action.depth, fallback_action.branch_factor,
                                      fallback_action.threshold))
    fallback_global_rate = (float(global_rates[fallback_index]) if fallback_index is not None else 0.0)
    selected: list[TreeAction] = []
    diagnostics: list[dict[str, object]] = []
    shrunk_by_bin: list[np.ndarray] = []
    for bin_index in range(bins):
        local_rates = token_sums[bin_index] / np.maximum(wall_sums[bin_index], 1.0)
        counts = pulse_counts[bin_index]
        weights = counts / (counts + 25.0)
        shrunk_rates = weights * local_rates + (1.0 - weights) * global_rates
        shrunk_by_bin.append(shrunk_rates)
        if fallback_index is None:
            reference_rate = fallback_global_rate
        else:
            reference_rate = float(shrunk_rates[fallback_index])
        best_index = int(np.argmax(shrunk_rates))
        use_best = (counts[best_index] >= 20.0 and
                    float(shrunk_rates[best_index]) >= reference_rate * 1.005)
        choice = actions[best_index] if use_best else fallback_action
        selected.append(choice)
        diagnostics.append({
            "bin": bin_index,
            "pulse_counts_by_action": {action.name: int(counts[i]) for i, action in enumerate(actions)},
            "shrunk_tokens_per_ns_by_action": {
                action.name: float(shrunk_rates[i]) for i, action in enumerate(actions)},
            "selected_action": choice.name,
            "selected_by_local_utility": bool(use_best),
            "fixed_reference_tokens_per_ns": reference_rate,
        })
    group_actions: dict[tuple[str, int], TreeAction] = {}
    group_diagnostics: list[dict[str, object]] = []
    for (context_group, bin_index), (tokens, wall, counts) in group_cells.items():
        local_rates = tokens / np.maximum(wall, 1.0)
        weights = counts / (counts + 50.0)
        shrunk_rates = weights * local_rates + (1.0 - weights) * shrunk_by_bin[bin_index]
        base_action = selected[bin_index]
        base_index = action_keys[(base_action.depth, base_action.branch_factor,
                                  base_action.threshold)]
        best_index = int(np.argmax(shrunk_rates))
        use_local = (counts[best_index] >= 20.0 and
                     float(shrunk_rates[best_index]) >= float(shrunk_rates[base_index]) * 1.005)
        if use_local and actions[best_index] != base_action:
            group_actions[(context_group, bin_index)] = actions[best_index]
            group_diagnostics.append({
                "context_group": context_group, "bin": bin_index,
                "selected_action": actions[best_index].name,
                "pulse_count": int(counts[best_index]),
                "shrunk_tokens_per_ns": float(shrunk_rates[best_index]),
                "bin_action_reference_tokens_per_ns": float(shrunk_rates[base_index]),
            })
    return tuple(selected), diagnostics, group_actions, group_diagnostics


def choose_depth(tree: list[dict], configured_depth: int, threshold: float) -> int:
    reach = [0.0] * len(tree)
    reach[0] = 1.0
    selected = 1
    for depth in range(1, configured_depth):
        level_mass = 0.0
        for parent_index, node in enumerate(tree):
            if node["depth"] + 1 != depth:
                continue
            q = node["q"]
            total = float(q.sum())
            for child_index, token in node["children"]:
                mass = reach[parent_index] * float(q[token]) / total
                reach[child_index] = mass
                level_mass += mass
        if level_mass < threshold:
            break
        selected = depth + 1
    return selected


def build_tree(model: ToyModel, context: tuple[int, ...], action: TreeAction) -> tuple[list[dict], int]:
    tree = [{"context": context, "depth": 0, "q": model.qrow(context), "children": []}]
    for parent_index, node in enumerate(tree):
        if node["depth"] + 1 >= action.depth:
            continue
        for token in model.retained_tokens(node["context"], action.branch_factor):
            child_context = node["context"] + (token,)
            child_index = len(tree)
            tree.append({"context": child_context, "depth": node["depth"] + 1,
                         "q": model.qrow(child_context), "children": []})
            tree[parent_index]["children"].append((child_index, token))
    selected = choose_depth(tree, action.depth, action.threshold)
    if selected < action.depth:
        tree = [node for node in tree if node["depth"] < selected]
        for node in tree:
            node["children"] = [(i, token) for i, token in node["children"] if i < len(tree)]
    return tree, selected


def verifier_step(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> tuple[int, bool, int]:
    proposal = sample_cat(q / q.sum(), rng)
    accept_prob = min(1.0, float(p[proposal]) / max(float(q[proposal]), 1e-300))
    accepted = bool(rng.random() < accept_prob)
    if accepted:
        return proposal, True, proposal
    residual = np.maximum(p - q, 0.0)
    total = float(residual.sum())
    if total == 0.0:
        raise RuntimeError("rejection occurred with zero residual mass")
    replacement = sample_cat(residual / total, rng)
    return replacement, False, proposal


def run_pulse(model: ToyModel, context: tuple[int, ...], action: TreeAction,
              rng: np.random.Generator, prompt_id: str, pulse_id: str,
              examples: list[NodeExample] | None = None, eager_tree: bool = False
              ) -> tuple[list[int], int, list[tuple[tuple[int, ...], float]], int, list[int]]:
    # With threshold zero the selected depth is always the configured depth.
    # Only the verifier's accepted path can be visited, so descendants of
    # unvisited siblings need not be materialized. This is distributionally
    # identical to the eager Stage 3 tree; nonzero thresholds stay eager.
    lazy = action.threshold == 0.0 and not eager_tree
    if lazy:
        tree: list[dict] = []
        selected_depth = action.depth
    else:
        tree, selected_depth = build_tree(model, context, action)
    emitted: list[int] = []
    accepted_count = 0
    visited: list[tuple[tuple[int, ...], float]] = []
    verifier_labels: list[int] = []
    node_index = 0
    node_context = context
    for depth in range(selected_depth):
        if lazy:
            q = model.qrow(node_context)
        else:
            node = tree[node_index]
            node_context = node["context"]
            q = node["q"]
        p = model.prow(node_context)
        token, accepted, proposal = verifier_step(p, q, rng)
        if examples is not None:
            alpha = float(np.minimum(p / p.sum(), q / q.sum()).sum())
            feats = draft_features(q, node_context, depth, model.v)
            visited.append((node_context, alpha))
            examples.append(NodeExample(
                features=feats, accepted=int(accepted), prompt_group=prompt_id,
                context_group=model.context_group(node_context), pulse_group=pulse_id,
                depth=depth, regime="high_overlap" if alpha >= 0.70 else "low_overlap"))
        verifier_labels.append(int(accepted))
        emitted.append(token)
        if not accepted:
            break
        accepted_count += 1
        if depth + 1 == selected_depth:
            break
        if lazy:
            if proposal not in model.retained_tokens(node_context, action.branch_factor):
                break
            node_context = node_context + (proposal,)
        else:
            child = next((index for index, child_token in node["children"] if child_token == proposal), None)
            if child is None:
                break
            node_index = child
    return emitted, accepted_count, visited, selected_depth, verifier_labels


def run_stream(model: ToyModel, prompt: tuple[int, ...], prompt_id: str, seed: int,
               token_limit: int, fixed_action: TreeAction | None = None,
               predictor: AcceptancePredictor | None = None,
               confidence_lookup: np.ndarray | None = None,
               action_lookup: tuple[TreeAction, ...] | None = None,
               scheduler: ConfidenceScheduler | None = None,
               heldout_groups: set[str] | None = None,
               fallback_action: TreeAction | None = None,
               online_policy: str | None = None,
               oracle: bool = False, examples: list[NodeExample] | None = None,
               pulse_offset: int = 0,
               pulse_traces: list[dict[str, object]] | None = None) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    context = prompt
    emitted_total = 0
    accepted_total = 0
    cycles = 0
    total_ns = 0
    decisions: dict[int, int] = {}
    online_actions = [TreeAction(1, 4, 0.0, "D1_B4"),
                      TreeAction(2, 4, 0.0, "D2_B4"),
                      TreeAction(4, 4, 0.0, "D4_B4")]
    ema_policy = EMAThresholdPolicy(online_actions, [0.60, 0.80], alpha=0.2, initial=0.5) \
        if online_policy == "ema" else None
    bandit_policy = UCB1BanditPolicy(online_actions, exploration=2.0) \
        if online_policy == "ucb1" else None
    while emitted_total < token_limit:
        trace_state = model.ctx3(context) if pulse_traces is not None else -1
        start_ns = time.perf_counter_ns()
        bandit_action_index: int | None = None
        if action_lookup is not None:
            action = action_lookup[model.ctx3(context)]
        elif fixed_action is not None:
            action = fixed_action
        elif ema_policy is not None:
            action = ema_policy.select()
        elif bandit_policy is not None:
            bandit_action_index = bandit_policy.select_index()
            action = online_actions[bandit_action_index]
        elif online_policy is not None:
            raise ValueError(f"unknown online policy {online_policy}")
        else:
            assert scheduler is not None and predictor is not None
            if heldout_groups is not None and model.context_group(context) not in heldout_groups:
                if fallback_action is None:
                    raise ValueError("adaptive evaluation requires a fixed fallback action")
                action = fallback_action
            else:
                if oracle:
                    q = model.qrow(context)
                    p = model.prow(context)
                    confidence = float(np.minimum(p / p.sum(), q / q.sum()).sum())
                elif confidence_lookup is None:
                    q = model.qrow(context)
                    confidence = float(predictor.predict(np.asarray([
                        draft_features(q, context, 0, model.v)]))[0])
                else:
                    confidence = float(confidence_lookup[model.ctx3(context)])
                action = scheduler.select(confidence)
        remaining = token_limit - emitted_total
        if action.depth > remaining:
            action = TreeAction(remaining, action.branch_factor, action.threshold, action.name)
        pulse_id = f"{prompt_id}:{seed}:{pulse_offset + cycles}"
        tokens, naccepted, _, used_depth, verifier_labels = run_pulse(
            model, context, action, rng, prompt_id, pulse_id, examples)
        if not tokens:
            raise RuntimeError("Stage 3 pulse failed to emit")
        for token in tokens:
            context += (token,)
        elapsed = time.perf_counter_ns() - start_ns
        if pulse_traces is not None:
            pulse_traces.append({"context3": trace_state, "depth": action.depth,
                                 "branch": action.branch_factor, "threshold": action.threshold,
                                 "emitted_tokens": len(tokens), "elapsed_ns": elapsed})
        if ema_policy is not None:
            for label in verifier_labels:
                ema_policy.update(label)
        if bandit_policy is not None and bandit_action_index is not None:
            measured_tokens_per_second = len(tokens) * 1e9 / max(elapsed, 1)
            bandit_policy.update(bandit_action_index,
                                 min(measured_tokens_per_second / 100000.0, 1.0))
        emitted_total += len(tokens)
        accepted_total += naccepted
        cycles += 1
        total_ns += elapsed
        decisions[used_depth] = decisions.get(used_depth, 0) + 1
    return {"prompt_id": prompt_id, "seed": seed, "tokens": emitted_total,
            "accepted": accepted_total, "cycles": cycles, "wall_ns": total_ns,
            "tokens_per_second": emitted_total * 1e9 / max(total_ns, 1),
            "decisions": decisions}


PROMPTS = [
    ("canonical_5_3_9", (5, 3, 9)),
    ("alternate_1_4_7", (1, 4, 7)),
    ("zero_0_0_0", (0, 0, 0)),
]


def write_nodes(path: Path, examples: list[NodeExample], probs: np.ndarray,
                train_idx: np.ndarray, eval_idx: np.ndarray) -> None:
    train_set, eval_set = set(train_idx.tolist()), set(eval_idx.tolist())
    fieldnames = ["prompt_group", "context_group", "pulse_group", "depth", "regime",
                  "accepted", "split", "predicted_acceptance"]
    if examples:
        fieldnames.extend(f"f{j}" for j in range(len(examples[0].features)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(examples):
            writer.writerow({"prompt_group": row.prompt_group, "context_group": row.context_group,
                             "pulse_group": row.pulse_group, "depth": row.depth,
                             "regime": row.regime, "accepted": row.accepted,
                             "split": "train" if i in train_set else "evaluation",
                             "predicted_acceptance": f"{probs[i]:.9f}",
                             **{f"f{j}": f"{value:.9g}" for j, value in enumerate(row.features)}})


def write_scheduler_nodes(path: Path, examples: list[NodeExample], probs: np.ndarray) -> None:
    fields = ["prompt_group", "context_group", "pulse_group", "depth", "regime",
              "accepted", "predicted_acceptance"]
    if examples:
        fields.extend(f"f{j}" for j in range(len(examples[0].features)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(examples, probs):
            writer.writerow({"prompt_group": row.prompt_group, "context_group": row.context_group,
                             "pulse_group": row.pulse_group, "depth": row.depth,
                             "regime": row.regime, "accepted": row.accepted,
                             "predicted_acceptance": f"{probability:.9f}",
                             **{f"f{j}": f"{value:.9g}" for j, value in enumerate(row.features)}})


def aggregate_runs(rows: list[dict[str, object]]) -> dict[str, float | int]:
    wall_ns = sum(int(row["wall_ns"]) for row in rows)
    tokens = sum(int(row["tokens"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    return {"tokens_per_second": tokens * 1e9 / max(wall_ns, 1),
            "tokens": tokens, "wall_ns": wall_ns,
            "mean_acceptance_per_emitted": accepted / max(tokens, 1),
            "runs": len(rows)}


class NativeRunner:
    """Stage 5-only native policy timer; calibration and audit replay stay Python."""
    def __init__(self, model_path: Path, out_dir: Path):
        self.model_path = model_path.resolve()
        self.executable = ROOT / "cpp/pulsar_stage5_native"
        subprocess.run(["make", "-C", str(ROOT / "cpp"), "pulsar_stage5_native"],
                       check=True, stdout=subprocess.DEVNULL)
        self.map_dir = out_dir / ".native_action_maps"
        self.map_dir.mkdir(parents=True, exist_ok=True)

    def _write_lookup(self, lookup: tuple[TreeAction, ...]) -> tuple[Path, str]:
        keys: list[tuple[int, int, float]] = []
        indices: dict[tuple[int, int, float], int] = {}
        encoded = bytearray()
        for action in lookup:
            key = (action.depth, action.branch_factor, action.threshold)
            if key not in indices:
                indices[key] = len(keys)
                keys.append(key)
            encoded.append(indices[key])
        action_text = ";".join(
            f"{depth},{branch},{threshold:.17g}" for depth, branch, threshold in keys)
        digest = hashlib.sha256(bytes(encoded) + action_text.encode("ascii")).hexdigest()
        path = self.map_dir / f"{digest}.bin"
        if not path.exists():
            path.write_bytes(encoded)
        return path, action_text

    def run(self, prompt_id: str, seed: int, token_limit: int,
            fixed_action: TreeAction | None = None,
            action_lookup: tuple[TreeAction, ...] | None = None,
            online_policy: str | None = None,
            trace_path: Path | None = None) -> dict[str, object]:
        command = [str(self.executable), "--model", str(self.model_path),
                   "--tokens", str(token_limit), "--prompt", prompt_id,
                   "--seed", str(seed)]
        if online_policy is not None:
            command.extend(["--online", online_policy])
        elif fixed_action is not None:
            command.extend(["--fixed-depth", str(fixed_action.depth),
                            "--fixed-branch", str(fixed_action.branch_factor),
                            "--fixed-threshold", f"{fixed_action.threshold:.17g}"])
        elif action_lookup is not None:
            path, action_text = self._write_lookup(action_lookup)
            command.extend(["--lookup", str(path), "--actions", action_text])
        else:
            raise ValueError("native timing requires a fixed action, compiled lookup, or online policy")
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            command.extend(["--trace", str(trace_path)])
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        tokens, accepted, cycles, wall_ns, depth1, depth2, depth4 = map(
            int, completed.stdout.strip().split(","))
        return {"prompt_id": prompt_id, "seed": seed, "tokens": tokens,
                "accepted": accepted, "cycles": cycles, "wall_ns": wall_ns,
                "tokens_per_second": tokens * 1e9 / max(wall_ns, 1),
                "decisions": {1: depth1, 2: depth2, 4: depth4},
                "backend": "native_cpp17"}


def write_policy_runs(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["phase", "policy", "backend", "prompt_group", "repetition", "seed", "tokens",
              "accepted", "cycles", "wall_ns", "tokens_per_second"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped Stage 5 calibration and held-out policy evaluation")
    parser.add_argument("--model", type=Path, default=ROOT / "cpp/inputs/model.bin")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper/results/stage5")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--calibration-repeats", type=int, default=20)
    parser.add_argument("--tuning-repeats", type=int, default=4)
    parser.add_argument("--evaluation-repeats", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--backend", choices=("native", "python"), default="native",
                        help="policy timing backend; calibration and audit collection remain Python")
    args = parser.parse_args()
    if (args.tokens <= 0 or args.calibration_repeats <= 0 or args.tuning_repeats < 2 or
            args.evaluation_repeats < 2 or args.bootstrap_samples < 100):
        raise SystemExit("tokens/calibration repeats must be positive; tuning repeats >=2, evaluation repeats >=2, and bootstrap samples >=100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = ToyModel(args.model)
    fixture_sha256 = hashlib.sha256(args.model.read_bytes()).hexdigest()
    native_runner = NativeRunner(args.model, args.out_dir) if args.backend == "native" else None

    def timed_stream(prompt_id: str, seed: int, fixed_action: TreeAction | None = None,
                     action_lookup: tuple[TreeAction, ...] | None = None,
                     online_policy: str | None = None,
                     trace_path: Path | None = None) -> dict[str, object]:
        if native_runner is not None:
            return native_runner.run(prompt_id, seed, args.tokens, fixed_action,
                                     action_lookup, online_policy, trace_path)
        pulse_traces: list[dict[str, object]] | None = [] if trace_path is not None else None
        if action_lookup is not None:
            result = run_stream(model, dict(PROMPTS)[prompt_id], prompt_id, seed, args.tokens,
                                action_lookup=action_lookup, pulse_traces=pulse_traces)
        elif online_policy is not None:
            result = run_stream(model, dict(PROMPTS)[prompt_id], prompt_id, seed, args.tokens,
                                online_policy=online_policy, pulse_traces=pulse_traces)
        else:
            result = run_stream(model, dict(PROMPTS)[prompt_id], prompt_id, seed, args.tokens,
                                fixed_action=fixed_action, pulse_traces=pulse_traces)
        if trace_path is not None and pulse_traces is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", newline="") as handle:
                fields = ["context3", "depth", "branch", "threshold", "emitted_tokens", "elapsed_ns"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(pulse_traces)
        return {**result, "backend": "python_reference"}

    # Calibration data, policy tuning episodes, and final test episodes use
    # disjoint deterministic seed namespaces. The predictor never sees test labels.
    seed_bases = {"calibration": 0x50554C5341520001,
                  "tuning": 0x10554C5341520002,
                  "test": 0x60554C5341520003}
    def episode_seed(phase: str, prompt_index: int, repetition: int) -> int:
        return (args.seed + seed_bases[phase] + prompt_index * 0x9E3779B97F4A7C15 +
                repetition * 0xD1B54A32D192ED03) & ((1 << 64) - 1)

    calibration_action = TreeAction(4, 4, 0.0, "calibration_D4_B4_t0")
    examples: list[NodeExample] = []
    for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
        for repetition in range(args.calibration_repeats):
            run_stream(model, prompt, prompt_id,
                       episode_seed("calibration", prompt_index, repetition), args.tokens,
                       fixed_action=calibration_action, examples=examples)

    train_idx, eval_idx = split_by_groups(examples, 0.25, seed=args.seed, group_by="context_group")
    x = np.asarray([row.features for row in examples], dtype=np.float64)
    y = np.asarray([row.accepted for row in examples], dtype=np.int64)
    predictor = AcceptancePredictor().fit(x[train_idx], y[train_idx])
    confidence_lookup = precompute_root_confidence(model, predictor)
    probs = predictor.predict(x)
    eval_labels = y[eval_idx]
    learned = calibration_report(probs[eval_idx], eval_labels)
    base_rate = float(y[train_idx].mean())
    constant_probs = np.full(len(eval_idx), base_rate)
    draft_top_probs = x[eval_idx, 2]
    constant = calibration_report(constant_probs, eval_labels)
    draft_top = calibration_report(draft_top_probs, eval_labels)
    eval_groups = [examples[int(i)].context_group for i in eval_idx]
    learned_vs_constant = clustered_brier_delta(
        probs[eval_idx], constant_probs, eval_labels, eval_groups,
        samples=args.bootstrap_samples, seed=args.seed + 11)
    learned_vs_draft_top = clustered_brier_delta(
        probs[eval_idx], draft_top_probs, eval_labels, eval_groups,
        samples=args.bootstrap_samples, seed=args.seed + 12)

    by_depth: dict[str, object] = {}
    by_regime: dict[str, object] = {}
    for depth in sorted({examples[int(i)].depth for i in eval_idx}):
        ix = np.asarray([i for i in eval_idx if examples[int(i)].depth == depth], dtype=int)
        if len(ix):
            by_depth[str(depth)] = calibration_report(probs[ix], y[ix])
    for regime in sorted({examples[int(i)].regime for i in eval_idx}):
        ix = np.asarray([i for i in eval_idx if examples[int(i)].regime == regime], dtype=int)
        if len(ix):
            by_regime[regime] = calibration_report(probs[ix], y[ix])

    grid = [TreeAction(d, b, t, f"D{d}_B{b}_T{t:.2f}")
            for d in (1, 2, 4) for b in (1, 2, 4) for t in (0.0, 0.2, 0.4)]
    heldout_groups = {examples[int(i)].context_group for i in eval_idx}
    heldout_root_scores = np.asarray([
        confidence_lookup[index]
        for index in range(model.v ** 3)
        if model.context_group((index // (model.v ** 2),
                                (index // model.v) % model.v,
                                index % model.v)) in heldout_groups], dtype=np.float64)
    if not len(heldout_root_scores):
        raise RuntimeError("no held-out root contexts are available for utility bins")
    score_edges = np.unique(np.quantile(heldout_root_scores, [0.25, 0.50, 0.75]))
    fit_repeats = max(1, args.tuning_repeats // 2)
    selection_repetitions = set(range(fit_repeats, args.tuning_repeats))
    trace_dir = args.out_dir / ".utility_fit_traces"
    shutil.rmtree(trace_dir, ignore_errors=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    adaptive_modes = [("confidence", "confidence_scheduler"),
                      ("oracle", "oracle_overlap_scheduler"),
                      ("ema", "ema_threshold_policy"),
                      ("ucb1", "ucb1_bandit_policy")]

    all_run_rows: list[dict[str, object]] = []
    tuning_runs: dict[str, list[dict[str, object]]] = {action.name: [] for action in grid}
    for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
        for repetition in range(args.tuning_repeats):
            seed = episode_seed("tuning", prompt_index, repetition)
            order = np.random.default_rng(seed ^ 0x424154434854554E).permutation(len(grid))
            for action_index in order:
                action = grid[int(action_index)]
                trace_path = (trace_dir / f"p{prompt_index}_r{repetition}_d{action.depth}_b{action.branch_factor}_t{action.threshold:.2f}.csv"
                              if repetition < fit_repeats else None)
                raw = timed_stream(prompt_id, seed, fixed_action=action, trace_path=trace_path)
                row = {"phase": "tuning", "policy": action.name, "prompt_group": prompt_id,
                       "repetition": repetition, "seed": seed, **raw}
                tuning_runs[action.name].append(row)
                all_run_rows.append(row)
    tuning_selection_runs = {
        name: [row for row in rows if int(row["repetition"]) in selection_repetitions]
        for name, rows in tuning_runs.items()}
    tuning_metrics = {name: aggregate_runs(rows) for name, rows in tuning_selection_runs.items()}
    selected_action = max(grid, key=lambda action: float(tuning_metrics[action.name]["tokens_per_second"]))
    fallback_action = selected_action
    utility_actions = tuple(action for action in grid
                            if action.branch_factor == fallback_action.branch_factor and
                            action.threshold == fallback_action.threshold)
    actions_by_bin, utility_diagnostics, group_actions, group_utility_diagnostics = fit_bin_actions(
        sorted(trace_dir.glob("*.csv")), model, confidence_lookup, heldout_groups,
        score_edges, utility_actions, fallback_action)
    confidence_policy_name = f"confidence_utility_bins{len(actions_by_bin)}"
    action_lookup = precompute_binned_action_lookup(
        model, confidence_lookup, heldout_groups, score_edges, actions_by_bin,
        fallback_action, group_actions)
    confidence_tuning_runs: dict[str, list[dict[str, object]]] = {confidence_policy_name: []}
    for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
        for repetition in sorted(selection_repetitions):
            seed = episode_seed("tuning", prompt_index, repetition)
            raw = timed_stream(prompt_id, seed, action_lookup=action_lookup)
            row = {"phase": "tuning", "policy": confidence_policy_name,
                   "prompt_group": prompt_id, "repetition": repetition,
                   "seed": seed, **raw}
            confidence_tuning_runs[confidence_policy_name].append(row)
            all_run_rows.append(row)
    confidence_tuning_metrics = {
        name: aggregate_runs(rows) for name, rows in confidence_tuning_runs.items()}
    selected_confidence_tuning_rate = float(
        confidence_tuning_metrics[confidence_policy_name]["tokens_per_second"])
    selected_fixed_tuning_rate = float(tuning_metrics[selected_action.name]["tokens_per_second"])
    confidence_fallback_to_fixed = selected_confidence_tuning_rate <= selected_fixed_tuning_rate
    oracle_confidence = precompute_oracle_confidence(model)
    oracle_action_lookup = precompute_binned_action_lookup(
        model, oracle_confidence, heldout_groups, score_edges, actions_by_bin,
        fallback_action, group_actions)
    shutil.rmtree(trace_dir, ignore_errors=True)

    test_runs: dict[str, list[dict[str, object]]] = {action.name: [] for action in grid}
    test_runs.update({name: [] for _, name in adaptive_modes})
    confidence_scheduler_nodes: list[NodeExample] = []
    for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
        for repetition in range(args.evaluation_repeats):
            seed = episode_seed("test", prompt_index, repetition)
            tasks = [("fixed", action.name, action) for action in grid]
            tasks.extend((mode, name, None) for mode, name in adaptive_modes)
            order = np.random.default_rng(seed ^ 0x4241544348544553).permutation(len(tasks))
            for task_index in order:
                mode, name, action = tasks[int(task_index)]
                if mode == "confidence" and confidence_fallback_to_fixed:
                    continue
                if mode == "fixed":
                    raw = timed_stream(prompt_id, seed, fixed_action=action)
                elif mode == "confidence":
                    raw = (timed_stream(prompt_id, seed, fixed_action=selected_action)
                           if confidence_fallback_to_fixed else
                           timed_stream(prompt_id, seed, action_lookup=action_lookup))
                elif mode == "oracle":
                    raw = timed_stream(prompt_id, seed, action_lookup=oracle_action_lookup)
                else:
                    raw = timed_stream(prompt_id, seed, online_policy=mode)
                raw.setdefault("backend", "python_reference")
                row = {"phase": "test", "policy": name, "prompt_group": prompt_id,
                       "repetition": repetition, "seed": seed, **raw}
                test_runs[name].append(row)
                all_run_rows.append(row)

    if confidence_fallback_to_fixed:
        # The selected policy is exactly the same fixed native path; reuse its
        # paired timings rather than adding a second noisy timing of identical work.
        test_runs["confidence_scheduler"] = [
            {**row, "policy": "confidence_scheduler"}
            for row in test_runs[selected_action.name]]
        all_run_rows.extend(test_runs["confidence_scheduler"])

    # Re-run confidence test episodes solely to collect visited-node audit
    # rows. Instrumentation is excluded from all policy timing measurements.
    for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
        for repetition in range(args.evaluation_repeats):
            seed = episode_seed("test", prompt_index, repetition)
            if confidence_fallback_to_fixed:
                run_stream(model, prompt, prompt_id, seed, args.tokens,
                           fixed_action=selected_action,
                           examples=confidence_scheduler_nodes)
            else:
                run_stream(model, prompt, prompt_id, seed, args.tokens,
                           action_lookup=action_lookup,
                           examples=confidence_scheduler_nodes)

    grid_results = []
    for action in grid:
        tune = tuning_metrics[action.name]
        test = aggregate_runs(test_runs[action.name])
        grid_results.append({"policy": action.name, "depth": action.depth,
                             "branch_factor": action.branch_factor, "threshold": action.threshold,
                             "tuning_tokens_per_second": tune["tokens_per_second"],
                             "test_tokens_per_second": test["tokens_per_second"],
                             "test_tokens": test["tokens"], "test_wall_ns": test["wall_ns"],
                             "test_runs": test["runs"],
                             "test_mean_acceptance_per_emitted": test["mean_acceptance_per_emitted"],
                             "selected_on_tuning_split": action.name == selected_action.name})
    selected_test = aggregate_runs(test_runs[selected_action.name])
    best_fixed = {"policy": selected_action.name, "depth": selected_action.depth,
                  "branch_factor": selected_action.branch_factor,
                  "threshold": selected_action.threshold,
                  "selection_phase": "independent tuning episodes",
                  "tuning_tokens_per_second": tuning_metrics[selected_action.name]["tokens_per_second"],
                  "test_tokens_per_second": selected_test["tokens_per_second"],
                  "tokens": selected_test["tokens"], "wall_ns": selected_test["wall_ns"],
                  "mean_acceptance_per_emitted": selected_test["mean_acceptance_per_emitted"]}
    test_best_fixed_descriptive = max(grid_results, key=lambda row: float(row["test_tokens_per_second"]))

    adaptive_results = []
    adaptive_comparisons = {}
    fixed_test_rows = test_runs[selected_action.name]
    for mode, name in adaptive_modes:
        rows = test_runs[name]
        aggregate = aggregate_runs(rows)
        comparison = paired_bootstrap_speedup(
            [int(row["tokens"]) for row in rows], [int(row["wall_ns"]) for row in rows],
            [int(row["tokens"]) for row in fixed_test_rows],
            [int(row["wall_ns"]) for row in fixed_test_rows],
            [str(row["prompt_group"]) for row in rows],
            samples=args.bootstrap_samples, seed=args.seed + 100 + len(adaptive_results))
        adaptive_comparisons[name] = comparison
        adaptive_results.append({"policy": name, "backend": rows[0]["backend"], **aggregate,
                                 "decision_histogram": {str(d): sum(int(r["decisions"].get(d, 0)) for r in rows)
                                                        for d in (1, 2, 4)},
                                 "paired_test_vs_tuned_best_fixed": comparison})

    evaluation_groups = heldout_groups
    scheduler_selected_nodes = [row for row in confidence_scheduler_nodes
                                 if row.context_group in evaluation_groups]
    if not scheduler_selected_nodes:
        raise RuntimeError("confidence scheduler visited no held-out context nodes")
    scheduler_x = np.asarray([row.features for row in scheduler_selected_nodes], dtype=np.float64)
    scheduler_y = np.asarray([row.accepted for row in scheduler_selected_nodes], dtype=np.int64)
    scheduler_prob = predictor.predict(scheduler_x)
    selected_calibration = calibration_report(scheduler_prob, scheduler_y)

    calibration_wins = (learned_vs_constant["ci95_high"] < 0 and
                        learned_vs_draft_top["ci95_high"] < 0)
    confidence_test = adaptive_comparisons["confidence_scheduler"]
    confidence_wins = confidence_test["ci95_low_percent"] > 0
    gate_reasons = []
    if not (learned_vs_constant["ci95_high"] < 0):
        gate_reasons.append("held-out learned Brier improvement over the constant baseline is not established at 95% cluster-bootstrap confidence")
    if not (learned_vs_draft_top["ci95_high"] < 0):
        gate_reasons.append("held-out learned Brier improvement over the draft-top baseline is not established at 95% cluster-bootstrap confidence")
    if not confidence_wins:
        gate_reasons.append("confidence scheduler does not beat the tuning-selected fixed policy with a positive 95% paired-bootstrap lower bound")
    gate_reasons.append("only one synthetic Markov fixture/workload is available; multiple-workload evidence is absent")
    exit_gate = {
        "heldout_calibration_beats_constant_with_95pct_cluster_ci": learned_vs_constant["ci95_high"] < 0,
        "heldout_calibration_beats_draft_top_with_95pct_cluster_ci": learned_vs_draft_top["ci95_high"] < 0,
        "heldout_calibration_beats_all_declared_baselines": calibration_wins,
        "confidence_scheduler_beats_tuned_best_fixed_with_95pct_paired_ci": confidence_wins,
        "multiple_models_or_workloads_verified": False,
        "overall_status": "PASS" if calibration_wins and confidence_wins else "NOT_PASSED",
        "blocking_reasons": gate_reasons,
        "note": "This runner evaluates one synthetic fixture. The three prompt strata are not counted as independent workloads.",
    }
    if not exit_gate["multiple_models_or_workloads_verified"]:
        exit_gate["overall_status"] = "NOT_PASSED: multi-model/workload gate is unverified"

    fixture_name = str(args.model.relative_to(ROOT) if args.model.is_relative_to(ROOT) else args.model)
    result = {
        "stage": 5,
        "fixture": fixture_name,
        "fixture_sha256": fixture_sha256,
        "fixture_scope": "single synthetic order-3 target/order-2 draft Markov model; no transformer or KV cache",
        "reference_path": ("Native C++17 Stage 3-semantic lazy/eager tree and frozen v1 verifier for fixed/confidence policy timing; "
                           "Python remains the calibration and audit reference" if args.backend == "native" else
                           "Python Stage 3-semantic tree/verifier simulation; wall times are not Stage 4 C++ timings"),
        "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                       "seed": args.seed, "command_protocol": "calibration/tuning/test use disjoint seed namespaces"},
        "settings": {"tokens_per_run": args.tokens,
                     "calibration_repeats_per_prompt": args.calibration_repeats,
                     "policy_tuning_repeats_per_prompt": args.tuning_repeats,
                     "utility_fit_repeats_per_prompt": fit_repeats,
                     "policy_selection_repeats_per_prompt": len(selection_repetitions),
                     "heldout_test_repeats_per_prompt": args.evaluation_repeats,
                     "bootstrap_samples": args.bootstrap_samples,
                     "policy_timing_backend": args.backend,
                     "calibration_policy": asdict(calibration_action),
                     "group_key": "last-two-token draft context",
                     "evaluation_groups": len(heldout_groups),
                     "training_groups": len({examples[int(i)].context_group for i in train_idx}),
                     "prompt_strata": [prompt_id for prompt_id, _ in PROMPTS]},
        "calibration": {"learned": learned, "constant_train_rate": constant,
                        "draft_top_probability_baseline": draft_top,
                        "paired_brier_delta_ci": {"learned_minus_constant": learned_vs_constant,
                                                   "learned_minus_draft_top": learned_vs_draft_top},
                        "by_depth": by_depth, "by_overlap_regime": by_regime,
                        "selected_node_distribution": selected_calibration,
                        "selected_node_count": len(scheduler_selected_nodes),
                        "note": "Brier differences are cluster-bootstrapped by held-out context group; negative favors the learned predictor."},
        "policy_tuning": {"selection_metric": ("aggregate native C++17 tokens/second" if args.backend == "native"
                                                 else "aggregate Python reference tokens/second"),
                          "fixed_policy_tuning_runs": sum(len(rows) for rows in tuning_runs.values()),
                          "confidence_scheduler_candidate_count": 1,
                          "confidence_scheduler_tuning_runs": sum(len(rows) for rows in confidence_tuning_runs.values()),
                          "confidence_scheduler_candidate_grid": {
                              "utility_action_grid": [asdict(action) for action in utility_actions],
                              "adaptation_scope": "depth only; branch factor and retained-level threshold match tuned fixed action",
                              "confidence_bin_edges": score_edges.tolist(),
                              "utility_fit_repeats_per_prompt": fit_repeats,
                              "policy_selection_repeats_per_prompt": len(selection_repetitions),
                              "utility_fit_diagnostics_by_bin": utility_diagnostics,
                              "context_group_adjustments": group_utility_diagnostics},
                          "selected_fixed_policy": selected_action.name,
                          "selected_confidence_scheduler": confidence_policy_name,
                          "confidence_fallback_to_fixed_if_not_faster_on_tuning": True,
                          "confidence_fallback_used": confidence_fallback_to_fixed,
                          "selection_is_separate_from_test": True},
        "confidence_scheduler_selection": {
            "policy": confidence_policy_name,
            "adaptation_scope": "depth only; branch factor and retained-level threshold match tuned fixed action",
            "score_bin_edges": score_edges.tolist(),
            "actions_by_score_bin": [asdict(action) for action in actions_by_bin],
            "tuning_tokens_per_second": selected_confidence_tuning_rate,
            "tuned_fixed_tokens_per_second": selected_fixed_tuning_rate,
            "fallback_to_tuned_fixed": confidence_fallback_to_fixed,
            "utility_fit_diagnostics_by_bin": utility_diagnostics,
            "context_group_adjustment_count": len(group_actions),
            "context_group_adjustments": group_utility_diagnostics,
            "effective_policy": (f"fixed_fallback:{selected_action.name}" if confidence_fallback_to_fixed
                                 else confidence_policy_name),
            "candidate_count": 1,
            "selection_phase": "utility fit on first tuning split; fixed/map selection on disjoint second split",
        },
        "fixed_policy_grid": grid_results,
        "best_fixed_policy": best_fixed,
        "test_best_fixed_policy_descriptive_only": test_best_fixed_descriptive,
        "adaptive_policies": adaptive_results,
        "policy_run_count": len(all_run_rows),
        "exit_gate": exit_gate,
    }

    nodes_path = args.out_dir / "stage5_nodes.csv"
    write_nodes(nodes_path, examples, probs, train_idx, eval_idx)
    scheduler_nodes_path = args.out_dir / "stage5_scheduler_nodes.csv"
    write_scheduler_nodes(scheduler_nodes_path, scheduler_selected_nodes, scheduler_prob)
    policy_runs_path = args.out_dir / "stage5_policy_runs.csv"
    write_policy_runs(policy_runs_path, all_run_rows)
    result_path = args.out_dir / "stage5_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    report_path = args.out_dir / "stage5_report.md"
    report = [
        "# Stage 5 — grouped acceptance calibration and adaptivity",
        "",
        f"**Status: {exit_gate['overall_status']}.** One synthetic Markov fixture; policy timing uses {('native C++17' if args.backend == 'native' else 'the Python reference')}, not transformer/KV-cache or multi-workload evidence.",
        "",
        "## Calibration on held-out context groups",
        "",
        f"Collected {len(examples)} verifier-visited nodes; trained on {result['settings']['training_groups']} context groups and held out {result['settings']['evaluation_groups']}. The split key is the last-two-token draft context.",
        "",
        "| Predictor/baseline | Brier | ECE |",
        "|---|---:|---:|",
        f"| Learned draft-only predictor | {learned['brier']:.6f} | {learned['ece']:.6f} |",
        f"| Constant train acceptance rate | {constant['brier']:.6f} | {constant['ece']:.6f} |",
        f"| Draft top-probability baseline | {draft_top['brier']:.6f} | {draft_top['ece']:.6f} |",
        f"| Actual confidence-scheduler visited nodes | {selected_calibration['brier']:.6f} | {selected_calibration['ece']:.6f} |",
        "",
        "Paired 95% context-cluster bootstrap intervals for learned-minus-baseline Brier (negative favors learned):",
        f"- Versus constant: {learned_vs_constant['estimate']:.6f} [{learned_vs_constant['ci95_low']:.6f}, {learned_vs_constant['ci95_high']:.6f}].",
        f"- Versus draft-top: {learned_vs_draft_top['estimate']:.6f} [{learned_vs_draft_top['ci95_low']:.6f}, {learned_vs_draft_top['ci95_high']:.6f}].",
        "",
        "## Fixed versus adaptive policies",
        "",
        f"The fixed policy `{selected_action.name}` was selected only on separate tuning episodes. The test set contains {args.evaluation_repeats} fresh repetitions per prompt; every policy is paired on the same prompt/seed episode, and run order is deterministically randomized.",
        "",
        "| Test policy | Tokens/s | Accepted/emitted | Paired speedup vs tuned fixed (95% CI) |",
        "|---|---:|---:|---:|",
        f"| Tuned fixed `{selected_action.name}` | {selected_test['tokens_per_second']:.1f} | {selected_test['mean_acceptance_per_emitted']:.4f} | reference |",
    ]
    for row in adaptive_results:
        ci = row["paired_test_vs_tuned_best_fixed"]
        report.append(f"| {row['policy']} | {row['tokens_per_second']:.1f} | {row['mean_acceptance_per_emitted']:.4f} | {ci['estimate_percent']:+.2f}% [{ci['ci95_low_percent']:+.2f}%, {ci['ci95_high_percent']:+.2f}%] |")
    report += [
        "",
        f"A test-set grid winner (`{test_best_fixed_descriptive['policy']}`) is reported descriptively only and is not substituted for the tuning-selected comparator. Detailed paired runs are in `stage5_policy_runs.csv`.",
        f"The confidence policy bins draft-only calibrated root scores at `{score_edges.tolist()}` and varies depth only, holding branch factor and retained-level threshold at the tuned fixed action; its measured-utility depth choices are `{[action.name for action in actions_by_bin]}` by bin, with {len(group_actions)} supported context-group/bin refinements. Pulse utilities were fit from the first {fit_repeats} tuning repeats per prompt; the map and tuned fixed comparator were selected on the disjoint remaining {len(selection_repetitions)} repeats. If the map does not beat fixed on that selection split, it uses the tuned fixed action unchanged; otherwise it applies the compiled map only on held-out calibration contexts and falls back to fixed elsewhere. EMA, UCB1, and exact-overlap binned utility remain separate controls. Fixed and confidence policy timers use the {('native C++17' if args.backend == 'native' else 'Python reference')} path; trace writing, map construction, audit collection, and model-forward/KV-cache costs are excluded from timed episodes.",
        "",
        "## Exit gate",
        "",
        f"- Calibration beats constant baseline with cluster-bootstrap evidence: **{learned_vs_constant['ci95_high'] < 0}**.",
        f"- Calibration beats draft-top baseline with cluster-bootstrap evidence: **{learned_vs_draft_top['ci95_high'] < 0}**.",
        f"- Confidence scheduler beats tuned fixed policy with paired evidence: **{confidence_wins}**.",
        "- Multi-model/multi-workload validation: **not verified** (only one synthetic Markov fixture is present). The three starting prompts are treated as strata, not independent workloads.",
        "",
        "Blocking reasons:",
    ]
    report.extend(f"- {reason}." for reason in gate_reasons)
    report += [
        "",
        "Reproduce with `python3 verification/experiments/test_stage5.py && python3 verification/experiments/run_stage5.py`. Defaults: 128 tokens/run, 20 calibration, 4 tuning, and 8 test repetitions per prompt, plus 2,000 bootstrap samples. Results are isolated under `paper/results/stage5/`; Stage 4 artifacts are untouched.",
    ]
    report_path.write_text("\n".join(report) + "\n")
    print(f"Stage 5 wrote {nodes_path}, {scheduler_nodes_path}, {policy_runs_path}, {result_path}, and {report_path}")
    print(f"Held-out Brier: learned={learned['brier']:.6f}, constant={constant['brier']:.6f}, draft-top={draft_top['brier']:.6f}")
    print(f"Confidence scheduler vs tuned fixed: {confidence_test['estimate_percent']:+.2f}% [{confidence_test['ci95_low_percent']:+.2f}%, {confidence_test['ci95_high_percent']:+.2f}%]; {exit_gate['overall_status']}")


if __name__ == "__main__":
    main()
