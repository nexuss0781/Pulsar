#!/usr/bin/env python3
"""Stage 6: real-model exactness and matched Transformers decoding benchmark.

Backends: target distilgpt2, draft sshleifer/tiny-gpt2, shared GPT-2 tokenizer.
The target law is softmax(target_logits / temperature); EOS is treated as an
ordinary token because this experiment uses a fixed new-token horizon.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pulsar.transformers_adapter import (  # noqa: E402
    KVState,
    TransformersCausalLM,
    assert_compatible_vocab,
    exactness_identity_error,
    top_positive,
    verified_sample,
    verify_proposal,
)

DEFAULT_OUT = ROOT / "paper/results/stage6_transformers"
TARGET_ID = "distilgpt2"
DRAFT_ID = "sshleifer/tiny-gpt2"
TEMPERATURE = 1.0
TUNE_PROMPTS = (
    "The history of science is",
    "In a small village,",
)
TEST_PROMPTS = (
    "When the lights went out,",
    "A recipe for bread begins",
    "The most important lesson I learned was",
)
TREE_ACTIONS = ((2, 1), (3, 1), (3, 2), (4, 2))  # (max verifier positions, branch)
CHAIN_GAMMAS = (2, 4, 6)


def categorical(row: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(row), p=row))


def normalize_row(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row, dtype=np.float64)
    if value.ndim != 1 or np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError("invalid categorical row")
    total = float(value.sum())
    if total <= 0:
        raise ValueError("categorical row has no mass")
    return value / total


def _new_output(prompt_ids: Sequence[int], token_limit: int) -> dict[str, Any]:
    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    return {"tokens": [], "pulses": 0, "accepted": 0, "trace": [],
            "token_limit": token_limit, "prompt_ids": tuple(int(x) for x in prompt_ids)}


def decode_ar(target: TransformersCausalLM, prompt_ids: Sequence[int],
              token_limit: int, seed: int) -> dict[str, Any]:
    """Vanilla autoregressive reference using the same target law and cache API."""
    target.reset_counters()
    start = time.perf_counter_ns()
    state = target.prefill(prompt_ids)
    rng = np.random.default_rng(seed)
    output = _new_output(prompt_ids, token_limit)
    while len(output["tokens"]) < token_limit:
        pulse_start = time.perf_counter_ns()
        row = target.row(state.next_logits)
        token = categorical(row, rng)
        output["tokens"].append(token)
        output["pulses"] += 1
        elapsed = max(time.perf_counter_ns() - pulse_start, 1)
        output["trace"].append({"confidence": float(np.max(row)), "emitted": 1,
                                "elapsed_ns": elapsed, "action": "AR"})
        if len(output["tokens"]) < token_limit:
            state = target.append(state, token)
    output["wall_ns"] = max(time.perf_counter_ns() - start, 1)
    output["target_counters"] = target.counter_snapshot()
    output["draft_counters"] = {"forward_calls": 0, "input_tokens": 0}
    return output


def decode_chain(target: TransformersCausalLM, draft: TransformersCausalLM,
                 prompt_ids: Sequence[int], token_limit: int, seed: int,
                 gamma: int) -> dict[str, Any]:
    """Standard exact speculative decoding, with one target block verification."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    target.reset_counters()
    draft.reset_counters()
    start = time.perf_counter_ns()
    target_state = target.prefill(prompt_ids)
    draft_state = draft.prefill(prompt_ids)
    rng = np.random.default_rng(seed)
    output = _new_output(prompt_ids, token_limit)
    while len(output["tokens"]) < token_limit:
        cycle_start = time.perf_counter_ns()
        tokens_before = len(output["tokens"])
        remaining = token_limit - len(output["tokens"])
        count = min(gamma, remaining)
        proposals: list[int] = []
        q_rows: list[np.ndarray] = []
        draft_states = [draft_state]
        work_state = draft_state
        for _ in range(count):
            q = draft.row(work_state.next_logits)
            token = categorical(q, rng)
            proposals.append(token)
            q_rows.append(q)
            work_state = draft.append(work_state, token)
            draft_states.append(work_state)
        p_rows, target_states = target.verify_chain(target_state, proposals)
        emitted: list[int] = []
        accepted = 0
        rejected_at: int | None = None
        residual_token: int | None = None
        for i, proposal in enumerate(proposals):
            _, token, is_accepted = verify_proposal(
                p_rows[i], q_rows[i], proposal, rng)
            if is_accepted:
                emitted.append(proposal)
                accepted += 1
                continue
            emitted.append(token)
            rejected_at = i
            residual_token = token
            break
        if rejected_at is not None:
            output["accepted"] += accepted
            output["tokens"].extend(emitted)
            if len(output["tokens"]) < token_limit:
                prefix = rejected_at
                target_state = target.append(target_states[prefix], int(residual_token))
                draft_state = draft.append(draft_states[prefix], int(residual_token))
        else:
            output["accepted"] += accepted
            output["tokens"].extend(emitted)
            # The target row after all accepted proposals is the exact bonus law.
            if len(output["tokens"]) < token_limit:
                bonus = categorical(p_rows[len(proposals)], rng)
                output["tokens"].append(bonus)
                if len(output["tokens"]) < token_limit:
                    target_state = target.append(target_states[-1], bonus)
                    draft_state = draft.append(draft_states[-1], bonus)
        output["pulses"] += 1
        elapsed = max(time.perf_counter_ns() - cycle_start, 1)
        output["trace"].append({"confidence": float(np.max(q_rows[0])),
                                "emitted": len(output["tokens"]) - tokens_before,
                                "elapsed_ns": elapsed, "action": f"chain_gamma_{gamma}"})
        # If this block reached the budget, no bonus/cache update is permitted.
        if len(output["tokens"]) > token_limit:
            raise AssertionError("speculative block exceeded token limit")
    output["wall_ns"] = max(time.perf_counter_ns() - start, 1)
    output["target_counters"] = target.counter_snapshot()
    output["draft_counters"] = draft.counter_snapshot()
    return output


def build_draft_tree(draft: TransformersCausalLM, root_state: KVState,
                     depth: int, branch: int,
                     root_q: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Build a bounded tree retaining top-q children while keeping complete rows."""
    if depth <= 0 or branch < 0:
        raise ValueError("invalid tree action")
    nodes: list[dict[str, Any]] = [{
        "context": root_state.context, "depth": 0,
        "q": draft.row(root_state.next_logits) if root_q is None else root_q,
        "draft_state": root_state, "target_state": None, "children": {},
    }]
    for level in range(depth - 1):
        parents = [i for i, node in enumerate(nodes) if node["depth"] == level]
        specifications: list[tuple[int, int]] = []
        parent_states: list[KVState] = []
        for parent_index in parents:
            parent = nodes[parent_index]
            for token in top_positive(parent["q"], branch):
                specifications.append((parent_index, token))
                parent_states.append(parent["draft_state"])
        if not specifications:
            continue
        child_states = draft.append_many(parent_states,
                                         [token for _, token in specifications])
        for (parent_index, token), state in zip(specifications, child_states):
            parent = nodes[parent_index]
            child_index = len(nodes)
            child = {
                "context": state.context, "depth": level + 1,
                "q": draft.row(state.next_logits), "draft_state": state,
                "target_state": None, "children": {},
            }
            nodes.append(child)
            parent["children"][token] = child_index
    return nodes


def populate_target_tree(target: TransformersCausalLM, root_state: KVState,
                         nodes: list[dict[str, Any]]) -> None:
    """Fetch true target rows and cache states for each tree context, level-batched."""
    nodes[0]["target_state"] = root_state
    max_level = max(node["depth"] for node in nodes)
    for level in range(1, max_level + 1):
        indexes = [i for i, node in enumerate(nodes) if node["depth"] == level]
        for begin in range(0, len(indexes), target.max_batch_size):
            selected = indexes[begin:begin + target.max_batch_size]
            states = target.prefill_many([nodes[i]["context"] for i in selected])
            for index, state in zip(selected, states):
                nodes[index]["target_state"] = state


def decode_tree_pulse(target: TransformersCausalLM, draft: TransformersCausalLM,
                      target_root: KVState, draft_root: KVState,
                      rng: np.random.Generator, depth: int, branch: int,
                      remaining: int, root_q: np.ndarray | None = None
                      ) -> tuple[list[int], int, KVState | None, KVState | None, float]:
    """Build/verify a tree pulse; return emitted tokens and committed cache states."""
    nodes = build_draft_tree(draft, draft_root, min(depth, remaining), branch, root_q)
    populate_target_tree(target, target_root, nodes)
    confidence = float(np.max(nodes[0]["q"]))
    current = 0
    emitted: list[int] = []
    accepted_count = 0
    commit_node: int | None = None
    commit_token: int | None = None
    while len(emitted) < min(depth, remaining):
        node = nodes[current]
        p = target.row(node["target_state"].next_logits)
        proposal, token, accepted = verified_sample(p, node["q"], rng)
        emitted.append(token)
        if not accepted:
            commit_node, commit_token = current, token
            break
        accepted_count += 1
        child = node["children"].get(proposal)
        if child is None or len(emitted) == min(depth, remaining):
            commit_node, commit_token = current, proposal
            break
        current = child
    if not emitted:
        raise AssertionError("tree pulse emitted no tokens")
    if len(emitted) < remaining:
        assert commit_node is not None and commit_token is not None
        target_next = target.append(nodes[commit_node]["target_state"], commit_token)
        draft_next = draft.append(nodes[commit_node]["draft_state"], commit_token)
    else:
        target_next = None
        draft_next = None
    return emitted, accepted_count, target_next, draft_next, confidence


def decode_tree(target: TransformersCausalLM, draft: TransformersCausalLM,
                prompt_ids: Sequence[int], token_limit: int, seed: int,
                fixed_action: tuple[int, int] | None = None,
                confidence_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    if (fixed_action is None) == (confidence_policy is None):
        raise ValueError("choose exactly one of fixed_action or confidence_policy")
    target.reset_counters()
    draft.reset_counters()
    start = time.perf_counter_ns()
    target_state = target.prefill(prompt_ids)
    draft_state = draft.prefill(prompt_ids)
    rng = np.random.default_rng(seed)
    output = _new_output(prompt_ids, token_limit)
    while len(output["tokens"]) < token_limit:
        pulse_start = time.perf_counter_ns()
        root_q = None
        if confidence_policy is None:
            depth, branch = fixed_action
        else:
            root_q = draft.row(draft_state.next_logits)
            confidence = float(np.max(root_q))
            bin_index = int(np.searchsorted(confidence_policy["thresholds"],
                                            confidence, side="right"))
            action_index = confidence_policy["bin_actions"][bin_index]
            depth, branch = confidence_policy["actions"][action_index]
        remaining = token_limit - len(output["tokens"])
        emitted, accepted, target_next, draft_next, confidence = decode_tree_pulse(
            target, draft, target_state, draft_state, rng, depth, branch, remaining,
            root_q=root_q)
        output["tokens"].extend(emitted)
        output["accepted"] += accepted
        output["pulses"] += 1
        elapsed = max(time.perf_counter_ns() - pulse_start, 1)
        label = (f"tree_{depth}_{branch}" if confidence_policy is None else
                 f"confidence_tree_{depth}_{branch}")
        output["trace"].append({"confidence": confidence, "emitted": len(emitted),
                                "elapsed_ns": elapsed, "action": label})
        if len(output["tokens"]) < token_limit:
            if target_next is None or draft_next is None:
                raise AssertionError("missing committed cache state before output limit")
            target_state, draft_state = target_next, draft_next
    output["wall_ns"] = max(time.perf_counter_ns() - start, 1)
    output["target_counters"] = target.counter_snapshot()
    output["draft_counters"] = draft.counter_snapshot()
    return output


def aggregate_rate(runs: Sequence[dict[str, Any]]) -> float:
    tokens = sum(len(run["tokens"]) for run in runs)
    elapsed = sum(int(run["wall_ns"]) for run in runs)
    if elapsed <= 0:
        raise ValueError("run timings must be positive")
    return tokens * 1e9 / elapsed


def paired_cluster_bootstrap(candidate: Sequence[dict[str, Any]],
                             baseline: Sequence[dict[str, Any]],
                             prompts: Sequence[str], samples: int,
                             seed: int) -> dict[str, float | int]:
    """Prompt-cluster paired bootstrap for aggregate tokens/s percent change."""
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired methods must have matching non-empty runs")
    for left, right in zip(candidate, baseline):
        if left["prompt"] != right["prompt"] or left["seed"] != right["seed"]:
            raise ValueError("benchmark rows are not paired by prompt and seed")
    strata = [np.array([i for i, run in enumerate(candidate) if run["prompt"] == p],
                       dtype=np.int64) for p in prompts]
    strata = [indices for indices in strata if len(indices)]
    if len(strata) < 2:
        raise ValueError("at least two held-out prompt clusters are required")

    def effect(indices: np.ndarray) -> float:
        c_tokens = sum(int(candidate[i]["tokens"]) for i in indices)
        c_time = sum(candidate[i]["wall_ns"] for i in indices)
        b_tokens = sum(int(baseline[i]["tokens"]) for i in indices)
        b_time = sum(baseline[i]["wall_ns"] for i in indices)
        return 100.0 * ((c_tokens / c_time) / (b_tokens / b_time) - 1.0)

    observed = effect(np.arange(len(candidate), dtype=np.int64))
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draws = [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        values[i] = effect(np.concatenate(draws))
    low, high = np.quantile(values, [0.025, 0.975])
    return {"estimate_percent": float(observed), "ci95_low_percent": float(low),
            "ci95_high_percent": float(high), "paired_runs": len(candidate),
            "prompt_clusters": len(strata), "bootstrap_samples": int(samples)}


def confidence_edges(scores: Sequence[float]) -> list[float]:
    if len(scores) < 3:
        raise ValueError("need at least three tuning pulse confidence scores")
    low, high = np.quantile(np.asarray(scores, dtype=np.float64), [1 / 3, 2 / 3])
    return [float(low), float(high)]


def fit_confidence_policy(actions: Sequence[tuple[int, int]],
                          traces_by_action: Sequence[Sequence[dict[str, Any]]],
                          fallback_action: tuple[int, int]) -> dict[str, Any]:
    """Fit a draft-top-probability bin-to-tree-action map on tuning traces only."""
    all_scores = [float(row["confidence"]) for rows in traces_by_action for row in rows]
    thresholds = confidence_edges(all_scores)
    bins: list[list[list[dict[str, Any]]]] = [[[] for _ in actions] for _ in range(3)]
    for action_index, rows in enumerate(traces_by_action):
        for row in rows:
            b = int(np.searchsorted(thresholds, float(row["confidence"]), side="right"))
            bins[b][action_index].append(row)
    fallback_index = list(actions).index(fallback_action)
    chosen: list[int] = []
    utility_table = []
    for action_rows in bins:
        rates = []
        for rows in action_rows:
            rates.append((sum(row["emitted"] for row in rows) * 1e9 /
                          sum(row["elapsed_ns"] for row in rows)) if rows else None)
        candidates = [i for i, rate in enumerate(rates) if rate is not None]
        selected = max(candidates, key=lambda i: float(rates[i])) if candidates else fallback_index
        chosen.append(int(selected))
        utility_table.append({"per_action_tokens_per_second": rates,
                              "counts": [len(rows) for rows in action_rows],
                              "selected_action_index": int(selected)})
    return {"actions": [list(action) for action in actions],
            "thresholds": thresholds, "bin_actions": chosen,
            "fallback_action": list(fallback_action), "utility_table": utility_table,
            "score_definition": "max draft-model softmax probability at pulse root"}


def run_record(method: str, prompt: str, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    return {"method": method, "prompt": prompt, "seed": int(seed),
            "tokens": len(result["tokens"]), "wall_ns": int(result["wall_ns"]),
            "tokens_per_second": len(result["tokens"]) * 1e9 / result["wall_ns"],
            "pulses": int(result["pulses"]), "accepted_proposals": int(result["accepted"]),
            "target_forward_calls": result["target_counters"]["forward_calls"],
            "target_input_tokens": result["target_counters"]["input_tokens"],
            "draft_forward_calls": result["draft_counters"]["forward_calls"],
            "draft_input_tokens": result["draft_counters"]["input_tokens"],
            "output_token_ids": " ".join(str(x) for x in result["tokens"])}


def audit_exactness(target: TransformersCausalLM, draft: TransformersCausalLM,
                    tokenizer: Any, draws: int, seed: int) -> dict[str, Any]:
    rows = []
    rng = np.random.default_rng(seed)
    for prompt in TEST_PROMPTS:
        context = tokenizer.encode(prompt, add_special_tokens=False)
        target_state = target.prefill(context)
        draft_state = draft.prefill(context)
        p = normalize_row(target.row(target_state.next_logits))
        q = normalize_row(draft.row(draft_state.next_logits))
        algebra_error = exactness_identity_error(p, q)
        residual = np.maximum(p - q, 0.0)
        residual_mass = float(residual.sum())
        p_cdf = np.cumsum(q)
        uniforms = rng.random(draws)
        proposals = np.searchsorted(p_cdf, uniforms, side="right").clip(max=len(q) - 1)
        acceptance = np.minimum(1.0, p[proposals] / q[proposals])
        accepted = rng.random(draws) < acceptance
        emitted = proposals.copy()
        if np.any(~accepted):
            if residual_mass <= 0.0:
                raise AssertionError("equal target/draft rows unexpectedly rejected")
            residual_cdf = np.cumsum(residual / residual_mass)
            replacements = np.searchsorted(residual_cdf, rng.random(int((~accepted).sum())),
                                           side="right").clip(max=len(q) - 1)
            emitted[~accepted] = replacements
        focus = np.argsort(-p)[:32]
        bin_index = np.full(len(p), 32, dtype=np.int64)
        bin_index[focus] = np.arange(32)
        expected = np.zeros(33, dtype=np.float64)
        observed = np.bincount(bin_index, weights=p, minlength=33)[:33]
        empirical = np.bincount(bin_index[emitted], minlength=33)[:33] / draws
        tv_grouped = float(0.5 * np.abs(empirical - observed).sum())
        rows.append({"prompt": prompt, "context_tokens": len(context),
                     "target_draft_tv": 0.5 * float(np.abs(p - q).sum()),
                     "residual_mass": residual_mass,
                     "pointwise_identity_max_abs_error": algebra_error,
                     "empirical_grouped_tv": tv_grouped,
                     "empirical_draws": draws,
                     "top32_plus_other_bins": 33})
        if algebra_error > 5e-12:
            raise AssertionError(f"pointwise exactness identity failed: {algebra_error}")
        if tv_grouped > 0.035:
            raise AssertionError(f"grouped empirical law check failed: TV={tv_grouped}")
    target.reset_counters()
    draft.reset_counters()
    return {"status": "PASS", "law": "softmax(target_logits / 1.0)",
            "rows": rows, "max_pointwise_identity_error": max(
                row["pointwise_identity_max_abs_error"] for row in rows),
            "max_grouped_empirical_tv": max(row["empirical_grouped_tv"] for row in rows),
            "sampling_note": "empirical check coalesces all but the 32 highest-p tokens into OTHER; deterministic pointwise identity covers the full vocabulary"}


def write_report(path: Path, results: dict[str, Any]) -> None:
    exact = results["exactness_audit"]
    tuning = results["tuning"]
    test = results["test_summary"]
    lines = [
        "# Stage 6 — Transformers backend integration and matched benchmark", "",
        f"**Run:** `{results['run_id']}`. **Exactness audit:** {exact['status']}", "",
        "## Falsifiable claim under test", "",
        ("On the reported CPU, Transformers version, `distilgpt2` target / "
         "`sshleifer/tiny-gpt2` draft, temperature-1 softmax law, prompt set, and "
         "fixed output horizon, the draft-top-probability scheduler selects a "
         "depth/branch tree action that improves held-out tokens/s over the "
         "tuning-selected fixed tree policy, while the exact verifier preserves "
         "the target categorical law. The claim is supported only if the paired "
         "95% lower confidence bound for scheduler-vs-fixed-tree speedup is > 0; "
         "otherwise it is not established for this operating condition."), "",
        "## Exactness against actual backend rows", "",
        "The target row is obtained from the loaded Transformers target model's "
        "logits after its declared temperature processor. The proposal row is "
        "obtained from the loaded draft model. The deterministic check applies "
        "the Stage 1 `q·a + Z·r = p` identity to every vocabulary token; the "
        "separate Monte Carlo check groups the 32 highest target-probability "
        "tokens and the remaining vocabulary into one `OTHER` bin.", "",
        "| Context | p–q TV | Pointwise max error | Grouped empirical TV |", "",
    ]
    for row in exact["rows"]:
        lines.append(f"| {row['prompt']} | {row['target_draft_tv']:.6f} | "
                     f"{row['pointwise_identity_max_abs_error']:.3g} | "
                     f"{row['empirical_grouped_tv']:.6f} |")
    lines += ["", "## Tuning and held-out benchmark", "",
              f"- Best chain block length: **γ={tuning['best_chain_gamma']}** "
              f"(candidates {tuning['chain_candidates']}).",
              f"- Best fixed tree: **depth={tuning['best_fixed_tree'][0]}, "
              f"branch={tuning['best_fixed_tree'][1]}** (grid "
              f"{tuning['tree_actions']}).",
              f"- Confidence map (thresholds on max draft probability): "
              f"{tuning['confidence_policy']['thresholds']}; bin actions "
              f"{tuning['confidence_policy']['bin_actions']}.", "",
              "All methods use the same target model, tokenizer, temperature, "
              "prompt/seed episodes, and generated-token budget. Tuning prompts "
              "and test prompts are disjoint. The fixed tree and speculative "
              "block length are selected on tuning runs only. Benchmark timing "
              "includes prompt prefill, draft expansion, target rows, exact "
              "verification, KV-cache state creation/commit, and output updates; "
              "model downloads/loading are excluded. EOS is treated as an ordinary "
              "token so every paired run has the same fixed horizon.", "",
              "| Method | Tokens/s | Paired change vs best fixed tree (95% CI) | Paired change vs AR (95% CI) |",
              "|---|---:|---:|---:|"]
    for name, row in test["methods"].items():
        fixed = "—" if name == "fixed_tree" else _fmt_ci(test["vs_fixed_tree"].get(name))
        ar = "—" if name == "autoregressive" else _fmt_ci(test["vs_ar"].get(name))
        lines.append(f"| {name} | {row['aggregate_tokens_per_second']:.2f} | {fixed} | {ar} |")
    claim = test["vs_fixed_tree"].get("confidence_tree", {})
    supported = claim.get("ci95_low_percent", -math.inf) > 0
    lines += ["", "**Scheduler claim:** " + ("SUPPORTED for this measured setup." if supported else
              "NOT ESTABLISHED; the paired 95% lower bound is not positive."), "",
              "### Per-episode results", "",
              "See `stage6_policy_runs.csv` for paired timings, seeds, output IDs, "
              "acceptance counts, and backend forward-call/input-token counters.", "",
              "## Backend adapter and cache behavior", "",
              "- `TransformersCausalLM.row`: actual model logits -> declared processed law.",
              "- `verify_chain`: one target call over the speculative block, with prefix KV caches cropped to the accepted/replacement point.",
              "- `append_many` + full-row draft distributions: batched branch expansion with complete q rows and top-q child retention.",
              "- `prefill_many`: level-batched target rows and per-node KV states for tree verification; accepted branches commit the matching target/draft cache, while residual/off-tree tokens append to the current state.",
              "- No custom tree-attention kernel or GPU execution is claimed. The tree adapter evaluates full candidate contexts batched by level; the call and input-token counters expose this cost.", "",
              "## Limitations and scope", "",
              "This is one CPU, batch-size-1, short-context study on GPT-2-family checkpoints. `tiny-gpt2` is a deliberately small draft, not a separately trained distillation of `distilgpt2`; the evaluation has only three held-out prompts and does not establish workload-wide statistical generality. No GPU, server batching, quantization, top-p/top-k, repetition penalties, long-context, or production serving claim is made. Confidence means max draft-token probability, not a calibrated estimate of verifier acceptance. Pulsar's rotor/quantum language is explanatory framing here; it is not part of the tested algorithm.", "",
              "Stage 5 remains **completed with throughput gate not met**; its separate pooled calibration result is unchanged. Stage 6 does not retroactively pass Stage 5.", "",
              "Reproduce with `python3 verification/experiments/run_stage6_transformers.py`. See `algorithm_v6_backend.md` for the adapter contract and experiment protocol.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_ci(value: dict[str, Any] | None) -> str:
    if not value:
        return "n/a"
    return (f"{value['estimate_percent']:+.2f}% "
            f"[{value['ci95_low_percent']:+.2f}%, {value['ci95_high_percent']:+.2f}%]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET_ID)
    parser.add_argument("--draft", default=DRAFT_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--tune-repeats", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--audit-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=60613)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if min(args.threads, args.tokens, args.tune_repeats, args.eval_repeats,
           args.audit_draws) <= 0 or args.bootstrap_samples < 100:
        parser.error("threads/tokens/repeats/audit draws must be positive and bootstrap >=100")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = TransformersCausalLM(args.target, temperature=TEMPERATURE, device=args.device)
    draft = TransformersCausalLM(args.draft, temperature=TEMPERATURE, device=args.device)
    assert_compatible_vocab(target, draft, len(tokenizer))
    print(f"Loaded target={args.target} draft={args.draft} device={args.device} "
          f"threads={args.threads}; tokenizer_vocab={len(tokenizer)}", flush=True)

    exactness = audit_exactness(target, draft, tokenizer, args.audit_draws,
                                args.seed + 5000)
    print("Actual-row exactness audit: PASS", flush=True)

    # Tuning episodes are fixed before evaluation; no test timings select an action.
    tune_prompts = list(TUNE_PROMPTS)
    tree_runs: list[list[dict[str, Any]]] = [[] for _ in TREE_ACTIONS]
    chain_runs: dict[int, list[dict[str, Any]]] = {g: [] for g in CHAIN_GAMMAS}
    tune_seed_list = [args.seed + 101 * i for i in range(args.tune_repeats)]
    for action_index, action in enumerate(TREE_ACTIONS):
        for prompt_index, prompt in enumerate(tune_prompts):
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            for repeat, seed in enumerate(tune_seed_list):
                print(f"Tune tree {action_index + 1}/{len(TREE_ACTIONS)} "
                      f"prompt {prompt_index + 1}/{len(tune_prompts)} "
                      f"repeat {repeat + 1}/{len(tune_seed_list)}", flush=True)
                result = decode_tree(target, draft, ids, args.tokens, seed,
                                     fixed_action=action)
                tree_runs[action_index].append({"prompt": prompt, "seed": seed,
                                                "result": result})
    chain_tune_runs = {}
    for gamma in CHAIN_GAMMAS:
        rows = []
        for prompt_index, prompt in enumerate(tune_prompts):
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            for repeat, seed in enumerate(tune_seed_list):
                print(f"Tune chain γ={gamma} prompt {prompt_index + 1}/"
                      f"{len(tune_prompts)} repeat {repeat + 1}/"
                      f"{len(tune_seed_list)}", flush=True)
                result = decode_chain(target, draft, ids, args.tokens, seed, gamma)
                rows.append({"prompt": prompt, "seed": seed, "result": result})
        chain_tune_runs[gamma] = rows
    best_tree_index = max(range(len(TREE_ACTIONS)),
                          key=lambda i: aggregate_rate([row["result"] for row in tree_runs[i]]))
    best_tree = TREE_ACTIONS[best_tree_index]
    best_gamma = max(CHAIN_GAMMAS,
                     key=lambda g: aggregate_rate([row["result"]
                                                  for row in chain_tune_runs[g]]))
    confidence_policy = fit_confidence_policy(
        TREE_ACTIONS, [[trace for row in rows for trace in row["result"]["trace"]]
                       for rows in tree_runs], best_tree)

    # Held-out paired benchmark. Randomized method order avoids systematic drift.
    test_methods = ["autoregressive", "tuned_chain", "fixed_tree", "confidence_tree"]
    records: dict[str, list[dict[str, Any]]] = {method: [] for method in test_methods}
    test_prompts = list(TEST_PROMPTS)
    for prompt_index, prompt in enumerate(test_prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        for repeat in range(args.eval_repeats):
            seed = args.seed + 10000 + prompt_index * 1000 + repeat
            order = list(test_methods)
            random.Random(seed ^ 0x51A6).shuffle(order)
            for method in order:
                print(f"Test {prompt_index + 1}/{len(test_prompts)} "
                      f"repeat {repeat + 1}/{args.eval_repeats}: {method}", flush=True)
                if method == "autoregressive":
                    result = decode_ar(target, ids, args.tokens, seed)
                elif method == "tuned_chain":
                    result = decode_chain(target, draft, ids, args.tokens, seed,
                                          best_gamma)
                elif method == "fixed_tree":
                    result = decode_tree(target, draft, ids, args.tokens, seed,
                                         fixed_action=best_tree)
                else:
                    result = decode_tree(target, draft, ids, args.tokens, seed,
                                         confidence_policy=confidence_policy)
                records[method].append(run_record(method, prompt, seed, result))
    prompts_by_method = {name: [run["prompt"] for run in runs]
                         for name, runs in records.items()}
    reference_prompts = prompts_by_method["autoregressive"]
    for method in test_methods:
        if prompts_by_method[method] != reference_prompts:
            raise AssertionError("held-out methods do not share paired episodes")
    rates = {name: {"aggregate_tokens_per_second": aggregate_rate([
                {"tokens": [None] * row["tokens"], "wall_ns": row["wall_ns"]}
                for row in runs]), "episodes": len(runs)}
             for name, runs in records.items()}
    # Compare candidates to the tuned fixed tree and AR baselines using the same episodes.
    comparisons = {"vs_fixed_tree": {}, "vs_ar": {}}
    for method in test_methods:
        if method not in ("fixed_tree",):
            comparisons["vs_fixed_tree"][method] = paired_cluster_bootstrap(
                records[method], records["fixed_tree"], test_prompts,
                args.bootstrap_samples, args.seed + 700 + test_methods.index(method))
        if method != "autoregressive":
            comparisons["vs_ar"][method] = paired_cluster_bootstrap(
                records[method], records["autoregressive"], test_prompts,
                args.bootstrap_samples, args.seed + 900 + test_methods.index(method))

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("stage6-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-seed{args.seed}"
    tuning = {
        "tune_prompts": tune_prompts,
        "tune_repeats_per_prompt": args.tune_repeats,
        "tokens_per_episode": args.tokens,
        "best_chain_gamma": int(best_gamma),
        "chain_candidates": list(CHAIN_GAMMAS),
        "chain_tuning_rates": {str(g): aggregate_rate([row["result"]
            for row in chain_tune_runs[g]]) for g in CHAIN_GAMMAS},
        "tree_actions": [list(action) for action in TREE_ACTIONS],
        "tree_tuning_rates": {str(list(action)): aggregate_rate([row["result"]
            for row in tree_runs[i]]) for i, action in enumerate(TREE_ACTIONS)},
        "best_fixed_tree": list(best_tree),
        "confidence_policy": confidence_policy,
    }
    test_summary = {"held_out_prompts": test_prompts,
                    "eval_repeats_per_prompt": args.eval_repeats,
                    "methods": rates, **comparisons}
    results = {
        "run_id": run_id,
        "metadata": {
            "target_model": args.target, "draft_model": args.draft,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__, "tokenizer_vocab_size": len(tokenizer),
            "device": str(target.device), "torch_threads": torch.get_num_threads(),
            "temperature": TEMPERATURE, "law": "softmax(logits / temperature)",
            "logits_processors": "none except temperature",
            "eos_policy": "treated as an ordinary token; fixed token horizon",
            "batch_size": 1, "tree_target_batch": "candidate contexts batched by tree level",
            "timed_scope": "prompt prefill through fixed-horizon token generation, including proposal/tree work, target verification, cache updates and output bookkeeping; model download/load excluded",
            "tune_prompts_disjoint_from_test": True,
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "logical_cpus": os.cpu_count()},
        },
        "exactness_audit": exactness,
        "tuning": tuning,
        "test_summary": test_summary,
    }
    result_path = out_dir / "stage6_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    with (out_dir / "stage6_policy_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(next(iter(records.values()))[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in test_methods:
            writer.writerows(records[method])
    (out_dir / "stage6_tuning.json").write_text(
        json.dumps(tuning, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "stage6_report.md", results)
    manifest = {}
    for name in ("stage6_results.json", "stage6_policy_runs.csv",
                 "stage6_tuning.json", "stage6_report.md"):
        manifest[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
    (out_dir / "stage6_manifest.json").write_text(
        json.dumps({"run_id": run_id, "sha256": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\nStage 6 complete: {out_dir}", flush=True)
    print(f"best chain γ={best_gamma}; best fixed tree={best_tree}; "
          f"confidence bins={confidence_policy['bin_actions']}", flush=True)
    for method, value in rates.items():
        print(f"{method}: {value['aggregate_tokens_per_second']:.2f} tokens/s", flush=True)
    claim = comparisons["vs_fixed_tree"].get("confidence_tree", {})
    print("confidence vs fixed-tree paired speedup: "
          f"{claim.get('estimate_percent', float('nan')):+.2f}% "
          f"95% CI [{claim.get('ci95_low_percent', float('nan')):+.2f}%, "
          f"{claim.get('ci95_high_percent', float('nan')):+.2f}%]", flush=True)


if __name__ == "__main__":
    main()
