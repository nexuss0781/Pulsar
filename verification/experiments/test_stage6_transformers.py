#!/usr/bin/env python3
"""Stage 6 integration checks; downloads the declared public model pair if needed."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "verification/experiments"))
from pulsar.transformers_adapter import (  # noqa: E402
    TransformersCausalLM,
    assert_compatible_vocab,
    exactness_identity_error,
)
from run_stage6_transformers import build_draft_tree, populate_target_tree  # noqa: E402


def close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    error = float(np.max(np.abs(actual - expected)))
    # Different batch widths can select a different float32 reduction kernel;
    # the maximum observed row discrepancy is expected to be a few e-6.
    if not np.allclose(actual, expected, rtol=3e-5, atol=3e-6):
        raise AssertionError(f"{label} mismatch: max_abs={error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="distilgpt2")
    parser.add_argument("--draft", default="sshleifer/tiny-gpt2")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = TransformersCausalLM(args.target, temperature=1.0, device="cpu")
    draft = TransformersCausalLM(args.draft, temperature=1.0, device="cpu")
    assert_compatible_vocab(target, draft, len(tokenizer))
    context = tokenizer.encode("A short cache contract test", add_special_tokens=False)
    token_ids = tokenizer.encode(" continues with branches", add_special_tokens=False)
    if len(token_ids) < 2:
        raise AssertionError("test tokenizer did not produce enough tokens")

    # A single-token KV append must match a full-sequence forward pass.
    root = target.prefill(context)
    appended = target.append(root, token_ids[0])
    full = target.prefill(context + [token_ids[0]])
    close(target.row(appended.next_logits), target.row(full.next_logits), "single cache append")

    # Batched branch updates must match independent, serial cache updates.
    branches = target.append_many([root, root], token_ids[:2])
    for i, token in enumerate(token_ids[:2]):
        serial = target.append(target.fork(root), token)
        close(target.row(branches[i].next_logits), target.row(serial.next_logits),
              f"batched branch append {i}")

    # Block verification rows and prefix caches must agree with direct target rows.
    draft_root = draft.prefill(context)
    proposals = []
    draft_state = draft_root
    for token in token_ids[:3]:
        proposals.append(token)
        draft_state = draft.append(draft_state, token)
    block_rows, block_states = target.verify_chain(root, proposals)
    for i in range(len(proposals) + 1):
        direct = target.prefill(context + proposals[:i])
        close(block_rows[i], target.row(direct.next_logits), f"chain target row {i}")
        close(target.row(block_states[i].next_logits), target.row(direct.next_logits),
              f"chain cached prefix {i}")

    # The candidate tree retains complete q rows and creates target states for all nodes.
    tree_draft_root = draft.prefill(context)
    nodes = build_draft_tree(draft, tree_draft_root, depth=3, branch=2)
    target_root = target.prefill(context)
    populate_target_tree(target, target_root, nodes)
    for i, node in enumerate(nodes):
        direct = target.prefill(node["context"])
        close(target.row(node["target_state"].next_logits), target.row(direct.next_logits),
              f"tree target row {i}")
        if node["q"].shape != (len(tokenizer),):
            raise AssertionError("tree node does not store the complete draft vocabulary row")

    # Exact v1 identity against actual target and draft distributions.
    target_state = target.prefill(context)
    draft_state = draft.prefill(context)
    error = exactness_identity_error(target.row(target_state.next_logits),
                                     draft.row(draft_state.next_logits))
    if error > 5e-12:
        raise AssertionError(f"actual-model verifier identity error {error}")

    print("Stage 6 Transformers integration checks: 5/5 PASS")
    print(f"  cache append and branch parity; chain prefix rows/caches; "
          f"tree row/caches ({len(nodes)} nodes); exact identity={error:.3g}")


if __name__ == "__main__":
    main()
