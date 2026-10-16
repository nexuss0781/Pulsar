#!/usr/bin/env python3
"""Integration smoke tests for a built Linux pulsar-generate binary.

Model files are supplied by the caller and are never copied into the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def run_checked(command: list[str], expected: int = 0, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode != expected:
        raise RuntimeError(
            f"command returned {result.returncode}, expected {expected}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def json_rows(stdout: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("CLI returned no JSONL rows")
    return rows


def generation_args(model: str, prompt: str, context: int, tokens: int = 8) -> list[str]:
    return [
        "--model", model, "--prompt", prompt,
        "--max-new-tokens", str(tokens), "--context-size", str(context),
        "--threads", "2", "--seed", "991", "--temperature", "0",
        "--top-k", "0", "--top-p", "1",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--llama-model", required=True,
                        help="TinyStories Llama-family GGUF")
    parser.add_argument("--llama-draft", required=True,
                        help="Tokenizer-compatible alternate-quant GGUF")
    parser.add_argument("--gpt2-model", required=True,
                        help="TinyStories GPT-2-family GGUF")
    parser.add_argument("--skip-resource-pressure", action="store_true")
    parser.add_argument("--skip-cancellation", action="store_true")
    args = parser.parse_args()

    binary = str(Path(args.binary).resolve())
    llama_model = str(Path(args.llama_model).resolve())
    llama_draft = str(Path(args.llama_draft).resolve())
    gpt2_model = str(Path(args.gpt2_model).resolve())
    evidence: dict[str, Any] = {"binary": binary, "checks": []}

    help_result = run_checked([binary, "--help"])
    if "Pulsar native CPU text generation" not in help_result.stdout:
        raise RuntimeError("installed CLI help banner is missing")
    evidence["checks"].append("help")

    bad_context = run_checked([
        binary, "--model", "/definitely/not/a/model.gguf", "--prompt", "x",
        "--context-size", "2147483648",
    ], expected=2)
    if "context" not in bad_context.stderr.lower():
        raise RuntimeError("oversized context was not rejected before model loading")

    missing = run_checked([
        binary, "--model", "/definitely/not/a/model.gguf", "--prompt", "x",
    ], expected=2)
    if "could not load GGUF model" not in missing.stderr:
        raise RuntimeError("missing GGUF did not return the documented load error")
    evidence["checks"].append("invalid_path_and_numeric_bounds")

    with tempfile.TemporaryDirectory(prefix="pulsar-phase5-") as temp_dir:
        temp = Path(temp_dir)
        malformed = temp / "truncated.gguf"
        malformed.write_bytes(b"GGUF\x00\x00")
        malformed_result = run_checked([
            binary, "--model", str(malformed), "--prompt", "x",
        ], expected=2, timeout=30)
        if "pulsar-generate:" not in malformed_result.stderr:
            raise RuntimeError("malformed GGUF did not produce a bounded CLI diagnostic")
        evidence["checks"].append("malformed_gguf")

        llama_prompt = "Once upon a time, a small test began."
        llama_rows = json_rows(run_checked(
            [binary, *generation_args(llama_model, llama_prompt, 256, 8)]).stdout)
        if llama_rows[0]["mode"] != "autoregressive":
            raise RuntimeError("Llama-family autoregressive smoke used the wrong mode")
        evidence["llama_family"] = {
            "prompt_tokens": llama_rows[0]["prompt_tokens"],
            "completion_tokens": llama_rows[0]["completion_tokens"],
            "finish_reason": llama_rows[0]["finish_reason"],
        }

        chain = json_rows(run_checked([
            binary, *generation_args(llama_model, llama_prompt, 256, 8),
            "--draft-model", llama_draft, "--speculative-tokens", "2",
        ]).stdout)[0]
        if chain["mode"] != "exact_speculative_chain":
            raise RuntimeError("speculative real-model smoke did not select chain mode")
        evidence["checks"].append("llama_quant_pair_chain_speculation")

        tree = json_rows(run_checked([
            binary, *generation_args(llama_model, llama_prompt, 256, 8),
            "--draft-model", llama_draft, "--tree-branching", "1", "--tree-depth", "2",
        ]).stdout)[0]
        if tree["mode"] != "exact_candidate_tree":
            raise RuntimeError("candidate-tree real-model smoke did not select tree mode")
        evidence["checks"].append("llama_quant_pair_candidate_tree")

        long_prompt = "Once upon a time " * 240
        gpt2_command = [binary, *generation_args(gpt2_model, long_prompt, 2048, 4)]
        gpt2_rows = json_rows(run_checked(gpt2_command).stdout)
        if gpt2_rows[0]["prompt_tokens"] < 256:
            raise RuntimeError("GPT-2 family long-context smoke did not exercise a long prompt")
        evidence["gpt2_family"] = {
            "prompt_tokens": gpt2_rows[0]["prompt_tokens"],
            "completion_tokens": gpt2_rows[0]["completion_tokens"],
            "finish_reason": gpt2_rows[0]["finish_reason"],
        }

        repeated = []
        for _ in range(4):
            repeated.append(json_rows(run_checked(gpt2_command).stdout)[0]["text"])
        if len(set(repeated)) != 1:
            raise RuntimeError("fixed-seed repeated warm runs were not deterministic")
        evidence["warm_runs"] = {"runs": len(repeated), "same_text": True,
                                 "performance_claim": False}
        evidence["checks"].append("gpt2_long_context_and_four_warm_runs")

        input_path = temp / "prompts.txt"
        output_path = temp / "results.jsonl"
        prompts = [f"A short batch prompt {i}." for i in range(16)]
        input_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        run_checked([
            binary, "--model", llama_model, "--input", str(input_path),
            "--output", str(output_path), "--batch-size", "4",
            "--context-size", "256", "--max-new-tokens", "4",
            "--threads", "2", "--seed", "73", "--temperature", "0",
            "--top-k", "0", "--top-p", "1",
        ])
        batch_rows = json_rows(output_path.read_text(encoding="utf-8"))
        if len(batch_rows) != len(prompts) or [row["id"] for row in batch_rows] != list(range(16)):
            raise RuntimeError("multi-microbatch CLI output lost, reordered, or duplicated requests")
        evidence["checks"].append("16_requests_in_four_microbatches_and_file_output")

        if not args.skip_cancellation:
            cancel_prompt = "Once upon a time " * 240
            cancel_command = [binary, *generation_args(gpt2_model, cancel_prompt, 2048, 1000000)]
            process = subprocess.Popen(cancel_command, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.05)
            if process.poll() is not None:
                raise RuntimeError("long generation finished before cancellation could be sent")
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=90)
            if process.returncode != 128 + signal.SIGINT:
                raise RuntimeError(f"SIGINT exit status was {process.returncode}: {stderr}")
            cancelled_rows = json_rows(stdout)
            if cancelled_rows[0]["finish_reason"] != "cancelled":
                raise RuntimeError(f"SIGINT did not return a cancelled JSONL result: {stdout}")
            evidence["checks"].append("sigint_partial_jsonl_cancellation")

        if not args.skip_resource_pressure:
            def limit_address_space() -> None:
                limit = 384 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

            pressure_command = [
                binary, "--model", llama_model, "--prompt", "x",
                "--context-size", "65536", "--batch-size", "64",
                "--batch-tokens", "512", "--threads", "1",
            ]
            pressured = subprocess.run(pressure_command, text=True, capture_output=True,
                                      timeout=60, preexec_fn=limit_address_space)
            if pressured.returncode == 0:
                raise RuntimeError("large context allocation unexpectedly succeeded under 384 MiB RLIMIT_AS")
            if pressured.returncode < 0:
                raise RuntimeError(f"memory-pressure run terminated by signal {pressured.returncode}")
            evidence["checks"].append("resource_limit_failure_is_non_signal_exit")
            evidence["resource_pressure"] = {
                "address_space_limit_mib": 384,
                "exit_status": pressured.returncode,
                "diagnostic": pressured.stderr[-1000:],
            }

    evidence["status"] = "PASS"
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"phase5_cli_smoke.py: FAIL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
