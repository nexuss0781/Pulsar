#!/usr/bin/env python3
"""Run the preregistered Phase 3 CPU benchmark and retain raw observations."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = Path(__file__).resolve().parent
PROMPT_MANIFEST = BENCH_DIR / "prompts.jsonl"
PROTOCOL = BENCH_DIR / "PHASE3_PROTOCOL.md"
UPSTREAM_REVISION = "cd0a5b6a180dff0048e5fd993098b21c0b2c91b8"
SOURCE_SNAPSHOT = "aca5c2502a76a41ee1022bd2f6d73a1dae42c473"
TUNING_SEEDS = [1701, 2801, 3907]
CONFIRMATORY_SEEDS = [5101, 6203, 7307, 8407, 9509, 10613, 11717, 12821, 13901, 15013]
BATCH_SIZES = [1, 2]
BUILTIN_CANDIDATES = [2, 4, 6]
TREE_CANDIDATES = [(1, 2), (2, 2), (2, 3)]
THREADS = 2
OUTPUT_HORIZON = 64
CONTEXT_SIZE = 256
BATCH_TOKENS = 512
SAMPLING = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
}
METRIC_SUFFIXES = {
    "prompt_tokens_total": "prompt_tokens_total",
    "prompt_seconds_total": "prompt_seconds_total",
    "tokens_predicted_total": "tokens_predicted_total",
    "tokens_predicted_seconds_total": "tokens_predicted_seconds_total",
    "n_decode_total": "n_decode_total",
    "n_tokens_max": "n_tokens_max",
    "spec_decode_num_draft_tokens_total": "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total": "spec_decode_num_accepted_tokens_total",
    "spec_decode_num_drafts_total": "spec_decode_num_drafts_total",
}
ACTIVE_RUN_DIR: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_prompts() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in PROMPT_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [(f"T{i:02d}", "tuning") for i in range(1, 5)] + [(f"H{i:02d}", "heldout") for i in range(1, 5)]
    if [(r["id"], r["split"]) for r in rows] != expected:
        raise RuntimeError("Prompt manifest changed from the preregistered eight-prompt split")
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise RuntimeError(f"Prompt {row['id']} has no non-empty prompt text")
        row["numeric_id"] = index
    return rows


def write_prompt_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            if "\t" in row["prompt"] or "\n" in row["prompt"]:
                raise ValueError("prompt text cannot contain tab or newline characters")
            stream.write(f"{row['numeric_id']}\t{row['id']}\t{row['prompt']}\n")


def command_version(executable: Path) -> str | None:
    try:
        result = subprocess.run([str(executable), "--version"], capture_output=True,
                                text=True, timeout=10, check=False)
        text = (result.stdout + result.stderr).strip()
        return text[:2000] if text else f"exit={result.returncode}"
    except Exception as exc:  # metadata only; the actual invocation still reports failures
        return f"unavailable: {exc}"


def host_metadata() -> dict[str, Any]:
    cpu_lines = Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
    model = next((line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")), None)
    mem_total = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_total = int(line.split()[1]) * 1024
            break
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": model,
        "logical_cpus_visible": os.cpu_count(),
        "cpu_affinity": affinity,
        "mem_total_bytes": mem_total,
        "python": sys.version,
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS")},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_native(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]],
               split: str, batch_size: int, seeds: list[int], mode: str,
               candidate: dict[str, int] | None = None) -> Path:
    label = mode
    if candidate:
        label += "_" + "_".join(f"{k}{v}" for k, v in candidate.items())
    label = safe_name(f"{split}_{label}_b{batch_size}")
    raw_path = run_dir / split / "raw" / f"{label}.jsonl"
    outputs_path = run_dir / split / "outputs" / f"{label}.jsonl"
    log_path = run_dir / split / "logs" / f"{label}.log"
    command_path = run_dir / split / "commands" / f"{label}.json"
    prompt_path = run_dir / "inputs" / f"{split}.tsv"
    warmup_path = run_dir / "inputs" / "warmup_tuning_T01.tsv"
    prompt_split = "tuning" if split == "tuning" else "heldout"
    selected = [row for row in prompts if row["split"] == prompt_split]
    warmup_rows = [row for row in prompts if row["id"] == "T01"]
    write_prompt_tsv(prompt_path, selected)
    write_prompt_tsv(warmup_path, warmup_rows)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.bench_bin), "--mode", "autoregressive" if mode == "pulsar-ar" else "pulsar-tree",
        "--split", split, "--model", str(args.target_model), "--prompts", str(prompt_path),
        "--warmup-prompts", str(warmup_path), "--raw-out", str(raw_path),
        "--outputs-out", str(outputs_path), "--seeds", ",".join(str(x) for x in seeds),
        "--batch-size", str(batch_size), "--max-new-tokens", str(OUTPUT_HORIZON),
        "--context-size", str(CONTEXT_SIZE), "--batch-tokens", str(BATCH_TOKENS),
        "--threads", str(THREADS), "--temperature", str(SAMPLING["temperature"]),
        "--top-k", str(SAMPLING["top_k"]), "--top-p", str(SAMPLING["top_p"]),
        "--repeat-penalty", str(SAMPLING["repeat_penalty"]),
        "--repeat-window", str(SAMPLING["repeat_last_n"]),
    ]
    if mode == "pulsar-tree":
        if not candidate:
            raise ValueError("fixed-tree run requires a tuning/selected configuration")
        command.extend([
            "--draft-model", str(args.draft_model),
            "--tree-branching", str(candidate["branching"]),
            "--tree-depth", str(candidate["depth"]),
        ])
    atomic_json(command_path, {
        "command": command,
        "target_model_sha256": sha256_file(args.target_model),
        "draft_model_sha256": sha256_file(args.draft_model) if mode == "pulsar-tree" else None,
        "seeds": seeds,
        "candidate": candidate,
        "protocol_revision": 1.1,
    })
    print(f"[{utc_now()}] native {label}: starting", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 text=True, check=False, timeout=900)
    if process.returncode != 0:
        raise RuntimeError(f"native benchmark failed ({process.returncode}); see {log_path}")
    elapsed = time.perf_counter() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    load_match = re.search(r"loaded_ms=([0-9.]+)", log_text)
    atomic_json(run_dir / split / "commands" / f"{label}_process.json", {
        "elapsed_process_seconds": elapsed,
        "model_load_ms": float(load_match.group(1)) if load_match else None,
        "peak_rss_bytes_stderr": int(re.search(r"peak_rss_bytes=(\d+)", log_text).group(1))
            if re.search(r"peak_rss_bytes=(\d+)", log_text) else None,
        "return_code": process.returncode,
        "raw_path": str(raw_path),
        "outputs_path": str(outputs_path),
    })
    rows = read_jsonl(raw_path)
    expected_batches = len(seeds) * len(selected) // batch_size
    if len(rows) != expected_batches:
        raise RuntimeError(f"native benchmark wrote {len(rows)} raw batches; expected {expected_batches}")
    print(f"[{utc_now()}] native {label}: retained {len(rows)} batches in {elapsed:.1f}s", flush=True)
    return raw_path


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def http_json_post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def parse_metrics(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        metric_name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        for suffix in METRIC_SUFFIXES:
            if metric_name.endswith(suffix):
                parsed[suffix] = value
    return parsed


def server_rss_hwm(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def build_server_command(args: argparse.Namespace, port: int, batch_size: int,
                         n_max: int) -> list[str]:
    return [
        str(args.llama_server), "--model", str(args.target_model),
        "--spec-draft-model", str(args.draft_model), "--spec-type", "draft-simple",
        "--spec-draft-n-max", str(n_max), "--threads", str(THREADS),
        "--threads-batch", str(THREADS), "--spec-draft-threads", str(THREADS),
        "--spec-draft-threads-batch", str(THREADS), "--ctx-size", str(CONTEXT_SIZE * batch_size),
        "--parallel", str(batch_size), "--batch-size", str(BATCH_TOKENS),
        "--ubatch-size", str(BATCH_TOKENS), "--n-gpu-layers", "0",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
        "--samplers", "penalties;temperature;top_k;top_p", "--no-warmup",
        "--host", "127.0.0.1", "--port", str(port),
    ]


def post_request(prompt: dict[str, Any], seed: int, base_url: str,
                 barrier: threading.Barrier) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt["prompt"], "n_predict": OUTPUT_HORIZON,
        "stream": False, "timings_per_token": True, "cache_prompt": False,
        "temperature": SAMPLING["temperature"], "top_k": SAMPLING["top_k"],
        "top_p": SAMPLING["top_p"], "repeat_penalty": SAMPLING["repeat_penalty"],
        "repeat_last_n": SAMPLING["repeat_last_n"], "seed": seed,
    }
    barrier.wait(timeout=60)
    started = time.perf_counter()
    response = http_json_post(base_url + "/completion", payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if "content" not in response:
        raise RuntimeError(f"llama-server returned no completion field: {response}")
    return {
        "prompt_id": prompt["id"], "prompt_numeric_id": prompt["numeric_id"],
        "request_seed_input": seed, "prompt": prompt["prompt"],
        "completion": response["content"],
        "completion_tokens": int(response.get("tokens_predicted", 0)),
        "prompt_tokens": int(response.get("tokens_evaluated", 0)),
        "finish_reason": response.get("stop_type"),
        "latency_ms": elapsed_ms,
        "server_timings": response.get("timings"),
    }


def send_server_batch(batch: list[dict[str, Any]], root_seed: int, base_url: str) -> tuple[list[dict[str, Any]], float]:
    barrier = threading.Barrier(len(batch))
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
        futures = [executor.submit(post_request, prompt, root_seed + prompt["numeric_id"], base_url, barrier)
                   for prompt in batch]
        responses = [future.result(timeout=650) for future in futures]
    response_completion_ms = (time.perf_counter() - started) * 1000.0
    responses.sort(key=lambda row: row["prompt_numeric_id"])
    return responses, response_completion_ms


def run_server(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]],
               split: str, batch_size: int, seeds: list[int], n_max: int) -> Path:
    label = safe_name(f"{split}_llama-draft-simple_n{n_max}_b{batch_size}")
    raw_path = run_dir / split / "raw" / f"{label}.jsonl"
    outputs_path = run_dir / split / "outputs" / f"{label}.jsonl"
    logs_dir = run_dir / split / "logs"
    command_path = run_dir / split / "commands" / f"{label}.json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_split = "tuning" if split == "tuning" else "heldout"
    prompt_rows = [row for row in prompts if row["split"] == prompt_split]
    warmup = next(row for row in prompts if row["id"] == "T01")
    port = reserve_port()
    base_url = f"http://127.0.0.1:{port}"
    command = build_server_command(args, port, batch_size, n_max)
    atomic_json(command_path, {
        "command": command,
        "target_model_sha256": sha256_file(args.target_model),
        "draft_model_sha256": sha256_file(args.draft_model),
        "seeds": seeds,
        "candidate": {"n_max": n_max},
        "protocol_revision": 1.1,
    })
    log_path = logs_dir / f"{label}_server.log"
    service_log = log_path.open("wb")
    start = time.perf_counter()
    print(f"[{utc_now()}] llama.cpp {label}: starting server", flush=True)
    process = subprocess.Popen(command, stdout=service_log, stderr=subprocess.STDOUT)
    startup_ms = None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup; see {log_path}")
            try:
                http_get(base_url + "/health", timeout=2)
                startup_ms = (time.perf_counter() - start) * 1000.0
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.25)
        if startup_ms is None:
            raise RuntimeError(f"llama-server health check timed out; see {log_path}")

        # Excluded warm-up uses only T01, never a held-out sample for configuration choice.
        warmup_responses, _ = send_server_batch([warmup], TUNING_SEEDS[0], base_url)
        warmup_path = run_dir / "warmups" / f"{label}.jsonl"
        write_jsonl_line(warmup_path, {"request": warmup_responses[0], "excluded_from_measurement": True})
        try:
            http_get(base_url + "/metrics", timeout=5)
            metrics_available = True
        except (urllib.error.URLError, TimeoutError, OSError):
            metrics_available = False

        with raw_path.open("w", encoding="utf-8") as raw, outputs_path.open("w", encoding="utf-8") as outputs:
            for trial, root_seed in enumerate(seeds):
                batch_group = 0
                for begin in range(0, len(prompt_rows), batch_size):
                    batch = prompt_rows[begin:begin + batch_size]
                    before = parse_metrics(http_get(base_url + "/metrics").decode("utf-8", errors="replace")) if metrics_available else {}
                    responses, response_completion_ms = send_server_batch(batch, root_seed, base_url)
                    total_tokens = sum(row["completion_tokens"] for row in responses)
                    output_start = time.perf_counter()
                    batch_id = f"{split}_b{batch_size}_s{root_seed}_g{batch_group}"
                    for response in responses:
                        outputs.write(json.dumps({
                            "batch_id": batch_id, "split": split,
                            "engine": "llama.cpp", "mode": "draft-simple",
                            **response,
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    outputs.flush()
                    os.fsync(outputs.fileno())
                    output_write_ms = (time.perf_counter() - output_start) * 1000.0
                    batch_wall_ms = response_completion_ms + output_write_ms
                    after = parse_metrics(http_get(base_url + "/metrics").decode("utf-8", errors="replace")) if metrics_available else {}
                    delta = {key: after[key] - before[key] for key in METRIC_SUFFIXES
                             if key in before and key in after}
                    ids = [row["prompt_id"] for row in responses]
                    row = {
                        "schema_version": 1, "batch_id": batch_id,
                        "engine": "llama.cpp", "mode": "draft-simple", "split": split,
                        "trial": trial, "seed_root": root_seed, "batch_index": batch_group,
                        "batch_size_configured": batch_size, "batch_prompt_ids": ids,
                        "request_seed_inputs": [row["request_seed_input"] for row in responses],
                        "prompt_tokens": [row["prompt_tokens"] for row in responses],
                        "completion_tokens": [row["completion_tokens"] for row in responses],
                        "finish_reasons": [row["finish_reason"] for row in responses],
                        "output_horizon": OUTPUT_HORIZON, "sampling": SAMPLING,
                        "request_latency_ms": [row["latency_ms"] for row in responses],
                        "generation_response_completion_ms": response_completion_ms,
                        "output_write_ms": output_write_ms,
                        "batch_completion_ms": batch_wall_ms,
                        "aggregate_tokens_per_second": total_tokens * 1000.0 / batch_wall_ms if batch_wall_ms else 0.0,
                        "total_completion_tokens": total_tokens,
                        "server_timings": [row["server_timings"] for row in responses],
                        "server_metrics_before": before if metrics_available else None,
                        "server_metrics_after": after if metrics_available else None,
                        "server_metrics_delta": delta if metrics_available else None,
                        "target_verifier_decode_tokens_derived": (
                            delta.get("spec_decode_num_draft_tokens_total", 0.0) +
                            delta.get("spec_decode_num_drafts_total", 0.0)
                        ) if metrics_available and "spec_decode_num_draft_tokens_total" in delta and
                             "spec_decode_num_drafts_total" in delta else None,
                        "draft_context_forward_tokens": None,
                        "draft_context_kv_tokens": None,
                        "target_decode_calls": None,
                        "cache_copy_rollback_counters": None,
                        "peak_process_rss_bytes": server_rss_hwm(process.pid),
                        "peak_live_kv_tokens": None,
                        "peak_kv_bytes_estimate": None,
                        "unavailable_counter_reason": "Pinned server metrics omit draft-context forwards/KV, exact target speculative decode-call and cache copy/rollback counts; response and global metrics retain exposed counters.",
                    }
                    raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                    raw.flush()
                    os.fsync(raw.fileno())
                    batch_group += 1
        atomic_json(run_dir / split / "commands" / f"{label}_process.json", {
            "startup_ms": startup_ms,
            "peak_rss_bytes": server_rss_hwm(process.pid),
            "metrics_available": metrics_available,
            "return_code_before_terminate": process.poll(),
            "raw_path": str(raw_path), "outputs_path": str(outputs_path),
        })
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        service_log.close()
    rows = read_jsonl(raw_path)
    expected_batches = len(seeds) * len(prompt_rows) // batch_size
    if len(rows) != expected_batches:
        raise RuntimeError(f"built-in server wrote {len(rows)} raw batches; expected {expected_batches}")
    print(f"[{utc_now()}] llama.cpp {label}: retained {len(rows)} batches", flush=True)
    return raw_path


def median_throughput(raw_path: Path) -> float:
    rows = read_jsonl(raw_path)
    if not rows:
        raise RuntimeError(f"no tuning observations found in {raw_path}")
    return statistics.median(float(row["aggregate_tokens_per_second"]) for row in rows)


def run_tuning(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {"protocol_revision": 1.1, "tuning_seeds": TUNING_SEEDS, "by_batch_size": {}}
    tuning_rows: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        ar_path = run_native(args, run_dir, prompts, "tuning", batch_size, TUNING_SEEDS, "pulsar-ar")
        ar_score = median_throughput(ar_path)
        tuning_rows.append({"batch_size": batch_size, "method": "autoregressive", "configuration": {},
                            "median_tokens_per_second": ar_score, "raw_path": str(ar_path)})
        builtin = []
        for n_max in BUILTIN_CANDIDATES:
            path = run_server(args, run_dir, prompts, "tuning", batch_size, TUNING_SEEDS, n_max)
            builtin.append({"n_max": n_max, "path": path, "score": median_throughput(path)})
            tuning_rows.append({"batch_size": batch_size, "method": "llama.cpp-draft-simple",
                                "configuration": {"n_max": n_max},
                                "median_tokens_per_second": builtin[-1]["score"], "raw_path": str(path)})
        builtin_best = max(builtin, key=lambda item: (item["score"], -item["n_max"]))
        trees = []
        for branching, depth in TREE_CANDIDATES:
            config = {"branching": branching, "depth": depth}
            path = run_native(args, run_dir, prompts, "tuning", batch_size, TUNING_SEEDS,
                              "pulsar-tree", config)
            node_bound = sum(branching ** level for level in range(depth + 1))
            trees.append({**config, "node_bound": node_bound, "path": path,
                          "score": median_throughput(path)})
            tuning_rows.append({"batch_size": batch_size, "method": "pulsar-fixed-tree",
                                "configuration": config,
                                "median_tokens_per_second": trees[-1]["score"], "raw_path": str(path)})
        tree_best = max(trees, key=lambda item: (item["score"], -item["node_bound"]))
        results["by_batch_size"][str(batch_size)] = {
            "autoregressive_raw": str(ar_path), "autoregressive_tuning_median_tokens_per_second": ar_score,
            "llama_cpp": {"n_max": builtin_best["n_max"], "raw": str(builtin_best["path"]),
                          "tuning_median_tokens_per_second": builtin_best["score"]},
            "pulsar_tree": {"branching": tree_best["branching"], "depth": tree_best["depth"],
                            "node_bound": tree_best["node_bound"], "raw": str(tree_best["path"]),
                            "tuning_median_tokens_per_second": tree_best["score"]},
        }
    freeze = {
        "status": "frozen_before_confirmatory_runs",
        "frozen_at_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "prompt_manifest_sha256": sha256_file(PROMPT_MANIFEST),
        "tuning_raw_sha256": {str(row["raw_path"]): sha256_file(Path(row["raw_path"])) for row in tuning_rows},
        "selection_rule": "highest median aggregate generated tokens/second across tuning batches; tie-break smaller n_max/node bound",
        "results": results,
        "candidate_scores": tuning_rows,
    }
    atomic_json(run_dir / "frozen_config.json", freeze)
    atomic_json(run_dir / "tuning_summary.json", results)
    return freeze


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def prompt_samples(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    samples: dict[tuple[str, int], float] = {}
    for row in rows:
        for prompt_id, latency in zip(row["batch_prompt_ids"], row["request_latency_ms"]):
            key = (prompt_id, int(row["seed_root"]))
            if key in samples:
                raise ValueError(f"duplicate prompt/seed pair in confirmatory records: {key}")
            samples[key] = float(latency)
    return samples


def batch_map(rows: list[dict[str, Any]]) -> dict[tuple[int, tuple[str, ...]], dict[str, Any]]:
    batches: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["seed_root"]), tuple(str(value) for value in row["batch_prompt_ids"]))
        if key in batches:
            raise ValueError(f"duplicate seed/prompt group in confirmatory records: {key}")
        batches[key] = row
    return batches


def bootstrap_ratio(ar_rows: list[dict[str, Any]], method_rows: list[dict[str, Any]],
                    statistic: str, *, q: float = 0.5) -> dict[str, Any]:
    ar_batches = batch_map(ar_rows)
    method_batches = batch_map(method_rows)
    if set(ar_batches) != set(method_batches):
        raise ValueError(f"confirmatory batch groups do not match: AR={len(ar_batches)} method={len(method_batches)}")
    keys = sorted(ar_batches)
    rng = random.Random(5555)
    draws: list[float] = []
    for _ in range(10_000):
        chosen = [keys[rng.randrange(len(keys))] for _ in keys]
        if statistic == "throughput":
            a_tokens = sum(float(ar_batches[key]["total_completion_tokens"]) for key in chosen)
            a_ms = sum(float(ar_batches[key]["batch_completion_ms"]) for key in chosen)
            m_tokens = sum(float(method_batches[key]["total_completion_tokens"]) for key in chosen)
            m_ms = sum(float(method_batches[key]["batch_completion_ms"]) for key in chosen)
            value = (m_tokens / m_ms) / (a_tokens / a_ms) if a_tokens and a_ms and m_ms else math.nan
        elif statistic == "batch_latency":
            a_ms = sum(float(ar_batches[key]["batch_completion_ms"]) for key in chosen)
            m_ms = sum(float(method_batches[key]["batch_completion_ms"]) for key in chosen)
            value = m_ms / a_ms if a_ms else math.nan
        else:
            a_values: list[float] = []
            m_values: list[float] = []
            for key in chosen:
                a_row, m_row = ar_batches[key], method_batches[key]
                a_values.extend(float(v) for v in a_row["request_latency_ms"])
                m_values.extend(float(v) for v in m_row["request_latency_ms"])
            a_stat, m_stat = percentile(a_values, q), percentile(m_values, q)
            value = m_stat / a_stat if a_stat else math.nan
        if math.isfinite(value):
            draws.append(value)
    return {"low_95": percentile(draws, 0.025), "high_95": percentile(draws, 0.975),
            "draws": len(draws), "seed": 5555, "cluster_count": len(keys)}


def classify_ratio(ratio: float, ci: dict[str, Any], favorable_direction: str) -> str:
    if 0.95 <= ratio <= 1.05:
        return "practical tie"
    if favorable_direction == "higher":
        if ratio >= 1.05 and ci["low_95"] > 1.0:
            return "practical win"
        if ratio <= 0.95 and ci["high_95"] < 1.0:
            return "practical loss"
    else:
        if ratio <= 0.95 and ci["high_95"] < 1.0:
            return "practical win"
        if ratio >= 1.05 and ci["low_95"] > 1.0:
            return "practical loss"
    return "inconclusive"


def parse_server_log_counters(log_path: Path, expected_requests: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    acceptance_re = re.compile(
        r"slot print_timing:\s*id\s+\d+\s+\|\s*task\s+(-?\d+)\s+\|\s*"
        r"draft acceptance\s*=\s*[0-9.]+\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated\)"
    )
    release_re = re.compile(
        r"slot\s+release:\s*id\s+\d+\s+\|\s*task\s+(-?\d+)\s+\|.*?"
        r"n_tokens\s*=\s*(\d+)"
    )
    acceptance_rows = [(int(task), int(accepted), int(proposed))
                       for task, accepted, proposed in acceptance_re.findall(text)]
    release_rows = [(int(task), int(tokens)) for task, tokens in release_re.findall(text)]
    if len(acceptance_rows) != expected_requests + 1 or len(release_rows) != expected_requests + 1:
        raise RuntimeError(
            f"server log task count mismatch in {log_path.name}: "
            f"acceptance={len(acceptance_rows)} release={len(release_rows)} expected={expected_requests + 1} including warm-up"
        )
    task_ids = [task for task, _, _ in acceptance_rows]
    release_task_ids = [task for task, _ in release_rows]
    if len(set(task_ids)) != len(task_ids) or len(set(release_task_ids)) != len(release_task_ids):
        raise RuntimeError(f"duplicate task IDs in server log: {log_path}")
    warmup_task_id = task_ids[0]
    measured_acceptance = [row for row in acceptance_rows if row[0] != warmup_task_id]
    measured_releases = [row for row in release_rows if row[0] != warmup_task_id]
    if len(measured_acceptance) != expected_requests or len(measured_releases) != expected_requests:
        raise RuntimeError(f"could not isolate the single warm-up task in {log_path}")
    if {task for task, _, _ in measured_acceptance} != {task for task, _ in measured_releases}:
        raise RuntimeError(f"server acceptance and release task IDs differ in {log_path}")
    proposed = sum(proposed for _, _, proposed in measured_acceptance)
    accepted = sum(accepted for _, accepted, _ in measured_acceptance)
    return {
        "source_log": str(log_path),
        "source_log_sha256": sha256_file(log_path),
        "requests_counted": expected_requests,
        "excluded_warmup_task_id": warmup_task_id,
        "draft_proposed_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "draft_acceptance_rate": accepted / proposed if proposed else None,
        "max_measured_sequence_tokens": max(tokens for _, tokens in measured_releases),
        "task_level_acceptance": [
            {"task_id": task, "accepted_tokens": accepted_count, "proposed_tokens": proposed_count}
            for task, accepted_count, proposed_count in measured_acceptance
        ],
    }


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(value) for row in rows for value in row["request_latency_ms"]]
    output_tokens = sum(int(row["total_completion_tokens"]) for row in rows)
    total_batch_ms = sum(float(row["batch_completion_ms"]) for row in rows)
    prompt_latency_values: dict[str, list[float]] = {}
    for row in rows:
        for prompt_id, value in zip(row["batch_prompt_ids"], row["request_latency_ms"]):
            prompt_latency_values.setdefault(prompt_id, []).append(float(value))
    rss = [int(row["peak_process_rss_bytes"]) for row in rows if row.get("peak_process_rss_bytes")]
    target_kv_tokens = [int(row["target_model_work"]["peak_live_kv_tokens"])
                        for row in rows if row.get("target_model_work") is not None]
    target_kv_bytes = [int(row["target_model_work"]["peak_kv_bytes_estimate"])
                       for row in rows if row.get("target_model_work") is not None]
    draft_kv_tokens = [int(row["draft_model_work"]["peak_live_kv_tokens"])
                       for row in rows if row.get("draft_model_work") is not None]
    draft_kv_bytes = [int(row["draft_model_work"]["peak_kv_bytes_estimate"])
                      for row in rows if row.get("draft_model_work") is not None]
    server_max_seq = [int((row.get("server_metrics_after") or {})["n_tokens_max"])
                      for row in rows if "n_tokens_max" in (row.get("server_metrics_after") or {})]
    native_target = [row["target_model_work"] for row in rows if row.get("target_model_work") is not None]
    native_draft = [row["draft_model_work"] for row in rows if row.get("draft_model_work") is not None]
    server_deltas = [row.get("server_metrics_delta") or {} for row in rows]

    def nested_total(model_rows: list[dict[str, Any]], field: str) -> int | None:
        return sum(int(item.get(field, 0)) for item in model_rows) if model_rows else None

    def server_total(metric: str) -> int | None:
        values = [delta[metric] for delta in server_deltas if metric in delta]
        return int(sum(values)) if values else None

    target_prefill_tokens = nested_total(native_target, "prefill_tokens")
    if target_prefill_tokens is None:
        target_prefill_tokens = server_total("prompt_tokens_total")
    if target_prefill_tokens is None:
        prompt_token_values = [int(value) for row in rows for value in row.get("prompt_tokens", [])]
        target_prefill_tokens = sum(prompt_token_values) if prompt_token_values else None
    target_decode_tokens = nested_total(native_target, "decode_tokens")
    draft_prefill_tokens = nested_total(native_draft, "prefill_tokens")
    draft_decode_tokens = nested_total(native_draft, "decode_tokens")
    target_verify_tokens = sum(int(row["target_verifier_decode_tokens"]) for row in rows
                               if row.get("target_verifier_decode_tokens") is not None)
    if not target_verify_tokens:
        target_verify_values = [row["target_verifier_decode_tokens_derived"] for row in rows
                                if row.get("target_verifier_decode_tokens_derived") is not None]
        target_verify_tokens = int(sum(target_verify_values)) if target_verify_values else None
    proposed_tokens = sum(int(row.get("proposed_tokens", 0)) for row in rows)
    if not proposed_tokens:
        proposed_tokens = server_total("spec_decode_num_draft_tokens_total")
    accepted_tokens = sum(int(row.get("accepted_proposals", 0)) for row in rows)
    if not accepted_tokens:
        accepted_tokens = server_total("spec_decode_num_accepted_tokens_total")
    work_counters = {
        "target_prefill_calls": nested_total(native_target, "prefill_calls"),
        "target_prefill_tokens": target_prefill_tokens,
        "target_decode_calls": nested_total(native_target, "decode_calls"),
        "target_decode_tokens": target_decode_tokens,
        "draft_prefill_calls": nested_total(native_draft, "prefill_calls"),
        "draft_prefill_tokens": draft_prefill_tokens,
        "draft_decode_calls": nested_total(native_draft, "decode_calls"),
        "draft_decode_tokens": draft_decode_tokens,
        "draft_proposed_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "target_verifier_decode_tokens": target_verify_tokens,
        "verification_steps": server_total("spec_decode_num_drafts_total"),
        "candidate_tree_nodes_built": sum(int(row.get("candidate_tree_nodes_built", 0)) for row in rows),
        "candidate_tree_pulses": sum(int(row.get("candidate_tree_pulses", 0)) for row in rows),
        "target_cache_copy_calls": nested_total(native_target, "sequence_copy_calls"),
        "target_cache_rollback_calls": nested_total(native_target, "sequence_rollback_calls"),
        "draft_cache_copy_calls": nested_total(native_draft, "sequence_copy_calls"),
        "draft_cache_rollback_calls": nested_total(native_draft, "sequence_rollback_calls"),
        "candidate_tree_build_inclusive_ms": sum(float(row.get("candidate_tree_build_inclusive_ms", 0.0)) for row in rows),
        "candidate_tree_verification_inclusive_ms": sum(float(row.get("candidate_tree_verification_inclusive_ms", 0.0)) for row in rows),
    }
    return {
        "batch_groups": len(rows), "requests": sum(len(row["batch_prompt_ids"]) for row in rows),
        "output_tokens": output_tokens,
        "aggregate_tokens_per_second": output_tokens * 1000.0 / total_batch_ms if total_batch_ms else 0.0,
        "total_batch_completion_seconds": total_batch_ms / 1000.0,
        "batch_completion_ms_p50": percentile([float(r["batch_completion_ms"]) for r in rows], 0.5),
        "batch_completion_ms_p95": percentile([float(r["batch_completion_ms"]) for r in rows], 0.95),
        "request_latency_ms_p50": percentile(latencies, 0.5),
        "request_latency_ms_p95": percentile(latencies, 0.95),
        "per_prompt_latency_ms": {key: {"n": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}
                                   for key, values in sorted(prompt_latency_values.items())},
        "peak_process_rss_bytes_max": max(rss) if rss else None,
        "target_peak_kv_tokens_max": max(target_kv_tokens) if target_kv_tokens else None,
        "target_peak_kv_bytes_estimate_max": max(target_kv_bytes) if target_kv_bytes else None,
        "draft_peak_kv_tokens_max": max(draft_kv_tokens) if draft_kv_tokens else None,
        "draft_peak_kv_bytes_estimate_max": max(draft_kv_bytes) if draft_kv_bytes else None,
        "server_max_observed_sequence_tokens": max(server_max_seq) if server_max_seq else None,
        "work_counters": work_counters,
    }


def compare_method(ar_rows: list[dict[str, Any]], method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ar_summary = summarize_method(ar_rows)
    method_summary = summarize_method(method_rows)
    throughput_ratio = method_summary["aggregate_tokens_per_second"] / ar_summary["aggregate_tokens_per_second"]
    ar_latency = prompt_samples(ar_rows)
    method_latency = prompt_samples(method_rows)
    if set(ar_latency) != set(method_latency):
        raise ValueError(f"confirmatory prompt/seed pairs do not match: AR={len(ar_latency)} method={len(method_latency)}")
    pairs = sorted(ar_latency)
    ar_p50 = percentile([ar_latency[key] for key in pairs], 0.5)
    method_p50 = percentile([method_latency[key] for key in pairs], 0.5)
    ar_p95 = percentile([ar_latency[key] for key in pairs], 0.95)
    method_p95 = percentile([method_latency[key] for key in pairs], 0.95)
    throughput_ci = bootstrap_ratio(ar_rows, method_rows, "throughput")
    batch_latency_ratio = method_summary["total_batch_completion_seconds"] / ar_summary["total_batch_completion_seconds"]
    batch_latency_ci = bootstrap_ratio(ar_rows, method_rows, "batch_latency")
    p50_ci = bootstrap_ratio(ar_rows, method_rows, "request_latency", q=0.5)
    p95_ci = bootstrap_ratio(ar_rows, method_rows, "request_latency", q=0.95)
    return {
        "ar": ar_summary, "method": method_summary,
        "matched_prompt_seed_pairs": len(pairs),
        "throughput_ratio_vs_ar": throughput_ratio,
        "throughput_ratio_95_ci": throughput_ci,
        "throughput_classification": classify_ratio(throughput_ratio, throughput_ci, "higher"),
        "total_batch_time_ratio_vs_ar": batch_latency_ratio,
        "total_batch_time_ratio_95_ci": batch_latency_ci,
        "batch_time_classification": classify_ratio(batch_latency_ratio, batch_latency_ci, "lower"),
        "request_p50_latency_ratio_vs_ar": method_p50 / ar_p50 if ar_p50 else math.nan,
        "request_p50_latency_ratio_95_ci": p50_ci,
        "request_p50_classification": classify_ratio(method_p50 / ar_p50, p50_ci, "lower"),
        "request_p95_latency_ratio_vs_ar": method_p95 / ar_p95 if ar_p95 else math.nan,
        "request_p95_latency_ratio_95_ci": p95_ci,
        "request_p95_classification": classify_ratio(method_p95 / ar_p95, p95_ci, "lower"),
    }


def write_report(run_dir: Path, freeze: dict[str, Any], comparisons: dict[str, Any]) -> None:
    lines = [
        "# Phase 3: fixed-policy and cost-profile benchmark",
        "",
        f"Run ID: `{run_dir.name}`  ",
        f"Confirmatory data completed: {utc_now()}  ",
        "Protocol: [PHASE3_PROTOCOL.md](../../../../cpp/production/benchmark/PHASE3_PROTOCOL.md)  ",
        "Raw evidence: `./confirmatory/raw/`, `./confirmatory/outputs/`, and `./confirmatory/logs/`.",
        "",
        "## Design and frozen settings",
        "",
        "This is a CPU-only TinyStories 15M GGUF fixture run, with target/draft Q2_K/Q3_K_M, 2 threads, batch sizes 1 and 2, eight fixed prompts split 4 tuning/4 held out, 64-token cap, 256-token context, common target sampling settings, and paired seed-root/prompt-ID inputs. Tuning used only T01–T04. No adaptive scheduling was implemented or evaluated. There is no speedup pass/fail gate.",
        "",
        "| Batch | Method | Frozen setting | Tuning median aggregate tok/s |",
        "|---:|---|---|---:|",
    ]
    for batch_size, config in freeze["results"]["by_batch_size"].items():
        lines.append(f"| {batch_size} | Autoregressive | target only | {config['autoregressive_tuning_median_tokens_per_second']:.2f} |")
        lines.append(f"| {batch_size} | llama.cpp built-in chain | draft-simple n_max={config['llama_cpp']['n_max']} | {config['llama_cpp']['tuning_median_tokens_per_second']:.2f} |")
        tree = config["pulsar_tree"]
        lines.append(f"| {batch_size} | Pulsar fixed tree | branching={tree['branching']}, depth={tree['depth']}, node bound={tree['node_bound']} | {tree['tuning_median_tokens_per_second']:.2f} |")
    lines += ["", "## Held-out paired results", ""]
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        lines += [f"### Batch size {batch_size}", "",
                  "| Method | Aggregate tok/s | Total batch time (s) | Request latency p50 / p95 (ms) | Peak RSS (MiB) | Peak KV tokens / estimated bytes | Paired throughput ratio (95% CI) | Throughput assessment |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        baseline = comparisons[key]["ar"]
        baseline_rss = baseline["peak_process_rss_bytes_max"]
        baseline_rss_mib = f"{baseline_rss / (1024 * 1024):.1f}" if baseline_rss is not None else "n/a"
        baseline_kv = (f"target {baseline['target_peak_kv_tokens_max']} / {baseline['target_peak_kv_bytes_estimate_max']} B"
                       if baseline["target_peak_kv_tokens_max"] is not None else "n/a")
        lines.append(
            f"| Pulsar autoregressive reference | {baseline['aggregate_tokens_per_second']:.2f} | "
            f"{baseline['total_batch_completion_seconds']:.3f} | "
            f"{baseline['request_latency_ms_p50']:.2f} / {baseline['request_latency_ms_p95']:.2f} | "
            f"{baseline_rss_mib} | {baseline_kv} | 1.000 (reference) | reference |"
        )
        for method_key, method_name in [("llama_cpp", "llama.cpp draft-simple"), ("pulsar_tree", "Pulsar fixed tree")]:
            row = comparisons[key][method_key]
            s = row["method"]
            rss = s["peak_process_rss_bytes_max"]
            rss_mib = f"{rss / (1024 * 1024):.1f}" if rss is not None else "n/a"
            if s["target_peak_kv_tokens_max"] is not None:
                kv = (f"target {s['target_peak_kv_tokens_max']} / {s['target_peak_kv_bytes_estimate_max']} B; "
                      f"draft {s['draft_peak_kv_tokens_max']} / {s['draft_peak_kv_bytes_estimate_max']} B")
            elif s["server_max_observed_sequence_tokens"] is not None:
                kv = f"max seq {s['server_max_observed_sequence_tokens']}; live total n/a"
            else:
                kv = "n/a (not exported)"
            ci = row["throughput_ratio_95_ci"]
            lines.append(
                f"| {method_name} | {s['aggregate_tokens_per_second']:.2f} | {s['total_batch_completion_seconds']:.3f} | "
                f"{s['request_latency_ms_p50']:.2f} / {s['request_latency_ms_p95']:.2f} | {rss_mib} | {kv} | "
                f"{row['throughput_ratio_vs_ar']:.3f} ({ci['low_95']:.3f}–{ci['high_95']:.3f}) | {row['throughput_classification']} |"
            )
        lines += ["", "| Method | Total batch-time ratio vs AR (95% CI) | Assessment | Request p50 latency ratio (95% CI) | Assessment | Request p95 latency ratio (95% CI) | Assessment |",
                  "|---|---:|---|---:|---|---:|---|"]
        for method_key, method_name in [("llama_cpp", "llama.cpp draft-simple"), ("pulsar_tree", "Pulsar fixed tree")]:
            row = comparisons[key][method_key]
            bt, p50, p95 = row["total_batch_time_ratio_95_ci"], row["request_p50_latency_ratio_95_ci"], row["request_p95_latency_ratio_95_ci"]
            lines.append(f"| {method_name} | {row['total_batch_time_ratio_vs_ar']:.3f} ({bt['low_95']:.3f}–{bt['high_95']:.3f}) | {row['batch_time_classification']} | {row['request_p50_latency_ratio_vs_ar']:.3f} ({p50['low_95']:.3f}–{p50['high_95']:.3f}) | {row['request_p50_classification']} | {row['request_p95_latency_ratio_vs_ar']:.3f} ({p95['low_95']:.3f}–{p95['high_95']:.3f}) | {row['request_p95_classification']} |")
        lines += ["", "Per-prompt request latency summaries (milliseconds; 40 held-out prompt/seed observations per method):", "",
                  "| Prompt | Autoregressive p50 / p95 | llama.cpp p50 / p95 | Pulsar tree p50 / p95 |", "|---|---:|---:|---:|"]
        for prompt_id in ("H01", "H02", "H03", "H04"):
            vals = []
            for method_key in ("ar", "llama_cpp", "pulsar_tree"):
                summary = comparisons[key][method_key] if method_key == "ar" else comparisons[key][method_key]["method"]
                s = summary["per_prompt_latency_ms"][prompt_id]
                vals.append(f"{s['p50']:.2f} / {s['p95']:.2f}")
            lines.append(f"| {prompt_id} | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines += ["", "Aggregate work-counter totals:", "",
                  "| Method | Target prefill / decode tokens | Draft prefill / decode tokens | Drafted / accepted tokens | Target verification tokens | Tree nodes / pulses | Tree build / verify inclusive ms | Cache copies / rollbacks (target + draft) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method_key, method_name in [("ar", "Pulsar autoregressive"),
                                        ("llama_cpp", "llama.cpp draft-simple"),
                                        ("pulsar_tree", "Pulsar fixed tree")]:
            summary = comparisons[key][method_key] if method_key == "ar" else comparisons[key][method_key]["method"]
            work = summary["work_counters"]

            def show(value: Any) -> str:
                return "n/a" if value is None else str(value)

            target_pair = f"{show(work['target_prefill_tokens'])} / {show(work['target_decode_tokens'])}"
            draft_pair = f"{show(work['draft_prefill_tokens'])} / {show(work['draft_decode_tokens'])}"
            proposals = f"{show(work['draft_proposed_tokens'])} / {show(work['accepted_draft_tokens'])}"
            tree = f"{work['candidate_tree_nodes_built']} / {work['candidate_tree_pulses']}"
            tree_ms = (f"{work['candidate_tree_build_inclusive_ms']:.1f} / "
                       f"{work['candidate_tree_verification_inclusive_ms']:.1f}")
            copies = work["target_cache_copy_calls"]
            rollbacks = work["target_cache_rollback_calls"]
            if work["draft_cache_copy_calls"] is not None:
                copies = (copies or 0) + work["draft_cache_copy_calls"]
                rollbacks = (rollbacks or 0) + (work["draft_cache_rollback_calls"] or 0)
            lines.append(
                f"| {method_name} | {target_pair} | {draft_pair} | {proposals} | "
                f"{show(work['target_verifier_decode_tokens'])} | {tree} | {tree_ms} | "
                f"{show(copies)} / {show(rollbacks)} |"
            )
        lines.append("")
    lines += [
        "## Analysis correction",
        "",
        "A post-run audit found that llama.cpp's raw `batch_index` is a run-global ordinal across seed roots, unlike the native per-seed ordinal. The initial paired analysis/report that keyed on `batch_index` is preserved under `attempt_failures/analysis_batch_index_pairing_bug/`. The corrected paired bootstrap uses the preregistered key `(seed_root, tuple(batch_prompt_ids))`, includes all 40 matched groups at batch size 1 and all 20 at batch size 2, and leaves every raw observation unchanged. The server's original `batch_index` remains in raw JSONL for traceability but is not used for pairing.",
        "",
        "## Work counters and caveats",
        "",
        "The raw JSONL is authoritative for batch observations. Native records expose target/draft prefill/decode work, tree node/pulse counts, candidate construction and verification time, sequence-cache copies/rollbacks, prompt/completion tokens, and inclusive batch timing. The llama.cpp server's /metrics endpoint was disabled for this run; exact proposed and accepted draft-token totals and maximum per-sequence lengths were recovered from retained per-task server timing logs, with the excluded warm-up task removed and log hashes recorded in the analysis JSON. Exact draft-context prefill/KV use, target verifier-step count, and cache copy/rollback operations are unavailable and remain null rather than inferred. Native KV bytes are an F16/equal-head-dimension estimate from live sequence lengths, not allocator telemetry; the server maximum sequence length is not total live KV occupancy. End-to-end batch wall time includes generation, verification, drafting, tree/cache work, response handling, and output record writes. Component timers overlap where documented in the protocol and must not be summed as disjoint stages.",
        "",
        "## Reproducibility and retained data",
        "",
        "`run_manifest.json` records model SHA-256s, source/upstream revisions, host/CPU/OS/build details and timestamps. `frozen_config.json` records tuning scores and raw-data hashes before any held-out run. Commands, process logs, warm-up records, prompt copies, generated outputs, per-batch raw JSONL, paired analysis JSON, and this report remain in the run directory. Bootstrap seed is 5555 with 10,000 paired cluster resamples. See [PHASE3_PROTOCOL.md](../../../../cpp/production/benchmark/PHASE3_PROTOCOL.md) for the predeclared margins, workloads, and timing contracts.",
        "",
        "## Interpretation",
        "",
        "Read results by workload, balancing throughput, total completion time, p50/p95 latency, memory, and exposed work counters. A practical win is not required for Phase 3 completion; practical ties, trade-offs, losses, or inconclusive estimates are all valid findings for this fixture.",
        "",
    ]
    (run_dir / "PHASE3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_confirmatory(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]], freeze: dict[str, Any]) -> dict[str, Any]:
    selected_raw: dict[str, dict[str, Path]] = {}
    for batch_size in BATCH_SIZES:
        config = freeze["results"]["by_batch_size"][str(batch_size)]
        ar_path = run_native(args, run_dir, prompts, "confirmatory", batch_size,
                             CONFIRMATORY_SEEDS, "pulsar-ar")
        builtin_n = int(config["llama_cpp"]["n_max"])
        builtin_path = run_server(args, run_dir, prompts, "confirmatory", batch_size,
                                  CONFIRMATORY_SEEDS, builtin_n)
        tree_config = {"branching": int(config["pulsar_tree"]["branching"]),
                       "depth": int(config["pulsar_tree"]["depth"])}
        tree_path = run_native(args, run_dir, prompts, "confirmatory", batch_size,
                               CONFIRMATORY_SEEDS, "pulsar-tree", tree_config)
        selected_raw[str(batch_size)] = {"ar": ar_path, "llama_cpp": builtin_path, "pulsar_tree": tree_path}
    comparisons = analyze_confirmatory(run_dir, freeze)
    atomic_json(run_dir / "run_status.json", {"status": "complete", "completed_at_utc": utc_now()})
    return comparisons


def analyze_confirmatory(run_dir: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "analysis_metadata": {
            "analysis_revision": "paired-prompt-group-v2",
            "paired_cluster_key": "(seed_root, tuple(batch_prompt_ids))",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 5555,
            "server_batch_index_used_for_pairing": False,
            "server_metrics_endpoint_enabled": False,
            "server_draft_counts_source": "retained per-task server timing logs, excluding the excluded warm-up task",
        }
    }
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        config = freeze["results"]["by_batch_size"][key]
        tree = config["pulsar_tree"]
        raw_dir = run_dir / "confirmatory" / "raw"
        paths = {
            "ar": raw_dir / f"confirmatory_pulsar-ar_b{batch_size}.jsonl",
            "llama_cpp": raw_dir / f"confirmatory_llama-draft-simple_n{config['llama_cpp']['n_max']}_b{batch_size}.jsonl",
            "pulsar_tree": raw_dir / f"confirmatory_pulsar-tree_branching{tree['branching']}_depth{tree['depth']}_b{batch_size}.jsonl",
        }
        ar_rows = read_jsonl(paths["ar"])
        builtin_rows = read_jsonl(paths["llama_cpp"])
        tree_rows = read_jsonl(paths["pulsar_tree"])
        expected = len(CONFIRMATORY_SEEDS) * 4 // batch_size
        for method_name, rows in (("ar", ar_rows), ("llama_cpp", builtin_rows), ("pulsar_tree", tree_rows)):
            if len(rows) != expected:
                raise RuntimeError(f"{method_name} batch count mismatch for B={batch_size}: {len(rows)} != {expected}")
        builtin_comparison = compare_method(ar_rows, builtin_rows)
        builtin_log = run_dir / "confirmatory" / "logs" / (
            f"confirmatory_llama-draft-simple_n{config['llama_cpp']['n_max']}_b{batch_size}_server.log"
        )
        server_log_counters = parse_server_log_counters(
            builtin_log, expected_requests=len(CONFIRMATORY_SEEDS) * 4
        )
        builtin_comparison["server_log_counters"] = server_log_counters
        builtin_summary = builtin_comparison["method"]
        builtin_summary["server_max_observed_sequence_tokens"] = server_log_counters["max_measured_sequence_tokens"]
        builtin_summary["work_counters"]["draft_proposed_tokens"] = server_log_counters["draft_proposed_tokens"]
        builtin_summary["work_counters"]["accepted_draft_tokens"] = server_log_counters["accepted_draft_tokens"]
        comparisons[key] = {
            "raw_paths": {name: str(path) for name, path in paths.items()},
            "ar": summarize_method(ar_rows),
            "llama_cpp": builtin_comparison,
            "pulsar_tree": compare_method(ar_rows, tree_rows),
        }
        for method_name, rows in (("ar", ar_rows), ("llama_cpp", builtin_rows), ("pulsar_tree", tree_rows)):
            if any(row.get("split") != "confirmatory" for row in rows):
                raise RuntimeError(f"{method_name} contains a non-confirmatory raw row for B={batch_size}")
    atomic_json(run_dir / "confirmatory_analysis.json", comparisons)
    write_report(run_dir, freeze, comparisons)
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "tuning", "confirmatory", "analyze"), default="all")
    parser.add_argument("--bench-bin", type=Path, default=Path("/tmp/pulsar-phase3-build/pulsar-phase3-bench"))
    parser.add_argument("--llama-server", type=Path, default=Path("/tmp/llama-phase3-upstream/bin/llama-server"))
    parser.add_argument("--target-model", type=Path, default=Path("/tmp/pulsar-smoke.gguf"))
    parser.add_argument("--draft-model", type=Path, default=Path("/tmp/pulsar-smoke-draft.gguf"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper" / "results" / "phase3")
    parser.add_argument("--run-dir", type=Path, help="existing run directory for --phase confirmatory or analyze")
    return parser.parse_args()


def main() -> int:
    global ACTIVE_RUN_DIR
    args = parse_args()
    if args.phase == "analyze":
        if args.run_dir is None:
            raise ValueError("--phase analyze requires --run-dir with completed confirmatory raw data")
        run_dir = args.run_dir.resolve()
        ACTIVE_RUN_DIR = run_dir
        freeze_path = run_dir / "frozen_config.json"
        if not freeze_path.exists():
            raise FileNotFoundError(f"frozen tuning configuration not found: {freeze_path}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "frozen_before_confirmatory_runs":
            raise ValueError("frozen configuration is not marked as frozen before confirmation")
        analyze_confirmatory(run_dir, freeze)
        status_path = run_dir / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update(status="complete", analysis_recomputed_at_utc=utc_now())
        atomic_json(status_path, status)
        print(run_dir.resolve())
        return 0
    for path, label in ((args.bench_bin, "native benchmark executable"),
                         (args.llama_server, "llama-server executable"),
                         (args.target_model, "target GGUF"), (args.draft_model, "draft GGUF")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    prompts = load_prompts()
    if args.phase == "confirmatory":
        if args.run_dir is None:
            raise ValueError("--phase confirmatory requires --run-dir from a completed tuning phase")
        run_dir = args.run_dir.resolve()
        ACTIVE_RUN_DIR = run_dir
        freeze_path = run_dir / "frozen_config.json"
        if not freeze_path.exists():
            raise FileNotFoundError(f"frozen tuning configuration not found: {freeze_path}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "frozen_before_confirmatory_runs":
            raise ValueError("frozen configuration is not marked as frozen before confirmation")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists; refusing to overwrite raw data: {run_dir}")
        run_dir.mkdir(parents=True)
        ACTIVE_RUN_DIR = run_dir
        (run_dir / "inputs").mkdir()
        (run_dir / "warmups").mkdir()
        shutil.copy2(PROMPT_MANIFEST, run_dir / "prompts.jsonl")
        shutil.copy2(PROTOCOL, run_dir / "PHASE3_PROTOCOL.md")
        metadata = {
            "run_id": run_id, "created_at_utc": utc_now(), "phase_requested": args.phase,
            "protocol_revision": 1.1, "protocol_sha256": sha256_file(PROTOCOL),
            "prompt_manifest_sha256": sha256_file(PROMPT_MANIFEST),
            "source_snapshot_commit": SOURCE_SNAPSHOT, "upstream_llama_cpp_revision": UPSTREAM_REVISION,
            "phase3_source_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in [
                    ROOT / "cpp/production/CMakeLists.txt",
                    ROOT / "cpp/production/include/pulsar/batch_generation.hpp",
                    ROOT / "cpp/production/include/pulsar/tree_generation.hpp",
                    ROOT / "cpp/production/src/llama_model.hpp",
                    ROOT / "cpp/production/src/llama_model.cpp",
                    ROOT / "cpp/production/src/phase3_bench.cpp",
                    PROTOCOL, PROMPT_MANIFEST, Path(__file__).resolve(),
                ]
            },
            "target_model": {"path": str(args.target_model), "size_bytes": args.target_model.stat().st_size,
                             "sha256": sha256_file(args.target_model)},
            "draft_model": {"path": str(args.draft_model), "size_bytes": args.draft_model.stat().st_size,
                             "sha256": sha256_file(args.draft_model)},
            "host": host_metadata(),
            "tools": {"pulsar_phase3_bench": str(args.bench_bin), "llama_server": str(args.llama_server),
                      "pulsar_phase3_bench_sha256": sha256_file(args.bench_bin),
                      "llama_server_sha256": sha256_file(args.llama_server),
                      "llama_server_version": command_version(args.llama_server),
                      "pulsar_cmake_cache_sha256": sha256_file(args.bench_bin.parent / "CMakeCache.txt"),
                      "llama_cpp_cmake_cache_sha256": sha256_file(args.llama_server.parent.parent / "CMakeCache.txt"),
                      "cxx_compiler": subprocess.run(["c++", "--version"], check=False,
                                                      capture_output=True, text=True).stdout.splitlines()[0]},
            "fixed_settings": {"threads": THREADS, "batch_sizes": BATCH_SIZES,
                               "output_horizon": OUTPUT_HORIZON, "context_size": CONTEXT_SIZE,
                               "batch_tokens": BATCH_TOKENS, "sampling": SAMPLING,
                               "tuning_seeds": TUNING_SEEDS, "confirmatory_seeds": CONFIRMATORY_SEEDS,
                               "builtin_candidates": BUILTIN_CANDIDATES,
                               "tree_candidates": TREE_CANDIDATES},
            "scope_note": "No adaptive scheduling; no changes to the shared publishing checkout.",
        }
        atomic_json(run_dir / "run_manifest.json", metadata)
        atomic_json(run_dir / "run_status.json", {"status": "started", "started_at_utc": utc_now()})
        freeze = {}
    if args.phase in ("all", "tuning"):
        freeze = run_tuning(args, run_dir, prompts)
        atomic_json(run_dir / "run_status.json", {
            "status": "tuning_complete_frozen" if args.phase == "tuning" else "confirmatory_starting",
            "updated_at_utc": utc_now(),
        })
    if args.phase in ("all", "confirmatory"):
        if not freeze:
            freeze = json.loads((run_dir / "frozen_config.json").read_text(encoding="utf-8"))
        run_confirmatory(args, run_dir, prompts, freeze)
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"phase3 benchmark failed: {exc}", file=sys.stderr)
        if ACTIVE_RUN_DIR is not None:
            status_path = ACTIVE_RUN_DIR / "run_status.json"
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    status.update(status="failed", updated_at_utc=utc_now(), failure=str(exc))
                    atomic_json(status_path, status)
                except Exception as status_exc:
                    print(f"could not update run status: {status_exc}", file=sys.stderr)
        raise
