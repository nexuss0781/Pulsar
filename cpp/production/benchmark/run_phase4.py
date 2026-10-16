#!/usr/bin/env python3
"""Run Phase 4 in calibration -> tuning/freeze -> locked evaluation -> analysis order."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "prompts_phase4.jsonl"
PROTOCOL = HERE / "PHASE4_PROTOCOL.md"
FIT_SCRIPT = HERE / "fit_calibration.py"
ANALYSIS_SCRIPT = HERE / "analyze_phase4.py"
TEST_SCRIPT = HERE / "test_phase4_tools.py"
BATCH_SIZES = [1, 2]
CALIBRATION_SEEDS = [211, 307, 401]
TUNING_SEEDS = [1701, 2801, 3907]
EVALUATION_SEEDS = [5101, 6203, 7307, 8407, 9509, 10613, 11717, 12821, 13901, 15013]
TREE_ACTIONS = [(1, 2), (2, 2), (2, 3)]
BUILTIN_ACTIONS = [2, 4, 6]
THREADS = 2
OUTPUT_HORIZON = 64
CONTEXT_SIZE = 256
BATCH_TOKENS = 512
SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95,
            "repeat_penalty": 1.0, "repeat_window": 64}
EXPECTED_MODEL_SHA256 = {
    "target": "f7e39dc9f26f3d39bf59e885349c6eec65880f685322d591f53e6cdb46ceb2e9",
    "draft": "58e4a0c44851897f4007f9c9018822348f7d74851790fa0ea1370ed82c36e003",
}


def load_phase3_helpers() -> Any:
    path = HERE / "run_phase3.py"
    spec = importlib.util.spec_from_file_location("phase3_benchmark_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Phase 3 built-in server helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P3 = load_phase3_helpers()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [(f"C{i:02d}", "calibration") for i in range(1, 9)] + \
               [(f"T{i:02d}", "tuning") for i in range(1, 9)] + \
               [(f"E{i:02d}", "evaluation") for i in range(1, 9)]
    if [(row.get("id"), row.get("split")) for row in rows] != expected:
        raise RuntimeError("Phase 4 prompt manifest differs from its frozen 8/8/8 group split")
    for index, row in enumerate(rows):
        if int(row.get("numeric_id", -1)) != index or not row.get("prompt", "").strip():
            raise RuntimeError(f"invalid Phase 4 prompt row at index {index}")
        row["numeric_id"] = index
    return rows


def write_prompts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            text = row["prompt"]
            if "\t" in text or "\n" in text:
                raise ValueError(f"prompt {row['id']} contains a TSV delimiter")
            stream.write(f"{row['numeric_id']}\t{row['id']}\t{text}\n")


def rows_for(prompts: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in prompts if row["split"] == split]


def native_run(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]],
               split: str, batch_size: int, seeds: list[int], mode: str,
               candidate: tuple[int, int] | None = None,
               verify_output_rows: bool = True) -> Path:
    selected = rows_for(prompts, split)
    if len(selected) != 8 or len(selected) % batch_size:
        raise RuntimeError(f"split {split} must contain eight groups divisible by batch size")
    if [int(row["numeric_id"]) for row in selected] != list(
            range(int(selected[0]["numeric_id"]), int(selected[0]["numeric_id"]) + 8)):
        raise RuntimeError(f"split {split} prompt IDs must be contiguous and ordered")
    if candidate is None:
        label = f"{split}_{mode}_b{batch_size}"
    else:
        label = f"{split}_{mode}_B{candidate[0]}_D{candidate[1]}_b{batch_size}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    raw_path = run_dir / split / "raw" / f"{label}.jsonl"
    outputs_path = run_dir / split / "outputs" / f"{label}.jsonl"
    observation_path = run_dir / split / "observations" / f"{label}.jsonl"
    log_path = run_dir / split / "logs" / f"{label}.log"
    command_path = run_dir / split / "commands" / f"{label}.json"
    prompt_path = run_dir / "inputs" / f"{split}.tsv"
    warmup_path = run_dir / "inputs" / "warmup_C01.tsv"
    write_prompts(prompt_path, selected)
    write_prompts(warmup_path, [prompts[0]])
    for path in (raw_path, outputs_path, log_path, command_path, observation_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.bench_bin), "--mode", mode, "--split", split,
        "--model", str(args.target_model), "--prompts", str(prompt_path),
        "--warmup-prompts", str(warmup_path), "--raw-out", str(raw_path),
        "--outputs-out", str(outputs_path), "--seeds", ",".join(map(str, seeds)),
        "--batch-size", str(batch_size), "--max-new-tokens", str(OUTPUT_HORIZON),
        "--context-size", str(CONTEXT_SIZE), "--batch-tokens", str(BATCH_TOKENS),
        "--threads", str(THREADS), "--temperature", str(SAMPLING["temperature"]),
        "--top-k", str(SAMPLING["top_k"]), "--top-p", str(SAMPLING["top_p"]),
        "--repeat-penalty", str(SAMPLING["repeat_penalty"]),
        "--repeat-window", str(SAMPLING["repeat_window"]),
    ]
    if mode != "autoregressive":
        command.extend(["--draft-model", str(args.draft_model),
                        "--observations-out", str(observation_path)])
        if candidate is not None:
            command.extend(["--tree-branching", str(candidate[0]), "--tree-depth", str(candidate[1])])
        else:
            command.extend(["--tree-branching", "2", "--tree-depth", "3"])
        if mode == "pulsar-adaptive":
            command.extend(["--calibrator", str(run_dir / "calibration" / "calibrator.tsv"),
                            "--action-costs", str(run_dir / "tuning" / "action_costs.tsv")])
    command_record = {
        "command": command, "split": split, "seeds": seeds, "candidate": candidate,
        "target_model_sha256": sha256_file(args.target_model),
        "draft_model_sha256": sha256_file(args.draft_model) if mode != "autoregressive" else None,
        "warmup_prompt_id": "C01", "protocol_sha256": sha256_file(PROTOCOL),
        "source_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                           capture_output=True, text=True, check=True).stdout.strip(),
    }
    artifacts = (raw_path, outputs_path, observation_path)
    if raw_path.exists():
        if not verify_output_rows:
            raise RuntimeError(f"locked evaluation artifact already exists; refusing to overwrite: {raw_path}")
        if not command_path.exists():
            raise RuntimeError(f"cannot verify existing benchmark command for {raw_path}")
        old_command = json.loads(command_path.read_text(encoding="utf-8"))
        expected_candidate = list(candidate) if candidate is not None else None
        if (old_command.get("command") != command or old_command.get("split") != split or
                old_command.get("seeds") != seeds or old_command.get("candidate") != expected_candidate or
                old_command.get("target_model_sha256") != command_record["target_model_sha256"] or
                old_command.get("draft_model_sha256") != command_record["draft_model_sha256"] or
                old_command.get("protocol_sha256") != command_record["protocol_sha256"]):
            raise RuntimeError(f"existing raw data does not match this frozen command: {raw_path}")
        expected = len(seeds) * len(selected) // batch_size
        raw_rows = read_jsonl(raw_path)
        if len(raw_rows) != expected or any(row.get("split") != split for row in raw_rows):
            raise RuntimeError(f"existing raw data is incomplete or has split leakage: {raw_path}")
        if not outputs_path.exists() or outputs_path.stat().st_size == 0:
            raise RuntimeError(f"existing benchmark outputs are missing: {outputs_path}")
        if mode != "autoregressive" and (not observation_path.exists() or observation_path.stat().st_size == 0):
            raise RuntimeError(f"existing verifier observations are missing: {observation_path}")
        print(f"[{utc_now()}] native {label}: reusing verified {len(raw_rows)} existing batches", flush=True)
        return raw_path
    if any(path.exists() for path in artifacts):
        raise RuntimeError(f"partial benchmark artifacts exist; refusing to overwrite {raw_path}")
    atomic_json(command_path, command_record)
    print(f"[{utc_now()}] native {label}: starting", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   check=False, timeout=1200)
    if completed.returncode != 0:
        raise RuntimeError(f"native Phase 4 benchmark failed ({completed.returncode}); see {log_path}")
    elapsed = time.perf_counter() - started
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise RuntimeError(f"native Phase 4 benchmark wrote no raw data: {raw_path}")
    load_match = None
    if verify_output_rows:
        process_text = log_path.read_text(encoding="utf-8", errors="replace")
        load_match = re.search(r"loaded_ms=([0-9.]+)", process_text)
    atomic_json(command_path.with_name(label + "_process.json"), {
        "elapsed_process_seconds": elapsed,
        "model_load_ms": float(load_match.group(1)) if load_match else None,
        "raw_path": str(raw_path), "outputs_path": str(outputs_path),
        "observations_path": str(observation_path) if mode != "autoregressive" else None,
        "return_code": completed.returncode,
    })
    if verify_output_rows:
        raw_rows = read_jsonl(raw_path)
        expected = len(seeds) * len(selected) // batch_size
        if len(raw_rows) != expected:
            raise RuntimeError(f"{label} retained {len(raw_rows)} raw batches; expected {expected}")
    print(f"[{utc_now()}] native {label}: finished in {elapsed:.1f}s", flush=True)
    return raw_path


def run_builtin(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]],
                split: str, batch_size: int, seeds: list[int], n_max: int,
                verify_output_rows: bool = True) -> Path:
    selected = rows_for(prompts, split)
    if len(selected) != 8 or len(selected) % batch_size:
        raise RuntimeError(f"split {split} must contain eight groups divisible by batch size")
    label = f"{split}_llama_draft_simple_n{n_max}_b{batch_size}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    raw_path = run_dir / split / "raw" / f"{label}.jsonl"
    outputs_path = run_dir / split / "outputs" / f"{label}.jsonl"
    log_path = run_dir / split / "logs" / f"{label}_server.log"
    command_path = run_dir / split / "commands" / f"{label}.json"
    warmup_record_path = run_dir / "warmups" / f"{label}.jsonl"
    for path in (raw_path, outputs_path, log_path, command_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    port = P3.reserve_port()
    server_args = SimpleNamespace(llama_server=args.llama_server,
                                  target_model=args.target_model,
                                  draft_model=args.draft_model)
    command = P3.build_server_command(server_args, port, batch_size, n_max)
    command_record = {
        "command": command, "split": split, "seeds": seeds, "n_max": n_max,
        "target_model_sha256": sha256_file(args.target_model),
        "draft_model_sha256": sha256_file(args.draft_model),
        "warmup_prompt_id": "C01", "protocol_sha256": sha256_file(PROTOCOL),
    }
    if raw_path.exists():
        if not verify_output_rows:
            raise RuntimeError(f"locked built-in evaluation artifact already exists; refusing to overwrite: {raw_path}")
        if not command_path.exists():
            raise RuntimeError(f"cannot verify existing built-in command for {raw_path}")
        old_command = json.loads(command_path.read_text(encoding="utf-8"))
        for field in ("split", "seeds", "n_max", "target_model_sha256", "draft_model_sha256",
                      "warmup_prompt_id", "protocol_sha256"):
            if old_command.get(field) != command_record[field]:
                raise RuntimeError(f"existing built-in run differs at {field}: {raw_path}")
        expected = len(seeds) * len(selected) // batch_size
        raw_rows = read_jsonl(raw_path)
        if len(raw_rows) != expected or any(row.get("split") != split for row in raw_rows):
            raise RuntimeError(f"existing built-in raw data is incomplete or has split leakage: {raw_path}")
        if not outputs_path.exists() or outputs_path.stat().st_size == 0 or not warmup_record_path.exists():
            raise RuntimeError(f"existing built-in outputs/warm-up record are missing: {raw_path}")
        print(f"[{utc_now()}] llama.cpp {label}: reusing verified {len(raw_rows)} existing batches", flush=True)
        return raw_path
    if any(path.exists() for path in (outputs_path, log_path, command_path, warmup_record_path)):
        raise RuntimeError(f"partial built-in benchmark artifacts exist; refusing to overwrite {raw_path}")
    P3.atomic_json(command_path, command_record)
    base_url = f"http://127.0.0.1:{port}"
    service_log = log_path.open("wb")
    process = subprocess.Popen(command, stdout=service_log, stderr=subprocess.STDOUT)
    startup_ms: float | None = None
    started = time.perf_counter()
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup; see {log_path}")
            try:
                P3.http_get(base_url + "/health", timeout=2)
                startup_ms = (time.perf_counter() - started) * 1000.0
                break
            except Exception:
                time.sleep(0.25)
        if startup_ms is None:
            raise RuntimeError(f"llama-server health check timed out; see {log_path}")
        warm_responses, _ = P3.send_server_batch([prompts[0]], seeds[0], base_url)
        warmup_record_path = run_dir / "warmups" / f"{label}.jsonl"
        warmup_record_path.parent.mkdir(parents=True, exist_ok=True)
        P3.write_jsonl_line(warmup_record_path, {
            'request': warm_responses[0], 'warmup_prompt_id': 'C01',
            'excluded_from_measurement': True})
        try:
            P3.http_get(base_url + "/metrics", timeout=5)
            metrics_available = True
        except Exception:
            metrics_available = False
        with raw_path.open("w", encoding="utf-8") as raw, outputs_path.open("w", encoding="utf-8") as outputs:
            for trial, root_seed in enumerate(seeds):
                batch_group = 0
                for begin in range(0, len(selected), batch_size):
                    batch = selected[begin:begin + batch_size]
                    before = P3.parse_metrics(P3.http_get(base_url + "/metrics").decode(
                        "utf-8", errors="replace")) if metrics_available else {}
                    responses, response_ms = P3.send_server_batch(batch, root_seed, base_url)
                    total_tokens = sum(row["completion_tokens"] for row in responses)
                    output_start = time.perf_counter()
                    batch_id = f"{split}_b{batch_size}_s{root_seed}_g{batch_group}"
                    for response in responses:
                        outputs.write(json.dumps({"batch_id": batch_id, "split": split,
                                                  "engine": "llama.cpp", "mode": "draft-simple",
                                                  **response}, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n")
                    outputs.flush()
                    os.fsync(outputs.fileno())
                    output_ms = (time.perf_counter() - output_start) * 1000.0
                    batch_ms = response_ms + output_ms
                    after = P3.parse_metrics(P3.http_get(base_url + "/metrics").decode(
                        "utf-8", errors="replace")) if metrics_available else {}
                    delta = {key: after[key] - before[key] for key in P3.METRIC_SUFFIXES
                             if key in before and key in after}
                    record = {
                        "schema_version": 1, "batch_id": batch_id, "engine": "llama.cpp",
                        "mode": "draft-simple", "split": split, "trial": trial,
                        "seed_root": root_seed, "batch_index": batch_group,
                        "batch_size_configured": batch_size,
                        "batch_prompt_ids": [row["prompt_id"] for row in responses],
                        "request_seed_inputs": [row["request_seed_input"] for row in responses],
                        "prompt_tokens": [row["prompt_tokens"] for row in responses],
                        "completion_tokens": [row["completion_tokens"] for row in responses],
                        "finish_reasons": [row["finish_reason"] for row in responses],
                        "output_horizon": OUTPUT_HORIZON, "sampling": SAMPLING,
                        "request_latency_ms": [row["latency_ms"] for row in responses],
                        "generation_response_completion_ms": response_ms,
                        "output_write_ms": output_ms, "batch_completion_ms": batch_ms,
                        "aggregate_tokens_per_second": total_tokens * 1000.0 / batch_ms if batch_ms else 0,
                        "total_completion_tokens": total_tokens,
                        "server_timings": [row["server_timings"] for row in responses],
                        "server_metrics_before": before if metrics_available else None,
                        "server_metrics_after": after if metrics_available else None,
                        "server_metrics_delta": delta if metrics_available else None,
                        "peak_process_rss_bytes": P3.server_rss_hwm(process.pid),
                        "peak_live_kv_tokens": None, "peak_kv_bytes_estimate": None,
                        "unavailable_counter_reason": "Pinned llama.cpp metrics do not expose draft-context KV or cache copy/rollback counters.",
                    }
                    raw.write(json.dumps(record, separators=(",", ":")) + "\n")
                    raw.flush()
                    os.fsync(raw.fileno())
                    batch_group += 1
        P3.atomic_json(command_path.with_name(label + "_process.json"), {
            "startup_ms": startup_ms, "metrics_available": metrics_available,
            "return_code_before_terminate": process.poll(), "raw_path": str(raw_path),
            "outputs_path": str(outputs_path),
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
    if verify_output_rows:
        raw_rows = read_jsonl(raw_path)
        expected = len(seeds) * len(selected) // batch_size
        if len(raw_rows) != expected:
            raise RuntimeError(f"{label} retained {len(raw_rows)} batches; expected {expected}")
    return raw_path


def host_metadata() -> dict[str, Any]:
    cpu = Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
    model = next((line.split(":", 1)[1].strip() for line in cpu if line.startswith("model name")), None)
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_model": model,
            "logical_cpus": os.cpu_count(), "python": sys.version}


def create_run_manifest(run_dir: Path, args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    runner_snapshot = run_dir / "tuning" / "run_phase4_used_for_tuning.py"
    runner_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not runner_snapshot.exists():
        shutil.copy2(HERE / "run_phase4.py", runner_snapshot)
    test_commands = [["ctest", "--test-dir", str(args.bench_bin.parent), "--output-on-failure"],
                     [sys.executable, str(TEST_SCRIPT)]]
    test_results = []
    for command in test_commands:
        tests = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        test_results.append({"command": command, "returncode": tests.returncode,
                             "stdout": tests.stdout, "stderr": tests.stderr})
        if tests.returncode != 0:
            break
    test_record = {"commands": test_results,
                   "status": "PASS" if len(test_results) == len(test_commands) and
                   all(row["returncode"] == 0 for row in test_results) else "FAIL",
                   "tested_at_utc": utc_now()}
    atomic_json(run_dir / "tests_status.json", test_record)
    if tests.returncode != 0:
        raise RuntimeError("Phase 4 correctness tests failed before measurement")
    manifest = {
        "phase": 4, "run_id": run_dir.name, "created_at_utc": utc_now(),
        "protocol_path": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "prompt_manifest_path": str(MANIFEST), "prompt_manifest_sha256": sha256_file(MANIFEST),
        "target_model": str(args.target_model), "draft_model": str(args.draft_model),
        "target_model_sha256": sha256_file(args.target_model),
        "draft_model_sha256": sha256_file(args.draft_model),
        "host": host_metadata(), "prompts_by_split": {
            split: [row["id"] for row in rows_for(prompts, split)]
            for split in ("calibration", "tuning", "evaluation")},
        "seeds": {"calibration": CALIBRATION_SEEDS, "tuning": TUNING_SEEDS,
                  "evaluation": EVALUATION_SEEDS},
        "evaluation_opened": False,
        "correctness_tests": test_record["status"],
    }
    atomic_json(run_dir / "run_manifest.json", manifest)


def run_calibration(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]]) -> None:
    for batch_size in BATCH_SIZES:
        for branching, depth in TREE_ACTIONS:
            native_run(args, run_dir, prompts, "calibration", batch_size,
                       CALIBRATION_SEEDS, "pulsar-tree", (branching, depth))
    calibration_dir = run_dir / "calibration"
    command = [sys.executable, str(FIT_SCRIPT), "--observations-dir",
               str(calibration_dir / "observations"), "--output-tsv",
               str(calibration_dir / "calibrator.tsv"), "--metadata",
               str(calibration_dir / "calibrator_metadata.json")]
    subprocess.run(command, check=True, timeout=300)


def run_tuning(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]]) -> None:
    calibrator_path = run_dir / "calibration" / "calibrator.tsv"
    if not calibrator_path.exists():
        raise RuntimeError("calibration must finish before tuning")
    scores: dict[str, Any] = {"by_batch_size": {}, "candidate_scores": []}
    raw_paths: dict[str, str] = {}
    costs: dict[tuple[int, int, int], float] = {}
    for batch_size in BATCH_SIZES:
        ar_path = native_run(args, run_dir, prompts, "tuning", batch_size,
                             TUNING_SEEDS, "autoregressive")
        ar_rows = read_jsonl(ar_path)
        scores["candidate_scores"].append({"batch_size": batch_size,
            "method": "autoregressive", "configuration": {},
            "median_tokens_per_second": __import__("statistics").median(
                float(row["aggregate_tokens_per_second"]) for row in ar_rows),
            "raw_path": str(ar_path)})
        raw_paths[f"ar_b{batch_size}"] = str(ar_path)
        builtin_rows = []
        for n_max in BUILTIN_ACTIONS:
            path = run_builtin(args, run_dir, prompts, "tuning", batch_size,
                               TUNING_SEEDS, n_max)
            rows = read_jsonl(path)
            score = __import__("statistics").median(
                float(row["aggregate_tokens_per_second"]) for row in rows)
            builtin_rows.append({"n_max": n_max, "path": path, "score": score})
            scores["candidate_scores"].append({"batch_size": batch_size,
                "method": "llama.cpp-draft-simple", "configuration": {"n_max": n_max},
                "median_tokens_per_second": score, "raw_path": str(path)})
            raw_paths[f"builtin_n{n_max}_b{batch_size}"] = str(path)
        builtin_best = max(builtin_rows, key=lambda item: (item["score"], -item["n_max"]))
        tree_rows = []
        for branching, depth in TREE_ACTIONS:
            path = native_run(args, run_dir, prompts, "tuning", batch_size,
                              TUNING_SEEDS, "pulsar-tree", (branching, depth))
            rows = read_jsonl(path)
            score = __import__("statistics").median(
                float(row["aggregate_tokens_per_second"]) for row in rows)
            pulse_cost_samples = [float(row["generation_wall_ms"]) /
                                  int(row["candidate_tree_pulses"])
                                  for row in rows if int(row["candidate_tree_pulses"]) > 0]
            if not pulse_cost_samples:
                raise RuntimeError(f"no verifier pulses found for {branching=}, {depth=}")
            action_cost = __import__("statistics").median(pulse_cost_samples)
            costs[(batch_size, branching, depth)] = action_cost
            tree_rows.append({"branching": branching, "depth": depth,
                              "node_bound": sum(branching ** level for level in range(depth)),
                              "path": str(path), "score": score, "median_generation_ms_per_pulse": action_cost})
            scores["candidate_scores"].append({"batch_size": batch_size,
                "method": "pulsar-fixed-tree", "configuration": {"branching": branching, "depth": depth},
                "median_tokens_per_second": score, "raw_path": str(path),
                "median_generation_ms_per_pulse": action_cost})
            raw_paths[f"tree_B{branching}_D{depth}_b{batch_size}"] = str(path)
        fixed_best = max(tree_rows, key=lambda item: (item["score"], -item["node_bound"]))
        scores["by_batch_size"][str(batch_size)] = {
            "autoregressive_raw": str(ar_path),
            "llama_cpp": {"n_max": builtin_best["n_max"], "raw": str(builtin_best["path"]),
                          "median_tokens_per_second": builtin_best["score"]},
            "pulsar_fixed_tree": {"branching": fixed_best["branching"],
                "depth": fixed_best["depth"], "node_bound": fixed_best["node_bound"],
                "raw": str(fixed_best["path"]), "median_tokens_per_second": fixed_best["score"]},
            "fixed_tree_candidates": tree_rows,
        }
    cost_path = run_dir / "tuning" / "action_costs.tsv"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# Phase 4 action costs: batch_size, branching, depth, median generation ms/pulse" + chr(10))
        for key in sorted(costs):
            stream.write(chr(9).join([str(key[0]), str(key[1]), str(key[2]),
                                      format(costs[key], ".17g")]) + chr(10))
    scores["action_costs_path"] = str(cost_path)
    runner_snapshot = run_dir / "tuning" / "run_phase4_used_for_tuning.py"
    if not runner_snapshot.exists():
        raise RuntimeError("the runner snapshot captured before measurements is missing")
    scores["tuning_runner_snapshot"] = str(runner_snapshot)
    scores["tuning_runner_sha256"] = sha256_file(runner_snapshot)
    scores["selection_rule"] = "maximum median batch tokens/s over tuning batches; ties prefer fewer actual tree nodes, then smaller n_max"
    atomic_json(run_dir / "tuning" / "tuning_selection.json", scores)
    raw_hashes = {key: sha256_file(Path(path)) for key, path in sorted(raw_paths.items())}
    freeze_path = run_dir / "freeze.json"
    if freeze_path.exists():
        raise RuntimeError("freeze.json already exists; do not overwrite a locked Phase 4 freeze")
    source_files = [
        PROTOCOL, MANIFEST, FIT_SCRIPT, ANALYSIS_SCRIPT, TEST_SCRIPT, HERE / "run_phase4.py",
        HERE / "run_phase3.py", runner_snapshot,
        ROOT / "cpp" / "production" / "CMakeLists.txt",
        ROOT / "cpp" / "production" / "src" / "phase4_bench.cpp",
        ROOT / "cpp" / "production" / "include" / "pulsar" / "adaptive_scheduler.hpp",
        ROOT / "cpp" / "production" / "include" / "pulsar" / "tree_generation.hpp",
        ROOT / "cpp" / "production" / "tests" / "runtime_tests.cpp",
        ROOT / "cpp" / "pulsar_core_v1.hpp", ROOT / "cpp" / "pulsar_tree_v3.hpp",
    ]
    source_hashes = {str(path): sha256_file(path) for path in source_files}
    binary = args.bench_bin
    freeze = {
        "status": "frozen_before_locked_evaluation",
        "frozen_at_utc": utc_now(), "protocol_revision": "1.1",
        "protocol_sha256": sha256_file(PROTOCOL), "prompt_manifest_sha256": sha256_file(MANIFEST),
        "analysis_script_sha256": sha256_file(ANALYSIS_SCRIPT),
        "fitter_script_sha256": sha256_file(FIT_SCRIPT),
        "source_hashes": source_hashes,
        "tuning_collection_runner_path": str(runner_snapshot),
        "tuning_collection_runner_sha256": sha256_file(runner_snapshot),
        "model_hashes": {"target": sha256_file(args.target_model), "draft": sha256_file(args.draft_model)},
        "binary_path": str(binary), "binary_sha256": sha256_file(binary),
        "build_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, check=True).stdout.strip(),
        "upstream_llama_cpp_revision": "cd0a5b6a180dff0048e5fd993098b21c0b2c91b8",
        "host": host_metadata(), "threads": THREADS, "batch_sizes": BATCH_SIZES,
        "sampling": SAMPLING, "output_horizon_tokens": OUTPUT_HORIZON,
        "context_size_tokens": CONTEXT_SIZE, "batch_token_capacity": BATCH_TOKENS,
        "warmup_prompt_id": "C01",
        "bootstrap": {"draws": 10000, "seed": 5555,
                       "clusters": {"batch_1": "individual evaluation context group",
                                    "batch_2": "adjacent E-group pairs"}},
        "evaluation_order_rng_seeds": {"1": 10001, "2": 10002},
        "seeds": {"calibration": CALIBRATION_SEEDS, "tuning": TUNING_SEEDS,
                  "evaluation": EVALUATION_SEEDS},
        "tuning_raw_sha256": raw_hashes,
        "calibrator_tsv_sha256": sha256_file(run_dir / "calibration" / "calibrator.tsv"),
        "calibrator_metadata_sha256": sha256_file(run_dir / "calibration" / "calibrator_metadata.json"),
        "action_costs_sha256": sha256_file(cost_path),
        "tests_status_sha256": sha256_file(run_dir / "tests_status.json"),
        "calibrator_metadata": json.loads((run_dir / "calibration" / "calibrator_metadata.json").read_text()),
        "action_costs_ms_per_pulse": {f"b{b}_B{branch}_D{depth}": value
                                       for (b, branch, depth), value in sorted(costs.items())},
        "selected_baselines": {key: {"llama_cpp_n_max": value["llama_cpp"]["n_max"],
                                      "fixed_tree": value["pulsar_fixed_tree"]}
                               for key, value in scores["by_batch_size"].items()},
        "adaptive_policy": {"actions": TREE_ACTIONS, "ema_alpha": 0.20,
                            "calibrated_weight": 0.50, "feature_set": ["root_q_max", "root_q_margin", "root_q_entropy"],
                            "utility": "expected emitted tokens divided by tuning median generation ms per verifier pulse",
                            "tie_break": "fewer actual nodes, shallower depth, lower branching"},
        "selection_rule": scores["selection_rule"],
    }
    atomic_json(freeze_path, freeze)
    freeze_path.chmod(0o444)
    print(f"[{utc_now()}] immutable tuning freeze written: {freeze_path}", flush=True)


def verify_freeze(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    path = run_dir / "freeze.json"
    if not path.exists():
        raise RuntimeError("locked evaluation is blocked until freeze.json exists")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    expected = {"protocol_sha256": sha256_file(PROTOCOL),
                "prompt_manifest_sha256": sha256_file(MANIFEST),
                "analysis_script_sha256": sha256_file(ANALYSIS_SCRIPT),
                "fitter_script_sha256": sha256_file(FIT_SCRIPT),
                "binary_sha256": sha256_file(args.bench_bin),
                "calibrator_tsv_sha256": sha256_file(run_dir / "calibration" / "calibrator.tsv"),
                "action_costs_sha256": sha256_file(run_dir / "tuning" / "action_costs.tsv"),
                "tests_status_sha256": sha256_file(run_dir / "tests_status.json")}
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise RuntimeError(f"locked evaluation freeze mismatch for {key}; evaluation was not started")
    for source_path, source_hash in freeze.get("source_hashes", {}).items():
        if sha256_file(Path(source_path)) != source_hash:
            raise RuntimeError(f"locked evaluation source changed after freeze: {source_path}")
    actual_models = {"target": sha256_file(args.target_model), "draft": sha256_file(args.draft_model)}
    if freeze.get("model_hashes") != actual_models:
        raise RuntimeError("locked evaluation model hashes differ from the pre-evaluation freeze")
    return freeze


def run_evaluation(args: argparse.Namespace, run_dir: Path, prompts: list[dict[str, Any]]) -> None:
    freeze = verify_freeze(args, run_dir)
    selected = freeze["selected_baselines"]
    for batch_size in BATCH_SIZES:
        batch_key = str(batch_size)
        tasks: list[tuple[str, Any]] = [
            ("autoregressive", lambda bs=batch_size: native_run(
                args, run_dir, prompts, "evaluation", bs, EVALUATION_SEEDS,
                "autoregressive", verify_output_rows=False)),
            ("builtin", lambda bs=batch_size, n=selected[batch_key]["llama_cpp_n_max"]: run_builtin(
                args, run_dir, prompts, "evaluation", bs, EVALUATION_SEEDS, n,
                verify_output_rows=False)),
        ]
        for branching, depth in TREE_ACTIONS:
            tasks.append((f"fixed_B{branching}_D{depth}",
                          lambda bs=batch_size, b=branching, d=depth: native_run(
                              args, run_dir, prompts, "evaluation", bs, EVALUATION_SEEDS,
                              "pulsar-tree", (b, d), verify_output_rows=False)))
        tasks.append(("adaptive", lambda bs=batch_size: native_run(
            args, run_dir, prompts, "evaluation", bs, EVALUATION_SEEDS,
            "pulsar-adaptive", None, verify_output_rows=False)))
        random.Random(10000 + batch_size).shuffle(tasks)
        order = []
        for label, task in tasks:
            print(f"[{utc_now()}] locked evaluation batch_size={batch_size}: {label}", flush=True)
            task()
            order.append(label)
        atomic_json(run_dir / "evaluation" / f"execution_order_b{batch_size}.json",
                    {"batch_size": batch_size, "order": order,
                     "freeze_sha256": sha256_file(run_dir / "freeze.json")})
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_completed_at_utc"] = utc_now()
    manifest["evaluation_opened"] = False
    manifest["evaluation_unlock_note"] = "All preregistered evaluation methods completed before analysis was allowed."
    atomic_json(manifest_path, manifest)
    atomic_json(run_dir / "evaluation" / "evaluation_complete.json", {
        "completed_at_utc": utc_now(), "methods_complete": True,
        "expected_methods_per_batch_size": ["autoregressive", "llama.cpp selected chain",
            "all three fixed trees", "adaptive"], "freeze_sha256": sha256_file(run_dir / "freeze.json")})


def analyze(args: argparse.Namespace, run_dir: Path) -> None:
    complete = run_dir / "evaluation" / "evaluation_complete.json"
    if not complete.exists():
        raise RuntimeError("analysis is blocked until every locked evaluation method has completed")
    subprocess.run([sys.executable, str(ANALYSIS_SCRIPT), "--run-dir", str(run_dir)],
                   check=True, timeout=900)


def new_run_dir(base: Path) -> Path:
    return base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-bin", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "paper" / "results" / "phase4")
    parser.add_argument("--phase", choices=("all", "calibration", "tuning", "evaluation", "analyze"),
                        default="all")
    args = parser.parse_args()
    for path in (args.bench_bin, args.llama_server, args.target_model, args.draft_model):
        if not path.is_file():
            raise RuntimeError(f"required executable/model does not exist: {path}")
    actual = {"target": sha256_file(args.target_model), "draft": sha256_file(args.draft_model)}
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"model hash mismatch; expected frozen TinyStories fixture, got {actual}")
    prompts = load_manifest()
    if args.run_dir is None:
        if args.phase != "all":
            raise RuntimeError("--run-dir is required when phases are run separately")
        run_dir = new_run_dir(args.results_root)
    else:
        run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        create_run_manifest(run_dir, args, prompts)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("prompt_manifest_sha256") != sha256_file(MANIFEST) or \
                manifest.get("target_model_sha256") != actual["target"] or \
                manifest.get("draft_model_sha256") != actual["draft"]:
            raise RuntimeError("existing run manifest does not match the frozen Phase 4 inputs")

    if args.phase in ("all", "calibration"):
        run_calibration(args, run_dir, prompts)
        if args.phase == "calibration":
            return
    if args.phase in ("all", "tuning"):
        run_tuning(args, run_dir, prompts)
        if args.phase == "tuning":
            return
    if args.phase in ("all", "evaluation"):
        run_evaluation(args, run_dir, prompts)
        if args.phase == "evaluation":
            return
    if args.phase in ("all", "analyze"):
        analyze(args, run_dir)
    print(f"[{utc_now()}] Phase 4 {args.phase} complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
