#!/usr/bin/env python3
"""Measure RTX 4090 shared_lat behavior and compare it with a simulator report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / (
    "simulator-remodeled/util/tuner/GPU_Microbenchmark/ubench/shd/"
    "shared_lat/shared_lat"
)
SIMULATOR_CASE_ID = "sm89_shared_lat_rtx4090_observation"
EXPECTED_APPLICATION_EXIT_CODE = 1
MIN_BASELINE_SAMPLES = 5

NCU_METRICS = (
    "gpc__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg",
    "sm__inst_executed.sum",
    "smsp__inst_executed.sum",
    "gpu__time_duration.sum",
)

LATENCY_PATTERN = re.compile(
    r"^Shared Memory Latency\s*=\s*([0-9]+(?:\.[0-9]+)?) cycles$",
    re.MULTILINE,
)
TOTAL_CLOCK_PATTERN = re.compile(
    r"^Total Clk number\s*=\s*([0-9]+)\s*$", re.MULTILINE
)


class HardwareProbeError(RuntimeError):
    """Raised when the hardware probe cannot produce a valid observation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_direct_output(output: str) -> Dict[str, Any]:
    latency = LATENCY_PATTERN.findall(output)
    total_clock = TOTAL_CLOCK_PATTERN.findall(output)
    if not latency or not total_clock:
        raise HardwareProbeError("shared_lat output is missing latency fields")
    return {
        "shared_memory_latency_cycles": float(latency[-1]),
        "timed_region_total_cycles": int(total_clock[-1]),
    }


def _parse_metric_number(value: str) -> Any:
    normalized = value.replace(",", "").strip()
    if not normalized:
        raise HardwareProbeError("Nsight metric value is empty")
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError as error:
            raise HardwareProbeError(f"Invalid Nsight metric value: {value}") from error


def parse_ncu_raw_csv(csv_output: str) -> Dict[str, Any]:
    lines = csv_output.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith('"ID",')
        )
    except StopIteration as error:
        raise HardwareProbeError("Nsight output has no raw CSV header") from error
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    if not reader.fieldnames:
        raise HardwareProbeError("Nsight output has no CSV header")
    missing_columns = sorted(set(NCU_METRICS) - set(reader.fieldnames))
    if missing_columns:
        raise HardwareProbeError(
            f"Nsight output is missing metrics: {', '.join(missing_columns)}"
        )

    for row in reader:
        if not row.get("Kernel Name"):
            continue
        if "shared_lat" not in row["Kernel Name"]:
            continue
        return {metric: _parse_metric_number(row[metric]) for metric in NCU_METRICS}
    raise HardwareProbeError("Nsight output has no shared_lat kernel row")


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise HardwareProbeError("Cannot summarize an empty sample set")
    fields = sorted(samples[0])
    summary: Dict[str, Any] = {"count": len(samples), "fields": {}}
    for field in fields:
        values = [sample[field] for sample in samples]
        summary["fields"][field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "samples": values,
        }
    return summary


def relative_error_percent(simulated: float, hardware: float) -> float:
    if hardware == 0:
        raise HardwareProbeError("Cannot compute relative error against zero")
    return round((simulated - hardware) / hardware * 100.0, 6)


def load_simulator_result(report_path: Path) -> Dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardwareProbeError(f"Cannot load simulator report: {error}") from error
    for result in report.get("results", []):
        if result.get("case_id") == SIMULATOR_CASE_ID:
            if result.get("failures"):
                raise HardwareProbeError("Simulator report contains run failures")
            return result
    raise HardwareProbeError(
        f"Simulator report does not contain case {SIMULATOR_CASE_ID}"
    )


def compare_with_simulator(
    simulator_result: Mapping[str, Any],
    ncu_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    simulator_stats = simulator_result["stats"]
    hardware_fields = ncu_summary["fields"]
    mappings = (
        (
            "whole_kernel_cycles",
            "gpu_tot_sim_cycle",
            "sm__cycles_elapsed.avg",
        ),
        (
            "executed_instructions",
            "gpu_tot_sim_insn",
            "sm__inst_executed.sum",
        ),
    )
    comparisons: Dict[str, Any] = {}
    for label, simulator_stat, hardware_metric in mappings:
        simulated = simulator_stats[simulator_stat]
        hardware = hardware_fields[hardware_metric]["median"]
        comparisons[label] = {
            "simulator_stat": simulator_stat,
            "simulator_value": simulated,
            "hardware_metric": hardware_metric,
            "hardware_median": hardware,
            "signed_relative_error_percent": relative_error_percent(
                simulated, hardware
            ),
            "absolute_relative_error_percent": abs(
                relative_error_percent(simulated, hardware)
            ),
        }
    return comparisons


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise HardwareProbeError(
            f"Command timed out after {timeout_seconds}s: {command[0]}"
        ) from error


def query_gpu(device_index: int) -> Dict[str, Any]:
    fields = (
        "index",
        "name",
        "uuid",
        "compute_cap",
        "driver_version",
    )
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise HardwareProbeError(completed.stderr.strip() or "nvidia-smi failed")
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise HardwareProbeError("Unexpected nvidia-smi GPU identity output")
    identity = {field: value.strip() for field, value in zip(fields, rows[0])}
    if identity["name"] != "NVIDIA GeForce RTX 4090":
        raise HardwareProbeError(
            f"Device {device_index} is not an RTX 4090: {identity['name']}"
        )
    if identity["compute_cap"] != "8.9":
        raise HardwareProbeError(
            f"Device {device_index} has unexpected compute capability: "
            f"{identity['compute_cap']}"
        )
    identity["index"] = int(identity["index"])
    return identity


def query_gpu_state(device_index: int) -> Dict[str, Any]:
    fields = ("clocks.current.sm", "temperature.gpu", "power.draw")
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise HardwareProbeError(completed.stderr.strip() or "nvidia-smi failed")
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise HardwareProbeError("Unexpected nvidia-smi GPU state output")
    values = [value.strip() for value in rows[0]]
    return {
        "sm_clock_mhz": int(values[0]),
        "temperature_c": int(values[1]),
        "power_draw_w": float(values[2]),
    }


def query_compute_processes(gpu_uuid: str) -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise HardwareProbeError(completed.stderr.strip() or "nvidia-smi failed")
    processes = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) != 3 or row[0].strip() != gpu_uuid:
            continue
        processes.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
            }
        )
    return processes


def ncu_version() -> str:
    completed = subprocess.run(
        ["ncu", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise HardwareProbeError("ncu --version failed")
    for line in completed.stdout.splitlines():
        if line.startswith("Version "):
            return line[len("Version ") :].strip()
    return completed.stdout.strip()


def cuda_environment(gpu_uuid: str) -> Dict[str, str]:
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu_uuid):
        raise HardwareProbeError(f"Invalid GPU UUID: {gpu_uuid}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return environment


def run_direct_samples(
    binary: Path,
    environment: Mapping[str, str],
    warmups: int,
    samples: int,
) -> List[Dict[str, Any]]:
    results = []
    for sample_index in range(warmups + samples):
        completed = _run(
            [str(binary)], environment=environment, timeout_seconds=30
        )
        if completed.returncode != EXPECTED_APPLICATION_EXIT_CODE:
            raise HardwareProbeError(
                f"shared_lat returned {completed.returncode}, expected "
                f"{EXPECTED_APPLICATION_EXIT_CODE}: {completed.stderr.strip()}"
            )
        parsed = parse_direct_output(completed.stdout + completed.stderr)
        if sample_index >= warmups:
            results.append(parsed)
    return results


def run_ncu_samples(
    binary: Path,
    environment: Mapping[str, str],
    samples: int,
) -> List[Dict[str, Any]]:
    command = [
        "ncu",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--metrics",
        ",".join(NCU_METRICS),
        "--csv",
        "--page",
        "raw",
        str(binary),
    ]
    results = []
    for _ in range(samples):
        completed = _run(command, environment=environment, timeout_seconds=180)
        if completed.returncode != EXPECTED_APPLICATION_EXIT_CODE:
            raise HardwareProbeError(
                f"ncu returned {completed.returncode}, expected "
                f"{EXPECTED_APPLICATION_EXIT_CODE}: {completed.stderr.strip()}"
            )
        results.append(parse_ncu_raw_csv(completed.stdout))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--ncu-samples", type=int, default=5)
    parser.add_argument("--simulator-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-busy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        binary = args.binary.resolve()
        if not binary.is_file():
            raise HardwareProbeError(f"Binary does not exist: {binary}")
        if args.warmups < 0 or args.samples <= 0 or args.ncu_samples <= 0:
            raise HardwareProbeError("Sample counts must be positive")

        gpu = query_gpu(args.device)
        active_processes = query_compute_processes(gpu["uuid"])
        if active_processes and not args.allow_busy:
            raise HardwareProbeError(
                f"GPU {args.device} has active compute processes: {active_processes}"
            )

        environment = cuda_environment(gpu["uuid"])
        simulator_result = load_simulator_result(args.simulator_report.resolve())
        state_before = query_gpu_state(args.device)

        started = time.monotonic()
        direct_samples = run_direct_samples(
            binary, environment, args.warmups, args.samples
        )
        ncu_samples = run_ncu_samples(binary, environment, args.ncu_samples)
        duration_seconds = round(time.monotonic() - started, 3)
        state_after = query_gpu_state(args.device)

        direct_summary = summarize_samples(direct_samples)
        ncu_summary = summarize_samples(ncu_samples)
        reasons = []
        if args.samples < MIN_BASELINE_SAMPLES:
            reasons.append("insufficient_direct_samples")
        if args.ncu_samples < MIN_BASELINE_SAMPLES:
            reasons.append("insufficient_ncu_samples")
        if simulator_result.get("status") != "passed":
            reasons.append("simulator_result_is_not_approved_golden")
        reasons.append("hardware_error_threshold_not_approved")

        report = {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "observed_not_accuracy_baseline",
            "provisional_reasons": reasons,
            "gpu": gpu,
            "gpu_state": {"before": state_before, "after": state_after},
            "toolchain": {
                "ncu_version": ncu_version(),
                "cuda_visible_devices": gpu["uuid"],
                "binary": {
                    "path": str(binary.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(binary),
                    "expected_exit_code": EXPECTED_APPLICATION_EXIT_CODE,
                },
            },
            "sampling": {
                "warmups": args.warmups,
                "direct_samples": direct_summary,
                "ncu_samples": ncu_summary,
                "duration_seconds": duration_seconds,
            },
            "simulator": {
                "case_id": simulator_result["case_id"],
                "status": simulator_result["status"],
                "inputs": simulator_result["inputs"],
                "stats": simulator_result["stats"],
            },
            "comparison": compare_with_simulator(simulator_result, ncu_summary),
        }
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(args.output)
        else:
            sys.stdout.write(rendered)
        return 0
    except HardwareProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
