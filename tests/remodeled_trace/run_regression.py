#!/usr/bin/env python3
"""Run deterministic remodeled trace observations and approved golden checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("cases.json")
DEFAULT_GOLDENS = Path(__file__).with_name("goldens.json")

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_RUN_FAILED = 3
EXIT_CHECK_FAILED = 4

COMPLETION_MARKER = "GPGPU-Sim: *** exit detected ***"
FAILURE_PATTERNS = {
    "assertion": re.compile(r"Assertion .* failed|assertion .* failed", re.IGNORECASE),
    "segmentation_fault": re.compile(r"Segmentation fault", re.IGNORECASE),
    "deadlock": re.compile(r"(?:deadlock detected|ERROR \*\* deadlock)", re.IGNORECASE),
}


def _integer(value: str) -> int:
    return int(value.strip())


def _floating(value: str) -> float:
    return float(value.strip())


STAT_PATTERNS: Mapping[str, Tuple[re.Pattern[str], Callable[[str], Any]]] = {
    "gpu_tot_sim_cycle": (
        re.compile(r"^gpu_tot_sim_cycle\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "gpu_tot_sim_insn": (
        re.compile(r"^gpu_tot_sim_insn\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "gpu_tot_ipc": (
        re.compile(r"^gpu_tot_ipc\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
        _floating,
    ),
    "l2_total_cache_accesses": (
        re.compile(r"^L2_total_cache_accesses\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "l2_total_cache_misses": (
        re.compile(r"^L2_total_cache_misses\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "total_dram_reads": (
        re.compile(r"^total dram reads\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "total_dram_writes": (
        re.compile(r"^total dram writes\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "gpgpu_n_shmem_bkconflict": (
        re.compile(r"^gpgpu_n_shmem_bkconflict\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_half_to_sp": (
        re.compile(r"^remodeled_dispatch_half_to_sp\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_half_to_int": (
        re.compile(r"^remodeled_dispatch_half_to_int\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_sp_to_sp": (
        re.compile(r"^remodeled_dispatch_sp_to_sp\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_sp_to_int": (
        re.compile(r"^remodeled_dispatch_sp_to_int\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_int_to_sp": (
        re.compile(r"^remodeled_dispatch_int_to_sp\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_int_to_int": (
        re.compile(r"^remodeled_dispatch_int_to_int\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_dp_to_dp": (
        re.compile(r"^remodeled_dispatch_dp_to_dp\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_dispatch_mem_to_mem": (
        re.compile(r"^remodeled_dispatch_mem_to_mem\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_shared_throttle_dp_events": (
        re.compile(r"^remodeled_shared_throttle_dp_events\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_shared_throttle_dp_cycles": (
        re.compile(r"^remodeled_shared_throttle_dp_cycles\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_shared_throttle_mem_events": (
        re.compile(r"^remodeled_shared_throttle_mem_events\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
    "remodeled_shared_throttle_mem_cycles": (
        re.compile(r"^remodeled_shared_throttle_mem_cycles\s*=\s*([0-9]+)\s*$", re.MULTILINE),
        _integer,
    ),
}


class HarnessError(RuntimeError):
    """Raised for invalid inputs or unsafe trace archives."""


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"Cannot load JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"Expected a JSON object in {path}")
    return value


def resolve_repo_path(raw_path: str, *, must_exist: bool = True) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise HarnessError(f"Path escapes repository root: {raw_path}") from error
    if must_exist and not path.exists():
        raise HarnessError(f"Required path does not exist: {raw_path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_stats(output: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name, (pattern, converter) in STAT_PATTERNS.items():
        matches = pattern.findall(output)
        if matches:
            stats[name] = converter(matches[-1])
    return stats


def evaluate_observation(
    output: str,
    exit_code: int,
    required_stats: Iterable[str],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    stats = parse_stats(output)
    missing = sorted(set(required_stats) - set(stats))
    failures: List[str] = []

    if exit_code != 0:
        failures.append(f"process_exit_code={exit_code}")
    if COMPLETION_MARKER not in output:
        failures.append("missing_completion_marker")
    failures.extend(
        name for name, pattern in FAILURE_PATTERNS.items() if pattern.search(output)
    )
    failures.extend(f"missing_stat:{name}" for name in missing)

    accesses = stats.get("l2_total_cache_accesses")
    misses = stats.get("l2_total_cache_misses")
    if accesses is not None and misses is not None and misses > accesses:
        failures.append("l2_misses_exceed_accesses")

    cycles = stats.get("gpu_tot_sim_cycle")
    instructions = stats.get("gpu_tot_sim_insn")
    ipc = stats.get("gpu_tot_ipc")
    if cycles is not None and instructions is not None and ipc is not None:
        if cycles == 0:
            failures.append("gpu_tot_sim_cycle_zero")
        else:
            derived_ipc = instructions / cycles
            if abs(derived_ipc - ipc) > 0.0001:
                failures.append("gpu_tot_ipc_inconsistent")

    return sorted(set(failures)), missing, stats


def compare_to_golden(
    observed: Mapping[str, Any],
    golden: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    expected_stats = golden.get("stats", {})
    tolerances = golden.get("absolute_tolerances", {})
    mismatches: List[Dict[str, Any]] = []

    for name, expected in expected_stats.items():
        if name not in observed:
            mismatches.append({"stat": name, "reason": "missing"})
            continue
        actual = observed[name]
        tolerance = tolerances.get(name, 0)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            matches = abs(actual - expected) <= tolerance
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(
                {
                    "stat": name,
                    "expected": expected,
                    "actual": actual,
                    "absolute_tolerance": tolerance,
                }
            )
    return mismatches


def validate_manifest(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise HarnessError("Unsupported manifest schema_version")
    for key in ("simulator_root", "binary", "environment_setup", "cases"):
        if key not in manifest:
            raise HarnessError(f"Manifest is missing {key}")

    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise HarnessError("Manifest cases must be a non-empty list")

    required_case_keys = {
        "id",
        "archive",
        "trace_root",
        "trace_file",
        "gpgpusim_config",
        "trace_config",
        "timeout_seconds",
        "omp_num_threads",
        "required_stats",
    }
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise HarnessError("Each manifest case must be an object")
        missing = sorted(required_case_keys - set(case))
        if missing:
            raise HarnessError(f"Case is missing fields: {', '.join(missing)}")
        if case["id"] in seen:
            raise HarnessError(f"Duplicate case id: {case['id']}")
        seen.add(case["id"])
        unknown_stats = sorted(set(case["required_stats"]) - set(STAT_PATTERNS))
        if unknown_stats:
            raise HarnessError(
                f"Case {case['id']} has unknown stats: {', '.join(unknown_stats)}"
            )
        trace_root = _safe_archive_path(str(case["trace_root"]))
        trace_file = _safe_archive_path(str(case["trace_file"]))
        if trace_root == PurePosixPath(".") or trace_file == PurePosixPath("."):
            raise HarnessError(f"Case {case['id']} has an empty trace path")

    resolve_repo_path(str(manifest["simulator_root"]))
    resolve_repo_path(str(manifest["binary"]))
    resolve_repo_path(str(manifest["environment_setup"]))
    for case in cases:
        resolve_repo_path(str(case["archive"]))
        resolve_repo_path(str(case["gpgpusim_config"]))
        resolve_repo_path(str(case["trace_config"]))
    return cases


def select_cases(
    cases: Sequence[Mapping[str, Any]], selected_ids: Sequence[str]
) -> List[Mapping[str, Any]]:
    if not selected_ids:
        return list(cases)
    by_id = {str(case["id"]): case for case in cases}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise HarnessError(f"Unknown case ids: {', '.join(unknown)}")
    return [by_id[case_id] for case_id in selected_ids]


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise HarnessError(f"Unsafe archive member: {name}")
    return path


def extract_trace_tree(archive_path: Path, trace_root: str, destination: Path) -> Path:
    root = _safe_archive_path(trace_root)
    matched = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = _safe_archive_path(member.name)
            if member_path != root and root not in member_path.parents:
                continue
            matched += 1
            target = destination.joinpath(*member_path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isreg():
                raise HarnessError(f"Unsupported archive member type: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise HarnessError(f"Cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if matched == 0:
        raise HarnessError(f"Trace root not found in archive: {trace_root}")
    return destination.joinpath(*root.parts)


def load_simulator_environment(setup_path: Path, simulator_root: Path) -> Tuple[Dict[str, str], List[str]]:
    command = [
        "bash",
        "-c",
        'setup_path="$1"; shift; export IS_SERT="${IS_SERT:-0}"; '
        'source "$setup_path" >/dev/null && env -0',
        "bash",
        str(setup_path),
    ]
    completed = subprocess.run(
        command,
        cwd=simulator_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise HarnessError(
            f"Environment setup failed with exit code {completed.returncode}: {message}"
        )
    environment = dict(os.environ)
    for entry in completed.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode()] = value.decode(errors="surrogateescape")
    warnings = [
        line
        for line in completed.stderr.decode("utf-8", errors="replace").splitlines()
        if "warning" in line.lower()
        or "error" in line.lower()
        or "operator expected" in line.lower()
    ]
    return environment, warnings


def git_identity() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--ignore-submodules", "HEAD", "--"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": dirty.returncode != 0,
    }


def input_identity(
    manifest: Mapping[str, Any], case: Mapping[str, Any]
) -> Dict[str, Dict[str, str]]:
    paths = {
        "binary": str(manifest["binary"]),
        "trace_archive": str(case["archive"]),
        "gpgpusim_config": str(case["gpgpusim_config"]),
        "trace_config": str(case["trace_config"]),
    }
    return {
        name: {"path": path, "sha256": sha256_file(resolve_repo_path(path))}
        for name, path in paths.items()
    }


def run_case(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, str],
    setup_warnings: Sequence[str],
    log_dir: Path | None,
) -> Dict[str, Any]:
    simulator_root = resolve_repo_path(str(manifest["simulator_root"]))
    binary = resolve_repo_path(str(manifest["binary"]))
    archive = resolve_repo_path(str(case["archive"]))
    gpgpusim_config = resolve_repo_path(str(case["gpgpusim_config"]))
    trace_config = resolve_repo_path(str(case["trace_config"]))

    case_environment = dict(environment)
    case_environment.update(
        {
            "LC_ALL": "C",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(case["omp_num_threads"]),
        }
    )

    with tempfile.TemporaryDirectory(prefix=f"remodeled-trace-{case['id']}-") as raw_tmp:
        temporary_root = Path(raw_tmp)
        extracted_root = extract_trace_tree(
            archive, str(case["trace_root"]), temporary_root
        )
        trace_file = _safe_archive_path(str(case["trace_file"]))
        trace_path = extracted_root.joinpath(*trace_file.parts)
        if not trace_path.is_file():
            raise HarnessError(f"Trace file was not extracted: {case['trace_file']}")

        command = [
            str(binary),
            "-trace",
            str(trace_path),
            "-config",
            str(gpgpusim_config),
            "-config",
            str(trace_config),
        ]
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=simulator_root,
                env=case_environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(case["timeout_seconds"]),
                check=False,
            )
            exit_code = completed.returncode
            output = completed.stdout
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_code = 124
            partial = error.stdout or ""
            output = partial.decode(errors="replace") if isinstance(partial, bytes) else partial
        duration_seconds = round(time.monotonic() - started, 3)

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{case['id']}.log").write_text(output, encoding="utf-8")

    failures, missing_stats, stats = evaluate_observation(
        output, exit_code, case["required_stats"]
    )
    if timed_out:
        failures.append("timeout")

    logical_trace = f"${{EXTRACTED_TRACE}}/{case['trace_root']}/{case['trace_file']}"
    inputs = input_identity(manifest, case)
    comparison_contract = {
        "execution_mode": "remodeled_trace",
        "trace": {
            "archive": inputs["trace_archive"],
            "root": case["trace_root"],
            "file": case["trace_file"],
        },
        "configs": {
            "gpgpusim": inputs["gpgpusim_config"],
            "trace": inputs["trace_config"],
        },
        "runtime": {
            "lc_all": "C",
            "omp_dynamic": "FALSE",
            "omp_num_threads": case["omp_num_threads"],
            "timeout_seconds": case["timeout_seconds"],
        },
    }
    return {
        "case_id": case["id"],
        "description": case.get("description", ""),
        "coverage": case.get("coverage", []),
        "status": "run_failed" if failures else "observed_not_golden",
        "command": [
            str(manifest["binary"]),
            "-trace",
            logical_trace,
            "-config",
            str(case["gpgpusim_config"]),
            "-config",
            str(case["trace_config"]),
        ],
        "runtime": {
            "timeout_seconds": case["timeout_seconds"],
            "omp_num_threads": case["omp_num_threads"],
            "duration_seconds": duration_seconds,
        },
        "process": {
            "exit_code": exit_code,
            "completion_marker_found": COMPLETION_MARKER in output,
            "timed_out": timed_out,
        },
        "inputs": inputs,
        "comparison_contract": comparison_contract,
        "stats": stats,
        "missing_stats": missing_stats,
        "failures": sorted(set(failures)),
        "environment_setup_warnings": list(setup_warnings),
        "raw_log_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def comparison_contract_mismatches(
    observed_contract: Mapping[str, Any], expected_contract: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    mismatches = []
    for name in sorted(set(observed_contract) | set(expected_contract)):
        expected = expected_contract.get(name)
        actual = observed_contract.get(name)
        if actual != expected:
            mismatches.append(
                {"contract": name, "expected": expected, "actual": actual}
            )
    return mismatches


def _validate_identity(
    identity: Any, label: str, errors: List[str]
) -> None:
    if not isinstance(identity, dict):
        errors.append(f"identity_not_object:{label}")
        return
    if not isinstance(identity.get("path"), str) or not identity["path"]:
        errors.append(f"invalid_identity_path:{label}")
    sha256 = identity.get("sha256")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        errors.append(f"invalid_identity_sha256:{label}")


def approved_golden_schema_errors(
    golden_case: Mapping[str, Any],
    required_stats: Iterable[str],
    observed_contract: Mapping[str, Any],
    golden_source_commit: Any,
) -> List[str]:
    errors = []
    if not isinstance(golden_source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", golden_source_commit
    ):
        errors.append("invalid_source_commit")

    expected_stats = golden_case.get("stats")
    if not isinstance(expected_stats, dict):
        errors.append("stats_not_object")
        expected_stats = {}
    for stat in sorted(set(required_stats)):
        if stat not in expected_stats:
            errors.append(f"missing_stat:{stat}")
        elif isinstance(expected_stats[stat], bool) or not isinstance(
            expected_stats[stat], (int, float)
        ):
            errors.append(f"invalid_stat_value:{stat}")

    tolerances = golden_case.get("absolute_tolerances", {})
    if not isinstance(tolerances, dict):
        errors.append("absolute_tolerances_not_object")
        tolerances = {}
    for stat, tolerance in tolerances.items():
        if stat not in expected_stats:
            errors.append(f"tolerance_without_stat:{stat}")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0
        ):
            errors.append(f"invalid_tolerance:{stat}")

    expected_contract = golden_case.get("comparison_contract")
    if not isinstance(expected_contract, dict):
        errors.append("comparison_contract_not_object")
        expected_contract = {}
    for name in sorted(observed_contract):
        if name not in expected_contract:
            errors.append(f"missing_contract:{name}")

    trace = expected_contract.get("trace")
    if not isinstance(trace, dict):
        errors.append("trace_contract_not_object")
    else:
        _validate_identity(trace.get("archive"), "trace.archive", errors)
        for name in ("root", "file"):
            if not isinstance(trace.get(name), str) or not trace[name]:
                errors.append(f"invalid_trace_contract:{name}")

    configs = expected_contract.get("configs")
    if not isinstance(configs, dict):
        errors.append("configs_contract_not_object")
    else:
        _validate_identity(configs.get("gpgpusim"), "configs.gpgpusim", errors)
        _validate_identity(configs.get("trace"), "configs.trace", errors)

    runtime = expected_contract.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime_contract_not_object")
    else:
        required_runtime = {
            "lc_all": str,
            "omp_dynamic": str,
            "omp_num_threads": int,
            "timeout_seconds": int,
        }
        for name, expected_type in required_runtime.items():
            value = runtime.get(name)
            if isinstance(value, bool) or not isinstance(value, expected_type):
                errors.append(f"invalid_runtime_contract:{name}")

    if expected_contract.get("execution_mode") != "remodeled_trace":
        errors.append("invalid_execution_mode")
    return sorted(set(errors))


def apply_golden(
    result: Dict[str, Any],
    golden_case: Mapping[str, Any] | None,
    golden_set_status: str | None,
    required_stats: Iterable[str],
    golden_source_commit: Any,
) -> None:
    result["golden_set_status"] = golden_set_status
    result["golden_schema_errors"] = []
    if result["failures"]:
        result["status"] = "run_failed"
        result["golden_mismatches"] = []
        return
    if (
        golden_set_status != "approved"
        or not golden_case
        or golden_case.get("approved") is not True
    ):
        result["status"] = "missing_approved_golden"
        result["golden_mismatches"] = []
        return
    schema_errors = approved_golden_schema_errors(
        golden_case,
        required_stats,
        result["comparison_contract"],
        golden_source_commit,
    )
    result["golden_schema_errors"] = schema_errors
    if schema_errors:
        result["status"] = "invalid_approved_golden"
        result["golden_mismatches"] = []
        return
    mismatches = compare_to_golden(result["stats"], golden_case)
    mismatches.extend(
        comparison_contract_mismatches(
            result["comparison_contract"],
            golden_case.get("comparison_contract", {}),
        )
    )
    result["golden_mismatches"] = mismatches
    if mismatches:
        result["status"] = "golden_mismatch"
    else:
        result["status"] = "passed"


def write_report(report: Mapping[str, Any], output_path: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is None:
        sys.stdout.write(rendered)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("list", "observe", "check"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--case", action="append", default=[], dest="case_ids")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_json(args.manifest.resolve())
        cases = select_cases(validate_manifest(manifest), args.case_ids)
        if args.mode == "list":
            report = {
                "schema_version": 1,
                "cases": [
                    {
                        "id": case["id"],
                        "description": case.get("description", ""),
                        "coverage": case.get("coverage", []),
                    }
                    for case in cases
                ],
            }
            write_report(report, args.output)
            return EXIT_OK

        goldens = load_json(args.goldens.resolve()) if args.mode == "check" else None
        simulator_root = resolve_repo_path(str(manifest["simulator_root"]))
        setup_path = resolve_repo_path(str(manifest["environment_setup"]))
        environment, setup_warnings = load_simulator_environment(
            setup_path, simulator_root
        )

        results = []
        for case in cases:
            result = run_case(
                manifest, case, environment, setup_warnings, args.log_dir
            )
            if args.mode == "check":
                apply_golden(
                    result,
                    goldens.get("cases", {}).get(case["id"]),
                    goldens.get("status"),
                    case["required_stats"],
                    goldens.get("source_commit"),
                )
            results.append(result)

        report = {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": args.mode,
            "source": git_identity(),
            "results": results,
            "summary": {
                "total": len(results),
                "passed": sum(result["status"] == "passed" for result in results),
                "observed_not_golden": sum(
                    result["status"] == "observed_not_golden" for result in results
                ),
                "failed": sum(
                    result["status"]
                    not in {"passed", "observed_not_golden"}
                    for result in results
                ),
            },
        }
        write_report(report, args.output)

        if args.mode == "observe":
            return (
                EXIT_RUN_FAILED
                if any(result["status"] == "run_failed" for result in results)
                else EXIT_OK
            )
        return (
            EXIT_OK
            if all(result["status"] == "passed" for result in results)
            else EXIT_CHECK_FAILED
        )
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
