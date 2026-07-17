#!/usr/bin/env python3
"""Build UUID-bound SM89 workloads and package deterministic trace fixtures."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("fixtures")
TRACER = REPO_ROOT / "simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.so"

WORKLOADS: Sequence[Mapping[str, str]] = (
    {
        "id": "half_pipeline",
        "source": "tests/remodeled_trace/workloads/half_pipeline.cu",
        "expected_opcode": "HADD2",
    },
    {
        "id": "fp64_dispatch",
        "source": "tests/remodeled_trace/workloads/fp64_dispatch.cu",
        "expected_opcode": "DADD",
    },
)


class FixtureGenerationError(RuntimeError):
    """Raised when hardware, tools, or trace output violate the fixture contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise FixtureGenerationError(
            f"Command failed with exit code {completed.returncode}: {rendered}\n"
            f"{completed.stdout}"
        )
    return completed


def parse_gpu_rows(output: str) -> List[Dict[str, str]]:
    rows = []
    for raw_line in output.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 5:
            continue
        rows.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "compute_capability": fields[3],
                "driver_version": fields[4],
            }
        )
    return rows


def resolve_gpu(device_index: int) -> Dict[str, str]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader",
        ]
    )
    matches = [row for row in parse_gpu_rows(query.stdout) if row["index"] == str(device_index)]
    if len(matches) != 1:
        raise FixtureGenerationError(f"Cannot resolve GPU index {device_index}")
    gpu = matches[0]
    if gpu["compute_capability"] != "8.9" or gpu["name"] != "NVIDIA GeForce RTX 4090":
        raise FixtureGenerationError(
            f"GPU {device_index} is not the required RTX 4090 SM89 device: {gpu}"
        )
    return gpu


def reject_busy_gpu(gpu_uuid: str) -> None:
    query = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ]
    )
    active = [line for line in query.stdout.splitlines() if line.strip().startswith(gpu_uuid)]
    if active:
        raise FixtureGenerationError(
            f"GPU {gpu_uuid} already has compute processes: {'; '.join(active)}"
        )


def deterministic_archive(
    source_root: Path,
    member_paths: Iterable[Path],
    archive_root: str,
    destination: Path,
) -> None:
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(filename="", fileobj=raw_output, mode="wb", mtime=0) as gzip_output:
            with tarfile.open(
                fileobj=gzip_output, mode="w", format=tarfile.GNU_FORMAT
            ) as archive:
                for relative_path in sorted(member_paths, key=lambda path: path.as_posix()):
                    source = source_root / relative_path
                    info = tarfile.TarInfo(
                        f"{archive_root}/{relative_path.as_posix()}"
                    )
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as payload:
                        archive.addfile(info, payload)


def tool_version(command: Sequence[str], pattern: str) -> str:
    output = run_command(command).stdout
    match = re.search(pattern, output)
    if not match:
        raise FixtureGenerationError(f"Cannot parse tool version from: {output}")
    return match.group(1)


def generate_workload(
    workload: Mapping[str, str],
    gpu: Mapping[str, str],
    output_dir: Path,
    generated_at: str,
    temporary_root: Path,
    nvcc_version: str,
) -> Dict[str, Any]:
    workload_id = workload["id"]
    source = REPO_ROOT / workload["source"]
    binary = temporary_root / "bin" / workload_id
    binary.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "nvcc",
            "-arch=sm_89",
            "-O3",
            "-lineinfo",
            str(source),
            "-o",
            str(binary),
        ]
    )

    sass = run_command(["cuobjdump", "--dump-sass", str(binary)]).stdout
    expected_opcode = workload["expected_opcode"]
    if not re.search(rf"\b{re.escape(expected_opcode)}\b", sass):
        raise FixtureGenerationError(
            f"{workload_id} does not contain required opcode {expected_opcode}"
        )

    run_root = temporary_root / "runs" / workload_id
    run_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu["uuid"],
            "CUDA_INJECTION64_PATH": str(TRACER),
            "LD_PRELOAD": str(TRACER),
        }
    )
    environment.pop("TRACES_FOLDER", None)
    environment.pop("USER_DEFINED_FOLDERS", None)
    traced_run = run_command([str(binary)], cwd=run_root, environment=environment)

    traces_root = run_root / "traces"
    required_members = [
        Path("dynamic_trace.pb"),
        Path("extra_info/enhanced_execution_info.json"),
    ]
    threadblock_members = sorted(
        path.relative_to(traces_root)
        for path in (traces_root / "threadblocks").rglob("*.pb")
    )
    members = required_members + threadblock_members
    missing = [path.as_posix() for path in members if not (traces_root / path).is_file()]
    if missing or not threadblock_members:
        raise FixtureGenerationError(
            f"Incomplete trace for {workload_id}; missing={missing}, "
            f"threadblocks={len(threadblock_members)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{workload_id}_sm89.tar.gz"
    archive_path = output_dir / archive_name
    temporary_archive = temporary_root / archive_name
    archive_root = f"{workload_id}_sm89/traces"
    deterministic_archive(traces_root, members, archive_root, temporary_archive)
    shutil.copyfile(temporary_archive, archive_path)

    member_hashes = {
        path.as_posix(): sha256_file(traces_root / path) for path in members
    }
    metadata = {
        "schema_version": 1,
        "fixture_id": f"{workload_id}_sm89_rtx4090_cuda{nvcc_version.replace('.', '_')}",
        "generated_at": generated_at,
        "archive": archive_name,
        "archive_sha256": sha256_file(archive_path),
        "hardware": {
            "requested_nvidia_smi_index": int(gpu["index"]),
            "name": gpu["name"],
            "uuid": gpu["uuid"],
            "uuid_binding_status": "verified_uuid_bound",
            "compute_capability": gpu["compute_capability"],
            "driver_version": gpu["driver_version"],
        },
        "toolchain": {
            "cuda_version": nvcc_version,
            "source": {
                "path": workload["source"],
                "sha256": sha256_file(source),
            },
            "binary": {"sha256": sha256_file(binary)},
            "tracer": {
                "path": str(TRACER.relative_to(REPO_ROOT)),
                "sha256": sha256_file(TRACER),
            },
            "required_sass_opcode": expected_opcode,
        },
        "trace_members": member_hashes,
        "traced_program_output_sha256": hashlib.sha256(
            traced_run.stdout.encode()
        ).hexdigest(),
    }
    metadata_path = output_dir / f"{workload_id}_sm89.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at", default=dt.date.today().isoformat())
    parser.add_argument(
        "--workload",
        action="append",
        choices=[workload["id"] for workload in WORKLOADS],
        dest="workload_ids",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not TRACER.is_file():
        raise FixtureGenerationError(f"Tracer does not exist: {TRACER}")
    gpu = resolve_gpu(args.device)
    reject_busy_gpu(gpu["uuid"])
    nvcc_version = tool_version(["nvcc", "--version"], r"release ([0-9]+\.[0-9]+)")
    output_dir = args.output_dir.resolve()
    selected_workloads = [
        workload
        for workload in WORKLOADS
        if not args.workload_ids or workload["id"] in args.workload_ids
    ]
    with tempfile.TemporaryDirectory(prefix="remodeled-semantic-fixtures-") as raw_tmp:
        temporary_root = Path(raw_tmp)
        results = [
            generate_workload(
                workload,
                gpu,
                output_dir,
                args.generated_at,
                temporary_root,
                nvcc_version,
            )
            for workload in selected_workloads
        ]
    print(json.dumps({"fixtures": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
