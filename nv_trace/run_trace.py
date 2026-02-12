#!/usr/bin/env python3
"""
NV Trace - Standalone GPU Trace Extraction Tool

Supports two modes:
  1. Benchmark Suite Mode (-B): Trace benchmarks defined in YAML suite definitions
  2. Single App Mode (--app):   Trace a single CUDA application directly

Examples:
    # --- Benchmark Suite Mode ---
    # Trace all rodinia_2.0-ft benchmarks on device 0
    ./run_trace.py -B rodinia_2.0-ft -D 0

    # Trace multiple suites
    ./run_trace.py -B rodinia_2.0-ft,GPU_Microbenchmark -D 0

    # Trace with kernel limit (max 50 kernels per benchmark)
    ./run_trace.py -B rodinia_2.0-ft -D 0 -l 50

    # List all available benchmark suites
    ./run_trace.py --list-suites

    # --- Single App Mode ---
    # Trace a single CUDA application
    ./run_trace.py --app /path/to/cuda_app --args "arg1 arg2" -D 0

    # Trace with kernel limits
    ./run_trace.py --app /path/to/cuda_app --kernel-start 1 --kernel-end 10
"""

import argparse
import os
import subprocess
import sys
import datetime
import re

# ============================================================================
# Default paths - adjust these for your system
# ============================================================================
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

# Default root of gpu-app-collection (benchmarks + data)
DEFAULT_GPUAPPS_ROOT = os.path.join(THIS_DIR, "gpu-app-collection")

# Default benchmark suite definition file
DEFAULT_APPS_YAML = os.path.join(THIS_DIR, "define-all-apps.yml")

# Default output root for traces
DEFAULT_TRACES_ROOT = os.path.join(THIS_DIR, "hw_run", "traces")


# ============================================================================
# CUDA version detection
# ============================================================================
def get_cuda_version():
    """Detect CUDA version from nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except Exception:
        pass
    # Fallback: check bin directories under GPUAPPS_ROOT
    gpuapps = os.environ.get("GPUAPPS_ROOT", DEFAULT_GPUAPPS_ROOT)
    bin_dir = os.path.join(gpuapps, "bin")
    if os.path.isdir(bin_dir):
        versions = [d for d in os.listdir(bin_dir) if os.path.isdir(os.path.join(bin_dir, d))]
        if versions:
            return sorted(versions)[-1]
    return "12.6"


# ============================================================================
# YAML parsing (minimal, no external dependency)
# ============================================================================
def load_yaml_simple(filepath):
    """
    Minimal YAML parser sufficient for the benchmark definition format.
    Uses PyYAML if available, otherwise falls back to a simple parser.
    """
    try:
        import yaml
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        return _parse_yaml_fallback(filepath)


def _parse_yaml_fallback(filepath):
    """Simple YAML parser for the specific benchmark definition format."""
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_suite = None
    current_key = None
    current_exec = None
    current_args_list = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        # Skip empty lines and comments
        if not stripped or stripped.lstrip().startswith('#'):
            i += 1
            continue

        indent = len(line) - len(line.lstrip())

        # Top-level suite (no indent, ends with ':')
        if indent == 0 and stripped.endswith(':') and not stripped.startswith(' '):
            suite_name = stripped[:-1].strip()
            current_suite = suite_name
            data[current_suite] = {}
            current_key = None
            current_exec = None
            i += 1
            continue

        if current_suite is None:
            i += 1
            continue

        content = stripped.lstrip()

        # Suite-level keys: exec_dir, data_dirs, execs
        if indent <= 4 and ':' in content and not content.startswith('-'):
            key, _, val = content.partition(':')
            key = key.strip()
            val = val.strip()
            if key in ('exec_dir', 'data_dirs'):
                data[current_suite][key] = val.strip('"').strip("'")
                current_key = key
            elif key == 'execs':
                data[current_suite]['execs'] = {}
                current_key = 'execs'
            i += 1
            continue

        # Executable entry: "- exec_name:"
        if current_key == 'execs' and content.startswith('- ') and content.endswith(':'):
            exec_name = content[2:-1].strip()
            data[current_suite]['execs'][exec_name] = []
            current_exec = exec_name
            i += 1
            continue

        # Args entry: "- args: ..."
        if current_exec and content.startswith('- args:'):
            args_val = content[len('- args:'):].strip()
            data[current_suite]['execs'][current_exec].append({'args': args_val})
            i += 1
            continue

        # Skip other keys (accel-sim-mem, qos, etc.)
        i += 1

    return data


def get_arg_foldername(args_str):
    """Convert args string to a safe folder name."""
    if not args_str or not args_str.strip():
        return "NO_ARGS"
    safe = args_str.strip()
    safe = re.sub(r'[/\\]', '_', safe)
    safe = re.sub(r'[^a-zA-Z0-9_.\-+]', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    return safe[:200] if safe else "NO_ARGS"


# ============================================================================
# Benchmark suite generation
# ============================================================================
def gen_benchmarks_from_suites(suite_names, apps_yaml, gpuapps_root, cuda_version):
    """
    Parse YAML and generate list of (exec_path, data_dir, run_name, args) tuples.

    YAML format: execs is a list of single-key dicts:
        execs:
            - exec_name:
                - args: "..."
    """
    if not os.path.exists(apps_yaml):
        sys.exit(f"Error: Benchmark definition file not found: {apps_yaml}")

    all_apps = load_yaml_simple(apps_yaml)
    benchmarks = []

    for suite in suite_names:
        if suite not in all_apps:
            print(f"Warning: Suite '{suite}' not found in {apps_yaml}. Available suites:")
            for s in sorted(all_apps.keys()):
                sdef = all_apps.get(s, {})
                if isinstance(sdef, dict) and 'execs' in sdef:
                    print(f"  - {s}")
            sys.exit(1)

        suite_def = all_apps[suite]
        exec_dir = suite_def.get('exec_dir', '')
        data_dirs = suite_def.get('data_dirs', '')
        execs = suite_def.get('execs', [])

        # Expand environment variables
        exec_dir = exec_dir.replace('$GPUAPPS_ROOT', gpuapps_root)
        exec_dir = exec_dir.replace('$CUDA_VERSION', cuda_version)
        data_dirs = data_dirs.replace('$GPUAPPS_ROOT', gpuapps_root)

        # execs is a list of single-key dicts: [{name: [args_list]}, ...]
        for exec_entry in execs:
            if isinstance(exec_entry, dict):
                for exec_name, args_list in exec_entry.items():
                    exec_path = os.path.join(exec_dir, exec_name)
                    if not args_list:
                        args_list = [{'args': ''}]
                    for arg_entry in args_list:
                        if isinstance(arg_entry, dict):
                            args_str = arg_entry.get('args', '') or ''
                        else:
                            args_str = str(arg_entry) if arg_entry else ''
                        benchmarks.append((exec_path, data_dirs, exec_name, args_str))
            elif isinstance(exec_entry, str):
                exec_path = os.path.join(exec_dir, exec_entry)
                benchmarks.append((exec_path, data_dirs, exec_entry, ''))

    return benchmarks


# ============================================================================
# Trace execution
# ============================================================================
def run_single_trace(exec_path, data_dir, run_name, args_str, traces_dir,
                     device, tracer_so, kernel_start, kernel_end,
                     terminate_upon_limit, compressed, post_processing_bin,
                     verbose, norun):
    """Run tracing for a single benchmark configuration."""
    arg_folder = get_arg_foldername(args_str)
    this_run_dir = os.path.join(traces_dir, run_name, arg_folder)
    this_trace_folder = os.path.join(this_run_dir, "traces")
    os.makedirs(this_trace_folder, exist_ok=True)

    # Link data directory if it exists
    if data_dir:
        data_candidates = [
            os.path.join(data_dir, run_name, "data"),
            os.path.join(data_dir, "data"),
        ]
        for dc in data_candidates:
            if os.path.isdir(dc):
                link_path = os.path.join(this_run_dir, "data")
                if os.path.lexists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(dc, link_path)
                except OSError:
                    pass
                break

        # Also link top-level data_dirs
        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        if os.path.isdir(data_dir):
            try:
                os.symlink(data_dir, all_data_link)
            except OSError:
                pass

    # Build the shell script
    sh = "#!/bin/bash\nset -e\n"

    if terminate_upon_limit:
        sh += "export TERMINATE_UPON_LIMIT=1\n"
    if kernel_end > 0:
        sh += f"export DYNAMIC_KERNEL_LIMIT_END={kernel_end}\n"
    if kernel_start > 0:
        sh += f"export DYNAMIC_KERNEL_LIMIT_START={kernel_start}\n"
    if verbose:
        sh += "export TOOL_VERBOSE=1\n"

    sh += f'export CUDA_VISIBLE_DEVICES="{device}"\n'
    sh += f'export TRACES_FOLDER="{this_trace_folder}"\n'
    sh += f'export USER_DEFINED_FOLDERS=1\n'

    sh += f'\necho "[NV Trace] Tracing: {run_name}"\n'
    sh += f'echo "[NV Trace] Args: {args_str}"\n'
    sh += f'echo "[NV Trace] Output: {this_trace_folder}"\n\n'

    # Main tracing command
    sh += f'LD_PRELOAD="{tracer_so}" {exec_path} {args_str}\n'

    # Post-processing for compressed mode
    if compressed and os.path.exists(post_processing_bin):
        try:
            import psutil
            available_mem = psutil.virtual_memory()[1] * 0.8
        except ImportError:
            available_mem = 4 * 1024 * 1024 * 1024
        kernelslist = os.path.join(this_trace_folder, "kernelslist")
        sh += f'\n{post_processing_bin} {kernelslist} {available_mem}\n'
        sh += f'rm -f {this_trace_folder}/*.trace\n'
        sh += f'rm -f {kernelslist}\n'

    sh += f'\necho "[NV Trace] Done: {run_name}"\n'

    # Write run script
    run_sh_path = os.path.join(this_run_dir, "run.sh")
    with open(run_sh_path, 'w') as f:
        f.write(sh)
    os.chmod(run_sh_path, 0o755)

    if norun:
        print(f"  [norun] Generated: {run_sh_path}")
        return True

    # Execute
    print(f"  Running: {run_name} ({args_str or 'no args'})")
    saved_dir = os.getcwd()
    os.chdir(this_run_dir)
    ret = subprocess.call(["bash", "run.sh"])
    os.chdir(saved_dir)

    if ret != 0:
        print(f"  ERROR: Tracing failed for {run_name} (exit code {ret})")
        return False
    else:
        print(f"  OK: Traces saved to {this_trace_folder}")
        return True


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NV Trace: Standalone GPU trace extraction tool for NVIDIA GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark suite mode
  %(prog)s -B rodinia_2.0-ft -D 0
  %(prog)s -B rodinia_2.0-ft,GPU_Microbenchmark -D 0 -l 50

  # Single app mode
  %(prog)s --app ./gpu-app-collection/bin/12.6/release/backprop-rodinia-2.0-ft --args "4096 ./data/result-4096.txt"

  # List available suites
  %(prog)s --list-suites
"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-B", "--benchmark-list",
        help="Comma-separated list of benchmark suites to trace (e.g. rodinia_2.0-ft,parboil)"
    )
    mode_group.add_argument(
        "--app", "-a",
        help="Path to a single CUDA application to trace"
    )
    mode_group.add_argument(
        "--list-suites",
        action="store_true",
        default=False,
        help="List all available benchmark suites and exit"
    )

    # Common options
    parser.add_argument(
        "-D", "--device",
        default="0",
        help="CUDA device number (default: 0)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output root directory for traces"
    )
    parser.add_argument(
        "-l", "--kernel-limit",
        type=int, default=0,
        help="Max number of kernels to trace per benchmark (0 = no limit)"
    )
    parser.add_argument(
        "--kernel-start",
        type=int, default=0,
        help="Start tracing from this kernel ID (0 = from beginning)"
    )
    parser.add_argument(
        "--kernel-end",
        type=int, default=0,
        help="Stop tracing after this kernel ID (0 = no limit, overridden by -l)"
    )
    parser.add_argument(
        "-t", "--terminate-upon-limit",
        action="store_true", default=False,
        help="Terminate the process once kernel limit is reached"
    )
    parser.add_argument(
        "-C", "--compressed",
        action="store_true", default=False,
        help="Run in compressed mode with post-processing"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", default=False,
        help="Enable verbose tracer output"
    )
    parser.add_argument(
        "-n", "--norun",
        action="store_true", default=False,
        help="Only generate run scripts, do not execute"
    )

    # Single-app specific
    parser.add_argument(
        "--args",
        default="",
        help="Arguments to pass to the application (single-app mode)"
    )

    # Path overrides
    parser.add_argument(
        "--gpuapps-root",
        default=None,
        help=f"Path to gpu-app-collection (default: {DEFAULT_GPUAPPS_ROOT})"
    )
    parser.add_argument(
        "--apps-yaml",
        default=None,
        help=f"Path to benchmark definition YAML (default: {DEFAULT_APPS_YAML})"
    )

    args = parser.parse_args()

    # Resolve paths
    gpuapps_root = os.path.abspath(args.gpuapps_root) if args.gpuapps_root else DEFAULT_GPUAPPS_ROOT
    apps_yaml = os.path.abspath(args.apps_yaml) if args.apps_yaml else DEFAULT_APPS_YAML
    cuda_version = get_cuda_version()

    # Set GPUAPPS_ROOT in environment (some scripts may need it)
    os.environ['GPUAPPS_ROOT'] = gpuapps_root
    os.environ.setdefault('CUDA_VERSION', cuda_version)

    # Tracer tool location
    tracer_so = os.path.join(THIS_DIR, "tracer_tool", "tracer_tool.so")
    post_processing_bin = os.path.join(
        THIS_DIR, "tracer_tool", "traces-processing", "post-traces-processing-compressed"
    )

    # Handle kernel limit shortcut
    kernel_end = args.kernel_end
    if args.kernel_limit > 0 and kernel_end == 0:
        kernel_end = args.kernel_limit

    # ---- List suites mode ----
    if args.list_suites:
        if not os.path.exists(apps_yaml):
            sys.exit(f"Error: YAML file not found: {apps_yaml}")
        all_apps = load_yaml_simple(apps_yaml)
        print(f"Available benchmark suites (from {apps_yaml}):\n")
        for suite_name in sorted(all_apps.keys()):
            suite_def = all_apps[suite_name]
            if isinstance(suite_def, dict) and 'execs' in suite_def:
                execs_list = suite_def['execs']
                num_execs = len(execs_list) if isinstance(execs_list, list) else 0
                total_configs = 0
                if isinstance(execs_list, list):
                    for entry in execs_list:
                        if isinstance(entry, dict):
                            for _, args_list in entry.items():
                                total_configs += max(1, len(args_list)) if isinstance(args_list, list) else 1
                        else:
                            total_configs += 1
                print(f"  {suite_name:40s}  ({num_execs} executables, {total_configs} configs)")
        return

    # ---- Validate mode ----
    if not args.benchmark_list and not args.app:
        parser.print_help()
        print("\nError: Must specify either -B <suite> or --app <path>")
        sys.exit(1)

    if not os.path.exists(tracer_so):
        sys.exit(
            f"Error: tracer_tool.so not found at {tracer_so}\n"
            "Please build first:\n"
            "  ./install_nvbit.sh && make"
        )

    # ---- Benchmark suite mode ----
    if args.benchmark_list:
        suites = [s.strip() for s in args.benchmark_list.split(",")]

        # Determine output directory
        if args.output:
            traces_root = os.path.abspath(args.output)
        else:
            traces_root = os.path.join(
                DEFAULT_TRACES_ROOT, f"device-{args.device}", cuda_version
            )

        print(f"=" * 70)
        print(f"NV Trace - Benchmark Suite Trace Extraction")
        print(f"=" * 70)
        print(f"  GPUAPPS_ROOT:  {gpuapps_root}")
        print(f"  CUDA version:  {cuda_version}")
        print(f"  Device:        {args.device}")
        print(f"  Suites:        {', '.join(suites)}")
        print(f"  Output root:   {traces_root}")
        if kernel_end > 0:
            print(f"  Kernel limit:  {kernel_end}")
        print(f"=" * 70)
        print()

        benchmarks = gen_benchmarks_from_suites(suites, apps_yaml, gpuapps_root, cuda_version)
        if not benchmarks:
            sys.exit("Error: No benchmarks found for the specified suites")

        print(f"Found {len(benchmarks)} benchmark configuration(s) to trace\n")

        ok_count = 0
        fail_count = 0
        skip_count = 0

        for exec_path, data_dir, exec_name, args_str in benchmarks:
            if not os.path.exists(exec_path):
                print(f"  SKIP: {exec_name} (executable not found: {exec_path})")
                skip_count += 1
                continue

            success = run_single_trace(
                exec_path=exec_path,
                data_dir=data_dir,
                run_name=exec_name,
                args_str=args_str,
                traces_dir=traces_root,
                device=args.device,
                tracer_so=tracer_so,
                kernel_start=args.kernel_start,
                kernel_end=kernel_end,
                terminate_upon_limit=args.terminate_upon_limit,
                compressed=args.compressed,
                post_processing_bin=post_processing_bin,
                verbose=args.verbose,
                norun=args.norun,
            )
            if success:
                ok_count += 1
            else:
                fail_count += 1

        print(f"\n{'=' * 70}")
        print(f"Summary: {ok_count} OK, {fail_count} failed, {skip_count} skipped")
        print(f"Traces root: {traces_root}")
        print(f"{'=' * 70}")

    # ---- Single app mode ----
    elif args.app:
        app_path = os.path.abspath(args.app)
        if not os.path.exists(app_path):
            sys.exit(f"Error: Application not found: {app_path}")

        app_name = os.path.basename(app_path)

        if args.output:
            output_dir = os.path.abspath(args.output)
        else:
            output_dir = os.path.join(os.getcwd(), "traces_output", app_name)

        print(f"NV Trace - Single Application Mode")
        print(f"  Application: {app_path}")
        print(f"  Device:      {args.device}")
        print(f"  Output:      {output_dir}")
        print()

        run_single_trace(
            exec_path=app_path,
            data_dir="",
            run_name=app_name,
            args_str=args.args,
            traces_dir=output_dir,
            device=args.device,
            tracer_so=tracer_so,
            kernel_start=args.kernel_start,
            kernel_end=kernel_end,
            terminate_upon_limit=args.terminate_upon_limit,
            compressed=args.compressed,
            post_processing_bin=post_processing_bin,
            verbose=args.verbose,
            norun=args.norun,
        )


if __name__ == "__main__":
    main()
