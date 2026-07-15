from __future__ import annotations

import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from tests.remodeled_trace import run_regression


class ParseStatsTest(unittest.TestCase):
    def test_uses_last_aggregate_value(self) -> None:
        output = """
gpu_tot_sim_cycle = 10
gpu_tot_sim_insn = 20
gpu_tot_ipc = 2.0000
gpu_tot_sim_cycle = 20
gpu_tot_sim_insn = 50
gpu_tot_ipc = 2.5000
"""

        self.assertEqual(
            run_regression.parse_stats(output),
            {
                "gpu_tot_sim_cycle": 20,
                "gpu_tot_sim_insn": 50,
                "gpu_tot_ipc": 2.5,
            },
        )

    def test_missing_required_stat_is_failure(self) -> None:
        failures, missing, _ = run_regression.evaluate_observation(
            run_regression.COMPLETION_MARKER,
            0,
            ["gpu_tot_sim_cycle"],
        )

        self.assertEqual(missing, ["gpu_tot_sim_cycle"])
        self.assertIn("missing_stat:gpu_tot_sim_cycle", failures)

    def test_ipc_invariant_accepts_print_rounding(self) -> None:
        output = """
gpu_tot_sim_cycle = 28795
gpu_tot_sim_insn = 804960
gpu_tot_ipc = 27.9549
GPGPU-Sim: *** exit detected ***
"""

        failures, _, _ = run_regression.evaluate_observation(
            output,
            0,
            ["gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpu_tot_ipc"],
        )

        self.assertEqual(failures, [])

    def test_zero_cycles_is_reported_without_division(self) -> None:
        output = """
gpu_tot_sim_cycle = 0
gpu_tot_sim_insn = 0
gpu_tot_ipc = 0.0000
GPGPU-Sim: *** exit detected ***
"""

        failures, _, _ = run_regression.evaluate_observation(
            output,
            0,
            ["gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpu_tot_ipc"],
        )

        self.assertIn("gpu_tot_sim_cycle_zero", failures)


class GoldenComparisonTest(unittest.TestCase):
    @staticmethod
    def comparison_contract() -> dict:
        identity = {"path": "input", "sha256": "b" * 64}
        return {
            "execution_mode": "remodeled_trace",
            "trace": {
                "archive": identity,
                "root": "suite/app/traces",
                "file": "dynamic_trace.pb",
            },
            "configs": {"gpgpusim": identity, "trace": identity},
            "runtime": {
                "lc_all": "C",
                "omp_dynamic": "FALSE",
                "omp_num_threads": 1,
                "timeout_seconds": 120,
            },
        }

    def test_exact_and_tolerant_fields(self) -> None:
        mismatches = run_regression.compare_to_golden(
            {"cycles": 100, "ipc": 1.00004},
            {
                "stats": {"cycles": 100, "ipc": 1.0},
                "absolute_tolerances": {"ipc": 0.0001},
            },
        )

        self.assertEqual(mismatches, [])

    def test_reports_mismatch(self) -> None:
        mismatches = run_regression.compare_to_golden(
            {"cycles": 101},
            {"stats": {"cycles": 100}},
        )

        self.assertEqual(mismatches[0]["stat"], "cycles")

    def test_missing_golden_does_not_hide_run_failure(self) -> None:
        result = {
            "status": "run_failed",
            "failures": ["timeout"],
            "stats": {},
            "inputs": {},
        }

        run_regression.apply_golden(result, None, "unapproved", [], None)

        self.assertEqual(result["status"], "run_failed")
        self.assertEqual(result["golden_set_status"], "unapproved")

    def test_empty_approved_golden_is_rejected(self) -> None:
        result = {
            "status": "observed_not_golden",
            "failures": [],
            "stats": {"cycles": 100},
            "inputs": {
                "binary": {"path": "simulator", "sha256": "b" * 64}
            },
            "comparison_contract": self.comparison_contract(),
        }

        run_regression.apply_golden(
            result,
            {"approved": True, "stats": {}, "comparison_contract": {}},
            "approved",
            ["cycles"],
            "a" * 40,
        )

        self.assertEqual(result["status"], "invalid_approved_golden")
        self.assertIn("missing_stat:cycles", result["golden_schema_errors"])
        self.assertIn(
            "missing_contract:execution_mode", result["golden_schema_errors"]
        )

    def test_complete_approved_golden_can_pass(self) -> None:
        result = {
            "status": "observed_not_golden",
            "failures": [],
            "stats": {"cycles": 100},
            "inputs": {
                "binary": {"path": "new-simulator", "sha256": "c" * 64}
            },
            "comparison_contract": self.comparison_contract(),
        }

        run_regression.apply_golden(
            result,
            {
                "approved": True,
                "stats": {"cycles": 100},
                "comparison_contract": self.comparison_contract(),
            },
            "approved",
            ["cycles"],
            "a" * 40,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["golden_schema_errors"], [])

    def test_runtime_contract_mismatch_cannot_pass(self) -> None:
        observed_contract = self.comparison_contract()
        golden_contract = self.comparison_contract()
        golden_contract["runtime"]["omp_num_threads"] = 2
        result = {
            "status": "observed_not_golden",
            "failures": [],
            "stats": {"cycles": 100},
            "inputs": {},
            "comparison_contract": observed_contract,
        }

        run_regression.apply_golden(
            result,
            {
                "approved": True,
                "stats": {"cycles": 100},
                "comparison_contract": golden_contract,
            },
            "approved",
            ["cycles"],
            "a" * 40,
        )

        self.assertEqual(result["status"], "golden_mismatch")
        self.assertEqual(result["golden_mismatches"][0]["contract"], "runtime")


class ArchiveExtractionTest(unittest.TestCase):
    def test_extracts_only_requested_trace_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            temporary_root = Path(raw_tmp)
            archive_path = temporary_root / "trace.tar.gz"
            payload = b"trace"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("suite/app/traces/dynamic_trace.pb")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                other = tarfile.TarInfo("suite/other/unrelated.txt")
                other.size = len(payload)
                archive.addfile(other, io.BytesIO(payload))

            destination = temporary_root / "output"
            trace_root = run_regression.extract_trace_tree(
                archive_path, "suite/app/traces", destination
            )

            self.assertEqual(
                (trace_root / "dynamic_trace.pb").read_bytes(), payload
            )
            self.assertFalse((destination / "suite/other/unrelated.txt").exists())

    def test_rejects_unsafe_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            temporary_root = Path(raw_tmp)
            archive_path = temporary_root / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("../suite/app/traces/dynamic_trace.pb")
                member.size = 0
                archive.addfile(member, io.BytesIO())

            with self.assertRaises(run_regression.HarnessError):
                run_regression.extract_trace_tree(
                    archive_path, "suite/app/traces", temporary_root / "output"
                )

    def test_manifest_rejects_trace_file_path_traversal(self) -> None:
        manifest = run_regression.load_json(run_regression.DEFAULT_MANIFEST)
        manifest["cases"][0]["trace_file"] = "../dynamic_trace.pb"

        with self.assertRaises(run_regression.HarnessError):
            run_regression.validate_manifest(manifest)


class EnvironmentSetupTest(unittest.TestCase):
    def test_setup_receives_no_harness_positional_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            temporary_root = Path(raw_tmp)
            setup = temporary_root / "setup.sh"
            setup.write_text(
                'if [ "$#" -ne 0 ]; then exit 17; fi\n'
                'export HARNESS_TEST_VALUE="loaded"\n',
                encoding="utf-8",
            )

            environment, warnings = run_regression.load_simulator_environment(
                setup, temporary_root
            )

            self.assertEqual(environment["HARNESS_TEST_VALUE"], "loaded")
            self.assertEqual(warnings, [])
            self.assertEqual(os.environ.get("HARNESS_TEST_VALUE"), None)


if __name__ == "__main__":
    unittest.main()
