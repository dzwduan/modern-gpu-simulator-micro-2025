from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.remodeled_trace import generate_semantic_fixtures


class GpuQueryTest(unittest.TestCase):
    def test_parses_gpu_identity_rows(self) -> None:
        rows = generate_semantic_fixtures.parse_gpu_rows(
            "0, GPU-a, NVIDIA GeForce RTX 4090, 8.9, 565.57.01\n"
        )

        self.assertEqual(
            rows,
            [
                {
                    "index": "0",
                    "uuid": "GPU-a",
                    "name": "NVIDIA GeForce RTX 4090",
                    "compute_capability": "8.9",
                    "driver_version": "565.57.01",
                }
            ],
        )


class DeterministicArchiveTest(unittest.TestCase):
    def test_repeated_archives_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            temporary_root = Path(raw_tmp)
            source_root = temporary_root / "source"
            source_root.mkdir()
            (source_root / "dynamic_trace.pb").write_bytes(b"trace")
            first = temporary_root / "first.tar.gz"
            second = temporary_root / "second.tar.gz"

            for destination in (first, second):
                generate_semantic_fixtures.deterministic_archive(
                    source_root,
                    [Path("dynamic_trace.pb")],
                    "fixture/traces",
                    destination,
                )

            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
