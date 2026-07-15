from __future__ import annotations

import os
import unittest

from tests.remodeled_trace import measure_shared_lat_rtx4090


class DirectOutputTest(unittest.TestCase):
    def test_parses_latency_and_total_clock(self) -> None:
        parsed = measure_shared_lat_rtx4090.parse_direct_output(
            "Shared Memory Latency  = 30.032715 cycles\n"
            "Total Clk number = 61507 \n"
        )

        self.assertEqual(parsed["shared_memory_latency_cycles"], 30.032715)
        self.assertEqual(parsed["timed_region_total_cycles"], 61507)


class NsightCsvTest(unittest.TestCase):
    def test_parses_shared_lat_kernel_row(self) -> None:
        metrics = measure_shared_lat_rtx4090.NCU_METRICS
        header = ["ID", "Kernel Name", *metrics]
        unit_row = ["", "", "cycle", "cycle", "inst", "inst", "ns"]
        data_row = [
            "0",
            "shared_lat(unsigned int *)",
            "270,595.82",
            "270,595.95",
            "32,885",
            "32,885",
            "121,120",
        ]
        csv_output = "==PROF== Connected\nprogram output\n" + "\n".join(
            ",".join(f'"{value}"' for value in row)
            for row in (header, unit_row, data_row)
        )

        parsed = measure_shared_lat_rtx4090.parse_ncu_raw_csv(csv_output)

        self.assertEqual(parsed["sm__cycles_elapsed.avg"], 270595.95)
        self.assertEqual(parsed["sm__inst_executed.sum"], 32885)


class SummaryAndComparisonTest(unittest.TestCase):
    def test_uses_median_and_computes_relative_error(self) -> None:
        summary = measure_shared_lat_rtx4090.summarize_samples(
            [
                {
                    "sm__cycles_elapsed.avg": 270600,
                    "sm__inst_executed.sum": 32885,
                },
                {
                    "sm__cycles_elapsed.avg": 270590,
                    "sm__inst_executed.sum": 32885,
                },
                {
                    "sm__cycles_elapsed.avg": 270595,
                    "sm__inst_executed.sum": 32885,
                },
            ]
        )
        comparison = measure_shared_lat_rtx4090.compare_with_simulator(
            {
                "stats": {
                    "gpu_tot_sim_cycle": 192072,
                    "gpu_tot_sim_insn": 32881,
                }
            },
            summary,
        )

        self.assertEqual(
            summary["fields"]["sm__cycles_elapsed.avg"]["median"], 270595
        )
        self.assertEqual(
            comparison["executed_instructions"]["hardware_median"], 32885
        )
        self.assertAlmostEqual(
            comparison["executed_instructions"][
                "signed_relative_error_percent"
            ],
            -0.012164,
        )


class DeviceSelectionTest(unittest.TestCase):
    def test_cuda_visible_devices_uses_gpu_uuid(self) -> None:
        gpu_uuid = "GPU-0a6e098f-a556-73f4-0a21-5151b876a20e"
        original = os.environ.get("CUDA_VISIBLE_DEVICES")

        environment = measure_shared_lat_rtx4090.cuda_environment(gpu_uuid)

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], gpu_uuid)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)


if __name__ == "__main__":
    unittest.main()
