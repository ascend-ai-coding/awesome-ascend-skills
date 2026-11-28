# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for per-case timing aggregation in msprof_perf_summary.

The paired-case tests reproduce a former issue where total_speedup summed
independent ref/asc lists and compared durations from different cases.
No NPU, profiling files, or subprocesses are required.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "msprof_perf_summary.py"
SPEC = importlib.util.spec_from_file_location("msprof_perf_summary_under_test", SCRIPT)
perf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = perf
SPEC.loader.exec_module(perf)


def _result(duration_us):
    """Build a synthetic _measure_one_impl result without running profiling."""
    if duration_us is None:
        return None, "synthetic measurement failure", None
    return duration_us, None, None


def _summary_for(case_results):
    """Run the production accumulator -> summary path for (ref, asc) cases."""
    accumulator = perf._LoopAccumulator(SimpleNamespace(keep_prof=True))
    for case_idx, (ref_us, asc_us) in enumerate(case_results):
        accumulator.record(
            case_idx,
            f"case_{case_idx}",
            "float32",
            _result(ref_us),
            _result(asc_us),
        )

    args = SimpleNamespace(warmup=0, repeats=1, seed=42)
    summary_input = perf._CompareSummaryInput(
        out_dir=Path("/tmp/synthetic-op"),
        rows=accumulator.rows,
        speedups=accumulator.speedups,
        ref_times=accumulator.ref_times,
        asc_times=accumulator.asc_times,
        n_cases=len(case_results),
        args=args,
        device_id=0,
        device_src="test",
    )
    return perf._compute_compare_summary(summary_input)


class TimingAggregationTest(unittest.TestCase):
    def test_total_speedup_when_all_cases_are_paired(self):
        """Normal control: (10 + 20) / (5 + 10) = 2x."""
        summary = _summary_for([(10.0, 5.0), (20.0, 10.0)])

        self.assertEqual(summary["n_cases_valid"], 2)
        self.assertAlmostEqual(summary["total_speedup"], 2.0)

    def test_total_speedup_ignores_unpaired_cases(self):
        """Only case 0 has both sides, so total_speedup must be 2x.

        Before the fix, the result was (10 + 100) / (5 + 1) = 18.333...x
        because the reference-only and AscendC-only cases were included.
        """
        summary = _summary_for([
            (10.0, 5.0),   # case 0: paired, speedup = 2x
            (100.0, None), # case 1: reference only; must not enter total_speedup
            (None, 1.0),   # case 2: AscendC only; must not enter total_speedup
        ])

        self.assertEqual(summary["n_cases_valid"], 1)
        self.assertAlmostEqual(summary["total_speedup"], 2.0)

    def test_total_speedup_requires_paired_cases(self):
        """Durations from disjoint cases must not produce a speedup.

        Before the fix, the result was 10 / 5 = 2x even though no case had
        both a valid reference and a valid AscendC measurement.
        """
        summary = _summary_for([
            (10.0, None), # case 0: reference only
            (None, 5.0),  # case 1: AscendC only
        ])

        self.assertEqual(summary["n_cases_valid"], 0)
        self.assertIsNone(summary["total_speedup"])


if __name__ == "__main__":
    unittest.main()
