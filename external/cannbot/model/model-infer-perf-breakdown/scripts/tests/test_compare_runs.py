# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_runs


def _run(label, cluster_name, cluster_value, total_value):
    return {
        "label": label,
        "sections": [
            {
                "component_type": "decoder",
                "sub_items": [
                    {
                        "name": cluster_name,
                        "kind": "cluster",
                        "metric": {"median_ms": cluster_value},
                    },
                    {"name": "bubble", "kind": "bubble", "metric": {"median_ms": 0.0}},
                ],
                "total": {"metric": {"median_ms": total_value}},
            }
        ],
    }


class CollectRowsTest(unittest.TestCase):
    def test_cluster_named_total_keeps_its_series_separate(self):
        rows = compare_runs.collect_rows(
            [
                _run("baseline", "TOTAL", 10.0, 100.0),
                _run("candidate", "TOTAL", 20.0, 120.0),
            ],
            "median_ms",
        )

        total_rows = [row for row in rows if row["sub_item"] == "TOTAL"]
        self.assertEqual(
            [
                (
                    row["kind"],
                    row["per_run"]["baseline"]["value"],
                    row["per_run"]["candidate"]["value"],
                )
                for row in total_rows
            ],
            [("cluster", 10.0, 20.0), ("total", 100.0, 120.0)],
        )

    def test_ordinary_cluster_name_keeps_existing_comparison(self):
        rows = compare_runs.collect_rows(
            [
                _run("baseline", "compute", 10.0, 100.0),
                _run("candidate", "compute", 20.0, 120.0),
            ],
            "median_ms",
        )

        compute = next(row for row in rows if row["sub_item"] == "compute")
        total = next(row for row in rows if row["kind"] == "total")
        self.assertEqual(compute["per_run"]["baseline"]["value"], 10.0)
        self.assertEqual(compute["per_run"]["candidate"]["value"], 20.0)
        self.assertEqual(total["per_run"]["baseline"]["value"], 100.0)
        self.assertEqual(total["per_run"]["candidate"]["value"], 120.0)

    def test_cluster_named_bubble_keeps_synthetic_bubble_separate(self):
        rows = compare_runs.collect_rows(
            [
                _run("baseline", "bubble", 10.0, 100.0),
            ],
            "median_ms",
        )

        bubble_rows = [row for row in rows if row["sub_item"] == "bubble"]
        self.assertEqual(
            [(row["kind"], row["per_run"]["baseline"]["value"]) for row in bubble_rows],
            [("cluster", 10.0), ("bubble", 0.0)],
        )


if __name__ == "__main__":
    unittest.main()
