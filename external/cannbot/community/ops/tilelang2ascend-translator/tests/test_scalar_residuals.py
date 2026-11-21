# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_scalar_residuals.py"
SPEC = importlib.util.spec_from_file_location("scalar_scan", SCRIPT)
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


class ScalarResidualTests(unittest.TestCase):
    def rules(self, source):
        return {r for finding in SCAN.scan_source(source) for r in finding["rules"]}

    def test_reduction_layouts_and_names(self):
        for header in ("for (int i=0; i<n; ++i)", "while (i<n)"):
            for body in (
                "{ tally += x[i]; }",
                "\n{\n tally += x[i];\n}",
                "\n tally += x[i];",
            ):
                with self.subTest(header=header, body=body):
                    self.assertIn(1, self.rules(header + body))
        self.assertIn(1, self.rules("do { total = total + x[i++]; } while(i<n);"))

    def test_nested_sort_crosses_lines(self):
        source = "for(int i=0;i<n;++i)\n{\nfor(int j=0;j<n;++j)\n{\nif(a[j]<a[j+1]) swap(a[j],a[j+1]);\n}\n}"
        self.assertIn(3, self.rules(source))
        self.assertEqual(len(SCAN.scan_source(source)), 2)

    def test_gather_non_order_variable(self):
        self.assertIn(4, self.rules("for(auto i: ids)\n dst[i] = src[permutation[i]];"))

    def test_getvalue_muls_crosses_lines(self):
        self.assertIn(
            5,
            self.rules(
                "for(int row=0;row<n;++row) {\n auto scale=x.GetValue(row);\n AscendC::Muls(y,y,scale,n);\n}"
            ),
        )

    def test_setvalue_getvalue_crosses_lines(self):
        self.assertIn(
            6,
            self.rules(
                "for(int j=0;j<n;++j)\n{ dst.SetValue(j,\n src.GetValue(2*j)); }"
            ),
        )

    def test_pointer_site_not_hidden_by_loop_scan(self):
        self.assertIn(
            2,
            self.rules("auto ptr = x.GetPhyAddr();\nfor(int i=0;i<n;++i) y[i]=ptr[i];"),
        )

    def test_comments_and_literals_are_ignored(self):
        source = """// for(i=0;i<n;++i) { sum += x[i]; }
/* while(x) { GetPhyAddr(); } */
auto a = "for (x) { reinterpret_cast<float*>(p); }";
auto b = R"tag(for(i=0;i<n;++i) { sum += x[i]; })tag";
char c = '}';
"""
        self.assertEqual(SCAN.scan_source(source), [])

    def test_legal_batch_loop_is_reviewed_not_deleted(self):
        source = (
            "// scalar-review: batch-control; each iteration processes one vector tile\n"
            "for(int b=0;b<blocks;++b) ProcessTile(b);"
        )
        result = SCAN.scan_source(source)
        self.assertEqual(result[0]["rules"], [7])
        self.assertEqual(result[0]["review"]["category"], "batch-control")
        self.assertEqual(result[0]["line"], 2)

    def test_threshold_fallback_can_be_reviewed(self):
        source = (
            "// scalar-review: threshold-fallback; sequential suppression control\n"
            "for(int k=0;k<n;++k) if(w4.GetValue(k)>threshold) flags[k]=true;"
        )
        finding = SCAN.scan_source(source)[0]
        self.assertIn(6, finding["rules"])
        self.assertIsNotNone(finding["review"])

    def test_parent_review_does_not_hide_inner_loop(self):
        source = (
            "// scalar-review: batch-control; batches\n"
            "for(int b=0;b<B;++b) {\n for(int j=0;j<n;++j) acc += x[j];\n}"
        )
        result = SCAN.scan_source(source)
        self.assertIsNotNone(result[0]["review"])
        self.assertIsNone(result[1]["review"])

    def test_review_needs_known_category_and_reason(self):
        for note in ("small-tile;", "anything; bypass", "small-tile"):
            with self.subTest(note=note):
                finding = SCAN.scan_source(
                    f"// scalar-review: {note}\nfor(;;) work();"
                )[0]
                self.assertIsNone(finding["review"])

    def test_string_cannot_supply_review(self):
        source = 'auto s="// scalar-review: small-tile; fake";\nfor(;;) work();'
        self.assertIsNone(SCAN.scan_source(source)[0]["review"])

    def test_do_while_has_one_inventory_entry(self):
        self.assertEqual(len(SCAN.scan_source("do { a += x[i++]; } while(i<n);")), 1)

    def test_unbraced_if_else_loop(self):
        self.assertIn(
            6,
            self.rules(
                "for(int i=0;i<n;++i) if(flag) f(); else dst.SetValue(i,src.GetValue(i));"
            ),
        )

    def test_malformed_input_fails_closed(self):
        with self.assertRaises(ValueError):
            SCAN.scan_source("for(;;) { sum += x[i];")

    def test_cli_exit_codes_and_header_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(*paths):
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT), *map(str, paths)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode, json.loads(proc.stdout)

            self.assertEqual(run(root)[0], 2)
            self.assertEqual(run(root / "missing.cpp")[0], 2)
            nested = root / "nested"
            nested.mkdir()
            source = nested / "compute.hpp"
            source.write_text("for(int i=0;i<n;++i) acc += x[i];")
            code, report = run(root)
            self.assertEqual(code, 1)
            self.assertEqual(report["files_scanned"], 1)
            source.write_text("AscendC::Muls(y,x,scale,n);")
            self.assertEqual(run(root)[0], 0)
            source.write_text(
                "// scalar-review: small-tile; guarded n<128\nfor(int i=0;i<n;++i) acc += x[i];"
            )
            self.assertEqual(run(root)[0], 0)
            source.write_text("for(;;) {")
            self.assertEqual(run(root)[0], 2)


if __name__ == "__main__":
    unittest.main()
