#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = TEST_ROOT.parent / "scripts" / "check_stl_op.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_stl_op", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParseDefinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _parse(self, content):
        with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as fp:
            fp.write(content)
            path = fp.name
        try:
            return self.module.parse_defines(path)
        finally:
            os.unlink(path)

    def test_leading_zero_decimal_does_not_crash(self):
        defines = self._parse("#define BLOCK 0100\n#define X 08\n#define UB 0x60000\n")
        self.assertEqual(defines["BLOCK"], 100)
        self.assertEqual(defines["X"], 8)
        self.assertEqual(defines["UB"], 0x60000)

    def test_non_numeric_define_is_skipped(self):
        defines = self._parse("#define BAD not_a_num\n#define OK 4\n")
        self.assertEqual(defines, {"OK": 4})

    def test_missing_file_returns_empty(self):
        self.assertEqual(self.module.parse_defines("/nonexistent/mem_info.h"), {})


class EvalExprTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_hex_and_decimal_constants(self):
        self.assertEqual(self.module.eval_expr("0x400", {}), 1024)
        self.assertEqual(self.module.eval_expr(" 16 ", {}), 16)

    def test_macro_arithmetic(self):
        defines = {"UB_SIZE": 0x40000}
        self.assertEqual(self.module.eval_expr("UB_SIZE * 2", defines), 0x80000)
        self.assertEqual(self.module.eval_expr("UB_SIZE / 2", defines), 0x20000)
        self.assertEqual(self.module.eval_expr("UB_SIZE + 0x10", defines), 0x40010)
        self.assertEqual(self.module.eval_expr("(UB_SIZE * 2) >> 4", defines), 0x8000)

    def test_unknown_identifier_returns_none(self):
        self.assertIsNone(self.module.eval_expr("blockDim * 0x40000", {}))
        self.assertIsNone(self.module.eval_expr("UNKNOWN_MACRO", {}))

    def test_comparison_is_not_evaluated(self):
        self.assertIsNone(self.module.eval_expr("1 < 2", {}))


class FindBlockStridesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_array_index_form_is_captured(self):
        text = "gmOut[block_idx * 0x400] = v;"
        strides = self.module.find_block_strides(text, {})
        self.assertEqual([v for v, _ in strides], [0x400])

    def test_parenthesized_expression(self):
        text = "block_idx * (UB_SIZE * 2);"
        strides = self.module.find_block_strides(text, {"UB_SIZE": 0x40000})
        self.assertEqual([v for v, _ in strides], [0x80000])

    def test_offset_after_stride_is_not_mixed(self):
        text = "dst_addr + block_idx * UB_SIZE + UB_SIZE / 2;"
        strides = self.module.find_block_strides(text, {"UB_SIZE": 0x40000})
        self.assertEqual([v for v, _ in strides], [0x40000])

    def test_source_context_is_retained(self):
        text = "gmOut[block_idx * 0x400] = v;"
        strides = self.module.find_block_strides(text, {})
        self.assertEqual(strides[0][1], "gmOut[block_idx * 0x400] = v;")


class BalanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_string_literal_does_not_break_balance(self):
        text = 'AML_LOGE("desc.outputs[{}] ({})", a);'
        self.assertTrue(self.module.balanced(self.module.strip_strings(text)))

    def test_unbalanced_is_detected(self):
        self.assertFalse(self.module.balanced("void f( {"))


class HostOutputPerBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_block_dim_times_size(self):
        cpp = "desc.blockDim = 2;\ndesc.outputs = {{2 * 0x40000}};"
        block_dim, per = self.module.host_output_per_block(cpp, {})
        self.assertEqual(block_dim, 2)
        self.assertEqual(per, [0x40000])

    def test_fixed_size_divided_by_block_dim(self):
        cpp = "desc.blockDim = 2;\ndesc.outputs = {{0x80000}};"
        block_dim, per = self.module.host_output_per_block(cpp, {})
        self.assertEqual(block_dim, 2)
        self.assertEqual(per, [0x40000])

    def test_unresolvable_block_dim_returns_none(self):
        cpp = "desc.blockDim = blockDim;\ndesc.outputs = {{0x80000}};"
        block_dim, per = self.module.host_output_per_block(cpp, {})
        self.assertIsNone(block_dim)
        self.assertEqual(per, [None])


class EndToEndTest(unittest.TestCase):
    def _run(self, op_dir):
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--op-dir", str(op_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )

    def _make_op_dir(self, root, cpp_outputs, kernel_text, block_dim="1"):
        ops = root / "ops"
        (ops / "ops_include").mkdir(parents=True)
        (ops / "ops_include" / "mem_info.h").write_text("#define UB_SIZE 0x40000\n")
        op_dir = ops / "aic_demo"
        op_dir.mkdir()
        (op_dir / "aic_demo.cpp").write_text(
            f"desc.blockDim = {block_dim};\n"
            'desc.name = "aic_demo";\n'
            f"desc.outputs = {{{cpp_outputs}}};\n"
            'ResolveKernelSymbol("AmlStlLaunchAicDemo");\n'
        )
        (op_dir / "aic_demo.asc").write_text(
            '#include "aic_demo.cce"\n'
            'extern "C" int AmlStlLaunchAicDemo() {}\n'
            "__global__ __aicore__ void aic_demo_impl() {}\n"
        )
        (op_dir / "aic_demo.cce").write_text(kernel_text)
        return op_dir

    def test_missing_op_dir_reports_error(self):
        result = self._run(Path("/nonexistent/op"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("[ERROR]", result.stdout)

    def test_op_dir_with_matched_host_and_kernel_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{1 * UB_SIZE}",
                "gmOut[block_idx * UB_SIZE] = v;\n",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("[OK]", result.stdout)

    def test_mismatched_stride_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{1 * UB_SIZE}",
                "gmOut[block_idx * (UB_SIZE / 2)] = v;\n",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("越界", result.stdout)
            self.assertNotIn("[OK]", result.stdout)

    def test_non_4_aligned_output_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{1 * (UB_SIZE + 2)}",
                "gmOut[block_idx * UB_SIZE] = v;\n",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("不是 4 的倍数", result.stdout)
            self.assertNotIn("[OK]", result.stdout)

    def test_unable_to_bound_check_does_not_print_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{1 * UB_SIZE}",
                "gmOut[0] = v;\n",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("[WARN]", result.stdout)
            self.assertNotIn("[OK]", result.stdout)

    def test_unresolvable_block_dim_reports_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{UB_SIZE * 2}",
                "gmOut[block_idx * UB_SIZE] = v;\n",
                block_dim="blockDim",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("[WARN]", result.stdout)
            self.assertIn("无法解析", result.stdout)
            self.assertNotIn("[OK]", result.stdout)

    def test_partial_unresolved_output_reports_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op_dir = self._make_op_dir(
                root,
                "{1 * UB_SIZE, unknown_placeholder}",
                "gmOut[block_idx * UB_SIZE] = v;\n",
            )
            result = self._run(op_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("[WARN]", result.stdout)
            self.assertNotIn("[OK]", result.stdout)


if __name__ == "__main__":
    unittest.main()
