# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Load CANN SparseFlashAttention's mathematical golden and precision checker.

CANN_BENCH_ROOT selects the external evaluator checkout. Its src/ directory
must be on PYTHONPATH so that the original kernel_eval checker is importable.
The test calls the golden with both FP64 and native FP16 inputs, preserving
the evaluator's cancellation and small-value rules.
"""

from pathlib import Path
import runpy


def load_evaluator(root):
    from kernel_eval.utils.compare import compare_tensors

    source = Path(root) / "tasks/level4/sparse_flash_attention/golden.py"
    golden = runpy.run_path(str(source))["sparse_flash_attention"]
    return golden, compare_tensors
