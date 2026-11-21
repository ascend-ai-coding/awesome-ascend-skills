# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Load the original CANNBot DSL evaluator with pinned source identities.

Only the worker-count constants are adapted to the device's available Cube cores.
The evaluator's mathematical reference and numerical criteria remain unchanged.
"""

import hashlib
import importlib.util
import re
import sys

UPSTREAM_COMMIT = "53f77e1b8ddce86584a6b65eabd22dac257594e8"
FILES = {
    "core": (
        "samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py",
        "e5152d247b0ddd26a186b42a0a40d3ecc5a6f026d9fd2b0f43a96e41df1e00e6",
    ),
    "metadata": (
        "samples/quant_lightning_indexer_metadata_dsl/quant_lightning_indexer_metadata_dsl.py",
        "f432d48d24a2a0adf600bd6baf6b014ecb4fe319d52531128e6ff6d8e0ea8f38",
    ),
    "tests": (
        "test/quant_lightning_indexer_dsl/test_quant_lightning_indexer_dsl.py",
        "9f87c21711b9fd9e5f1431d40cc98ea5da17eb68acdd89cbacd664c6faf6f996",
    ),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(root, temporary, workers):
    loaded = {}
    # Register the upstream import helper without replacing PYTHONPATH.
    load_module(root / "test/_samples_path.py", "_samples_path")
    for role, (relative, expected_hash) in FILES.items():
        text = (root / relative).read_text()
        if role == "core":
            text = re.sub(
                r"^MAX_CUBE_WORKERS = \d+$", "MAX_CUBE_WORKERS = 32", text, flags=re.M
            )
        elif role == "metadata":
            text = re.sub(r"^    workers = \d+$", "    workers = 32", text, flags=re.M)
        if hashlib.sha256(text.encode()).hexdigest() != expected_hash:
            raise RuntimeError(f"{relative} does not match upstream {UPSTREAM_COMMIT}")
        if role == "core":
            text = text.replace(
                "MAX_CUBE_WORKERS = 32", f"MAX_CUBE_WORKERS = {workers}"
            )
        elif role == "metadata":
            text = text.replace("    workers = 32", f"    workers = {workers}")
        target = temporary / f"{role}.py"
        target.write_text(text)
        loaded[role] = load_module(target, f"_qli_evaluator_{role}")
    return loaded
