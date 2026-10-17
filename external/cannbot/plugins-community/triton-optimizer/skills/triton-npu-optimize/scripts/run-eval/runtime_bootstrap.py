# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
import os
import sys


_TORCH_BACKEND_AUTOLOAD_ENV = "TORCH_DEVICE_BACKEND_AUTOLOAD"


def bootstrap_torch_npu() -> None:
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None and hasattr(loaded_torch, "npu"):
        return

    previous = os.environ.get(_TORCH_BACKEND_AUTOLOAD_ENV)
    os.environ[_TORCH_BACKEND_AUTOLOAD_ENV] = "0"
    try:
        importlib.import_module("torch")
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            pass
    finally:
        if previous is None:
            os.environ.pop(_TORCH_BACKEND_AUTOLOAD_ENV, None)
        else:
            os.environ[_TORCH_BACKEND_AUTOLOAD_ENV] = previous
