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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script_module(
    script_path: Path,
    module_name: str,
    *,
    remove_after_load: bool,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load bench runtime helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script_path.parent)
    added = script_dir not in sys.path
    if added:
        sys.path.insert(0, script_dir)
    sys.modules[module_name] = module
    discard_module = remove_after_load
    try:
        spec.loader.exec_module(module)
    except Exception:
        discard_module = True
        raise
    finally:
        if discard_module:
            sys.modules.pop(module_name, None)
        if added:
            sys.path.remove(script_dir)
    return module
