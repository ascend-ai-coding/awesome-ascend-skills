#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""verify.py / benchmark.py 共用的工具函数，避免重复实现。"""
from typing import Any, Dict, List


def move_to_device(x: Any, device: Any) -> Any:
    """递归把 x 中的所有 torch.Tensor（含嵌套 list/tuple 内的）迁移到 device。

    - torch.Tensor: 直接 .to(device)
    - list: 递归迁移每个元素，保留 list 类型
    - tuple: 递归迁移每个元素，保留 tuple 类型
    - 其他（标量 / None）: 原样返回

    verify.py / benchmark.py 共用，避免在两个脚本中重复实现递归迁移逻辑。
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [move_to_device(e, device) for e in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(e, device) for e in x)
    return x


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    try:
        import torch
    except Exception:
        torch = None

    descs: List[Dict[str, Any]] = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs
