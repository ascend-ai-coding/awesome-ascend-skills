# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""torchair FX->GE converter 模板（图模式，不入图可不要本文件）。

文件名与算子名同名：xops/converter/npu_x_custom_op.py
占位符：npu_x_custom_op（torch 算子名）、XCustomOp（GE op_type，须与 OpDef 的 OP_ADD 完全一致）
"""
from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor


@register_fx_node_ge_converter(torch.ops.custom.npu_x_custom_op.default)
def convert_npu_x_custom_op(
    x1: Tensor,
    x2: Tensor,
    out: Tensor,
    *,
    meta_outputs: Any = None,
):
    # 函数签名必须与 schema 逐参对齐（含默认值），末尾额外接 *, meta_outputs=None
    # 三个 dict/list 的名字一律取自 OpDef，不是 aclnn 形参名：
    #   inputs  的 key    == OpDef 的 this->Input("...")
    #   attrs   的 key    == OpDef 的 this->Attr("...")，OpDef 中不存在的参数不要放进来
    #   outputs 的顺序    == OpDef 的 this->Output("...") 声明顺序
    # 可选输入为 None 时不要放进 inputs，否则 GE 建图报缺输入。
    inputs = {"x1": x1, "x2": x2}
    attrs = {}
    outputs = ["y"]
    return torchair.ge.custom_op("XCustomOp", inputs=inputs, attrs=attrs, outputs=outputs)
