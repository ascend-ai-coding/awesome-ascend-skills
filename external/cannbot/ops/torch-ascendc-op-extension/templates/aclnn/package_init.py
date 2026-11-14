# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import warnings

import torch
import torch_npu

# 加载编译产物，触发 csrc 里 TORCH_LIBRARY / TORCH_LIBRARY_IMPL 的静态注册
from . import xops_lib

# 注册 FX->GE converter（图模式）。多算子时在这里逐个列全，漏掉的算子 eager 可用但入图报错。
# 不做图模式时删掉这一行，并删掉 converter/ 目录。
from .converter import npu_x_custom_op

# 把 torch.ops.custom.* 镜像到 torch_npu，使 torch_npu.npu_x_custom_op(...) 等价可用
custom_ops_module = getattr(torch.ops, 'custom', None)
if custom_ops_module is None:
    warnings.warn("torch.ops.custom not found, mount custom ops to torch_npu failed. "
                  "Please use torch.ops.custom.xxx instead of torch_npu.xxx.")
else:
    for op_name in dir(custom_ops_module):
        if not op_name.startswith('_'):
            setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))
