# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU FP64 autograd reference for the unrounded matrix chain."""

import torch


def reference(a, b, c, upstream):
    operands = [value.detach().cpu().double().requires_grad_() for value in (a, b, c)]
    output = torch.softmax(operands[0] @ operands[1], dim=1) @ operands[2]
    gradients = torch.autograd.grad(output, operands, upstream.detach().cpu().double())
    return output.detach(), tuple(value.detach() for value in gradients)


def scalar_objective(a, b, c, upstream):
    return ((torch.softmax(a @ b, dim=1) @ c) * upstream).sum()
