# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent integer ordering; no floating-point comparison keys."""

import torch


def reference(indices):
    if indices.dtype != torch.int32 or indices.ndim != 2 or indices.shape[1] != 512:
        raise ValueError("expected INT32 [rows,512]")
    if bool(((indices < 0) | (indices > 2**24)).any()):
        raise ValueError("the exact FP32-key domain is [0,2**24]")
    return torch.sort(indices, dim=-1).values
