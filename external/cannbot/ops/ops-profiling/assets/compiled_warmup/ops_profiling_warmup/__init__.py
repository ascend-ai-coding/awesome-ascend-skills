# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Project-local compiled warmup used by the ops-profiling runner."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401 - initialize the PrivateUse1 backend

from . import _C

WARMUP_SHAPE = (128, 1024, 1024)
WARMUP_BYTES = 256 * 1024 * 1024
KERNEL_NAME = "OpsProfilingWarmup"

torch.library.define("ops_profiling::compiled_warmup", "(Tensor x) -> Tensor")


@torch.library.impl("ops_profiling::compiled_warmup", "PrivateUse1")
def _compiled_warmup_npu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((), dtype=x.dtype, device=x.device)
    _C.compiled_warmup_npu(x, out)
    return out


@torch.library.impl("ops_profiling::compiled_warmup", "Meta")
def _compiled_warmup_meta(x: torch.Tensor) -> torch.Tensor:
    return _C.compiled_warmup_meta(x)


def make_warmup_tensor(device: object) -> torch.Tensor:
    """Allocate the exact reusable tensor required by the profiling protocol."""
    value = torch.empty(WARMUP_SHAPE, dtype=torch.float16, device=device)
    if value.numel() * value.element_size() != WARMUP_BYTES:
        raise RuntimeError("compiled warmup tensor must be exactly 256 MiB")
    return value


def compiled_warmup(value: torch.Tensor) -> torch.Tensor:
    """Run one full-device read/add/write pass over the reusable tensor."""
    return torch.ops.ops_profiling.compiled_warmup(value)


__all__ = [
    "KERNEL_NAME",
    "WARMUP_BYTES",
    "WARMUP_SHAPE",
    "compiled_warmup",
    "make_warmup_tensor",
]
