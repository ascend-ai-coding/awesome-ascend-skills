# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""选卡与 A5 门槛，所有样例共用。

用 pytest 的 conftest 而不是样例库自己的 runner：样例保持为普通 pytest 文件，
``python -m pytest examples/matmul/<name>/test.py`` 就是全部用法。卡号取自
``TILE_FWK_DEVICE_ID``——框架自己的系统测试读的也是这个变量，所以能跑那些测试的
环境就能跑这里的样例。
"""

from __future__ import annotations

import os

import pytest

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


@pytest.fixture(scope="session")
def device() -> str:
    """要跑的 A5 卡；没有就 skip。

    是 skip 不是 fail：样例是「碰巧能执行的文档」，没有卡的机器至少应该还能把它
    收集起来、把说明读完。
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    try:
        torch.npu.set_device(DEVICE)
    except (RuntimeError, AttributeError) as exc:
        pytest.skip(f"没有可用的 NPU：{exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"这些样例面向 A5，当前卡是 {name}")
    return DEVICE
