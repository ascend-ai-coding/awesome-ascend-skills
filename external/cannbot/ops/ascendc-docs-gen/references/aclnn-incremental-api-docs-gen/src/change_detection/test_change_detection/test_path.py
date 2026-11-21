# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""存在性检查草稿: 验证 addcmul 快照输出目录命名。

历史版本硬编码了本机绝对路径, 现改为经 CANN_DATA_PATH 环境变量提供
(指向 snapshot 输出根目录, 如 <workspace>/data/outputs/snapshot)。

注意: 历史路径示例中 operator_storage 的父目录名有两种拼写
(addmul_modified / addcmul_modified), 当时用于排查目录名不一致问题。
"""

import os
from pathlib import Path

snapshot_root = Path(os.environ.get("CANN_DATA_PATH", "")) / "addcmul"
if not snapshot_root:
    raise SystemExit("请设置 CANN_DATA_PATH 环境变量指向快照输出根目录")

# 历史上两种拼写并存: addmul_modified(疑似笔误) vs addcmul_modified(正确)
is_exist = (snapshot_root / "operator_storage" / "addmul_modified").exists()
print(is_exist)
is_exist_real = (snapshot_root / "operator_storage" / "addcmul_modified").exists()
print(is_exist_real)
