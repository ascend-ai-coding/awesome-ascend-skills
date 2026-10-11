#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""wm_log.py — 世界模型脚本共享的数据输出 logger。

DATA_LOGGER 专用于 CLI 数据输出（JSON/表格/验证报告），输出到 stdout
（agent 调用协议通道），与各模块自身的 LOGGER（stderr 进度/警告）分离，
避免 lint G.LOG.02 误报 print。wm_ops / wm_audit 共用此定义，消除重复代码。
"""

import logging
import sys

DATA_LOGGER = logging.getLogger("wm.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)
