# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mathematical values exposed by the debugging probes.

Copy probes preserve each input coordinate. The VF register probe multiplies
each coordinate by 100 before publishing its tile for dump_data. Log counts,
assertion messages and trap behavior are checked separately in test.py.
"""


def reference_copy(values):
    return values.clone()


def reference_register(values):
    return values * 100.0
