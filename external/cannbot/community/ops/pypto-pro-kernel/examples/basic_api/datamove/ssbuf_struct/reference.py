# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SSBUF 交接后 GM 上该是什么。

**为什么可以 `rtol=0, atol=0` 比。** 全程只搬整数标量，一个算术都没有。用 INT32 而不是浮点:
这一例判的是"字段有没有串"，整数能一眼认出串到了哪个字段。

三个值 `8 / 1 / 32768` 互不相同、都不是 0，也互不为倍数——**串了能定位到具体哪个槽**。
如果三个值取成 `1, 2, 3`，串成 `[1, 2, 2]` 和 `[1, 2, 3]` 差一眼看不出。
"""

SSBUF_BYTES_PER_FIELD = 8
"""每个字段在 SSBUF 里占的字节数。**实测值，不是推测**：

`offset` 的单位（字节）和这个槽宽都没有官方文档。模型由 5 个偏移的实测结果拟合、
再用另一个字段数的 4 个偏移**预测验证**，7 个配置逐位吻合。证据表在 `test.py` §四。
"""


def spacing(field_count):
    """`field_count` 个字段的 struct 在 SSBUF 里占多少字节 = 第二个 struct 的最小偏移。"""
    return SSBUF_BYTES_PER_FIELD * field_count


def clobbered_at(first, second, slot_index):
    """第二个 struct 压在第一个的第 `slot_index` 个槽上时，第一个读回来是什么。

    从 `slot_index` 起，第一个 struct 的字段被第二个的字段**按顺序**顶掉——因为两者的槽宽相同,
    偏移又正好对齐在槽边界上。这个函数就是那条规则的形式化。
    """
    out = list(first)
    for i in range(slot_index, len(out)):
        out[i] = second[i - slot_index]
    return out
