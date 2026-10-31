# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""数据在两种核之间往返：三个事件、四种 pipe。

**解决什么.** 这是手写跨核事件的双向交接：vector 算完的东西要交回 cube。这里算
``out = ((Q @ K) + X1) @ V + X2``：

```
cube    mm1 = Q @ K            --事件 0-->  vector
vector  v1  = mm1 + X1         --事件 2-->  cube        <-- 反方向
cube    mm2 = v1 @ V           --事件 1-->  vector
vector  out = mm2 + X2
```

三块跨核共享的 Tile 声明在**两个 section 之外**（`mm1_ub`、`mm2_ub`、`v1_l1`），
各自 section 内部的 Tile 声明在自己的 section 里。这是跨核共享 buffer 的惯例位置。

**为何这样同步：三个事件，按每条边选择写入和读取的 pipe.**

| 事件 | 方向 | 生产侧 `set` | 为什么 | 消费侧 `wait` | 为什么 |
|---|---|---|---|---|---|
| 0 | cube → vec | **FIX** | `move(mm1_ub, c1)` 源在 `Acc` | **V** | 本侧第一个读它的是 VF `_add` |
| 2 | **vec → cube** | **MTE3** | `insert(v1_l1, nz_ub)` 是 UB → L1 | **MTE1** | 本侧第一个碰它的是 `move`（Mat → Left） |
| 1 | cube → vec | **FIX** | `move(mm2_ub, c2)` 源在 `Acc` | **V** | 同事件 0 |

规则始终是同一条：**生产侧填最后一个写它的 op 的 pipe，消费侧填第一个读它的 op 的
pipe。** 这张表就是这条规则在三条交接边上的展开——记住规则比记住表有用，因为换一种
数据通路，pipe 就跟着变。

**事件 2 是反方向的那一个。** 这里 vector 产出的 `v1_l1` 要交给 cube 消费，
是完整的第二个方向。它和
「缓冲复用需要的反向同步」是**同一个机制的两种用法**：一个是交数据，一个是还额度。

**事件 0 的消费侧那个 `V` 是本样例最容易写错的地方。** ``wait_cross_core`` **只挡它
点名的那条队列**。`mm1_ub` 在 vector 侧第一个被读取的位置是 VF `_add`，跑在 V 上；如果填成
MTE3（"我要等搬运完成"这种直觉），`add` 照跑，读到的是搬运还没落盘的字节。

本工作区实测踩过一次：最大差恰好等于漏掉的那一项（0.5），IR 里 `@block.adds` 和它的
mutex 全都在，**看不出任何异常**。

**核内那一层交给了 `auto_mutex`.** 显式地址组使用 `mutex_ids="auto", depth=1`，框架规划 ID 并插入核内同步。
手写同步是三个事件各自的 set/wait，共六次调用。

上游 ST 是 `@pl.jit()` 不开 auto_mutex 的手工同步版，每两个 op 之间还要成对写
`sync_src` / `sync_dst`——那是另一个主题（核内流水），在这里只会盖过跨核这条线。

**两侧独立规划 mutex ID.** Cube 和 Vector 编译成独立的二进制，各有独立的 32 个 ID 空间。同一数值可在两侧复用；共享组在两侧必须保持一致的物理地址和跨核事件契约，不要求 mutex ID 数值相同。

**`DualModeSplitM` 与 `insert`.** 两个 AIV 各处理一半 M 行，所以：

- cube 搬出累加器时用 `acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM`，把 `[64, 64]`
  按 M 劈成两半分别发给两个 AIV，于是 vector 侧的 Tile 是 `[32, 64]`；
- vector 送回 L1 时用 `pl.insert(v1_l1, nz_ub, [off, 0])`，各写 `v1_l1` 的一半；
- 送回之前要先 `pl.move(nz_ub, res_ub)` 做 **ND → NZ**——L1 上要当矩阵乘的左操作数，
  必须是 NZ。

**如何独立验证.** 四条：

1. **最终结果**与参考实现比。
2. **第一趟的中间量**单独验：把 `x2` 置零、`v` 置成单位阵，则 `out` 就等于
   `(Q@K) + X1`。这样能把「事件 0 或事件 2 坏了」和「事件 1 坏了」分开——只比最终结果
   的话，中间哪一环坏了都是同一个现象。
3. **跑两遍逐位相同。** 跨核同步不足的表现是不稳定，单跑一遍可能恰好是对的。
4. **两个 AIV 各写了自己那一半。**

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_add_matmul_add.py``
的 ``matmul_add_matmul_add``（那里是手工核内同步版）。多次交接与缓冲复用的更大例子见
同目录的 ``test_matmul_add_buffer_manager.py``。

**自动插同步那条路.** 这三条交接边的事件也可以不手写：把 kernel 拆成 `@pl.pipeline.stage`
函数、给跨核 Tile 声明 `fwd_ids` / `bwd_ids`、`pl.jit(pipeline=PipelineConfig(preload=0))`，
框架会自己插。代价是要满足 stage 的结构约束。两条路怎么选见
[api/crosscore-contract.md](../../../../references/api/crosscore-contract.md)。
"""

import torch

from .kernel import SIZE, matmul_add_chain
from .reference import chain_reference, mm1_reference


def _run(q, k, v, x1, x2):
    out = torch.zeros([SIZE, SIZE], device=q.device, dtype=torch.float32)
    matmul_add_chain(q, k, v, x1, x2, out)
    torch.npu.synchronize()
    return out


def _inputs(device, seed=0):
    torch.manual_seed(seed)

    def g():
        return torch.randn([SIZE, SIZE], device=device, dtype=torch.float32) * 0.1

    return g(), g(), g(), g(), g()


def test_the_whole_chain(device):
    q, k, v, x1, x2 = _inputs(device)

    out = _run(q, k, v, x1, x2)

    torch.testing.assert_close(
        out, chain_reference(q, k, v, x1, x2), rtol=1e-2, atol=1e-2
    )


def test_the_first_handover_alone(device):
    """把 V 置成单位阵、X2 置零，out 就等于第一趟的结果 (Q@K) + X1。

    只比最终结果的话，事件 0 / 2 / 1 哪一个坏了都是同一个现象。这条把前两个
    事件单独隔出来。
    """
    q, k, _, x1, _ = _inputs(device, seed=1)
    eye = torch.eye(SIZE, device=device, dtype=torch.float32)
    zero = torch.zeros([SIZE, SIZE], device=device, dtype=torch.float32)

    out = _run(q, k, eye, x1, zero)

    torch.testing.assert_close(out, mm1_reference(q, k, x1), rtol=1e-2, atol=1e-2)


def test_two_runs_agree_bit_for_bit(device):
    """跨核同步不足的表现是不稳定，单跑一遍可能恰好是对的。"""
    q, k, v, x1, x2 = _inputs(device, seed=2)

    first = _run(q, k, v, x1, x2)
    second = _run(q, k, v, x1, x2)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_both_aivs_wrote_their_half(device):
    """漏掉 get_subblock_idx 的话有一半是零。"""
    q, k, v, x1, x2 = _inputs(device, seed=3)

    out = _run(q, k, v, x1, x2)

    half = SIZE // 2
    assert torch.count_nonzero(out[:half]) > 0, "AIV0 负责的上半没有写"
    assert torch.count_nonzero(out[half:]) > 0, "AIV1 负责的下半没有写"
