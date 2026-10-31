# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""任意 M/N/K：一个编译产物服务所有形状。

**解决什么.** 形状写死在注解里，每换一组 M/N/K 就是一份新的二进制。`pl.DYNAMIC` 让 GM 的
尺寸留到运行期，从张量上读；Tile 的**容量**仍然是编译期常量（cube 指令要求如此）。判据只有
一条：**喂进去好几种形状，而编译缓存里只有一份**。

**和 `tail_extent` 的界线.** 两者都在处理"最后一块不够整块"，但形状什么时候确定不同：

| 样例 | 覆盖 | 形状什么时候确定 |
| --- | --- | --- |
| [`tail_extent`](../../tail_extent/test.py) | `pl.gemm(m=/n=/k=)` 收窄，和 `compact` 的强制关系 | **解析期常量** |
| 本样例 | `pl.DYNAMIC` + `out.shape[0]`，一次编译多种形状 | **运行期** |

**用 `pl.gemm` 写，不用 `phase`.** 同样的活手写 `pl.matmul` 要多三样，这里都省了：

| | 手写 `pl.matmul` | `pl.gemm` |
| --- | --- | --- |
| L0 暂存 | 每个 K 步两条 `pl.move` | gemm 内部做 |
| 收窄 | 每个 K 步 `set_validshape` × 4~8（L1、L0、acc 各自设，容易各说各话） | gemm 给**参与的五块**一起设；只剩 L1 那两条——docstring 明说故意不碰 L1 |
| K 累加 | `matmul` / `matmul_acc` + `phase` 的多臂分支 | `init=True/False`，**完全不用 `phase`** |

去掉 `phase` 不是省事：`pl.matmul` 的 `phase` 本来就是 `Optional[...] = None`，而
`_gemm_lowering.py` 里 `phase` 出现 **0 次**（`init=True` → `matmul`，`init=False` →
`matmul_acc`）。框架自己也有一份无 `phase` 的多轮 M/N/K 上板回归
（`test_matmul_hf8_fp32.py` 的 `make_matmul_no_phase`）。

**两个写法约束**（都查过实现，不是试出来的）：

1. **`init` 必须是解析期的 `bool`**——lowering 里是 `def matmul_stmt(init_step: bool)`。
   所以写不出 `init=(ki == 0)`，K 循环里要一个两臂 `if`。两臂做的搬运完全一样，
   ping-pong 是平衡的（ASW 那份 kernel 警告过的是**条件跳过**某个子块，那才会让
   `auto_mutex` 在 K 尾上死锁）。
2. **`splitk` 用不了**——它把 `k` 变成循环上界，而那必须是编译期值。动态 K 的循环只能自己
   写。`splitn` 同理，而且它和 `m` 互斥。

反过来，`m` / `n` / `k` 这三个收窄参数**明确接受运行期值**（docstring：*"An extent may be
a runtime value, unlike a split size"*），这正是动态形状需要的那一半。

**`compact=1` 是强制的，而且是拒绝而不是静默.** 被 `m`/`n`/`k` 收窄的每块 tile 都要
`compact=1`，否则 `pl.gemm() needs '<name>' declared with compact=1 ...`。挡掉的错误值得看
一眼——`_gemm_require_compact` 的 docstring 记了实测（Ascend950PR，64×64 Tile 里放 48×40）：

```
两块都不 compact          结果错了约 50
只有累加器 compact        约 47
只有 L0 暂存 compact      约 62
两块都 compact            对到 1.7e-05
```

**每种情形下边界都是被尊重的，范围外什么都没被写**——只是范围内的值放错了位置，所以后面
没有任何检查能抓到。这也是为什么 gemm 宁可在解析期拒掉。

收窄的分工不对称，也不能靠直觉补：`m` 收窄 `l0a` 和 `acc`，`n` 收窄 `l0b` 和 `acc`，
`k` 收窄 `l0a` 和 `l0b`（K 被归约掉，不出现在输出形状里，所以不碰 `acc`）。gemm 把这张表
实现成一处，这是它比手写更值的地方。

**如何独立验证.** 四条：

1. **一份编译产物**——六种形状跑完，`len(kernel._compiled_by_signature) == 1`。这条是这个
   样例的重点：少了它，"支持动态形状"和"每种形状各编一份"看起来一样。
2. **六种形状各自与参考比**，含三个方向单独的尾块、三个方向一起的尾块，以及每个方向都只有
   一个不完整块的极小形状。
3. **IR 里没有 `phase`**，而且 `matmul` 和 `matmul_acc` 都在——钉住 `init` 的两条路都走到了。
4. **同一形状跑两遍逐位相同。**

来源：`pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py`
（281 行，两级 K 循环 + `phase` + 手写 `move`）和 `.../test_matmul_mx.py`（全动态，但混着
MX）。这里换成 `pl.gemm` 重写，把两级 K 循环留给以后单独的性能样例——那讲的是性能结构，
不是动态形状。
"""

import pytest
import torch

from .kernel import matmul_dynamic
from .reference import matmul_reference

# (M, N, K)。TILE=64，所以「尾块」就是不被 64 整除的那些。
SHAPES = [
    ("exact", (128, 128, 128)),
    ("m_tail", (100, 128, 128)),
    ("n_tail", (128, 100, 128)),
    ("k_tail", (128, 128, 100)),
    ("all_tails", (100, 76, 52)),
    ("one_partial_block_each", (37, 23, 19)),
]


def _run(device, m, n, k, seed=0):
    torch.manual_seed(seed)
    a = torch.rand([m, k], device=device, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([k, n], device=device, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([m, n], device=device, dtype=torch.float32)
    matmul_dynamic(a, b, out)
    torch.npu.synchronize()
    return out.cpu(), a.cpu(), b.cpu()


def test_the_ir_has_no_phase_and_both_init_paths():
    """不需要卡。钉住 gemm 降级出来的东西。"""
    import pypto_pro as pp

    program, _ = matmul_dynamic.to_kernel_def().parse_target_program(
        pp.ir.SectionKind.Cube
    )
    text = pp.ir.python_print(program)

    assert "phase" not in text.lower(), "gemm 不该产出 phase"
    assert "matmul_acc" in text, "init=False 那一臂应当降成 matmul_acc"
    # matmul_acc 里含 'matmul'，所以要数「不是 matmul_acc 的 matmul」。
    plain = text.count("matmul") - text.count("matmul_acc")
    assert plain > 0, "init=True 那一臂应当降成 matmul"


@pytest.mark.parametrize("mnk", [s[1] for s in SHAPES], ids=[s[0] for s in SHAPES])
def test_matches_the_reference(device, mnk):
    """每种形状各自与参考比。"""
    m, n, k = mnk

    out, a, b = _run(device, m, n, k)

    assert out.shape == (m, n)
    torch.testing.assert_close(out, matmul_reference(a, b), rtol=1e-2, atol=1e-2)


def test_one_compiled_kernel_serves_every_shape(device):
    """这条是重点：六种形状跑完，编译缓存里只有一份。

    少了它，「支持动态形状」和「每种形状各编一份」在测试里看起来一样——两者都会通过上面
    那组比对。
    """
    for _, (m, n, k) in SHAPES:
        _run(device, m, n, k)

    variants = len(matmul_dynamic._compiled_by_signature)
    assert variants == 1, (
        f"{len(SHAPES)} 种形状应当共用一份编译产物，实际编了 {variants} 份——"
        "说明有形状漏进了静态签名"
    )


def test_two_runs_agree_bit_for_bit(device):
    """同一形状跑两遍，逐位相同。"""
    m, n, k = SHAPES[-2][1]

    first, _, _ = _run(device, m, n, k, seed=3)
    second, _, _ = _run(device, m, n, k, seed=3)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
