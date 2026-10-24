# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""有效范围：算一片比 Tile 小的区域，以及它和 ``compact`` 的强制关系。

**解决什么.** 真实形状很少刚好是分块大小的整数倍。最后一块只有半行、半列或半段
归约——但 Tile 是按整块声明的。``pl.gemm`` 的 ``m`` / ``n`` / ``k`` 让这一次调用
只算 Tile 里的一片，**声明不变、搬运不变**，变的只有这一行。

五个 case 按「收窄哪一维」铺开：

| case | 参数 | 被收窄的 Tile | 输出的有效区域 |
|---|---|---|---|
| `m` | `m=72` | `l0a`、`acc` | `[:72, :]` |
| `n` | `n=88` | `l0b`、`acc` | `[:, :88]` |
| `k` | `k=104` | `l0a`、`l0b` | **整块** |
| `mnk` | 三个一起 | 三块都是 | `[:72, :88]` |
| `odd` | `37 / 23 / 19` | 三块都是 | `[:37, :23]` |

**这张表就是那条规则**，它不对称，也不能靠直觉补：

```
m 收窄 l0a 和 acc          n 收窄 l0b 和 acc          k 收窄 l0a 和 l0b
```

``k`` 那一行是唯一一个不碰累加器的——K 是被归约掉的那一维，不出现在输出形状里，
所以 ``k`` case 的累加器**不该**声明 compact，也确实没声明。反过来，``m`` 和
``n`` 都要碰累加器：只盯着被切维度对应的 L0 会漏掉它。

**为什么必须 compact.** matmul 永远按 compact 布局取操作数、留结果。``compact=1``
让搬进 L0 的那一步按**收窄后的步长**写，也让累加器搬出去的那一步按它读。少了它，
数据保持整块的物理布局，而 matmul 按有效范围走——两边对不上元素在哪。

漏掉会**编译期失败，不会静默算错**：

```text
pl.gemm() needs 'l0a' declared with compact=1 to take an effective extent
```

这条检查存在的全部理由就是「静默」两个字。框架侧在 A5 上用 64x64 的 Tile 算
48x40 实测过四种组合（记在 ``_gemm_lowering.py`` 的 ``_gemm_require_compact``）：
两块都不 compact 错约 50，只有累加器 compact 错约 47，只有 L0 暂存 compact 错约
62，两块都有才对到 1.7e-05。**四种情况都没有越界**，错的全在范围之内——所以任何
「看看有没有写出界」的事后检查都拦不住它。

**物理访问.** 有效范围不改变任何一块 Tile 的 shape，也不改变 ``pl.load`` 从 GM
搬多少：L1 上仍然是完整的 ``[M, K]`` 和 ``[K, N]``。收窄只发生在 L1 → L0 的搬运
和那条矩阵乘指令上。要少搬 GM 是 ``pl.load`` 的偏移与形状的事，和这里无关——两件
事经常被混成一件。

``compact`` 本身只是布局属性，**不改变 Tile 占多少字节**：槽位大小算的是
``prod(shape) * bits / 8``，与 compact 无关。所以加不加 compact 不影响地址规划。

**为何这样同步.** 五个显式地址组使用 ``mutex_ids="auto"``，由 ``auto_mutex`` 插入同步。有效范围不产生
新的 Tile 也不改变依赖关系——它只是让同一条 move / matmul 少碰一些字节，所以同步
与整块算时完全一样。

**如何独立验证.** 每个 case 比两件事，**缺一不可**：

1. 有效范围内的值与参考实现一致；
2. 有效范围**之外**一个非零都没有。

只比第 1 条会放过「把邻居的缓冲涂掉了」；只比第 2 条会放过「边界对、里面的值全错
位」。而这正是实测到的失败形态——上游 ST 的原话是「每一种配错有效范围的情况都保住
了边界、错的是里面的值」。所以两条都留着。

``k`` case 没有第 2 条可比（输出是整块的），它比的是**整片**结果与「只用 A 的前
104 列、B 的前 104 行」算出来的结果一致——如果 gemm 偷偷按整个 K 归约了，这一条
会失败。

四个 case 源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_gemm_matrix_st.py``
的 ``g_n`` / ``g_k`` / ``g_mnk`` / ``g_odd``。``m`` 单独一个 case **上游没有**
（那里 ``m=EM`` 只和 splitk、trans_a 组合出现），是本样例新加的，已上板验证。
"""

import pytest
import torch

from .kernel import EM, EN, K, M, N, OM, ON
from .kernel import tail_k
from .kernel import tail_m
from .kernel import tail_mnk
from .kernel import tail_n
from .kernel import tail_odd
from .reference import reference_k
from .reference import reference_m
from .reference import reference_mnk
from .reference import reference_n
from .reference import reference_odd

# case 名 -> (kernel, reference, 有效行数, 有效列数)。None 表示这一维没被收窄。
CASES = {
    "m": (tail_m, reference_m, EM, None),
    "n": (tail_n, reference_n, None, EN),
    "k": (tail_k, reference_k, None, None),
    "mnk": (tail_mnk, reference_mnk, EM, EN),
    "odd": (tail_odd, reference_odd, OM, ON),
}


def _run(kernel, a, b):
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    return out


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
def test_extent_computes_its_region_and_touches_nothing_else(case, device):
    kernel, reference, rows, cols = CASES[case]
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    out = _run(kernel, a, b)
    expected = reference(a, b)

    # 1. 范围内的值。
    torch.testing.assert_close(
        out[: expected.shape[0], : expected.shape[1]], expected, rtol=2e-2, atol=2e-2
    )

    # 2. 范围外一个非零都没有。out 是零初始化的，所以非零只可能是 kernel 写的。
    if rows is not None:
        assert torch.count_nonzero(out[rows:]) == 0, f"{case} 写到了第 {rows} 行之外"
    if cols is not None:
        assert torch.count_nonzero(out[:, cols:]) == 0, f"{case} 写到了第 {cols} 列之外"


def test_k_really_shortens_the_reduction(device):
    """``k`` case 的输出是整块的，「范围外为零」这条用不上，换一条。

    要排除的错法是「gemm 忽略了 ``k``，照整个 K 归约」。上面那条 parametrize
    已经能抓住它（比的是截断后的参考，多归约 24 项不可能还落在 2e-2 里），
    但那是间接的。这里直接要求：收窄后的结果与按整个 K 算的结果**必须不同**。
    """
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    out = _run(tail_k, a, b)
    full_k = torch.matmul(a.float(), b.float())

    assert not torch.allclose(out, full_k, rtol=2e-2, atol=2e-2), (
        "收窄 K 之后的结果与按整个 K 归约的结果一致，说明 k 没有生效"
    )


def test_the_compact_rule_is_not_symmetric():
    """把规则本身钉住：``k`` 的累加器不 compact，``m`` / ``n`` 的必须 compact。

    这条不上板，读的是源码。规则写在注释里会随注释腐烂，写成断言不会。
    """
    import ast
    import inspect

    from .kernel import k as k_mod
    from .kernel import m as m_mod
    from .kernel import n as n_mod

    def _compact_blocks(mod):
        """返回源码里带 compact=1 的那几块的变量名。"""
        blocks = set()
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if ast.unparse(node.value.func) != "pl.make_tile_group":
                continue
            tile_type = next(kw.value for kw in node.value.keywords if kw.arg == "type")
            if any(
                kw.arg == "compact"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == 1
                for kw in tile_type.keywords
            ):
                blocks.add(node.targets[0].id)
        return blocks

    assert _compact_blocks(m_mod) == {"l0a", "acc"}
    assert _compact_blocks(n_mod) == {"l0b", "acc"}
    assert _compact_blocks(k_mod) == {"l0a", "l0b"}
