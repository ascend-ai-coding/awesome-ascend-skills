# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""kernel 入口、循环、cube 段与自动流水。

**写法：原始。**

## 一、五个东西

| API | 是什么 | ST |
| --- | --- | --- |
| ``@pl.jit(...)`` | **唯一**的 kernel 入口 | 到处都是 |
| ``pl.range(...)`` | kernel 体里的循环迭代器 | `control_flow`:385 |
| ``pl.section_cube()`` | cube 流水的 section | `matmul`:495 |
| ``pl.pipeline`` | `@pl.pipeline.stage` + `PipelineConfig(preload=n)` | **75 次**，见 §五 |
| ``@pl.inline`` | **已废弃**，只报警告后原样返回 | ut 里 1 条，断言它报警告 |

## 二、如何独立验证

1. ``loop_over_blocks``：第 k 块乘 `k + 1`，四块全对。
2. ``loop_with_a_step``：`pl.range(0, 4, 2)` 只碰 0、2 号块，1、3 号仍是哨兵。
3. ``loop_with_python_range``：内置 `range` 被解析期拒。
4. ``make_pipelined(p)``：`p` 取 0/1/2/3，**四个结果两两逐位相同**，且都等于参考。
5. ``cube_section_moves_through_l1``：混合 kernel 跑通，vector 段的搬运对得上。
6. ``make_pipelined_with_an_aliased_memory_space``：跨核 buffer 的 `target_memory`
   写成别名被拒。
7. ``PipelineConfig`` 的四种非法 `preload`：`-1`、`1.5`、`True`、`()`。
8. ``@pl.inline`` 报 `DeprecationWarning`，**而它指的替代品 `pl.fn` 不存在**。

判据 1 乘的是**轮次号**而不是常数：常数的话循环跑错次数、跑错顺序都看不出来。
判据 2 靠哨兵——不填哨兵只能证「碰到的对」，证不到「没碰的没动」。

## 三、`preload` 换了结果必须一样（判据 4）

| `preload` | 语义 | 实测 |
| ---: | --- | --- |
| 0 | 不提前拉任何 stage，串行 + 跨核同步 | 对 |
| 1 / 2 / 3 | 上游提前 n 轮 | 对，且**和 0 逐位相同** |

`config.py` 的 docstring 说「Zero pulls no stage ahead of any other……Start there to confirm
the serial kernel is correct, then raise it」。**判据 4 把这句话变成一条会红的测试**:
四个 `preload` 的结果两两逐位相同。

为什么这是这一例最有分量的判据：它验的不是某个 API 的返回值，而是**流水变换不改变语义**——
而那正是整个 `pl.pipeline` 存在的前提。逐位相同而不是 `allclose`：`preload` 改的是指令排布,
不是算术，有任何数值差都说明语义被改了。

`preload=3` 也在里面（`NT` 只有 4）：它让填充和排空占掉大半个循环,
是边界最紧的那一档。

流水 stage 里的 `scale` 是 kernel 体按轮次算的（`scale = ki + 2`），
所以判据同时在验「第 k 轮看到的是为第 k 轮算的那个 scale」，不是别的节拍的。

## 四、跨核 buffer 的 `target_memory` 必须是字面量

```
pipeline: cross-core buffer 'shared' has no resolvable memory space. Its
make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X)) must set
target_memory to a literal pl.MemorySpace.<X>.
```

`_cross_core_scanner.py:775`。**写这一例的探针时四个 `preload` 因此全红**,
因为探针顶上有 `MAT = pl.MemorySpace.Mat` 这样的别名——而本仓每个样例顶上都有这类别名,
所以这是个会反复踩的坑。

**这是本仓第三次撞到「检查只认 AST 字面量」**：

| 位置 | 症状 |
| --- | --- |
| `pl.gemm_mx` 的 `k % 64` 守卫 | 写成常量名**绕过**检查（假阴性） |
| `pl.const` 的第一个参数 | 常量名**被拒**（假阳性，见 `support/scalar_const` §六） |
| 跨核 buffer 的 `target_memory` | 别名**被拒**（假阳性） |

同一个实现习惯（`isinstance(node, ast.Constant)` / 读 AST 而不是求值），两种相反的后果。

## 五、`pl.pipeline` 不是「没 ST 的 API」——那是数法假象

`references/tooling/example-coverage.md` §五 原来把 `pl.pipeline` 列进「12 个 API 框架自己也没测过」,
数法是 `grep -rho 'pl\.pipeline('`。**那个数法对模块不成立**：`pl.pipeline` 是
`pypto_pro.runtime.pipeline` 这个**模块**，从来不被当函数调用。

| 搜法 | 命中 |
| --- | ---: |
| `pl.pipeline(` | **0** |
| `pl.pipeline.` | **75**（`stage` 40、`PipelineConfig` 35） |

`st/.../auto_pipeline/` 整整一组十来个文件都在用它。这一例照它们抄。

**[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 那条「同一侧换名字搜也算一次独立搜索」在这里又对了一次**,
而且这次的变形不是前缀后缀，是「它是模块不是函数」。同一次复核还改掉了 `pl.inline`
那条（4 次 ut 用量，见 §六），留下 `pl.get_spr` 那条（见 `support/spr_ctrl` §六:
名字相近的 ST 其实在测 `vf.get_mask_spr`）。**三条里两条是数错的。**

## 六、`@pl.inline` 已废弃，而它指的替代品不存在

```
@pl.inline is deprecated and will be removed.
Use @pl.fn with type annotations, or pass the function as an annotated callable.
```

`decorator.py:30`。装饰器本身是个 no-op：报完警告原样返回 `fn`。

**`pl.fn` 不存在**（2026-09-28 核）：

| 搜法 | 结果 |
| --- | --- |
| `pl.__all__` 198 项里有 `fn` 吗 | **没有**；f 开头的只有 `fillpad` / `fused_mul_add` / `fused_mul_add_relu` / `fill_index` |
| `_api.py` 里有 `def fn` 吗 | 没有 |
| 整个树里 `pl.fn` 出现在哪 | **只有那条废弃消息自己**（`decorator.py:31`） |
| C++ 侧、`docs/` 侧 | 无 |

所以跟着这条消息走会拿到 `AttributeError`。消息的后半句（「pass the function as an
annotated callable」）是真的——普通函数在 kernel 体里调用就是现在的写法,
本仓每个样例的 `_tile_type()` 之类辅助函数都是。**按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)，这条落在
`pypto/`**:
废弃消息指向一个不存在的 API 是基建问题。

**已在 `pypto/` 侧改掉**（`dev/pro-agentkit-build`）：新消息是

```
@pl.inline is deprecated and will be removed; it is already a no-op.
Drop the decorator: a plain function called from a @pl.jit kernel body is inlined
by the parser. Use @pl.vector_function for a vf/SIMT body.
```

并在 `test_decorator.py` 补了两条 UT：一条断言消息里不再出现 `@pl.fn`、
提到的名字都真实存在，一条断言装饰器是 no-op。

判据 8 只断言消息**以「@pl.inline is deprecated」开头**、以及 `pl.fn` 不存在,
所以改完之后它照样绿（实测）。**故意不断言整句**：整句会随措辞变,
而这条判据要守的是「不指向不存在的 API」这个性质。

## 七、这一例不讲什么

- **`pl.jit` 的全部参数**——`arch=` / `auto_mutex=` / `tiling_key` / `compile_config` 那一堆。
  这里只用到 `auto_mutex=` 和 `pipeline=`。`auto_mutex` 的边界在
  `references/diagnostics/validation-traps.md` #7 和 `scripts/_checks/synchronisation.py`。
- **跨核同步的语义**——`fwd_ids` / `bwd_ids` 传给 `make_tile_group` 之后流水自己排同步,
  手工的那套在 `datamove/move_sync` 和 `references/api/crosscore-contract.md`。
- **`preload` 的性能效果**——**只验了数值不变，没测快了多少**。要测得先证明仪器能分辨
  （`references/verification/board-runs.md` §七），而这一例的 kernel 小到流水填充占大头,
  量出来的数说明不了什么。**这是明说跳过。**
- **多个流水循环时 `preload` 给元组**——`PipelineConfig` 支持一个循环一个值,
  这一例只有一个循环，**没测**元组形式的分配规则。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 四节。
"""

import warnings

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro._errors import InvalidArgument, InvalidVal, NotSupported, OutOfRange

from .kernel import (
    FULL_M,
    NT,
    PRELOADS,
    STEP,
    TM,
    TN,
    cube_section_moves_through_l1,
    loop_over_blocks,
    loop_with_a_step,
    loop_with_python_range,
    make_pipelined,
    make_pipelined_with_an_aliased_memory_space,
)
from .reference import (
    SENTINEL,
    all_equal,
    make_input,
    make_input_fp16,
    scaled_by_block_index,
)

PIPELINE_BASE = 2
"""流水 stage 里 `scale = ki + 2`，照 ST。参考用同一个 base。"""


def _run_loop(kernel, device, sentinel=False):
    a = make_input(FULL_M, TN)
    if sentinel:
        out = torch.full(
            (FULL_M, TN), float(SENTINEL), dtype=torch.float32, device=device
        )
    else:
        out = torch.zeros((FULL_M, TN), dtype=torch.float32, device=device)
    kernel[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


def _run_pipelined(preload, device):
    a = make_input_fp16(FULL_M, TN)
    out = torch.zeros((FULL_M, TN), dtype=torch.float16, device=device)
    make_pipelined(preload)[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# 一、pl.range
# ---------------------------------------------------------------------------


def test_the_loop_visits_every_block_in_order(device):
    """验证 1。第 k 块乘 `k + 1`。

    **乘轮次号而不是常数**：常数的话跑错次数、跑错顺序都看不出来。
    """
    a, got = _run_loop(loop_over_blocks, device)
    np.testing.assert_allclose(got, scaled_by_block_index(a, TM, NT), rtol=1e-6, atol=0)


def test_a_step_skips_the_odd_blocks(device):
    """验证 2。`pl.range(0, 4, 2)` 只碰 0、2 号块，1、3 号仍是哨兵。

    **哨兵是这条判据的全部**：不填哨兵只能证「碰到的对」，证不到「没碰的没动」。
    """
    a = make_input(FULL_M, TN)
    out = torch.full((FULL_M, TN), float(SENTINEL), dtype=torch.float32, device=device)
    loop_with_a_step[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    got = out.cpu().numpy()

    untouched = np.full_like(a, SENTINEL)
    expected = scaled_by_block_index(a, TM, NT, step=STEP, untouched=untouched)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0)
    for block in range(1, NT, STEP):
        segment = got[slice(block * TM, (block + 1) * TM)]
        assert (segment == SENTINEL).all(), (
            f"第 {block} 块被动了：{segment[0][:4].tolist()}"
        )


def test_a_python_range_is_refused():
    """验证 3。内置 `range` 被拒，消息直接点出该用什么。

    这个 kernel 和 ``loop_over_blocks`` **只差一个词**，所以它验的确实是 `range` 本身。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(loop_with_python_range)
    assert "must use pl.range()" in str(raised.value), str(raised.value)


# ---------------------------------------------------------------------------
# 二、section_cube
# ---------------------------------------------------------------------------


def test_a_mixed_kernel_runs(device):
    """验证 5。cube 段搬一趟 L1，vector 段把原值送出去。

    cube 段那一趟的结果不出去——**它的作用是让这个 kernel 成为混合 kernel**,
    而那件事的可观测后果在 ``support/block_grid`` 验证 4（`subblock_num` 变成 2）。
    """
    a = make_input_fp16(TM, TN)
    out = torch.zeros((TM, TN), dtype=torch.float16, device=device)
    cube_section_moves_through_l1[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), a)


# ---------------------------------------------------------------------------
# 三、pl.pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preload", PRELOADS)
def test_each_preload_gives_the_right_answer(preload, device):
    """验证 4 前半。每个 `preload` 各自对上参考。

    参考里没有任何流水的概念——**一份参考对四个 `preload`**，这正是要验的点。
    """
    a, got = _run_pipelined(preload, device)
    expected = scaled_by_block_index(a, TM, NT, base=PIPELINE_BASE)
    np.testing.assert_allclose(
        got.astype(np.float32), expected.astype(np.float32), rtol=1e-3, atol=1e-3
    )


def test_every_preload_gives_the_same_bits(device):
    """验证 4 后半，**这一例最有分量的一条**。四个 `preload` 的结果两两逐位相同。

    它验的不是某个 API 的返回值，而是**流水变换不改变语义**——那是整个 `pl.pipeline`
    存在的前提。逐位而不是 `allclose`：`preload` 改的是指令排布，不是算术。

    `preload=0` 在里面是因为 `config.py` 的 docstring 说它是「先确认串行版对」的起点;
    `preload=3` 在里面是因为它大于 `NT/2`，填充和排空占掉大半个循环。
    """
    results = [_run_pipelined(preload, device)[1] for preload in PRELOADS]
    assert all_equal(results), (
        "不同 preload 给出不同的结果——流水变换改变了语义。"
        + "；".join(
            f"preload={p} 首元素 {r.ravel()[0]}" for p, r in zip(PRELOADS, results)
        )
    )


def test_an_aliased_memory_space_is_refused_on_a_cross_core_buffer(device):
    """验证 6。跨核 buffer 的 `target_memory` 写成别名被拒。

    **别名是很自然的写法**（本仓每个样例顶上都有 `FP32 = pl.DT_FP32` 这类），
    而它只在这一个位置会炸。消息里会直接说「must set target_memory to a literal」。

    这条要走一遍编译（流水变换在 jit 里做，不在 `parse_target_program` 里）。
    """
    a = make_input_fp16(FULL_M, TN)
    out = torch.zeros((FULL_M, TN), dtype=torch.float16, device=device)
    with pytest.raises(InvalidVal) as raised:
        make_pipelined_with_an_aliased_memory_space()[None, 1](
            torch.from_numpy(a).to(device), out
        )
    message = str(raised.value)
    assert "no resolvable memory space" in message, message
    assert "literal pl.MemorySpace" in message, message


@pytest.mark.parametrize(
    ("bad", "error"),
    [(-1, OutOfRange), (1.5, ValueError), (True, ValueError), ((), InvalidArgument)],
)
def test_an_invalid_preload_is_refused(bad, error):
    """验证 7。四种非法 `preload`，每种的异常类型都不同。

    `True` 单独一条：它 `isinstance(True, int)` 为真，**所以不特判就会被当成 1 放过**。
    框架特判了（消息里点名 bool），这条钉住它。
    """
    with pytest.raises(error) as raised:
        pl.pipeline.PipelineConfig(preload=bad)
    assert "preload" in str(raised.value), str(raised.value)


# ---------------------------------------------------------------------------
# 四、@pl.inline
# ---------------------------------------------------------------------------


def test_inline_is_deprecated_and_its_replacement_does_not_exist():
    """验证 8。`@pl.inline` 报 `DeprecationWarning`，而消息指的 `pl.fn` 不存在。

    两半分开断言：**警告在**是框架的现状，**`pl.fn` 不在**是那条消息的缺陷。
    后者按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 落在 `pypto/`（已改消息并补 UT），所以**这条判据会在那个改动
    落下来之后红**——那时候该改的是这里，不是代码。见 §六。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @pl.inline
        def helper(x):
            return x

    assert len(caught) == 1, [str(w.message) for w in caught]
    assert caught[0].category is DeprecationWarning
    assert "@pl.inline is deprecated" in str(caught[0].message)
    assert helper(3) == 3, "废弃的装饰器应该原样返回函数，它是个 no-op"

    assert not hasattr(pl, "fn"), (
        "`pl.fn` 现在存在了——那条废弃消息不再是缺陷，§六 和这条判据都该更新"
    )


def test_pl_pipeline_is_a_module_not_a_callable():
    """§五 的判据。`pl.pipeline` 是模块，所以按 `pl.pipeline(` 搜必然是 0 次。

    **这条钉住的是「那个 0 不是证据」**，不是 API 的行为。它不上板,
    留在这儿是因为一条数错的结论比没有结论更危险。
    """
    assert not callable(pl.pipeline), "pl.pipeline 变成可调用的了？那 §五 的数法要重看"
    assert hasattr(pl.pipeline, "stage")
    assert hasattr(pl.pipeline, "PipelineConfig")
