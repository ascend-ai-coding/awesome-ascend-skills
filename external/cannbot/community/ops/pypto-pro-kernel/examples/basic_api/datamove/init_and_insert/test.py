# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""GM 就地初始化（`pl.init_output`）与小块插进大块（`pl.insert`）。

**写法：原始。**

## 一、两个 API 的单位不一样

| | 偏移单位 | 形状从哪来 | 作用在 |
| --- | --- | --- | --- |
| ``pl.init_output(tensor, offset=, size=, value=)`` | **元素**，一维（行优先拉平） | 没有形状，只有一段区间 | **GM 张量** |
| ``pl.insert(dst_tile, src_tile, [row, col])`` | **二维元素坐标** | `src_tile` 自己的形状 | **片上 tile 之间**（TINSERT） |

`init_output` 没有"形状"这个概念——它只认一段 `[offset, offset+size)`。验证 2 刻意让这段
**跨行且不落在整行边界上**：落在整行上的话，「按元素」和「按行」两种理解会给出同样的期望值，
判据就分不出来。

## 二、`init_output` 自己带同步，但只带一半

docstring：「The framework allocates a temporary UB buffer, fills it with *value* via the V
pipeline (Duplicate), then copies it to GM via the MTE3 pipeline (DataCopyPad). **V→MTE3
synchronization is inserted automatically.**」——所以它不需要手工配同步。

但下一段接着说：「This op **never** emits a cross-core barrier」。数据要给别的核或 cube 子核用
（比如后面跟一个 `pl.matmul`）时，**两侧都得显式** `pl.system.sync_all(core_type=MIX)`。
这一例是单核单 block，用不上——**这一点明说，免得有人以为"单核跑通了"就等于这条不存在**。

## 三、`init_output` 需要一个 vector section，空的也行（实测）

它内部用 V 流水填一块临时 UB，再走 MTE3 搬到 GM。**kernel 里没有 `pl.section_vector()` 时
编译失败**，而报出来的是 C++ 层的两条错，看不出和 section 有关：

```
kernel.cpp:23:13: error: the ranges of 1st parameter must be [0, 0], [2, 5], [10, 10]
    get_buf(PIPE_V, 0, 0);
arch_cce_intrinsic.hpp:306:5: error: function type '...' of 'copy_ubuf_to_gm_align_v2'
    does not support the given target feature
```

两条都在说同一件事：**这个核上没有 V 流水**。加 `with pl.section_vector(): pass` 就好——
实测四种形状：

| 形状 | 结果 |
| --- | --- |
| 只有 `init_output` | **编译失败** |
| `init_output` + 空的 `section_vector` | 通过 |
| `init_output` + `make_tile` + 有内容的 section | 通过 |
| 部分区间 + 空 section | 通过，区间外保持哨兵 |

所以这一例里那几个空的 `section_vector` **是必需的，不是装饰**。

## 四、如何独立验证

1. ``init_whole_tensor``：整张等于 `-1.0`，一个哨兵都不剩。
2. ``init_partial_region``：`[32, 96)` 这 64 个元素是 `7.5`，**其余仍是哨兵**。
3. ``init_after_tile_declaration``：与验证 1 结果相同——docstring 说的顺序约束解析期不拦。
   **但「没有差别」只对这个 kernel 成立**，原因见 §五。
4. ``insert_small_into_big``：大块等于 `x`，但 `[2, 4]` 起的 `4 × 8` 被 `x` 右下角那块覆盖。
5. ``insert_flush_to_the_corner``：偏移 `[4, 8]` 恰好贴住右下角，必须合法。
6. `insert` 的偏移给一维或三维被解析期拒（**这是 `insert` 唯一被检查的东西**）。
7. 越界的 `insert` 偏移 **不被**拒（不上板，见 §六）。
8. `init_output` 的 `size` 超过张量大小 **不被**拒（不上板，见 §七）。
9. ``init_int_tensor_with_whole_float``：INT32 张量 + `value=3.0`，**通过**，填进去是 `3`。
10. ``init_int_tensor_with_fraction``：INT32 张量 + `value=1.5`，**编译期**被拒，
    判据去日志里抓 `changes value from 1.5 to 1`。
11. ``make_load_then_init(addr)``：先 load 再 `init_output`，`0x0` / `0x1000` 处的 tile
    被覆写成填充值，`0x2000` 完好。**这才是那句 docstring 要防的事**，见 §五。

第 1 条单独不够：它只证明"填了"。**第 2 条才证明 offset / size 的单位是元素**，也才证明
`init_output` 不会顺手写掉区间之外的东西。第 4 条同理——把源和目的都放在原点的话，
行列写反也会给出正确结果，所以源取右下角、目的取 `[2, 4]`。

第 5 条是边界那一格：越界检查如果写成 `>=` 而不是 `>`，这个合法用法会被误拒。

## 五、`init_output` 的底层：一块写死在 UB `0x0` 的 8 KB 临时 tile

`backend_cce_block_out_ops.cpp:548` 的 codegen，逐条读生成的 CCE（这个 kernel 的原文）：

```cpp
// 用户的 src，8×16 float = 512 B
__inline_0_return_val_1(8, 16);              TASSIGN(__inline_0_return_val_1, 0x0);
// init_output 的临时 tile，2048 float = 8 KB
Tile<Vec, float, 1, 2048, ...> out_0__io0_tile(1, 2048);
                                             TASSIGN(out_0__io0_tile, 0x0);
get_buf(PIPE_V, 0, 0);
TEXPANDS(out_0__io0_tile, -1.0f);            // V：填满整个 8 KB
rls_buf(PIPE_V, 0, 0);
get_buf(PIPE_MTE3, 0, 0);
  TSTORE(out_0__io0, out_0__io0_tile);       // MTE3：从 UB 0x0 读 512 B → GM
rls_buf(PIPE_MTE3, 0, 0);
TLOAD(__inline_0_return_val_1, x_0);         // MTE2：从 GM 读 512 B → UB 0x0
```

### 五点一、覆写宽度 = `(256 × 32) / sizeof(T)` 个元素，与 `size=` 无关

`chunk_elems = (256 * 32) / dtype_bytes`（`MAX_REPEAT_TIMES × ONE_BLK_SIZE`）,
b8/b16/b32 都折成 **8192 字节**；b64 被夹到 64 个元素 = 512 字节
（注释说 `TEXPANDS` 对 b64 走 `Int64Fill`，一条 `vsts` 最多 512 字节）。

**关键是 `TEXPANDS` 在 `SetValidShape` 之前**：`SetValidShape(1, __io_chunk)` 在循环里,
而 `TEXPANDS` 在循环外，按声明形状 `(1, 2048)` 填。所以给 `size=128` 也会覆写 8 KB。

验证 11 在板上量了三个地址，`0x2000` 是边界：

| tile 地址 | 模型 | 实测 |
| --- | --- | --- |
| `0x0` | 被覆写 | 被覆写（全 `-1.0`） |
| `0x1000` | 被覆写 | 被覆写 |
| `0x2000` | 安全 | **完好** |

**这就是 docstring 那句「Must be called before Tile buffer allocation」的内容**——
它不是风格建议，是「临时 tile 占着 UB `0x0` 起的 8 KB」。而那个数字**文档里没有**。

### 五点二、`TSTORE` 之后没有 MTE3 → MTE2 的同步

`init_output` 的 docstring 只保证 **V→MTE3**（`rls_buf(PIPE_V,0,0)` + `get_buf(PIPE_MTE3,0,0)`
那一对），并单独提醒跨核可见性要自己 `sync_all`。**它没提 UB `0x0` 上的后续写。**

生成的代码里 `TLOAD`（MTE2 写 UB `0x0`）前面**什么同步都没有**。于是：

- MTE3 正在从 UB `[0x0, 0x200)` 读 512 字节往 GM 搬;
- MTE2 正在往同一个 `[0x0, 0x200)` 写 512 字节。

**这是 WAR 竞争。** 竞争没发生 → GM 拿到 `-1.0`；发生 → GM 里出现 `x` 的元素。

**用户没法自己修**：临时 tile 是框架内部的，用户既不知道它存在、也不知道它在哪,
写不出针对它的同步。按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 这条落在 `pypto/`；三个可能的改法和各自的代价
记在 [`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #11。

### 五点三、把窗口放宽就复现了：12 次里 6 次

原尺寸的窗口只有**一轮** 512 字节的 `TSTORE`，隔离态连跑 100 次干净。
`size=65536` → **32 轮**，每轮 8 KB，紧跟一条落在同一片 UB 的 `pl.load`：

| kernel | 12 次里出错 |
| --- | ---: |
| `@pl.jit()` + 裸 `make_tile(0x0)` | **6 次**（错的值全是输入 `x` 的元素） |
| 同上 + `pl.system.bar_all()` | 0 次 |
| 同上 + `pl.system.bar_mte3()` | **6 次** |
| `auto_mutex=True` + 裸 `make_tile(0x0)` | **6 次** |
| `@pl.jit()` + `make_tile_group(addrs=0x0, mutex_ids=[0])` | 0 次 |
| `auto_mutex=True` + 同一个 group | 0 次 |
| `@pl.jit()` + 裸 `make_tile(0x2000)` | 0 次 |

**决定因素是地址和 tile 的声明方式，不是调用顺序**——上表全是 docstring 要求的合法顺序,
照样 6/12。救它的是 `make_tile_group`（它发 `get_buf(PIPE_MTE2, 0, 0)`，和 `init_output` 的
`get_buf(PIPE_MTE3, 0, 0)` 同一个 buffer id），**`auto_mutex` 在这里一条指令都没加**
——生成的 cpp 两对各自逐字相同。我先前把安全归给了 `auto_mutex`
（`references/verification/evidence.md` #19）。

静态门禁：`scripts/_checks/init_output.py`。本文件的探针 kernel 带 `反例：` 声明放行,
因为它们量的就是这件事。

### 五点四、2026-09-28 那次失败：诊断怎么留

验证 3 那条判据在一次整份 `pytest examples -q` 里失败过（`1 failed, 580 passed`）,
**而差异没抓到**——那次命令用了 `| tail -3`，只剩汇总行。
这正撞在 `references/verification/board-runs.md` §一「输出落文件再 grep，别在本地 `| tail`」上。

之后的复核：整份复跑 581 passed、单例 10 passed、整章 61 passed、
那个 kernel 同一进程连跑 **100 次**（40 + 60）全干净——**原尺寸下复现不了**,
窗口太窄（见五点三：放宽到 32 轮就稳定 6/12）。

当时做了两件事，不改判据的严格程度：

1. **把诊断写进断言消息**——下次失败会直接说「其中 N 个是输入 x 的元素 → WAR 竞争」
   还是「N 个是哨兵 → `init_output` 没盖满」，不用再等抓 `--tb`;
2. 记下**可证伪的预测**：如果根因是五点二那条竞争，错误值应当是 `x` 的元素,
   **不是哨兵 `-999.0`**。（我上一轮猜的是哨兵，那个猜法基于
   `support/block_grid` 的多 block 回写现象——**读了 codegen 之后它不成立了**,
   这个 kernel 的问题在单核内，和 56 个 block 无关。）

## 六、这一例不讲什么

- **`pl.move`（tile↔tile 跨 memory space 搬运）**——它的路径表和随路量化面在 `fixpipe/` 几例
  和 `references/api/gemm.md`。`insert` 是它的"带偏移"版本，但这一例只用 UB→UB。
- **`insert` 的随路量化参数**（`scale` / `quant_offset` / `relu_pre_mode` / `phase`）——
  那些只在 Acc→Mat 路径上有意义，在 `fixpipe/` 一章。
- **有效区**——`pl.set_validshape` 在 ``tile_view``；填充区在 ``dma_pad``。

## 七、文档写了、解析期不检查的两条（实测）

| docstring 怎么说 | 解析期实际 |
| --- | --- |
| `init_output`「Must be called before Tile buffer allocation」 | **通过**（在 `make_tile` 之后调也过） |
| `insert` 的偏移应让 `src` 落在 `dst` 界内 | **通过**（`[5, 8]` 越界一行也过） |

**`value` 的类型兼容性是第三条，但它不属于这张表**：docstring 说 value「must be type-compatible
with tensor's dtype」，解析期确实不拦，**但编译期拦了**，而且规则比 docstring 说的更准——

| INT32 张量 + | 结果 |
| --- | --- |
| `value=5` | 通过，填 `5` |
| `value=3.0` | **通过**，填 `3` |
| `value=1.5` | **编译期拒** |

规则不是「类型要一致」，是**转换不能改变值**：生成的 C++ 里那个字面量要过
`-Werror,-Wliteral-conversion`，它只在值变了时才报。`3.0 → 3` 没丢东西所以过。

诊断在这里有个缺口：消息本身很准（`changes value from 1.5 to 1`），但**框架抛的异常只有
`Failed to compile kernel '<name>'`**，那行在 `jit.py:1591` 的日志里。验证 10 用 `caplog`
去抓日志而不是匹配异常——只断言"编译失败"的判据太钝，任何编译错误都会让它绿。

`insert` 唯一被检查的是偏移的**元素个数**：

```
offset must contain exactly 2 elements
```

`block_ops.py:1109`。

**两处明说跳过上板**，理由不一样：

- **越界的 `insert`** 会写到目的 tile 之外的 UB。越界落在哪儿取决于布局，**没有判据可写**
  ——一个没有判据的上板用例只会产生一条看着绿的噪声。
- **`size` 超界的 `init_output`** 会写到 GM 张量之外，越界的目标不是我们自己的 UB，
  而是设备内存里别的东西。在共享机上试它不值当。

两条都只钉住「解析期不拦」这件事——**框架哪天加上检查，这两条测试会红**，那正是我们想收到的
通知（`references/verification/evidence.md` #15：没有东西会让它变红的句子是叙述，不是不变量）。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import logging

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidArgument, RuntimeFailure

from .kernel import (
    COLS,
    INIT_OFFSET,
    INIT_OUTPUT_CLOBBER_BYTES,
    INIT_SIZE,
    INIT_VALUE,
    INSERT_COL,
    INSERT_ROW,
    ROWS,
    SMALL_COLS,
    SMALL_ROWS,
    SMALL_SRC_COL,
    SMALL_SRC_ROW,
    VA_SAFE,
    make_load_then_init,
    init_after_tile_declaration,
    init_int_tensor_with_fraction,
    init_int_tensor_with_whole_float,
    init_partial_region,
    init_size_beyond_the_tensor,
    init_whole_tensor,
    insert_flush_to_the_corner,
    insert_offset_with_one_element,
    insert_past_the_edge,
    insert_small_into_big,
)
from .reference import SENTINEL, filled_region, filled_whole, inserted, make_input


def _sentinel_out(device):
    """输出张量先填哨兵。**这是这一例大半判据的前提**：不填就证不了"不该动的没动"。"""
    return torch.full((ROWS, COLS), float(SENTINEL), device=device, dtype=torch.float32)


def _run_init(kernel, device):
    out = _sentinel_out(device)
    kernel(out)
    torch.npu.synchronize()
    return out.cpu().numpy()


def _run_with_input(kernel, device):
    x = make_input(ROWS, COLS)
    out = _sentinel_out(device)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def _parse(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)
    return pp.ir.python_print(program)


def test_init_output_fills_the_whole_tensor(device):
    """验证 1。整张都是 `-1.0`，一个哨兵都不剩。

    这条单独只证明"填了"。断言"没有哨兵剩下"比只比 golden 多一层意思：哨兵还在就说明有一段
    没被覆盖，而那正是 `size` 算错的表现。
    """
    got = _run_init(init_whole_tensor, device)
    np.testing.assert_array_equal(got, filled_whole(ROWS, COLS, -1.0))
    assert not (got == SENTINEL).any(), "还有哨兵没被覆盖，说明 size 没盖满整张"


def test_init_output_offsets_are_element_counts(device):
    """验证 2，这一例的主题之一。`[32, 96)` 是元素区间，跨行且不落在整行边界上。

    如果 `offset` / `size` 被当成行数或字节数，这条会在两端露出来——差的不是一两个元素，
    而是整段错位。参考值用 `ravel()` 算，本身就是这条判据的形式化。
    """
    got = _run_init(init_partial_region, device)
    np.testing.assert_array_equal(
        got, filled_region(ROWS, COLS, INIT_OFFSET, INIT_SIZE, INIT_VALUE)
    )


def test_declaring_a_tile_first_still_gives_the_right_output(device):
    """验证 3。`init_output` 放在 tile 声明之后，解析期不拦，**这个 kernel 的输出仍然对**。

    **但不是因为「放后面也行」。** `VA_BIG` 是 `0x0000`，而 `init_output` 的临时 tile 也
    写死在 `0x0`——生成的 CCE 里是两个 `TASSIGN(..., 0x0)`。输出对只是因为 `src` 载入之后
    再没被读过。把顺序倒过来（先 load 再 init）就会看到覆写，那是验证 7。

    **这条只证到 `make_tile` 这一档。** docstring 点名的是 `make_tile_group` 那种自动分配。

    失败时的诊断写在断言消息里（见 §五点二）：错误值是 `x` 的元素 → UB `0x0` 上的
    MTE3/MTE2 WAR 竞争；是哨兵 → `init_output` 没盖满。**2026-09-28 见过一次失败,
    但没抓到差异**（那次跑用了 `| tail -3`），所以这条消息是为下一次准备的。
    """
    got, x = _run_with_input(init_after_tile_declaration, device)
    expected = filled_whole(ROWS, COLS, -1.0)
    if not np.array_equal(got, expected):
        wrong = got[got != np.float32(-1.0)]
        from_input = np.isin(wrong, x)
        pytest.fail(
            f"{wrong.size}/{ROWS * COLS} 个元素不是 -1.0。出现的值 "
            f"{sorted(set(wrong.tolist()))[:8]}。\n"
            f"其中 {int(from_input.sum())} 个是输入 x 的元素 → "
            f"UB 0x0 上的 WAR 竞争（MTE3 还在读，MTE2 已经在写），见 §五点二；\n"
            f"其中 {int((wrong == SENTINEL).sum())} 个是哨兵 {SENTINEL} → "
            f"init_output 没盖满那段 GM，那是另一回事。"
        )


@pytest.mark.parametrize("addr", [0x0000, 0x1000, VA_SAFE])
def test_init_output_clobbers_8k_of_ub_at_address_zero(device, addr):
    """验证 7，**「必须在 tile 分配之前」那句话的真正后果**。

    先 load 再 `init_output`：`0x0` 和 `0x1000` 处的 tile 内容变成填充值,
    `0x2000` 完好。

    数字是从 codegen 算出来再在板上验边界的：临时 tile 声明成
    `Tile<Vec, T, 1, (256*32)/sizeof(T)>` 并 `TASSIGN(..., 0x0)`，而 `TEXPANDS` 在
    `SetValidShape` **之前**，所以它填满整个 chunk——FP32 是 8192 字节，**和 `size=` 无关**。

    **`0x2000` 这个边界点必须测**：只测 `0x0` 和一个很远的地址，宽度算错一倍也看不出来。
    """
    x = make_input(ROWS, COLS)
    out = torch.zeros((ROWS, COLS), dtype=torch.float32, device=device)
    scratch = torch.zeros((ROWS, COLS), dtype=torch.float32, device=device)
    make_load_then_init(addr)[None, 1](
        torch.from_numpy(np.ascontiguousarray(x)).to(device), out, scratch
    )
    torch.npu.synchronize()
    got = out.cpu().numpy()

    if addr < INIT_OUTPUT_CLOBBER_BYTES:
        np.testing.assert_array_equal(
            got,
            filled_whole(ROWS, COLS, -1.0),
            err_msg=f"tile 在 {addr:#x}（覆写范围 [0, {INIT_OUTPUT_CLOBBER_BYTES:#x}) 内）"
            f"却没被覆写 —— 覆写宽度变小了？",
        )
    else:
        np.testing.assert_array_equal(
            got,
            x,
            err_msg=f"tile 在 {addr:#x}（覆写范围之外）被动了 —— 覆写宽度变大了？",
        )


def test_insert_places_the_small_tile_at_the_offset(device):
    """验证 4。`[2, 4]` 起的 `4 × 8` 被 `x` 右下角那块覆盖，其余等于 `x`。

    源窗口 `[4, 8]`、目的窗口 `[2, 4]`，两个都不在原点：**都在原点的话行列写反也会"对"**。
    """
    got, x = _run_with_input(insert_small_into_big, device)
    np.testing.assert_array_equal(
        got,
        inserted(
            x,
            SMALL_SRC_ROW,
            SMALL_SRC_COL,
            SMALL_ROWS,
            SMALL_COLS,
            INSERT_ROW,
            INSERT_COL,
        ),
    )


def test_an_offset_flush_to_the_corner_is_legal(device):
    """验证 5。`4 + 4 = 8`、`8 + 8 = 16`，恰好贴边。

    边界那一格要单独测：越界检查写成 `>=` 而不是 `>` 的话，这个合法用法会被误拒。
    """
    got, x = _run_with_input(insert_flush_to_the_corner, device)
    np.testing.assert_array_equal(
        got,
        inserted(x, 0, 0, SMALL_ROWS, SMALL_COLS, ROWS - SMALL_ROWS, COLS - SMALL_COLS),
    )


def test_an_offset_that_is_not_two_elements_is_refused():
    """验证 6。**这是 `insert` 唯一被检查的东西。**"""
    with pytest.raises(InvalidArgument) as raised:
        _parse(insert_offset_with_one_element)
    assert "offset must contain exactly 2 elements" in str(raised.value), str(
        raised.value
    )


def test_an_out_of_bounds_insert_is_not_refused():
    """验证 7。越界一行，解析期通过。**断言的是"没有检查"。**

    框架哪天加上界检查，这条会红——那正是我们想收到的通知。不上板的理由见 §六：
    越界写到目的 tile 之外的 UB，落在哪儿取决于布局，没有判据可写。
    """
    _parse(insert_past_the_edge)  # 不抛就是这条测试的全部内容


def test_an_init_size_beyond_the_tensor_is_not_refused():
    """验证 8。`size` 是张量元素数的 4 倍，解析期通过。**断言的是"没有检查"。**

    不上板，理由和验证 7 不一样：它往 GM 张量之外写，越界目标不是我们自己的 UB。
    在共享机上试它不值当——这是明说跳过。
    """
    _parse(init_size_beyond_the_tensor)


def test_a_float_value_that_fits_an_int_tensor_is_accepted(device):
    """验证 9。INT32 张量 + `value=3.0` 通过，填进去是 `3`。

    这条和验证 10 必须一起读：单看验证 10 会得出「INT32 张量不能传浮点」，那是错的结论。
    **两条一起才定出真正的规则：转换不能改变值。**
    """
    out = torch.full((ROWS, COLS), -999, device=device, dtype=torch.int32)
    init_int_tensor_with_whole_float(out)
    torch.npu.synchronize()
    got = out.cpu().numpy()
    np.testing.assert_array_equal(got, np.full((ROWS, COLS), 3, dtype=np.int32))


def test_a_fractional_value_into_an_int_tensor_fails_to_compile(device, caplog):
    """验证 10。`value=1.5` 进 INT32 张量：**编译期**被拒。

    判据去日志里抓 `changes value from 1.5 to 1`，**不匹配异常消息**——异常只有
    `Failed to compile kernel '<name>'`，任何编译错误都长这样，用它当判据等于只断言"炸了"。
    这是一处诊断缺口：消息本身很准，但没有传到异常里（`jit.py:1591` 写日志，`jit.py:2465` 抛异常）。
    """
    out = torch.full((ROWS, COLS), -999, device=device, dtype=torch.int32)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeFailure):
            init_int_tensor_with_fraction(out)
    assert "changes value from 1.5 to 1" in caplog.text, caplog.text[-2000:]
