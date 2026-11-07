# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""`pl.gemm_mx`：一条调用顶四条 move 加一条 `matmul_mx`，展开与手写逐条相同。

**解决什么.** MX 每个 K 步要写的东西是普通 matmul 的两倍：每个操作数都成对到场（FP8 数据 +
E8M0 scale），每对都要搬两次。`pl.gemm` 把两条 move 收成一条调用，`pl.gemm_mx` 把**四条**
收成一条。

**接口 = `gemm` + 两个 L1 scale + 两个 plane.**

```python
pl.gemm_mx(acc, l1a, l1b, l1a_scale, l1b_scale,      # 调用方已经 load 好的
           l0a=..., l0b=..., scale_a=..., scale_b=...,  # 糖要写进去的暂存
           m=..., n=..., k=..., init=True, trans_a=False, trans_b=False)
```

五个位置参数是**调用方装好的**，四个关键字参数是**糖写进去的暂存**——分界就是斜杠。
`scale_a` / `scale_b` 这两个名字和 `pl.matmul_mx(dst, lhs, rhs, /, scale_a, scale_b)` 一致，
因为它们就是原封不动传给那个 op 的东西。层级写在前面（`l1a` / `l0a`），一次调用里就能对齐。

**没有 bias / splitn / splitk.** `pl.matmul_mx` 没有 bias 形参；切分要沿 plane 的组轴切，
而组是不可分的单位。三个都**按名字拒绝并说明原因**，不是当成拼写错误。

**收窄的分工是推出来的，不是选的.**

`scale_a` 是 `[m, k/32]`、`scale_b` 是 `[k/32, n]`，所以：

| 参数 | 收窄谁 |
| --- | --- |
| `m` | `l0a`、`acc`、**`scale_a` 的行轴** |
| `n` | `l0b`、`acc`、**`scale_b` 的列轴** |
| `k` | `l0a`、`l0b`、**两个 plane 的组轴** |

**`k` 那一行是实测纠正过来的**（2026-09-26，Ascend950PR，MXFP8，声明 K=128、`k=64`）：

| 做法 | 结果 |
| --- | --- |
| plane 保持四个组（本样例最初的写法） | **错**，max\|diff\| 183.2，14980/16384 个不对 |
| plane 组轴收到两个 | **逐位相等** |
| 完全不收窄（`k=128`） | 逐位相等 |

原来的推理是「一个组 32 个元素、不可分，所以 `k` 动不了组轴」。那句话对**半个组**成立，对
**整数个组**什么都没说——`kernel.py` 里 `mx_sugar_k_half` 就是把这条钉住的用例。

**`m` / `n` 那两行是弱得多的声明。** K 整着的时候，plane 的 `valid_shape` 实测**毫无作用**：
收它的行轴、甚至把组轴收到 1 个（只有 2 个），两者都与参考逐位相等。所以那两条是为了让描述符
和配对的数据 tile 保持一致而发的，不是结果依赖它。**为什么同一个组轴在 M 收窄下无所谓、在 K
收窄下却是决定性的，没有查清**——两边都是测量，不是机制。

两个后果：

- plane **不要求** `compact=1`。那条要求的理由是「乘法读收窄后的范围」；plane 的配对走地址，
  要求它会误拒正确的 kernel。
- `k` 必须是 64 的倍数，这条是**硬件要求**：实测 `k=32` 和 `k=96`（都是整数个组）各自把设备打成
  **AICore trap 507015**，而 64 和 128 逐位相等。所以单位是 64，不是组。

**四条 move 成对相邻.** 数据那条和它 plane 那条在硬件上是**同一条** `dma.l1_to_l0.mx`
（ascriptor 的 `mx.py` 模块 docstring），所以拆开写等于把一条指令的操作数散在两处。Pro 手写
惯例发的是「Left 数据、Right 数据、Left plane、Right plane」，糖改成成对——**这是个顺序改动，
所以上板验了**（见下）。

**如何独立验证.**

1. **糖的展开 == 手写**，op + 同步序列逐条相同（`memref_id` 和 SSA 名要抹掉，前者是进程内
   全局递增）。比较生成 IR 时需保留地址、布局和同步字段。
2. **四条 move 的顺序是成对的**，从 IR 里读操作数名。
3. **收窄落在该落的五块上**，`scale_a` 是 `[m, 2]`、`scale_b` 是 `[2, n]`——组轴是 2，一个不少。
4. **糖和手写在板上逐位相同**。
5. **糖和参考逐位相等**（`rtol=0, atol=0`）——操作数取 E4M3 单位阶梯配 `{0.5, 1, 2}` 的 scale，
   整条 K=128 的和在 FP32 里精确可表示，细节见
   [`../fundamental_usage/matmul_mx/reference.py`](../fundamental_usage/matmul_mx/reference.py)。
6. **收窄版也逐位相等**，比的是有效范围那 48 行。
7. **自动 ID 版的 plane 落在它自己的 `l0a` / `l0b` 地址 `>> 4` 上**，逐槽——两个槽才看得出槽间距。
8. **`k` 收窄版与参考逐位相等**——这条路曾经静默算错，见上面那张表。
9. **FP4 的 plane 槽间距是 `0x80`**，和官方 ST 里的 `scale_slot_addr = 0x80 if a_is_fp4` 对上。
10. **不是 64 倍数的 `k` 被拒**——这条边界背后是个会打崩设备的硬件陷阱，所以要有一条会失败的
    断言，而不是一句注释。
11. **FP4 的两张值表是推出来的**，和官方 ST 的 `_FP4_VALUES` 逐值相同；打包方向用 unpack 自证。
12. **FP4 与参考逐位相等**（`rtol=0, atol=0`）——同一个「整条和精确可表示」的论证，换了阶梯。
    编码约定见 [`reference.py`](reference.py)。
13. **混合配对 `e2m1 × e1m2` 也逐位相等**——把推出来的第二张表也跑上板。**推出来没跑过的表不算
    验过**，这条用例存在的唯一理由就是这个。

**FP4 的 GM 载体使用打包字节，注解按逻辑元素计数。** 本例沿用 `pl.DYNAMIC`，Tile 形状仍为逻辑元素数。当前框架同样支持静态逻辑形状 `[64, 64]` 接受 `[64, 32]` 的字节载体；冷调用、缓存命中
及非法调用前后的边界由 [dtype_carriers](../../basic_api/support/dtype_carriers/test.py) 验证。

**plane 地址归调用方.** 糖不接管声明（沿用 `pl.gemm` 的「L0 归调用方」决定），
两版的 `scale_a` 地址表均逐槽取 `l0a` 地址 `>> 4`。写错了框架会拒——`matmul_mx`
上那条逐槽检查照样作用在糖展开出来的调用上，本文件有一条测试专门钉这一点。

**显式声明 plane 地址与深度**：

```python
pl.make_tile_group(
    type=pl.TileType(shape=[TILE, SCALE_K], dtype=E8M0,
                     target_memory=pl.MemorySpace.ScaleLeft, compact=1),
    addrs=[0, (TILE * TILE) >> 4], mutex_ids="auto", depth=2,
)
```

这里配对的 L0A 数据地址为 `[0, TILE*TILE]`。plane 保留独立的 mutex 所有权，不能用数据组的 ID 代替。自动规划 ID 不负责分配这些地址。

两个 L1 的 scale Mat 要给 `layout=`（`Mat` 默认 NZ，MX 的 scale 视图 A 侧 ZZ、B 侧 NN）；plane
自己不用给，空间已经把 layout 定死了。

接口契约：[api/gemm-mx.md](../../../references/api/gemm-mx.md)。
"""

import ast
import re

import numpy as np
import pytest
import torch

from ..fundamental_usage.matmul_mx.reference import make_operands, matmul_reference
from .reference import (
    E1M2_VALUES,
    E2M1_VALUES,
    NONZERO_E2M1,
    make_fp4_operands,
    pack_fp4,
    unpack_fp4,
)
from .kernel import mx_placed_fp4_mixed
from .kernel import (
    GROUP_SIZE,
    K,
    K_HALF,
    M,
    M_TAIL,
    N,
    PAIR,
    SCALE_K,
    TILE,
    _FP4_PLANE_SLOT,
    mx_placed_fp4,
    mx_manual,
    mx_sugar,
    mx_sugar_k_half,
    mx_sugar_m_tail,
)


def _cube_ir(kernel):
    import pypto_pro as pp

    program, _ = kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)
    return pp.ir.python_print(program.get_function(kernel.__name__))


def _op_sequence(text):
    """block op 与同步操作的顺序，去掉操作数名。

    忽略暂存变量、地址和进程内递增的 `memref_id`。自动规划与手动 ID 允许重命名，
    比较每条同步指令的 pipe、分组候选 ID 和共享关系。
    """
    sequence = []
    canonical_ids = {}

    def normalize_ids(value):
        if isinstance(value, list):
            return tuple(normalize_ids(item) for item in value)
        return canonical_ids.setdefault(value, len(canonical_ids))

    for line in text.splitlines():
        lock = re.search(r"@(system\.mutex_(?:un)?lock)_dyn\(", line)
        if lock:
            pipe = re.search(r'pipe="pipe": (\d+)', line)
            ids = re.search(r", (\[\[[0-9,\s\[\]]*\]\]), pipe=", line)
            assert ids is not None, f"Unrecognized grouped mutex operands: {line}"
            normalized = normalize_ids(ast.literal_eval(ids.group(1)))
            sequence.append(f"{lock.group(1)} pipe={pipe.group(1)} ids={normalized}")
            continue
        block = re.search(r"@(block\.[a-z_0-9]+)\(", line)
        if block and not block.group(1).startswith("block.make_tile"):
            sequence.append(block.group(1))
    return sequence


def _run(device, kernel, ops, rows=M):
    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    out = torch.zeros([M, N], device=device, dtype=torch.float32)
    kernel(
        dev(ops["a_codes"]),
        dev(ops["b_codes"]),
        dev(ops["scale_a"]),
        dev(ops["scale_b"]),
        out,
    )
    torch.npu.synchronize()
    return out.cpu()[:rows]


def test_the_sugar_expands_to_the_hand_written_form():
    """验证 1。op + 同步序列逐条相同。"""
    assert _op_sequence(_cube_ir(mx_sugar)) == _op_sequence(_cube_ir(mx_manual))


def test_the_four_moves_come_out_in_pairs():
    """验证 2。数据、它的 plane、另一侧数据、它的 plane。

    成对是因为一条 `dma.l1_to_l0.mx` 同时发数据和 plane 两个 load；拆开写读者看不出配对。
    """
    moves = re.findall(r"@block\.move\(_pl_gemm\d+_(\w+?)_\d+,", _cube_ir(mx_sugar))

    # 两臂各一份（init 必须编译期已知），每份四条。
    assert len(moves) == 8
    assert moves[:4] == ["l0a", "scale_a", "l0b", "scale_b"]
    assert moves[4:] == ["l0a", "scale_a", "l0b", "scale_b"]


def test_only_m_narrows_three_tiles_and_the_group_axis_stays_whole():
    """验证 3。只给了 `m`，所以收窄落在 l0a、acc 和 plane_a 的行轴上，l0b / plane_b 不动。"""
    shapes = re.findall(
        r"@block\.set_validshape\([^,]+, ([^)]*)\)", _cube_ir(mx_sugar_m_tail)
    )

    assert len(shapes) == 6  # 两臂各三块
    assert shapes[:3] == [
        f"{M_TAIL}, {TILE}",  # l0a：k 取声明值
        f"{M_TAIL}, {TILE}",  # acc：n 取声明值
        f"{M_TAIL}, {SCALE_K}",  # plane_a：收行，组轴仍是 SCALE_K=2
    ]
    assert shapes[3:] == shapes[:3]  # 另一臂一样


def test_the_plane_pairing_is_still_checked_through_the_sugar():
    """糖展开出的是真的 `pl.matmul_mx`，所以逐槽地址检查照样作用。

    把 plane 的 `addrs` 写成标量（那个按 dtype 变脸的陷阱，见
    [`../fundamental_usage/matmul_mx/test.py`](../fundamental_usage/matmul_mx/test.py)），期望被拒而不是静默算错。
    """
    import pypto_pro as pp
    import pypto_pro.language as pl
    from pypto_pro._errors import InvalidVal

    from .kernel import E8M0, FP8

    @pl.jit(arch="a5", auto_mutex=True, name="gemm_mx_scalar_plane_addrs")
    def bad(
        a: pl.Tensor[[M, K], FP8],
        b: pl.Tensor[[K, N], FP8],
        sa: pl.Tensor[[M, K // PAIR, 2], E8M0],
        sb: pl.Tensor[[K // PAIR, N, 2], E8M0],
        out: pl.Tensor[[M, N], pl.DT_FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=FP8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x00000,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=FP8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x10000,
            mutex_ids=[2, 3],
        )
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, SCALE_K],
                dtype=E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=0x20000,
            mutex_ids=[4, 5],
        )
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SCALE_K, TILE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=0x21000,
            mutex_ids=[6, 7],
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=FP8,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                compact=1,
            ),
            addrs=0x00000,
            mutex_ids=[8, 9],
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                compact=1,
            ),
            addrs=0x00000,
            mutex_ids=[10, 11],
        )
        # 标量：按 plane 自己的 128 字节递推，而硬件要的是数据 4096 >> 4 = 256
        plane_a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, SCALE_K],
                dtype=E8M0,
                target_memory=pl.MemorySpace.ScaleLeft,
                layout=pl.ZZ,
                compact=1,
            ),
            addrs=0x00000,
            mutex_ids=[12, 13],
        )
        plane_b = pl.make_tile_group(
            type=pl.TileType(
                shape=[SCALE_K, TILE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.ScaleRight,
                layout=pl.NN,
                compact=1,
            ),
            addrs=0x00000,
            mutex_ids=[14, 15],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                compact=1,
            ),
            addrs=0x00000,
            mutex_ids=[16],
        )
        with pl.section_cube():
            cur_acc = acc.current()
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            cur_sa = sa_l1.next()
            cur_sb = sb_l1.next()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.load(cur_sa, sa, [0, 0, 0])
            pl.load(cur_sb, sb, [0, 0, 0])
            pl.gemm_mx(
                cur_acc,
                cur_a,
                cur_b,
                cur_sa,
                cur_sb,
                l0a=l0a,
                l0b=l0b,
                scale_a=plane_a,
                scale_b=plane_b,
            )
            pl.store(out, cur_acc, [0, 0])

    with pytest.raises(InvalidVal) as raised:
        bad.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)

    message = raised.value.message
    assert "0x80" in message and "0x100" in message, message


def test_the_sugar_and_the_hand_written_form_agree_on_board(device):
    """验证 4。同一份操作数，两种写法逐位相同。"""
    ops = make_operands(M, K, N)

    sugared = _run(device, mx_sugar, ops)
    manual = _run(device, mx_manual, ops)

    torch.testing.assert_close(sugared, manual, rtol=0, atol=0)


def test_the_sugar_matches_the_reference_bit_for_bit(device):
    """验证 5。不给容差——这组操作数下整条和在 FP32 里精确可表示。"""
    ops = make_operands(M, K, N)

    out = _run(device, mx_sugar, ops)

    torch.testing.assert_close(
        out,
        torch.from_numpy(matmul_reference(ops["a_values"], ops["b_values"])),
        rtol=0,
        atol=0,
    )


def test_the_m_narrowed_sugar_matches_the_reference(device):
    """验证 6。`m=` 这条路上板成立。

    只比有效范围那 48 行：范围外累加器里是什么，`m=` 这个接口不作承诺。

    **这条不证明 plane 的收窄是必要的**——模块 docstring 里那个三版本对照说明它实测无效。
    这里验的是「糖发出来的那一组收窄整体上是对的」。
    """
    ops = make_operands(M, K, N, seed=3)

    out = _run(device, mx_sugar_m_tail, ops, rows=M_TAIL)

    want = matmul_reference(ops["a_values"][:M_TAIL], ops["b_values"])
    torch.testing.assert_close(out, torch.from_numpy(want), rtol=0, atol=0)


def test_planes_pair_with_their_data_group():
    """验证 7。显式 plane 地址逐槽等于数据地址 `>> 4`。

    两个槽才看得出槽间距：一个槽时两边都是 0，`0 >> 4 == 0` 恒成立。
    """
    rows = {}
    for line in _cube_ir(mx_sugar).splitlines():
        m = re.search(
            r"_tg_(?:__inline_\d+_)?(\w+?)_tiles_(\d+)_0 = .*?MemorySpace\.(\w+).*?"
            r'memref_addr="memref_addr": (\d+)',
            line,
        )
        if m:
            rows.setdefault(m.group(1), {})[int(m.group(2))] = int(m.group(4))

    for data, plane in (("l0a", "plane_a"), ("l0b", "plane_b")):
        want = [rows[data][slot] >> 4 for slot in sorted(rows[data])]
        got = [rows[plane][slot] for slot in sorted(rows[plane])]
        assert len(got) == 2, f"{plane} 应当一个数据槽一个 plane 槽"
        assert got == want, (
            f"{plane} 在 {[hex(a) for a in got]}，应当是 {data} 的 >> 4 "
            f"= {[hex(a) for a in want]}"
        )


def test_a_narrowed_k_matches_the_reference(device):
    """验证 8。`k` 收窄要连 plane 的组轴一起收——这条曾经静默算错。

    声明 K=128，只算前 64。参考取 `A[:, :64] @ B[:64, :]`。plane 保持四个组的那个版本在同一块卡
    上错 183.2，所以这条断言不是形式主义。
    """
    ops = make_operands(TILE, K, TILE, seed=17)

    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    mx_sugar_k_half(
        dev(ops["a_codes"]),
        dev(ops["b_codes"]),
        dev(ops["scale_a"]),
        dev(ops["scale_b"]),
        out,
    )
    torch.npu.synchronize()

    want = matmul_reference(ops["a_values"][:, :K_HALF], ops["b_values"][:K_HALF, :])
    torch.testing.assert_close(out.cpu(), torch.from_numpy(want), rtol=0, atol=0)


def test_the_fp4_plane_stride_is_half_the_fp8_one():
    """验证 9。FP4 的 tile 形状是逻辑元素数，所以差别只落在字节地址上。

    官方 ST（`test_matmul_mx.py`）自己写的是 `scale_slot_addr = 0x80 if a_is_fp4 else 0x100`。
    显式声明的 plane 槽间距应当就是这个数，逻辑形状保持不变。
    """
    rows = {}
    for line in _cube_ir(mx_placed_fp4).splitlines():
        m = re.search(
            r"_tg_(\w+?)_tiles_(\d+)_0 = .*?MemorySpace\.(\w+).*?"
            r'memref_addr="memref_addr": (\d+)',
            line,
        )
        if m:
            rows.setdefault(m.group(1), {})[int(m.group(2))] = int(m.group(4))

    plane = [rows["plane_a"][slot] for slot in sorted(rows["plane_a"])]
    data = [rows["l0a"][slot] for slot in sorted(rows["l0a"])]

    assert len(plane) == 2
    assert plane[1] - plane[0] == _FP4_PLANE_SLOT == 0x80, plane
    assert data[1] - data[0] == TILE * TILE // 2, "FP4 的 64x64 块是 2048 字节"
    assert plane == [a >> 4 for a in data]
    # plane 的形状和 FP8 那版一模一样：每 32 个逻辑元素一个组
    assert SCALE_K == TILE // GROUP_SIZE


def test_a_k_that_is_not_a_multiple_of_64_is_refused():
    """验证 10。这条拒绝挡的是一个**会打崩设备**的写法，所以它必须是拒绝而不是文档。

    实测（Ascend950PR，2026-09-26）：手写 `pl.matmul_mx` 配 `k=32` 和 `k=96`——**两者都是整数个
    组**，按「组不可分」那个理由本该没问题——各自触发 **AICore trap 507015**。而 `k=64` / `k=128`
    逐位相等。所以单位是 64，不是 32。

    只有字面量 `k` 查得到：解析一个运行期表达式会把它的 IR 发两遍（展开出来的 `set_validshape`
    还要再解析一次），所以运行期 `k` 这条证不了，模块 docstring 里说明了。
    """
    import pypto_pro as pp
    import pypto_pro.language as pl
    from pypto_pro._errors import InvalidVal

    from .kernel import E8M0, FP8, GROUP_SIZE

    @pl.jit(arch="a5", auto_mutex=True, name="gemm_mx_k_not_64")
    def bad(
        a: pl.Tensor[[TILE, K], FP8],
        b: pl.Tensor[[K, TILE], FP8],
        sa: pl.Tensor[[TILE, K // PAIR, 2], E8M0],
        sb: pl.Tensor[[K // PAIR, TILE, 2], E8M0],
        out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, TILE], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K // GROUP_SIZE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=1,
        )
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K // GROUP_SIZE, TILE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=0x4100,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, TILE],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        plane_a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K // GROUP_SIZE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0x0],
            mutex_ids="auto",
            depth=1,
        )
        plane_b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K // GROUP_SIZE, TILE],
                dtype=E8M0,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0x0],
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            cur_acc = acc.current()
            cur_a = a_l1.current()
            cur_b = b_l1.current()
            cur_sa = sa_l1.current()
            cur_sb = sb_l1.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.load(cur_sa, sa, [0, 0, 0])
            pl.load(cur_sb, sb, [0, 0, 0])
            pl.gemm_mx(
                cur_acc,
                cur_a,
                cur_b,
                cur_sa,
                cur_sb,
                l0a=l0a,
                l0b=l0b,
                scale_a=plane_a,
                scale_b=plane_b,
                k=GROUP_SIZE,
            )  # 一个整组，但不是 64 的倍数
            pl.store(out, cur_acc, [0, 0])

    with pytest.raises(InvalidVal) as raised:
        bad.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)

    message = raised.value.message
    assert "'k' is 32" in message
    assert "64" in message, message


def test_the_fp4_value_tables_are_derived_and_match_the_official_ones():
    """验证 11。两张表从位域推出来，再和官方 ST 的常数核对。

    官方 `_FP4_VALUES`（`test_matmul_mx.py:885`）是这两张表的独立来源。推导写在
    `reference.py` 的 docstring 里：两种格式都是 `bias = 1`，E2M1 的四个阶码全部用于有限
    值——4 位放不下 inf / nan。
    """
    official_e2m1 = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    official_e1m2 = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        -0.0,
        -0.25,
        -0.5,
        -0.75,
        -1.0,
        -1.25,
        -1.5,
        -1.75,
    ]

    assert E2M1_VALUES.tolist() == official_e2m1
    assert E1M2_VALUES.tolist() == official_e1m2
    # ST 给的两个单位码也对得上
    assert E2M1_VALUES[0x02] == 1.0
    assert E1M2_VALUES[0x04] == 1.0
    # 阶梯的性质：14 个非零码，值都是 0.5 的倍数、绝对值不超过 6
    assert NONZERO_E2M1.size == 14
    magnitudes = np.abs(E2M1_VALUES[NONZERO_E2M1])
    assert np.all(magnitudes * 2 == np.rint(magnitudes * 2))
    assert magnitudes.max() == 6.0


def test_the_fp4_packing_direction_round_trips():
    """验证 11 的后半。偶数逻辑下标进低半字节——写反了整个操作数就转置了半个字节。"""
    codes = np.array([[0x1, 0x2, 0x3, 0x4]], dtype=np.uint8)

    packed = pack_fp4(codes)

    assert packed.shape == (1, 2)
    assert packed.tolist() == [[0x21, 0x43]], "偶数下标在低位，奇数在高位"
    assert unpack_fp4(packed, codes.shape).tolist() == codes.tolist()
    # 随机一轮往返，确认不是只对这四个码成立
    big = np.random.default_rng(0).integers(0, 16, size=(8, 64), dtype=np.uint8)
    assert unpack_fp4(pack_fp4(big), big.shape).tolist() == big.tolist()


def test_fp4_matches_the_reference_bit_for_bit(device):
    """验证 12。E2M1 × E2M1，K=64 一步，不给容差。

    整条和精确可表示：乘积项是 `2^-4` 的倍数、绝对值 ≤ 144，64 项的和 ≤ 9216，
    `9216 / 2^-4 = 147456 < 2^24`。
    """
    ops = make_fp4_operands(TILE, TILE, TILE, seed=23)

    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    mx_placed_fp4(
        dev(ops["a_packed"]),
        dev(ops["b_packed"]),
        dev(ops["scale_a"]),
        dev(ops["scale_b"]),
        out,
    )
    torch.npu.synchronize()

    want = matmul_reference(ops["a_values"], ops["b_values"])
    torch.testing.assert_close(out.cpu(), torch.from_numpy(want), rtol=0, atol=0)


def test_the_mixed_fp4_pair_matches_the_reference(device):
    """验证 13。A 侧 E2M1、B 侧 E1M2，两种 4 位格式混用。

    这条跑的是 `reference.py` 推出来的**第二张**表。上一条只用了 E2M1，E1M2 那张表当时只和
    官方常数比过——**推出来没跑过的表不算验过**。

    和依然精确可表示：乘积项是 `2^-5` 的倍数、绝对值 ≤ 42，K=64 的和 ≤ 2688 < `2^24 · 2^-5`。
    """
    ops = make_fp4_operands(TILE, TILE, TILE, seed=31, a_format="e2m1", b_format="e1m2")

    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    mx_placed_fp4_mixed(
        dev(ops["a_packed"]),
        dev(ops["b_packed"]),
        dev(ops["scale_a"]),
        dev(ops["scale_b"]),
        out,
    )
    torch.npu.synchronize()

    want = matmul_reference(ops["a_values"], ops["b_values"])
    torch.testing.assert_close(out.cpu(), torch.from_numpy(want), rtol=0, atol=0)
