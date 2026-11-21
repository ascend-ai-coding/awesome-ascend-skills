# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GM 初始化区间与 UB insert 的独立验证。

init_output 在 Vector section 内执行，offset/size 均按元素计数。整张填充与部分区间检查保留
原始 NumPy golden 和区间外哨兵；INT32 目标按向零截断转换浮点常量，正负输入分别检查。

UB insert 检查非零源/目的偏移及恰好贴边的合法区域。偏移维数错误与静态 GM 初始化越界只解析
不执行；静态 offset+size 超过张量元素数必须抛出 InvalidShape。越界 UB insert 同样不执行。

load→init_output→store 探针检查初始化临时 UB 区间的覆写边界，使用 0、0x1000、0x2000 三个
显式地址并保留所需的流水同步。该结果只说明当前工具链的临时存储行为，不构成任意裸 Tile 的
自动同步保证。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidArgument, InvalidShape

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
    init_int_tensor_with_negative_fraction,
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
    return pp.ir.text_dump(program)


def test_init_output_fills_the_whole_tensor(device):
    """整张 GM 张量填成 -1.0，任何未覆盖哨兵均视为失败。"""
    got = _run_init(init_whole_tensor, device)
    np.testing.assert_array_equal(got, filled_whole(ROWS, COLS, -1.0))
    assert not (got == SENTINEL).any(), "还有哨兵没被覆盖，说明 size 没盖满整张"


def test_init_output_offsets_are_element_counts(device):
    """按拉平的元素区间 [32,96) 检查初始化，区间外保持原始哨兵。"""
    got = _run_init(init_partial_region, device)
    np.testing.assert_array_equal(
        got, filled_region(ROWS, COLS, INIT_OFFSET, INIT_SIZE, INIT_VALUE)
    )


def test_declaring_a_tile_first_still_gives_the_right_output(device):
    """bare Tile 声明之后的初始化仍须完整写入 GM；错误值保留用于定位覆写或漏写。"""
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
    """三个显式 UB 地址检查初始化临时区间 [0,8192) 的覆写边界。"""
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
    """非零源区域插入非零目的偏移，精确检查整张输出。"""
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
    """源区域恰好贴住右下边界时，搬运必须合法且精确。"""
    got, x = _run_with_input(insert_flush_to_the_corner, device)
    np.testing.assert_array_equal(
        got,
        inserted(x, 0, 0, SMALL_ROWS, SMALL_COLS, ROWS - SMALL_ROWS, COLS - SMALL_COLS),
    )


def test_an_offset_that_is_not_two_elements_is_refused():
    """二维 insert 偏移必须恰好包含两个元素。"""
    with pytest.raises(InvalidArgument) as raised:
        _parse(insert_offset_with_one_element)
    assert "offset must contain exactly 2 elements" in str(raised.value), str(
        raised.value
    )


def test_an_out_of_bounds_insert_is_not_refused():
    """越界 UB insert 只检查当前解析边界，不编译或执行。"""
    _parse(insert_past_the_edge)  # 不抛就是这条测试的全部内容


def test_an_init_size_beyond_the_tensor_is_refused():
    """静态 offset+size 超过 GM 张量元素数时，在解析期拒绝。"""
    pattern = (
        rf"offset \(0\) \+ size \({ROWS * COLS * 4}\) exceeds "
        rf"tensor element count \({ROWS * COLS}\)"
    )
    with pytest.raises(InvalidShape, match=pattern):
        _parse(init_size_beyond_the_tensor)


def test_a_float_value_that_fits_an_int_tensor_is_accepted(device):
    """INT32 目标接受无损的 3.0 字面量，输出精确等于整数 3。"""
    out = torch.full((ROWS, COLS), -999, device=device, dtype=torch.int32)
    init_int_tensor_with_whole_float(out)
    torch.npu.synchronize()
    got = out.cpu().numpy()
    np.testing.assert_array_equal(got, np.full((ROWS, COLS), 3, dtype=np.int32))


@pytest.mark.parametrize(
    "kernel,value",
    [
        (init_int_tensor_with_fraction, 1.5),
        (init_int_tensor_with_negative_fraction, -1.5),
    ],
)
def test_fractional_values_into_an_int_tensor_truncate_toward_zero(device, kernel, value):
    """正负浮点常量向零截断，输出逐元素精确等于独立 NumPy 参考。"""
    out = torch.full((ROWS, COLS), -999, device=device, dtype=torch.int32)
    kernel(out)
    torch.npu.synchronize()
    expected = np.full((ROWS, COLS), np.trunc(value), dtype=np.int32)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
