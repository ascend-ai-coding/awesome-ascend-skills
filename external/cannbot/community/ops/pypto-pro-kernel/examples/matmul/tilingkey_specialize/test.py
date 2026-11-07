# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""编译期特化：一个 kernel 源码，几个 key，几份各自特化过的二进制。

**解决什么.** 同一个 matmul 要支持 FP16 和 BF16、块大小要能在 64 和 128 之间选。写成
运行期分支会把两条路都编进去，而且 `TileType` 的 dtype 和 shape 本来就必须是编译期常量
——运行期的值放不进去。`tiling_key` 的做法是：把这几个选择声明成一组字段，**parser 在
解析时把字段引用折成常量**，每组具体值编出一份自己的二进制。

**最反直觉的三件事.** 光看签名一个都看不出来：

| | 实际是什么 |
| --- | --- |
| key 怎么进 kernel | **不是形参**。`Dtype` / `TileBlock` 在函数体当**自由变量**用，parser 注入。所以 `kernel.py` 顶上要有 `# ruff: noqa: F821` |
| 形参写什么类型 | 只能是 `pl.Ptr[pl.DT_UINT8]`。dtype 是 key 决定的，绑定前写不出 `pl.Tensor[..., io_dtype]`；类型由体内的 `pl.make_tensor` 补 |
| 打包后的 key 是什么 | `values` 里那个值的**下标**，不是值本身。`TileBlock=128` 打出来是 1。和 AscendC 的模板 tiling key 对齐 |

**启动语法**：`kernel[stream, block_dim, {"Dtype": 0, "TileBlock": 64}](...)`——第三个
下标位是 key 字典。少了它不是默认值，是报错（实测消息见下面那条测试）。

**`is_valid` 的两个用途**。声明在 key 类上，签名 `is_valid(self, key)`，`key` 是按声明
顺序排的**具体值元组**：

1. 启动时拒掉非法的具体 key；
2. 二进制交付路径枚举全部组合时，只留下合法的那些。

**它不能碰 `self` 上的任何东西**：schema 用 `object.__new__(cls)` 造实例、**不走
`__init__`**，所以 `__init__` 里设的属性在这里都不存在。这一条有一个专门的反例测试。

**和 `preload` 的一个区别.** 编译缓存的键是 `(static_signature, dtype_hash,
tilingkey_packed)`，build 目录也带 key 后缀——所以不同 key **不会**互相喂上一份二进制。
`pipeline=` 的 `preload` 就不在这个键里，那边必须一个 preload 一个 kernel（见
`examples/algorithms/gemm_bias_silu_banded/kernel.py` 的 kernel 工厂注意事项）。这里不用。

**如何独立验证.** 五条，前四条不需要卡：

1. **key 真的变成了 IR 里的常量**——解析两个 `TileBlock`，在 IR 文本里读那个标量。这条
   防的是「传了 key 但什么都没特化」。
2. **打包用下标不用值**——`values=[64, 128]` 让这两者可区分。
3. **`is_valid` 过滤了枚举**，而且非法 key 在校验时被拒。
4. **`is_valid` 碰 `self` 会炸**，反例钉住。
5. **三个合法 key 各自与参考比**，并且编译缓存里确实是三份、packed 值是 `{0, 2, 3}`。

来源：``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_tilingkey.py``（163
行，`Dtype` / `TileBlock` 两个字段，M=N=1024）。这里缩到 512 省编译时间，并补上源文件
没有的 `is_valid`。

**源文件带的 `@pypto.options(pass_options={"enable_slice": False})` 这里没有。** 在
M=N=512 / K=128 这个形状上不加也全过（2026-09-26 上板）。**没有验过 1024 上加不加的
区别**，所以这不是「那条选项没用」，只是「这个形状不需要」。

**读 IR 时的一个坑.** `program.get_function(name)` 的 `name` 是**被装饰的 Python 函数
名**，不是 `pl.jit(name=)` 给的名字；名字写错不报错，返回一个一行的空壳。要看全程序就直接
`pypto_pro.ir.python_print(program)`，别点名。
"""

import pytest
import torch

from pypto_pro._errors import InvalidArgument, InvalidVal
from pypto_pro.runtime.tilingkey import TilingKeySchema

from .kernel import K, M, N, MatmulKey, NeedsInitKey, matmul_keyed
from .reference import TOLERANCE, matmul_reference

# 三个合法组合：BF16 + 64 被 is_valid 挡掉了。
VALID_KEYS = [
    ({"Dtype": 0, "TileBlock": 64}, torch.float16),
    ({"Dtype": 0, "TileBlock": 128}, torch.float16),
    ({"Dtype": 1, "TileBlock": 128}, torch.bfloat16),
]
REJECTED_KEY = {"Dtype": 1, "TileBlock": 64}


def _key_id(case):
    key, _ = case
    return f"Dtype{key['Dtype']}_Tile{key['TileBlock']}"


def _ir_text(key: dict) -> str:
    """解析成 Cube 侧的程序再打印。整程序打印，不点名函数。"""
    import pypto_pro as pp

    program, _ = matmul_keyed.to_kernel_def(key).parse_target_program(
        pp.ir.SectionKind.Cube
    )
    return pp.ir.python_print(program)


def test_the_key_became_a_constant_in_the_ir():
    """先验这条：否则「传了 key」和「特化了」是两件事，后面几条都测不到东西。

    `tile = TileBlock` 这一行在 IR 里就是一个字面量标量，`if Dtype == 0` 那一支也已经
    被折掉——留下的 dtype 只有一个。
    """
    small = _ir_text({"Dtype": 0, "TileBlock": 64})
    large = _ir_text({"Dtype": 0, "TileBlock": 128})

    assert "ir.Scalar[ir.INT64] = 64" in small
    assert "ir.Scalar[ir.INT64] = 128" in large
    # dtype 也折了：FP16 那份里不该出现 BF16。
    assert "ir.FP16" in small and "ir.BF16" not in small
    bf16 = _ir_text({"Dtype": 1, "TileBlock": 128})
    assert "ir.BF16" in bf16


def test_the_packed_key_is_the_value_index_not_the_value():
    """`values=[64, 128]` 让「下标」和「值」可区分——都写 0/1 就看不出差别。"""
    schema = TilingKeySchema(MatmulKey)

    assert schema.field_names() == ["Dtype", "TileBlock"]
    assert schema.pack({"Dtype": 0, "TileBlock": 64}) == 0
    assert schema.pack({"Dtype": 0, "TileBlock": 128}) == 2  # 下标 1 移到第 1 位
    assert schema.pack({"Dtype": 1, "TileBlock": 128}) == 3


def test_is_valid_filters_the_enumeration_and_rejects_a_concrete_key():
    """两个用途各验一个。"""
    schema = TilingKeySchema(MatmulKey)

    combos = schema.enumerate_valid()
    assert len(combos) == 3, f"四个组合里该留三个，实际 {combos}"
    assert (1, 64) not in combos

    schema.validate_concrete({"Dtype": 0, "TileBlock": 64})  # 合法的不抛
    with pytest.raises(InvalidVal, match="did not pass MatmulKey.is_valid"):
        schema.validate_concrete(REJECTED_KEY)


def test_a_value_outside_the_candidates_is_refused():
    """不在 `values` 里的值是另一类错，报的也是另一条消息。"""
    schema = TilingKeySchema(MatmulKey)

    with pytest.raises(InvalidArgument, match="is not a candidate"):
        schema.validate_concrete({"Dtype": 0, "TileBlock": 96})


def test_is_valid_must_not_read_self():
    """schema 用 `object.__new__` 造实例、不走 `__init__`。

    构造 schema 这一步**过得去**——它只绑定不调用。等到真的调 `is_valid` 才炸，所以
    这个坑会在枚举或校验的时候才现形。
    """
    schema = TilingKeySchema(NeedsInitKey)  # 这一步不抛

    with pytest.raises(AttributeError, match="has no attribute 'allowed'"):
        schema.enumerate_valid()


def test_launching_without_a_key_is_refused():
    """少了 key 字典不是取默认值，是报错。"""
    a = torch.zeros([M, K], dtype=torch.float16)
    with pytest.raises(
        InvalidArgument, match=r"Expected kernel\[stream, block_dim, tilingkey\]"
    ):
        matmul_keyed[None, 1](a, a, a)


@pytest.mark.parametrize("case", VALID_KEYS, ids=[_key_id(c) for c in VALID_KEYS])
def test_matches_the_reference(device, case):
    """三个合法 key 各自与参考比。容差按 dtype 分，理由见 reference.py。"""
    key, torch_dtype = case
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch_dtype)
    b = torch.randn([K, N], device=device, dtype=torch_dtype)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    matmul_keyed[None, 1, key](a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(
        out.cpu(), matmul_reference(a.cpu(), b.cpu()), **TOLERANCE[torch_dtype]
    )


def test_each_key_got_its_own_binary(device):
    """编译缓存的键含 packed key，所以三个 key 是三份，不会互相喂。

    这条和 `pipeline_*` 那两个样例的注意事项正好相反：`preload` **不在**这个键里。
    """
    for key, torch_dtype in VALID_KEYS:
        a = torch.randn([M, K], device=device, dtype=torch_dtype)
        b = torch.randn([K, N], device=device, dtype=torch_dtype)
        out = torch.zeros([M, N], device=device, dtype=torch.float32)
        matmul_keyed[None, 1, key](a, b, out)
    torch.npu.synchronize()

    packed = {entry[2] for entry in matmul_keyed._compiled_by_signature}
    assert packed == {0, 2, 3}, f"三个 key 的 packed 值应当是 0/2/3，实际 {packed}"
