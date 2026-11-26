# torch\_npu.npu\_transpose\_quant\_batchmatmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：完成张量x1与张量x2量化的矩阵乘计算。仅支持三维的tensor传入。tensor支持转置，转置序列根据传入的序列进行变更。perm\_x1代表张量x1的转置序列，perm\_x2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。
- 计算公式：

    T1、T2、Ty分别通过参数perm\_x1、perm\_x2、perm\_y描述转置序列。

    $$
    out=((x1^{T1}@x2^{T2}+bias)*x2\_scale*x1\_scale)^{Ty}
    $$

## 函数原型

```python
torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype, *, bias=None, x1_scale=None, x2_scale=None, group_sizes=None, perm_x1=None, perm_x2=None, perm_y=None, batch_split_factor=1, x1_dtype=None, x2_dtype=None) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：必选参数，表示矩阵乘的第一个矩阵。数据格式支持ND。仅支持3维输入，shape要求为（m, b, k）。数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。
- **x2**（`Tensor`）：必选参数，表示矩阵乘的第二个矩阵。数据格式支持ND和NZ。仅支持3维输入，shape要求为（b, k, n）或（b, n, k）。`x2`的k维度需要与`x1`的k维度大小相等。数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。
- **dtype**（`int`）：必选参数，表示output的数据类型，支持传值`torch.float16`、`torch.bfloat16`和`torch_npu.hifloat8`。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **bias**（`Tensor`）：可选参数，表示矩阵乘的偏置矩阵，当前版本暂不支持该参数，使用默认值即可。
- **x1\_scale**（`Tensor`）：可选参数，表示左矩阵的量化系数。数据格式支持ND。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`和`torch.int64`。shape支持1维或4维。
- **x2\_scale**（`Tensor`）：可选参数，表示右矩阵的量化系数。数据格式支持ND。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`和`torch.int64`。shape支持1维或4维。
- **group\_sizes**（List\[int\]）：可选参数，表示量化分组大小，数据类型为`torch.int32`。默认值为None。
  - 仅支持三维列表，形如\[group\_m, group\_n, group\_k\]，其分别表示在m、n、k维度上的量化分组情况。以group\_m为例，其表示在m维度上每group\_m个数对应一个量化参数。
  - 当\[group\_m, group\_n, group\_k\]中有1个或多个为0时，接口会根据`x1`、`x2`、`x1_scale`、`x2_scale`输入shape重新设置该值。计算原理：假设group\_m=0，表示m方向量化分组值由接口推断，推断公式为group\_m=m/scale\_m（保证m能被scale\_m整除），m与`x1` shape中的m一致，scale_m与`x1_scale` shape中的m一致。
  - 仅在mx量化模式需要传入，目前\[group\_m, group\_n, group\_k\]仅支持\[0,0,32\]、\[0,1,32\]、\[1,0,32\]、\[1,1,32\]。

- **perm\_x1**（List\[int\]）：可选参数，表示矩阵乘第一个矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持ND，只支持\[1, 0, 2\]。
- **perm\_x2**（List\[int\]）：可选参数，表示矩阵乘第二个矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持ND。支持\[0, 1, 2\]或\[0, 2, 1\]。
- **perm\_y**（List\[int\]）：可选参数，表示矩阵乘输出矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持ND，只支持\[1, 0, 2\]。
- **batch\_split\_factor**（`int`）：可选参数，用于指定矩阵乘输出矩阵中b维的切分大小。数据类型为`torch.int32`。默认值为1，当前仅支持配置为1。
- **x1_dtype**（`int`）：可选参数，表示x1的数据类型，支持传值`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。
- **x2_dtype**（`int`）：可选参数，表示x2的数据类型，支持传值`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。

## 返回值说明

**y**（`Tensor`）：表示最终计算结果，公式中的$out$，数据格式支持ND，shape维度支持3维。shape为\(m, b, n\)；数据类型支持`torch.float16`、`torch.bfloat16`和`torch_npu.hifloat8`。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式。
- 图模式支持K-C、mx量化模式。
- K-C量化场景下：
  - `x1_scale`、`x2_scale`仅支持1维输入，`x1_scale`要求shape为\(m, \)，`x2_scale`要求shape为\(n, \)。
  - `x2`仅支持ND格式输入。
  - k仅支持512，n仅支持128。
  - `perm_x2`只支持\[0, 1, 2\]。

- mx量化场景下：
  - `x1`、`x2`仅支持`torch.float8_e4m3fn`输入。k仅支持64的倍数。
  - `x1_scale`、`x2_scale`仅支持4维输入，`x1_scale`要求shape为\(m, b, k/64, 2\)；`perm_x2`为\[0,1,2\]时，`x2_scale`要求shape为\(b, k/64, n, 2\)，`perm_x2`为\[0,2,1\]时，`x2_scale`要求shape为\(b, n, k/64, 2\)。
  - 支持将`x2`转为昇腾亲和的数据排布以提高搬运效率。需要调用`torch_npu.npu_format_cast`完成输入`x2`（weight）为昇腾亲和的数据排布功能。

- T-C量化场景下：
  - `x1`、`x2`仅支持`torch_npu.hifloat8`输入。
  - `x2`仅支持ND格式输入。
  - `x1_scale`、`x2_scale`仅支持1维输入；`x1_scale`支持为空，非空时要求shape为\(1, \)，`x2_scale`要求shape为\(n, \)。

## 调用示例

- 单算子模式调用
  - K-C量化场景：

  ```python
  import torch
  import torch_npu

  M, K, N, Batch = 32, 512, 128, 32
  x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()

  x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32).npu()
  x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32).npu()
  y = torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype=torch.float16, x1_scale=x1_scale,
                                          x2_scale=x2_scale, perm_x1=[1, 0, 2],
                                          perm_x2=[0, 1, 2], perm_y=[1, 0, 2])
  ```

- 图模式调用
  - K-C量化场景：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M, K, N, Batch = 32, 512, 128, 32
    x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()

    x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32).npu()
    x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32).npu()

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x1_scale, x2_scale, dtype, perm_x1, perm_x2, perm_y, batch_split_factor=1):
            output = torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype, x1_scale=x1_scale, x2_scale=x2_scale, perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y, batch_split_factor=batch_split_factor)
            return output

    model = MyModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), torch.float16, (1, 0, 2), (0, 1, 2), (1, 0, 2)).to("cpu")
    ```
