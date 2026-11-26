# torch\_npu.npu\_add\_quant\_gmm\_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：

  在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatmul操作接InplaceAdd操作的融合场景。本算子（QuantGroupedMatmulInplaceAdd）将上述算子融合起来，以提高网络性能。

- **计算公式**：

  不同量化场景公式如下，更多关于量化技术的介绍参见《CANN算子库》中的[量化介绍](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/quant_mode_introduction.md)。

  - T-T/T-C量化场景：

    $$
    \mathit{self} = \mathit{self} + \mathit{x1} \times \mathit{x2} * \mathit{x2\_scale} * \mathit{x1\_scale}
    $$

  - mx量化场景：

    $$
    \begin{aligned}
    \mathit{self}[m,n] = {} &\mathit{self}[m,n] + \\
    &\sum_{j=0}^{k\_loops-1}\left(\left(\sum_{k=0}^{gsk-1}(\mathit{x1\_slice}_i \times \mathit{x2\_slice}_i)\right) \times \left(\mathit{x1\_scale}_i[m,j] \times \mathit{x2\_scale}_i[j,n]\right)\right)
    \end{aligned}
    $$

    其中gsk代表K轴的mx量化的block size即32，K<sub>i</sub>为每个分组的K的大小，x1\_slice<sub>i</sub>代表x1<sub>i</sub>第m行长度为gsk的向量，x2\_slice<sub>i</sub>代表x2<sub>i</sub>第n列长度为gsk的向量，K轴均从j\*gsk起始切片，j的取值范围\[0, k\_loops\)，k\_loops=ceil\(K<sub>i</sub>/gsk\)，支持最后的切片长度不足gsk。

## 函数原型

```python
torch_npu.npu_add_quant_gmm_(self, x1, x2, x2_scale, group_list, *, x1_scale=None, group_list_type=0, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None)
```

## 参数说明

> 参数说明里Shape使用的变量说明：
>
> - g：表示分组数目，取值范围为1-1024。
> - M：输出矩阵的倒数第二维大小，取值范围为1-2147483647。
> - N：输出矩阵的倒数第一维大小，取值范围为1-2147483647。
> - K：矩阵乘法reduce轴的大小，取值范围为1-2147483647。
> 注：M、N、K如为输入矩阵最内轴时需小于2097152。

- **`self`**（`Tensor`）：**必选参数**，待累加矩阵，数据类型支持`torch.float32`，tensor支持3维，shape为\(g, M, N\)。数据格式支持$ND$。
- **`x1`**（`Tensor`）：**必选参数**，表示矩阵乘法中的左矩阵，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，tensor支持2维，shape为\(K, M\)，其中`torch_npu.hifloat8`需配置可选参数`x1_dtype`为对应类型，此时`x1`本身的`dtype`不再生效，但仍需保证`x1`本身的`dtype`为8bit位的数据类型，以保证shape正确。数据格式支持$ND$。
- **`x2`**（`Tensor`）：**必选参数**，表示矩阵乘法中的右矩阵，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，tensor支持2维，shape为\(K, N\)，其中`torch_npu.hifloat8`需配置可选参数`x2_dtype`为对应类型，此时`x2`本身的`dtype`不再生效，但仍需保证`x2`本身的`dtype`为8bit位的数据类型，以保证shape正确。数据格式支持$ND$。
- **`x2_scale`**（`Tensor`）：**必选参数**，表示矩阵乘法中的右矩阵的缩放因子，数据类型支持`torch_npu.float8_e8m0fnu`、`torch.float32`，shape支持1-3维，其中`torch_npu.float8_e8m0fnu`需配置可选参数`x2_scale_dtype`为对应类型，此时`x2_scale`本身的`dtype`不再生效，但仍需保证`x2_scale`本身的`dtype`为8bit位的数据类型，以保证shape正确。数据格式支持$ND$。
- **`group_list`**（`Tensor`）：**必选参数**，代表输入和输出分组轴方向的matmul大小分布，数据类型支持`torch.int64`，1维tensor，shape为\(g，\)，数据格式支持$ND$。当`group_list_type`为0时，`group_list`必须为非负单调非递减数列，并且最后一个值不大于输入tensor的第一维；当`group_list_type`为1时，`group_list`必须为非负数列，并且数值的总和不大于输入tensor的第一维。`group_list`中的值约束了输出数据的有效部分，`group_list`中未指定的部分将不会参与更新。
- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **`x1_scale`**（`Tensor`）：**可选参数**，表示矩阵乘法中的左矩阵的缩放因子，数据类型支持`torch_npu.float8_e8m0fnu`、`torch.float32`，shape支持1维、2维或3维，其中`torch_npu.float8_e8m0fnu`需配置可选参数`x1_scale_dtype`为对应类型，此时`x1_scale`本身的`dtype`不再生效，但仍需保证`x1_scale`本身的`dtype`为8bit位的数据类型，以保证shape正确。数据格式支持$ND$。
- **`group_list_type`**（`int`）：**可选参数**，代表`group_list`的表达形式。数据类型支持`torch.int32`。

  - 0：默认值，`group_list`中数值为分组轴大小的cumsum结果（累积和）。
  - 1：`group_list`中数值为分组轴上每组大小。

- **`group_sizes`**（`List[int]`）：**可选参数**，列表大小为3，分别表示`M`、`N`、`K`三个轴上多少个参数共用一个缩放因子。数据类型支持`torch.int32`，目前只支持默认值None。
- **`x1_dtype`**（`int`）：**可选参数**，可用于在`x1`无法用torch原生数据类型表示时显式指定`x1`的数据类型。

  - None：默认值，表示输入`x1`真实的数据类型与输入`x1`的`dtype`相同。
  - 赋予输入`x1`真实的数据类型，当前仅支持`torch_npu.hifloat8`。

- **`x2_dtype`**（`int`）：**可选参数**，可用于在`x2`无法用torch原生数据类型表示时显式指定`x2`的数据类型。

  - None：默认值，表示输入`x2`真实的数据类型与输入`x2`的`dtype`相同。
  - 赋予输入`x2`真实的数据类型，当前仅支持`torch_npu.hifloat8`。

- **`x1_scale_dtype`**（`int`）：**可选参数**，用于在`x1_scale`无法用torch原生数据类型表示时显式指定`x1_scale`的数据类型。

  - None：默认值，表示输入`x1_scale`真实的数据类型与输入`x1_scale`的`dtype`相同。
  - 赋予输入`x1_scale`真实的数据类型，当前仅支持`torch_npu.float8_e8m0fnu`。

- **`x2_scale_dtype`**（`int`）：**可选参数**，用于在`x2_scale`无法用torch原生数据类型表示时显式指定`x2_scale`的数据类型。

  - None：默认值，表示输入`x2_scale`真实的数据类型与输入`x2_scale`的`dtype`相同。
  - 赋予输入`x2_scale`真实的数据类型，当前仅支持`torch_npu.float8_e8m0fnu`。

## 返回值说明

**`self`**（`Tensor`）：groupedMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，数据类型、shape、数据格式均与输入`self`保持一致。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `group_list`第1维最大支持1024，即最多支持1024个group。
- 数据类型约束：

  | 场景 | x1 | x2 | x2_scale | x1_scale | self |
  | --- | --- | --- | --- | --- | --- |
  | T-T/T-C量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | `torch.float32` |
  | mx量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | `torch.float32` |

- shape约束

  | 场景 | x2_scale | x1_scale |
  | --- | --- | --- |
  | T-T/T-C量化 | pertensor场景：2维tensor或1维tensor，shape为(g, 1)或(g,)；perchannel场景：2维tensor，shape为(g, N) | 2维tensor或1维tensor，shape为(g, 1)或(g,) |
  | mx量化 | 3维tensor，shape为((K / 64) + g, N, 2) | 3维tensor，shape为((K / 64) + g, M, 2) |

## 调用示例

- 单算子模式调用
  - T-C量化场景

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu()
    x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu()
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
    print(y.cpu())
    ```

  - mx量化

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu()
    x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0,1)
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu)
    print(y.cpu())
    ```

- 图模式调用
  - T-C量化场景

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 576
    K = 512
    N = 7168
    g = 4
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_dtype, x2_dtype):
            return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=group_list_type, x1_dtype=x1_dtype, x2_dtype=x2_dtype)

    def main():
        y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu()
        x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu()
        x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu()
        group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
        group_list_type = 0
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.hifloat8, torch_npu.hifloat8)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```

  - mx量化

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 576
    K = 512
    N = 7168
    g = 4
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale.transpose(0, 1), group_list_type=group_list_type, x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype)

    def main():
        y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu()
        group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
        group_list_type = 0
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```
