# torch\_npu.npu\_add\_quant\_matmul\_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：

    在micro-batch训练场景，需要做micro-batch的梯度累积，会存在大量QuantBatchMatmul操作接InplaceAdd操作的场景。本算子（QuantBatchMatmulInplaceAdd）将上述操作融合起来，以提高网络性能。

- 计算公式：

    mx量化场景公式如下，更多关于量化技术的介绍参见《CANN 算子库》中的[量化介绍](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/quant_mode_introduction.md)。

    ![](figures/zh-cn_formulaimage_0000002521244910.png)

    其中gsk代表K轴的mx量化的block size即32，x1\_slice<sub>i</sub>代表x1<sub>i</sub>第m行长度为gsk的向量，x2\_slice<sub>i</sub>代表x2<sub>i</sub>第n列长度为gsk的向量，K轴均从j\*gsk起始切片，j的取值范围\[0, k\_loops\)，k\_loops=ceil\(K<sub>i</sub>/gsk\)，支持最后的切片长度不足gsk。

    T-T量化场景计算公式如下。

    ![](figures/zh-cn_formulaimage_0000002594063675.png)

## 函数原型

```python
torch_npu.npu_add_quant_matmul_(self, x1, x2, x2_scale, *, x1_scale=None, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> torch.Tensor
```

## 参数说明

- **self**（`Tensor`）：必选参数，待累加矩阵，数据类型支持`torch.float32`，tensor支持2维，shape为\(M, N\)，数据格式支持ND。
- **x1**（`Tensor`）：必选参数，表示矩阵乘法中的左矩阵，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，tensor支持2维，shape为\(K, M\) ，数据格式支持ND。
- **x2**（`Tensor`）：必选参数，表示矩阵乘法中的右矩阵，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，tensor支持2维，shape为\(K, N\)，数据格式支持ND。
- **x2\_scale**（`Tensor`）：必选参数，表示矩阵乘法中的右矩阵的缩放因子，数据类型支持`torch_npu.float8_e8m0fnu`、`torch.float32`，shape支持3维，其中`torch_npu.float8_e8m0fnu`需配置可选参数`x2_scale_dtype`为对应类型，此时`x2_scale`本身的dtype不再生效，但仍需保证`x2_scale`本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **x1\_scale**（`Tensor`）：可选参数，表示矩阵乘法中的左矩阵的缩放因子，数据类型支持`torch_npu.float8_e8m0fnu`、`torch.float32`，shape支持3维，其中`torch_npu.float8_e8m0fnu`需配置可选参数`x1_scale_dtype`为对应类型，此时`x1_scale`本身的dtype不再生效，但仍需保证`x1_scale`本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
- **group\_sizes**（`List[int]`）：可选参数。默认值为None。
  - 非None时，仅支持三维列表，形如\[group\_m, group\_n, group\_k\]，分别表示在m、n、k维度上的量化分组情况。以group\_m为例，表示在m维度上group\_m个数对应一个量化参数。
  - 当\[group\_m, group\_n, group\_k\]中有1个或多个为0时，接口会根据`x1`、`x2`、`x1_scale`、`x2_scale`输入shape重新设置该值。计算原理：假设group\_m=0，表示m方向量化分组值由接口推断，推断公式为group\_m=m/scale\_m（保证m能被scale\_m整除），m与x1 shape中的m一致，scale\_m与x1\_scale shape中的m一致。
  - 目前\[group\_m, group\_n, group\_k\]mx量化支持的取值仅为\[1,1,32\]，T-T量化仅支持取值\[0,0,0\]。

- **x1\_scale\_dtype**（`int`）：可选参数，用于在`x1_scale`无法用torch原生数据类型表示时显式指定`x1_scale`的数据类型。None：默认值，表示输入真实数据类型与输入`x1_scale`的dtype相同。当前仅支持`torch_npu.float8_e8m0fnu`。

- **x2\_scale\_dtype**（`int`）：可选参数，用于在`x2_scale`无法用torch原生数据类型表示时显式指定`x2_scale`的数据类型。None：默认值，表示输入真实数据类型与输入`x2_scale`的dtype相同。当前仅支持`torch_npu.float8_e8m0fnu`。

## 返回值说明

**self**（`Tensor`）：QuantBatchMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，支持数据类型、shape、数据格式均与输入self保持一致。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 数据类型约束：

    | 场景 | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | mx量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | `torch.float32` |
    | T-T量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | `torch.float32` |

- shape约束：

    | 场景 | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | mx量化 | (K,M) | (K,N) | (ceil(K/64),N,2) | (ceil(K/64),M, 2) | (M,N) |
    | T-T量化 | (K,M) | (K,N) | (1) | (1) | (M,N) |

## 调用示例

- 单算子模式调用
  - mx量化单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    N = 7168
    K = 512
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2_scale = torch.randint(-1, 1, (math.ceil(K/64), N, 2), dtype=torch.int8).npu()
    x1_scale = torch.randint(-1, 1, (math.ceil(K/64), M, 2), dtype=torch.int8).npu().transpose(0,1)
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes = [1,1,32])
    ```

  - T-T量化单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    M = 16
    N = 16
    K = 16
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu().transpose(0,1)
    x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
    x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale,x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale_dtype=None, x2_scale_dtype=None, group_sizes = [0,0,0])
    ```

- 图模式调用
  - mx量化图模式调用

    ```python
    import math
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
    N = 7168
    K = 512
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_scale = x1_scale.transpose(0, 1), x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2_scale = torch.randint(-1, 1, (math.ceil(K/64),N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(-1, 1, (math.ceil(K/64),M, 2), dtype=torch.int8).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu)
        print(y.cpu())
     
    if __name__ == '__main__':
        main()
    ```

  - T-T量化图模式调用

    ```python
    import math
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 16
    N = 16
    K = 16
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale = x1_scale, x1_scale_dtype=None, x2_scale_dtype=None)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu()
        x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
        x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, None, None)
        print(y.cpu())
    
    if __name__ == '__main__':
        main()
    ```
