# torch_npu.npu_quant_matmul_reduce_sum

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。

- 计算公式：

$$
out = \sum_{i=0}^{batch - 1}(x1_i @ x2_i) * x1Scale_i * x2Scale
$$

## 函数原型

```python
torch_npu.npu_quant_matmul_reduce_sum(x1, x2, *, x1_scale, x2_scale) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，数据类型支持`torch.int8`，数据格式支持$ND$，shape支持3维，形状为$(batch, m, k)$。

- **x2** (`Tensor`)：必选参数，数据类型支持`torch.int8`，数据格式仅支持$NZ$，shape支持3维，形状为$(batch, k, n)$。如果输入数据是$ND$格式，可通过`x2 = torch_npu.npu_format_cast(x2.contiguous(), 29)`将$ND$格式的`x2`转换为$NZ$格式，其中29为NZ格式的枚举值。

- **x1_scale** (`Tensor`)：必选关键字参数，对应公式中的$x1Scale$。数据类型支持`torch.float32`，数据格式支持$ND$，shape支持2维，形状为$(batch, m)$。在实际计算时，`x1_scale`会被广播到$(batch,m,n)$。

- **x2_scale** (`Tensor`)：必选关键字参数，对应公式中的$x2Scale$。数据类型支持`torch.bfloat16`，数据格式支持$ND$，shape支持1维，形状为$(n,)$。在实际计算时，`x2_scale`会被广播到$(batch，m，n)$。

## 返回值说明

`Tensor`

公式中的$out$，算子的计算结果。输出的数据类型为`torch.bfloat16`，数据格式为ND，shape为2维，形状为$(m, n)$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持静态图模式。
- 传入的`x1`、`x2`、`x1_scale`、`x2_scale`不能为空。
- 输入和输出支持以下数据类型组合：

  | x1   | x2   | x1_scale | x2_scale  | out      |
  |------|------|---------|----------|----------|
  | `torch.int8` | `torch.int8` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |

## 调用示例

- 单算子调用

  ```python
  import torch
  import torch_npu

  torch_npu.npu.config.allow_internal_format = True

  b,m,k,n = (2,3,4,5)
  x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
  x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
  x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
  x1_scale = torch.ones((b, m), dtype=torch.float32).npu()
  x2_scale = torch.ones((n,), dtype=torch.bfloat16).npu()
  y = torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=x1_scale, x2_scale=x2_scale)
  ```

- 图模式调用

  ```python
  import torch
  import torch_npu
  import torchair as tng
  from torchair.ge_concrete_graph import ge_apis as ge
  from torchair.configs.compiler_config import CompilerConfig
  import logging
  from torchair.core.utils import logger

  logger.setLevel(logging.DEBUG)
  import os
  import numpy as np

  # "ENABLE_ACLNN"是否走aclnn, true: 回调走aclnn, false: 在线编译
  os.environ["ENABLE_ACLNN"] = "false"
  config = CompilerConfig()
  npu_backend = tng.get_npu_backend(compiler_config=config)

  class MyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
 
      def forward(self, x1, x2, scale, pertoken_scale):
          return torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=pertoken_scale, x2_scale=scale)

  cpu_model = MyModel()
  model = cpu_model.npu()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  b,m,k,n = (2,3,4,5)
  x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
  x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
  x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
  pertoken_scale = torch.ones((b, m), dtype=torch.float32).npu()
  scale = torch.ones((n,), dtype=torch.bfloat16).npu()
  npu_out = model(x1, x2, scale, pertoken_scale)
  print(npu_out)
  ```
