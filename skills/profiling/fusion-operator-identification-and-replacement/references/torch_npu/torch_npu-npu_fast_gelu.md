# torch_npu.npu_fast_gelu

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

- API功能：快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算`FastGelu`的前向结果。
- 计算公式：

  <!-- npu="910,310p" id6 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>公式如下：

  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702 \mid x\mid}} e^{0.851 x(x- \mid x\mid)}
  $$
  <!-- end id6 -->
  <!-- npu="950,A3,910b" id7 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>公式如下：

  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702x}}
  $$
  <!-- end id7 -->
  
## 函数原型

```python
torch_npu.npu_fast_gelu(self) -> Tensor
```

## 参数说明

- **self**(`Tensor`)：必选参数，对应公式中的$x$。数据格式支持$ND$，支持非连续的Tensor。输入最大支持8维。支持空Tensor。

  <!-- npu="910" id8 -->
  - <term>Atlas训练系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
  <!-- end id8 -->
  <!-- npu="910b" id9 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
  <!-- end id9 -->
  <!-- npu="A3" id10 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
  <!-- end id10 -->
  <!-- npu="310p" id11 -->
  - <term>Atlas推理系列产品</term>：数据类型仅支持`torch.float16`、`torch.float32`。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
  <!-- end id12 -->

## 返回值说明

`Tensor`

代表`fast_gelu`的计算结果。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `self`输入不能为None。

## 调用示例

- 单算子调用

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np

    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y = torch_npu.npu_fast_gelu(x).cpu().numpy()
    ```

- 图模式调用

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    os.environ["ENABLE_ACLNN"] = "false"
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
        def forward(self, x): 
            y = torch_npu.npu_fast_gelu(x)
            return y
            
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y =npu_mode(x).cpu().numpy()
    print("shape of y:",y.shape)
    print("dtype of y:",y.dtype)
    
    # 执行上述代码的输出类似如下
    shape of y: (4, 2048, 16, 128)
    dtype of y: float32
    ```
