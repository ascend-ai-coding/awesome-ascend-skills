# torch_npu.npu_scatter_nd_update_

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

将`updates`中的值按指定的索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`中的数据被改变。

## 函数原型

```python
torch_npu.npu_scatter_nd_update_(input, indices, updates) -> Tensor(a!)
```

## 参数说明

- **input** (`Tensor`)：必选输入，源数据张量，数据格式支持$ND$，支持非连续的Tensor，数据类型需要与`updates`一致，维度只能是1~8维。

  <!-- npu="310p" id6 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`。
  <!-- end id6 -->
  <!-- npu="910" id7 -->
  - <term>Atlas训练系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`。
  <!-- end id7 -->
  <!-- npu="910b" id8 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`。
  <!-- end id8 -->
  <!-- npu="A3" id9 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`。
  <!-- end id10 -->

- **indices** (`Tensor`)：必选输入，索引张量，数据类型支持`torch.int32`、`torch.int64`，数据格式支持$ND$，支持非连续的Tensor，`indices`中的索引数据不支持越界。
- **updates** (`Tensor`)：必选输入，更新数据张量，数据格式支持$ND$，支持非连续的Tensor，数据类型需要与`input`一致。

  <!-- npu="310p" id11 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`。
  <!-- end id11 -->
  <!-- npu="910" id12 -->
  - <term>Atlas训练系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`。
  <!-- end id12 -->
  <!-- npu="910b" id13 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`。
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`。
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bool`、`torch.bfloat16`、`torch.int64`、`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`。
  <!-- end id15 -->

## 返回值

`Tensor`

代表`input`被更新后的结果。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- `indices`至少是2维，其最后1维的大小不能超过`input`的维度大小。
- 假设`indices`最后1维的大小是a，则`updates`的shape等于`indices`除最后1维外的shape加上`input`除前a维外的shape。举例：`input`的shape是$(4, 5, 6)$，`indices`的shape是$(3, 2)$，则`updates`的shape必须是$(3, 6)$。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    
    data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
    var = torch.from_numpy(data_var).to(torch.float16).npu()
       
    data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
    indices = torch.from_numpy(data_indices).to(torch.int32).npu()
    
    data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
    updates = torch.from_numpy(data_updates).to(torch.float16).npu()
    
    out = torch_npu.npu_scatter_nd_update_(var, indices, updates)
    print(out)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[1.8271, 1.4551, 1.3154,  ..., 1.9854, 1.4365, 1.0732],
            [1.9492, 1.6455, 1.6504,  ..., 1.5957, 1.6201, 1.4385],
            [0.0742, 0.1982, 0.8945,  ..., 0.4912, 0.6753, 0.1120],
            ...,
            [0.1113, 0.6255, 0.7686,  ..., 0.0247, 0.2490, 0.6909],
            [0.4312, 0.7954, 0.7339,  ..., 0.1154, 0.6440, 0.3342],
            [0.9570, 0.2869, 0.6489,  ..., 0.7451, 0.0234, 0.8843]],
            device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import os
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch.nn as nn
    import torch
    import numpy as np
    import numpy
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    os.environ["ENABLE_ACLNN"] = "false"
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
    
        def forward(self, var, indices, update):
            # 调用目标接口
            res = torch_npu.npu_scatter_nd_update_(var, indices, update)
            return res
            
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    
    dtype = np.float32
    x = [33 ,5]
    indices = [33,25,1]
    update = [33,25,5]
    
    data_x = np.random.uniform(0, 1, x).astype(dtype)
    data_indices = np.random.uniform(0, 10, indices).astype(dtype)
    data_update = np.random.uniform(0, 1, update).astype(dtype)
    
    tensor_x = torch.from_numpy(data_x).to(torch.float16)
    tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
    tensor_update = torch.from_numpy(data_update).to(torch.float16)
    
    # 传参
    out= npu_mode (tensor_x.npu(), tensor_indices.npu(), tensor_update.npu())
    print(out.shape, out.dtype)

    # 执行上述代码的输出类似如下
    torch.Size([33, 5]) torch.float16
    ```
