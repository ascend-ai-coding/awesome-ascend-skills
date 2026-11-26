# torch_npu.npu_fused_matmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：矩阵乘与通用向量计算融合，减少数据搬运提升性能。

- **计算公式**：

  $$
  y = \operatorname{op}((x1 @ x2 + bias),\ x3)
  $$

  op运算类型由`fused_op_type`输入定义，支持如下：

  | fused_op_type | 计算公式 |
  | :--- | :--- |
  | add | $y = (x1 @ x2 + bias) + x3$ |
  | mul | $y = (x1 @ x2 + bias) \times x3$ |
  | gelu_erf | $y = \operatorname{gelu\_erf}(x1 @ x2)$ |
  | gelu_tanh | $y = \operatorname{gelu\_tanh}(x1 @ x2)$ |
  | relu | $y = \operatorname{relu}(x1 @ x2 + bias)$ |

  - `x1`是矩阵乘matmul的左矩阵。
  - `x2`是矩阵乘matmul的右矩阵。
  - `x3`是向量计算的矩阵，当`fused_op_type`是`"add"`和`"mul"`时生效。
  - `bias`是矩阵乘的偏置。
  - `fused_op_type`是输入的string类型，支持`""`（表示不做融合）、`"16cast32"`（表示不做融合，并且输出为`torch.float32`，当输入`torch.float16`、`torch.bfloat16`时会把输出转成`torch.float32`）、`"mul"`、`"gelu_erf"`、`"gelu_tanh"`和`"relu"`。

## 函数原型

```python
torch_npu.npu_fused_matmul(x1, x2, *, bias=None, x3=None, fused_op_type) -> Tensor
```

## 参数说明

- **x1**(`Tensor`)：必选参数，即矩阵乘中的`x1`。数据格式支持$ND$，支持非连续的Tensor，当`fused_op_type`为`"relu"`或`""`时，支持输入为两到六维，为`"add"`或`"mul"`时，支持输入为两维(M,K)和三维(B,M,K)，其他情况支持输入维度为两维(M, K)，多维场景下不满足broadcast条件时batch维度需要保持一致。

  <!-- npu="950" id2 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`（HF32）。
  <!-- end id2 -->

- **x2**(`Tensor`)：必选参数，即矩阵乘中的`x2`。数据格式支持$ND$，支持非连续的Tensor，当`fused_op_type`为`"relu"`或`""`时，支持输入为两到六维，为`"add"`或`"mul"`时，支持输入为两维(K,N)和三维(B,K,N)，其他情况支持输入维度为两维(K, N)，维度需与`x1`保持一致，多维场景下不满足broadcast条件时batch维度需要保持一致。

  <!-- npu="950" id3 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型必须与输入的`x1`保持一致。
  <!-- end id3 -->

- \*：代表其之前的变量支持按位置输入，也可使用键值对赋值；之后的变量仅支持使用键值对赋值，其中带默认值的变量不赋值时使用默认值，不带默认值的变量必须赋值。
- **bias**(`Tensor`)：**可选参数**，即矩阵乘中的bias，默认值为None。数据格式支持$ND$，不支持非连续的Tensor，支持输入维度为两维(1, N)或一维(N, )。仅当`fused_op_type`为`""`、`"relu"`、`"add"`、`"mul"`时生效，其他情况请使用默认None。

  <!-- npu="950" id4 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。当`x1`数据类型为`torch.bfloat16`，bias可为`torch.float32`、`torch.bfloat16`；当`x1`数据类型为`torch.float16`，bias需为`torch.float16`。
  <!-- end id4 -->

- **x3**(`Tensor`)：**可选参数**，即矩阵乘中的`x3`，默认值为None。数据格式支持$ND$，支持非连续的Tensor。如果`fused_op_type`为`"add"`或`"mul"`，则`x3`为必选，其他情况则`x3`必为None。`x3`的维度需与矩阵乘输出`y`的形状一致或满足batch维广播关系，若输出维度为两维(M,N)，`x3`支持两维(M,N)；若输出维度为三维(B,M,N)，`x3`支持(B,M,N)、(1,M,N)或(M,N)。输入维度为(1,M,N)和(M,N)时，`x3`需在batch维进行广播，必须和输出维度保持一致，不支持M或N维广播。

  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型必须和输入的`x1`保持一致。
  <!-- end id5 -->

- **fused_op_type**(`str`)：必选参数，表示算子融合类型。支持取值：`""`（表示不做融合）、`"16cast32"`（表示不做融合，并且输出为`torch.float32`，当输入`torch.float16`、`torch.bfloat16`时会把输出转成`torch.float32`）、`"add"`、`"mul"`、`"gelu_tanh"`、`"gelu_erf"`和`"relu"`。

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持取值`""`、`"16cast32"`、`"add"`、`"mul"`、`"gelu_tanh"`、`"gelu_erf"`和`"relu"`。
  <!-- end id6 -->
  - 当`fused_op_type`取值为`"gelu_erf"`、`"gelu_tanh"`时，`x1`、`x2`数据类型必须为`torch.bfloat16`、`torch.float16`。
  - 当`fused_op_type`取值为`""`、`"relu"`时，`x1`、`x2`数据类型必须为`torch.bfloat16`、`torch.float16`、`torch.float32`（HF32）。
  - 当`fused_op_type`取值为`"add"`、`"mul"`时，`x1`、`x2`、`x3`数据类型必须为`torch.bfloat16`、`torch.float16`、`torch.float32`（HF32）。

## 返回值说明

`Tensor`

表示最终的计算结果，当`fused_op_type`为`"16cast32"`时，`y`的数据类型为`torch.float32`；当`fused_op_type`非`"16cast32"`时，数据类型与输入`x1`保持一致。数据格式支持$ND$，当`fused_op_type`为`"relu"`或`""`时，支持输出为两到六维，为`"add"`或`"mul"`时，支持输出为两维(M,N)和三维(B,M,N)，其他情况支持输出维度为两维(M, N)。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和静态图模式。
- 当`x1`、`x2`数据类型为`torch.float32`时，调用当前接口必须开启HF32，否则不支持。
- 当`fused_op_type`取值为`"add"`、`"mul"`时，在BMM（三维）场景下，`x1`、`x2`和`y`支持三维；`x3`支持2-3维，二维`x3`可按矩阵广播用于三维输出，三维`x3`的batch轴需要与`y`一致或为1。

## 调用示例

- 单算子模式调用

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np
    M, K, N = 128, 1, 16
    x1 = torch.randn((M, K), dtype=torch.float16)
    x2 = torch.randn((K, N), dtype=torch.float16)
    bias = None
    x3 = torch.randn((M, N), dtype=torch.float16)
    fused_op_type = "add"
    y = torch_npu.npu_fused_matmul(x1.npu(), x2.npu(), bias=bias, x3=x3.npu(), fused_op_type=fused_op_type)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import os
    import sys, getopt
    import random
    if torch.__version__ >= "2.0.0":
        from torch_npu.dynamo import torchair as tng
        from torch_npu.dynamo.torchair import CompilerConfig
        torch._dynamo.config.suppress_errors = True
    os.environ["ENABLE_ACLNN"] = "false"
    class FusedMatmulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, x3, fused_op_type):
            out = torch_npu.npu_fused_matmul(x1, x2, bias=None, x3=x3, fused_op_type=fused_op_type)
            return out
    def executor(device):
        backend = "inductor"
        if device == "npu":
            config = CompilerConfig()
            backend = tng.get_npu_backend(compiler_config=config)
        return backend
    if __name__ == '__main__':
        argv = sys.argv[1:]
        opts, ars = getopt.getopt(argv, "hi:o:", ["npu_mode="])
        npu_mode = ""
        print("begin to run torch_stc_model")
        model = FusedMatmulModel()
        model = model.npu()
        model = torch.compile(model, backend=executor("npu"), dynamic=False, fullgraph=True)

        m = 2
        n = 3
        k = 4
        input_shape_0 = (m, k)
        input_shape_1 = (k, n)
        fused_op_type = random.choice(["", "add", "mul", "gelu_tanh", "gelu_erf", "relu"])
        fmap = torch.from_numpy(np.ones(input_shape_0).astype(np.float16))
        fmap_npu = fmap.npu()
        weight = torch.from_numpy(np.ones(input_shape_1).astype(np.float16))
        weight_npu = weight.npu()
        x3 = torch.from_numpy(np.random.rand(m, n)).to(torch.float16)
        x3_npu = x3.npu()
        if "gelu" in fused_op_type or fused_op_type == "" or fused_op_type == "relu":
            x3_npu = None
        output_data = model(fmap_npu, weight_npu, x3_npu, fused_op_type)
        print("successed!")
        print(output_data.cpu())
    ```
