# torch\_npu.npu\_interleave\_rope

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

## 功能说明

- API功能：针对单输入`x`进行旋转位置编码。
- 计算公式：

    $$q = \text{reshape}(x, [B, N, S, D/2, 2]).\text{transpose}(-1, -2).\text{reshape}([B, N, S, D])$$

    $$q_{\text{embed}} = q \cdot \cos + \text{RotateHalf}(q) \cdot \sin$$

     其中：RotateHalf\(q\) 将张量 `q` 在最后一个维度 D 上执行如下变换：

     1. 将D维平分为前后两部分，前半部分长度为 $D/2$，后半部分长度也为 $D/2$。
     2. 后半部分元素整体前移并取相反数，前半部分元素整体后移，保持原始值。

     数学表示：设 $q = [q_0, q_1, \ldots, q_{D/2-1}, q_{D/2}, q_{D/2+1}, \ldots, q_{D-1}]$，则

     $$\text{RotateHalf}(q) = [-q_{D/2}, -q_{D/2+1}, \ldots, -q_{D-1}, q_0, q_1, \ldots, q_{D/2-1}]$$

     示例：设 $q = [1, 2, 3, 4]$，则前半部分为 $[1, 2]$，后半部分为 $[3, 4]$，变换后得到 $\text{RotateHalf}(q) = [-3, -4, 1, 2]$。

    $$\text{RotateHalf}(q)_i = \begin{cases} -q_{i+D/2}, & i < D/2 \\ q_{i-D/2}, & i \geq D/2 \end{cases}$$

## 函数原型

```python
torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，表示待处理张量。要求为4维张量，shape为\(B, N, S, D\)，数据格式支持$ND$。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`，不支持非连续的Tensor。
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，支持非连续的Tensor。
    <!-- end id5 -->

- **cos** (`Tensor`)：必选参数，表示RoPE旋转位置编码的余弦分量。要求为4维张量，shape为\(B, N, S, D\)，S可以为1或与`x`的S相同，数据类型、数据格式与`x`一致。

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`，不支持非连续的Tensor。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，支持非连续的Tensor。
    <!-- end id7 -->

- **sin** (`Tensor`)：必选参数，表示RoPE旋转位置编码的正弦分量。shape、数据类型、数据格式需要与`cos`保持一致。

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`，不支持非连续的Tensor。
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，支持非连续的Tensor。
    <!-- end id9 -->

## 返回值说明

`Tensor`

表示旋转编码后的结果。shape、数据类型、数据格式与输入`x`保持一致，不支持非连续的Tensor。

## 约束说明

- 该接口仅支持推理场景。
- 该接口支持图模式。

<!-- npu="A3,910b" id10 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：输入`x`、`cos`、`sin`的D维度均必须等于64，且`cos`、`sin`的N维度必须等于1。
<!-- end id10 -->
<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - 输入张量`x`、`cos`、`sin`及输出张量的D维度大小必须等于64。
    - 输入张量`cos`和`sin`的shape必须完全相同，且N维度必须等于1；S维度可以为1或与`x`的S维度相同。
<!-- end id11 -->

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    
    # 生成随机数据
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    # 调用InterleaveRope算子
    q_embed_npu = torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    
    # 生成随机数据
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_npu, cos_npu, sin_npu):
            return torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)
    
    # 调用InterleaveRope算子
    q_embed_npu = model(x_npu, cos_npu, sin_npu)
    ```
