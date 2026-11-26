# torch_npu.npu_masked_causal_conv1d

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：对hidden层的token之间进行带mask的因果一维分组卷积操作。

- 计算公式：

    对给定的输入张量`input`，shape为[S, B, H]，卷积权重`weight`，shape为[W, H]（W=3），进行以下计算：

    1. 对`input`沿序列维度进行因果零填充（在序列起始处填充W-1个零），然后执行逐通道（depthwise）一维卷积：

        $$
        \text{output}[s, b, h] = \sum_{k=0}^{W-1} \text{weight}[k, h] \cdot \text{input}[s - (W-1-k), b, h]
        $$

        其中越界的`input`索引视为0（因果零填充）。

    2. 若提供了`mask`（shape为[B, S]，true表示有效位置），则对输出进行掩码操作：

        $$
        \text{output}[s, b, :] = 0, \quad \text{if } \text{mask}[b, s] = \text{false}
        $$

## 函数原型

```python
torch_npu.npu_masked_causal_conv1d(input, weight, *, mask=None) -> Tensor
```

## 参数说明

> **说明：**
>
> - `input`、`weight`参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、W（Window Size）表示卷积窗口大小。

- **input** (`Tensor`)：必选参数，表示卷积输入张量。支持非连续Tensor，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`，shape为[S, B, H]。
- **weight** (`Tensor`)：必选参数，表示卷积权重张量。支持非连续Tensor，数据格式支持$ND$，数据类型与`input`一致，shape为[W, H]，W目前只支持3。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **mask** (`Tensor`)：可选参数，表示卷积输出的掩码。不支持非连续的Tensor，数据格式支持$ND$，数据类型支持`torch.bool`，shape为[B, S]，true表示有效位置，false表示需要置零的位置。默认值为None，表示不进行掩码操作。

## 返回值说明

`Tensor`

代表公式中的`output`，表示因果卷积的输出结果，shape和数据类型与`input`一致，数据格式为$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    S, B, H, W = 2048, 4, 768, 3
    input  = torch.randn(S, B, H, dtype=torch.bfloat16).npu()
    weight = torch.randn(W, H, dtype=torch.bfloat16).npu()
    mask   = torch.rand(B, S).npu() > 0.3  # bool [B, S]

    output = torch_npu.npu_masked_causal_conv1d(input, weight, mask=mask)
    # output shape: [S, B, H]
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MaskedCausalConv1dModel(torch.nn.Module):
        def forward(self, input, weight, mask):
            return torch_npu.npu_masked_causal_conv1d(input, weight, mask=mask)

    S, B, H, W = 2048, 4, 768, 3
    input  = torch.randn(S, B, H, dtype=torch.bfloat16).npu()
    weight = torch.randn(W, H, dtype=torch.bfloat16).npu()
    mask   = torch.rand(B, S).npu() > 0.3

    model = MaskedCausalConv1dModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(input, weight, mask)
    ```
