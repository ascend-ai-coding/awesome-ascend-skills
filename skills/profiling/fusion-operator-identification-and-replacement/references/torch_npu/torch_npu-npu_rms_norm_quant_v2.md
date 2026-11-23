# torch_npu.npu_rms_norm_quant_v2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：RMSNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuantV2算子将RMSNorm算子以及RMSNorm后的Quantize算子融合起来，减少搬入搬出操作。同时在RmsNormQuant算子的基础上新增了Rstd的输出。
- 计算公式：
  - RMSNorm计算过程：

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 量化计算过程：
    - 当`div_mode`为True时：

    $$
    y=round((quant\_in/scale)+offset)
    $$

    - 当`div_mode`为False时：

    $$
    y=round((quant\_in*scale)+offset)
    $$

## 函数原型

```python
torch_npu.npu_rms_norm_quant_v2(x, gamma, scale, *, offset, beta, epsilon=1e-06, div_mode=True, dst_dtype=None) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，表示标准化过程中的源数据张量，对应公式中的$x$，数据格式支持$ND$，shape支持1-8维，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，支持非连续的Tensor，不支持空Tensor。

- **gamma** (`Tensor`)：必选参数，表示标准化过程中的缩放张量，对应公式中的$gamma$，数据格式支持$ND$，shape支持1-2维，若shape为1维，则需与`x`最后一维维度一致；若shape为2维，则第一维必须为1，第二维需与`x`最后一维维度一致。数据类型需与`x`保持一致，支持非连续Tensor，不支持空Tensor。

- **scale** (`Tensor`)：必选参数，表示量化过程中得到`y`的scales张量，对应公式中的$scale$。数据格式支持$ND$，shape为1，维度为1。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，支持非连续的Tensor，不支持空Tensor。该参数的值不能为0。

- **offset** (`Tensor`)：可选参数，表示量化过程中的偏移张量，对应公式中的$offset$，数据格式支持$ND$，shape需与`scale`保持一致，数据类型支持`torch.int8`、`torch.int32`、`torch.float16`、`torch.bfloat16`、`torch.float32`，支持非连续Tensor，不支持空Tensor。

- **beta** (`Tensor`)：可选参数，表示标准化过程中的偏移张量，对应公式中的$beta$，shape、数据类型和数据格式需要与`gamma`保持一致，支持非连续Tensor，不支持空Tensor。

- **epsilon** (`float`)：可选参数，对应公式中的$epsilon$，用于防止除零错误。建议传入较小的正数，默认值为1e-6。

- **div_mode** (`bool`)：可选参数，对应公式中的$div\_mode$，决定量化公式是否使用除法的参数，默认值为True。

- **dst_dtype** (`int`)：可选参数，指定量化输出`y`的类型，默认值为None。支持取值`torch.int8`、`torch.quint4x2`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`，传None时当做`torch.int8`处理。

## 返回值说明

- **y** (`Tensor`)：返回结果，对应公式中的$y$，即最终量化输出张量，数据类型由`dst_dtype`指定。当`dst_dtype`是`torch.quint4x2`时，`y`的数据类型为`torch.int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`torch.int32`元素包含8个`torch_npu.int4`结果。其他场景下`y`形状与输入`x`一致。

- **rstd** (`Tensor`)：返回结果，对应公式中$Rms(x)$的倒数，表示归一化后均方根的倒数。数据类型仅支持`torch.float32`。维度数与`x`保持一致，不需要norm的维度与`x`对应维度保持一致，需要norm的维度为1。`rstd`shape与`x`shape、`gamma`shape关系举例：若`x`shape=(2,3,4,8)，`gamma`shape=(8,)，`rstd`shape=(2,3,4,1)。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口仅支持单算子模式调用。
- 当`dst_dtype`取`torch.quint4x2`时，`x`、`gamma`以及`beta`的最后一维必须为偶数，并且`x`最后一维必须能够被8整除。
- 当`y`的数据类型为`torch.int32`时，`y`的最后一维必须是`x`最后一维的1/8。
- 该接口支持数据类型说明：

    | x | gamma | scale | offset | beta | epsilon | y | rstd |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.int8` | `torch.float16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8` | `torch.bfloat16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float32` | `torch.int32` | `torch.float16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.int32` | `torch.bfloat16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.float16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.bfloat16` | torch.double | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |

- **单算子模式下**，算子通过读取输入Tensor(x)的requires_grad属性来决定是否输出有效的rstd。requires_grad是PyTorch Tensor的标准属性，默认值为False，当requires_grad=False时，算子不写出rstd，接口返回shape[0]的空Tensor，此时rstd为无效占位输出。

## 调用示例

```python
import torch
import torch_npu

x = torch.randn([8, 64], dtype=torch.float16, requires_grad=True).npu()
gamma = torch.randn([64], dtype=torch.float16).npu()
scale = torch.ones(1, dtype=torch.float16).npu()
offset = torch.zeros(1, dtype=torch.float16).npu()
beta = torch.randn([64], dtype=torch.float16).npu()
y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x, gamma, scale, offset=offset, beta=beta, epsilon=1e-6, div_mode=True, dst_dtype=torch.int8)
y = y_npu.cpu()
rstd = rstd_npu.cpu()
print(f"y dtype = {y.dtype}, y shape = {y.shape}")
print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
```
