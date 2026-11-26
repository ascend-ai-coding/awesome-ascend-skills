# torch_npu.npu_fused_floyd_attention

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：训练场景下，npu_fused_floyd_attention相较于传统FA(npu_fusion_attention)主要是计算QK/PV注意力时，会额外将维度(seq)作为batch轴处理，从而将注意力计算转换为批量矩阵乘法(batchMatmul)。
- 计算公式：

$$
\text{weights} = \text{Softmax}\left(\text{atten\_mask} + \text{scale\_value} \cdot \left(\text{einsum}(\text{query}, \text{key}_1^T) + \text{einsum}(\text{query}, \text{key}_2^T)\right)\right)
$$

$$
\text{attention\_out} = \text{einsum}(\text{weights}, \text{value}_1) + \text{einsum}(\text{weights}, \text{value}_2)
$$

## 函数原型

```python
torch_npu.npu_fused_floyd_attention(query_ik, key_ij, value_ij, key_jk, value_jk, *, atten_mask=None, scale_value=1.) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **query_ik** (`Tensor`)：必选参数，输入张量，对应公式中的$query$，数据类型支持`torch.bfloat16`、`torch.float16`。数据格式支持$ND$，输入shape支持[B,H,N,M,D]。
- **key_ij** (`Tensor`)：必选参数，输入张量，对应公式中的$key_1$，代表从节点i到其直接邻居j的关系或特征，数据类型支持`torch.bfloat16`、`torch.float16`。数据格式支持$ND$，输入shape支持[B,H,N,K,D]。
- **value_ij** (`Tensor`)：必选参数，输入张量，对应公式中的$value_1$，代表从节点i到其直接邻居j的信息内容，数据类型支持`torch.bfloat16`、`torch.float16`。数据格式支持$ND$，输入shape支持[B,H,N,K,D]。
- **key_jk** (`Tensor`)：必选参数，输入张量，对应公式中的$key_2$，代表从直接邻居j到支点k的关系或特征，数据类型支持`torch.bfloat16`、`torch.float16`，数据格式支持$ND$，输入shape支持[B,H,K,M,D]。
- **value_jk** (`Tensor`)：必选参数，输入张量，对应公式中的$value_2$，代表从节点j到其直接邻居k的信息内容，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$，输入shape支持[B,H,K,M,D]。
- **atten_mask** (`Tensor`)：可选参数，输入张量，对应公式中的$atten\_mask$，数据类型支持`torch.bool`、`torch.uint8`，数据格式支持$ND$，输入shape类型需为[B,1,N,1,K]，取值为1代表该位不参与计算，为0代表该位参与计算。默认值为None。
- **scale_value** (`float`)：可选参数，代表缩放系数，对应公式中的$scale\_value$，数据类型支持`float`。默认值为1。

## 返回值说明

- **softmax_max_out** (`Tensor`)：输出张量，Softmax计算的Max中间结果，用于反向计算。数据类型支持`torch.float`，输出的shape类型为[B,H,N,M,8]。数据格式支持$ND$。
- **softmax_sum_out** (`Tensor`)：输出张量，Softmax计算的Sum中间结果，用于反向计算。数据类型支持`torch.float`，输出的shape类型为[B,H,N,M,8]。数据格式支持$ND$。
- **attention_out** (`Tensor`)：输出张量，计算公式的最终输出，对应公式中的$attention\_out$。数据类型支持`torch.bfloat16`、`torch.float16`。数据类型和shape类型与`query_ik`保持一致，数据格式支持$ND$，输出shape支持[B,H,N,M,D]。

## 约束说明

- 关于数据shape的约束，其中：

    B：取值范围为1~2K。

    H：取值范围为1~256。

    N：取值范围为16~1M且N%16==0。

    M：取值范围为128~1M且M%128==0。

    K：取值范围为128~1M且K%128==0。

    D：只支持32/64/128。

- `query_ik`与`key_ij`的第0/2/4轴需相同。
- `key_ij`与`value_ij`的shape需相同。
- `key_jk`与`value_jk`的shape需相同。
- `softmax_max_out`与`softmax_sum_out`的shape需相同。
- `query_ik`、`key_ij`、`value_ij`、`key_jk`和`value_jk`的数据类型需保持一致。
- 支持PyTorch2.6.0及以上版本。
- 该接口的反向计算不支持确定性。

## 调用示例

```python
import torch
import torch_npu
import math

def truncated_normal(mean, std, min, max, size):
    x = torch.normal(mean, std, size=size)
    x = torch.where((x < min) | (x > max), torch.tensor(0.0), x)
    return x
B, N, S1, S2, S3, D = 1, 1, 16, 256, 256, 64
Q = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()
K1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
K2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
V1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
V2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
atten_mask = torch.randint(0, 2, [B, 1, S1, 1, S3]).to(torch.bool).npu()
scale = 1.0/math.sqrt(D)
x_max_npu, x_sum_npu, output_npu = torch_npu.npu_fused_floyd_attention(
    Q,
    K1,
    V1,
    K2,
    V2,
    atten_mask = atten_mask,
    scale_value = scale
)
```
