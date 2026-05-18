# layer_norm

[📄 查看源码](https://gitcode.com/Ascend/triton-ascend/blob/main/third_party/ascend/tutorials/03-layer-norm.py)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     x    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：LayerNorm 算子最早在[BA2016]中被提出，用于提高序列模型（如Transformer）或小批量神经网络的性能。它接收一个向量:math: x 作为输入，并产生一个形状相同的向量:math: y 作为输出。归一化过程是通过减去:math: x 的均值并除以其标准差来实现的。归一化之后，会应用一个带有可学习权重:math: w 和偏置:math: b 的线性变换。
- 计算公式：

  $$
  y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
  $$

## 函数原型

  y = layer_norm(x, w_shape, weight, bias, epsilon)


## 参数说明

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入向量:math: x。</td>
    </tr>
    <tr>
      <td>w_shape</td>
      <td>输入</td>
      <td>权重向量:math: w 的形状。</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>权重向量:math: w。</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>偏置向量:math: b。</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>小常量:math: \epsilon，用于避免除0错误。</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td> 
      <td>Layer Norm归一化输出:math: y。</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 各平台支持数据类型说明：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
    | `x`数据类型 | `weight`数据类型 | `bias`数据类型 | `epsilon`数据类型 |
    | -------- | -------- | -------- | -------- |
    | FLOAT16 | FLOAT32  | FLOAT32 | FLOAT16 |
    | BFLOAT16 | FLOAT32 | FLOAT32 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 |
    | FLOAT32 | FLOAT32  | FLOAT32 | FLOAT32  |

## 调用示例

```python
def _layer_norm(M, N, dtype, eps=1e-5, device='npu'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print(f"y_tri: {y_tri}")
    print(f"y_ref: {y_ref}")
    print(f"Layer Normalization {M},{N} {dtype} PASSED!")
}
```