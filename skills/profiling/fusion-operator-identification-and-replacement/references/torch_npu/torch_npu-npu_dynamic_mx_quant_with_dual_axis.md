# torch_npu.npu_dynamic_mx_quant_with_dual_axis

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：在输入张量的-1轴和-2轴上同时进行目标数据类型为`float4`（`torch_npu.float4_e1m2fn_x2`，`torch_npu.float4_e2m1fn_x2`）、`float8`（`torch.float8_e4m3fn`，`torch.float8_e5m2`）的MX量化。在-1轴和-2轴上，每32个数计算出对应的量化尺度mxscale1、mxscale2作为输出mxscale1、mxscale2的对应部分，然后分别将两组数所有元素除以对应的量化尺度，根据round\_mode转换到对应的dst\_type，得到量化结果y1和y2。
  - 合轴说明：算子实现时，会对-2轴（不包含）之前的所有轴进行合轴处理。即对于输入shape为$(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$的张量，-2轴之前的维度$(d_0, d_1, ..., d_{n-3})$会被合并为一个维度，等效于将输入reshape为$(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$后再进行量化计算。
- 计算公式：
  - 场景1，当scale\_alg为0时，即OCP Microscaling Formats (Mx) Specification实现：
    - 将输入x在-1轴上按照32个数进行分组，一组32个数$\{\{V_i\}_{i=1}^{32}\}$量化为$\{mxscale1, \{P_i\}_{i=1}^{32}\}$：

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      mxscale1 = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - 同时，将输入x在-2轴上按照32个数进行分组，一组32个数$\{\{V_j\}_{j=1}^{32}\}$量化为$\{mxscale2, \{P_j\}_{j=1}^{32}\}$：

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      mxscale2 = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - -1轴量化后的$P_i$按对应的$V_i$的位置组成输出y1，mxscale1按对应的-1轴维度上的分组组成输出mxscale1。-2轴量化后的$P_j$按对应的$V_j$的位置组成输出y2，mxscale2按对应的-2轴维度上的分组组成输出mxscale2。
    - emax：对应数据类型的最大正规数的指数位，对应关系如下表：

      | dst_type | emax |
      | --- | --- |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - 场景2，当scale\_alg为1时，只涉及float8（CuBALS Scale计算算法）：
    - -1轴量化：将输入x在-1轴上按照32个数进行分组，每组长度为32，对每组单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把组内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型float8。如果最后一组不足32个元素，把缺失值视为0，按照完整组处理。找到该组中数值的最大绝对值：

      $$
      Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{32})
      $$

      将`torch.float32`映射到目标数据类型为float8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：

      $$
      S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
      $$

      将块缩放因子$S_{`torch.float32`}^b$转换为fp8格式下可表示的缩放值$S_{ue8m0}^b$，从$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，为保证量化时不溢出，对指数进行向上取整，且在fp8可表示的范围内：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

      计算块缩放因子$S_{ue8m0}^b=2^{E_{int}^b}$，块转换因子$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$，对每个组内元素$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$，最终-1轴输出的量化结果是$\left(S^b, [d^i]_{i=1}^{32}\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$）。
    - -2轴量化：同时，将输入x在-2轴上按照32个数进行分组，采用与-1轴相同的CuBALS Scale计算算法，对每组独立计算块缩放因子并量化，-2轴输出的量化结果是$\left(S^b, [d^j]_{j=1}^{32}\right)$。
    - -1轴量化结果组成输出y1，对应的块缩放因子组成输出mxscale1。-2轴量化结果组成输出y2，对应的块缩放因子组成输出mxscale2。

  - 场景3，当scale\_alg为2时，只涉及`torch_npu.float4_e2m1fn_x2`类型：
    - 当dst\_type\_max为0.0/6.0/7.0时，将输入x在-1轴和-2轴上分别按照32个数进行分组，动态量化为$\{mxscale, \{P\}\}$：

      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前1/2位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{否则} \end{cases}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - 当dst\_type\_max不为上述特殊取值时，将输入x在-1轴和-2轴上分别按照32个数进行分组，每组长度为32，采用与场景2相同的块缩放算法：对每组单独计算块缩放因子$S_{`torch.float32`}^b$，映射到目标低精度类型`torch_npu.float4_e2m1fn_x2`。其中$Amax(DType)$在dst\_type\_max为0时是目标精度能表示的最大值，不为0时是dst\_type\_max传入值：

      $$
      S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
      $$

      指数向上取整规则为：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

      其余步骤（计算$S_{ue8m0}^b$、$R_{`torch.float32`}^b$、$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$）与场景2相同。
    - -1轴量化结果组成输出y1，对应的块缩放因子组成输出mxscale1。-2轴量化结果组成输出y2，对应的块缩放因子组成输出mxscale2。

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant_with_dual_axis(input, *, round_mode="rint", dst_type=296, scale_alg=0,
                                              dst_type_max=0.0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，表示输入x。维度支持2-7维，数据格式支持ND，数据类型支持`torch.float16`、`torch.bfloat16`。支持非连续Tensor。当dst\_type为float4（即296、297对应的类型）时，input的最后一维必须是偶数。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round\_mode**（`str`）：可选参数，表示数据转换的模式，对应公式中的$round\_mode$。默认值为"rint"。
  - 当dst\_type为`torch.float8_e5m2`、`torch.float8_e4m3fn`时，仅支持"rint"。
  - 当dst\_type为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`时，支持"rint"、"floor"、"round"。
- **dst\_type**（`int`）：可选参数，表示数据转换后y1和y2的数据类型。默认值为296，支持取值：
  - 23：`torch.float8_e5m2`。
  - 24：`torch.float8_e4m3fn`。
  - 296：`torch_npu.float4_e2m1fn_x2`。
  - 297：`torch_npu.float4_e1m2fn_x2`。
- **scale\_alg**（`int`）：可选参数，表示mxscale1和mxscale2的计算方法。支持取值为0、1、2，默认值为0。
  - 0：场景1（OCP MX规格实现），float8、float4均支持。
  - 1：场景2（CuBALS Scale计算算法），仅float8支持。
  - 2：场景3（fp4动态量化），仅`torch_npu.float4_e2m1fn_x2`支持。
- **dst\_type\_max**（`float`）：可选参数，表示maxType的取值，对应公式中的$Amax(DType)$。默认值为0.0。仅支持在dst\_type为`torch_npu.float4_e2m1fn_x2`且scale\_alg为2时设置该值，支持取值0.0和6.0-12.0。取值为0.0代表Amax(DType)为量化结果数据类型的最大值，取值为6.0-12.0代表Amax(DType)为传入值。

## 返回值说明

- **y1**（`Tensor`）：表示输入x量化-1轴后的对应结果。shape和输入x一致。当dst\_type为float8（23，24）时，数据类型为`torch.float8_e5m2`或`torch.float8_e4m3fn`；当dst\_type为float4（296，297）时，实际返回的数据类型为`torch.uint8`，最后一维为输入x最后一维的一半（每个`torch.uint8`元素打包2个float4数据），查看具体值需自行解包，如`y1.view(torch_npu.float4_e2m1fn_x2)`。不支持非连续Tensor。
- **mxscale1**（`Tensor`）：表示-1轴每个分组对应的量化尺度，对应公式中的$mxscale1$。数据类型为`torch.float8_e8m0fnu`，实际返回的数据类型为`torch.uint8`，查看具体值需自行转换，如`mxscale1.view(torch.float8_e8m0fnu)`。shape为输入x的-1轴的值除以32向上取整，并对其进行偶数pad（pad填充值为0），最后追加一维大小为2。不支持非连续Tensor。
- **y2**（`Tensor`）：表示输入x量化-2轴后的对应结果。shape和输入x一致。数据类型与y1相同（float4时实际返回的数据类型为`torch.uint8`，最后一维为输入x最后一维的一半）。不支持非连续Tensor。
- **mxscale2**（`Tensor`）：表示-2轴每个分组对应的量化尺度，对应公式中的$mxscale2$。数据类型为`torch.float8_e8m0fnu`，实际返回的数据类型为`torch.uint8`，查看具体值需自行转换。shape为输入x的-2轴的值除以32向上取整，并对其进行偶数pad（pad填充值为0），最后追加一维大小为2，且输出需要对每两行数据进行交织处理。不支持非连续Tensor。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 确定性计算：默认确定性实现。
- dst\_type、scale\_alg、round\_mode、dst\_type\_max的参数组合约束如下表：

| dst_type | scale_alg | round_mode | dst_type_max |
| --- | --- | --- | --- |
| `torch.float8_e5m2` | 0、1 | "rint" | 不支持设置（仅默认值0.0） |
| `torch.float8_e4m3fn` | 0、1 | "rint" | 不支持设置（仅默认值0.0） |
| `torch_npu.float4_e2m1fn_x2` | 0、2 | "rint"/"floor"/"round" | 仅scale_alg=2时生效，支持0.0和6.0-12.0 |
| `torch_npu.float4_e1m2fn_x2` | 0 | "rint"/"floor"/"round" | 不支持设置（仅默认值0.0） |

- 关于input、mxscale1、mxscale2的shape约束说明如下：
  - rank(mxscale1) = rank(input) + 1。
  - rank(mxscale2) = rank(input) + 1。
  - mxscale1.shape[-2] = (ceil(input.shape[-1] / 32) + 2 - 1) // 2。
  - mxscale2.shape[-3] = (ceil(input.shape[-2] / 32) + 2 - 1) // 2。
  - mxscale1.shape[-1] = 2。
  - mxscale2.shape[-1] = 2。
  - 其他维度与输入input一致。
  - 举例：输入input的shape为[B, M, N]，目的数据类型为float8时，对应的y1和y2的shape为[B, M, N]，mxscale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，mxscale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用示例

- 单算子模式调用，float8量化（`scale_alg`使用默认值0）

    ```python
    import torch
    import torch_npu
    
    # input的shape为[1, 4]，取值为(0, 8, 64, 512)
    input = torch.tensor([[0, 8, 64, 512]], dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        input, dst_type=torch.float8_e4m3fn)
    print(y1)
    print(mxscale1.shape, mxscale1.dtype)
    print(mxscale1.cpu().view(torch.float8_e8m0fnu))
    print(y2.shape, y2.dtype)
    print(mxscale2.shape, mxscale2.dtype)
    print(mxscale2.cpu().view(torch.float8_e8m0fnu))
    ```

    输出如下所示

    ```text
    tensor([[  0.,   4.,  32., 256.]], device='npu:0', dtype=torch.float8_e4m3fn)
    torch.Size([1, 1, 2]) torch.uint8
    tensor([[[2., 0.]]], dtype=torch.float8_e8m0fnu)
    torch.Size([1, 4]) torch.float8_e4m3fn
    torch.Size([1, 4, 2]) torch.uint8
    tensor([[[0.0000, 0.0000],
             [0.0312, 0.0000],
             [0.2500, 0.0000],
             [2.0000, 0.0000]]], dtype=torch.float8_e8m0fnu)
    ```

- 单算子模式调用，float4量化（`scale_alg`=2）

    ```python
    import torch
    import torch_npu
    
    # dst_type为float4时，input的最后一维必须为2的整数倍
    input = torch.randn((64, 256), dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        input, dst_type=torch_npu.float4_e2m1fn_x2, scale_alg=2, dst_type_max=6.0)
    print(y1.shape, y1.dtype)
    print(mxscale1.shape)
    print(mxscale2.shape)
    ```

    输出如下所示

    ```text
    torch.Size([64, 128]) torch.uint8
    torch.Size([64, 4, 2])
    torch.Size([1, 256, 2])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch_npu.npu_dynamic_mx_quant_with_dual_axis(x, dst_type=torch.float8_e4m3fn)

    model = Model().npu()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    input = torch.randn((64, 256), dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = model(input)
    print(y1.shape, y1.dtype)
    print(mxscale1.shape)
    print(y2.shape, y2.dtype)
    print(mxscale2.shape)
    ```

    输出如下所示

    ```text
    torch.Size([64, 256]) torch.float8_e4m3fn
    torch.Size([64, 4, 2])
    torch.Size([64, 256]) torch.float8_e4m3fn
    torch.Size([1, 256, 2])
    ```
