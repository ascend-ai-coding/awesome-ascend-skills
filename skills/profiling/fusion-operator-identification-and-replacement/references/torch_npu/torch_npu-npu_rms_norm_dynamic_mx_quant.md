# torch_npu.npu_rms_norm_dynamic_mx_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：实现RmsNorm算子与DynamicMxQuant算子融合，先对输入进行RmsNorm归一化，将RmsNorm结果与beta相加后再进行动态MX量化，减少搬入搬出操作。其中RmsNorm是大模型常用的归一化操作，相比LayerNorm其去掉了减去均值的部分。

- **计算公式**：
    1. RmsNorm计算

        $$
        \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
        $$

    2. beta与RmsNorm结果相加

        $$
        y=\operatorname{RmsNorm}(x)+beta
        $$

    3. DynamicMxQuant量化
        - **场景1：当scale_alg为0时，支持fp4、fp8的动态MX量化。**

            将RmsNorm输出y在尾轴维度上按k=blocksize个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$，1<=i<=k，blocksize=32。

            $$
            shared\_exp = floor(log_2(max_i(|V_i|))) - emax
            $$

            $$
            mxscale = 2^{shared\_exp}
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
            $$

            - 量化后的$P_i$按$V_i$在原始x张量中的位置组成输出y，mxscale按对应axis维度上的分组组成输出mxscale_out。

            - 公式中emax指对应数据类型的最大正规数的指数位，对应关系如下：

                | dst_type | emax |
                | --- | --- |
                | `torch.float4_e2m1fn_x2` | 2 |
                | `torch.float4_e1m2fn_x2` | 0 |
                | `torch.float8_e4m3fn` | 8 |
                | `torch.float8_e5m2` | 15 |

        - **场景2：当scale_alg为1时，支持Ffp8的动态MX量化。**
            - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型fp8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
            - 找到该块中数值的最大绝对值：

                $$
                Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
                $$

            - 将`torch.float32`映射到目标数据类型fp8可表示的范围内，其中Amax\(DType\)是目标精度能表示的最大值

                $$
                S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
                $$

            - 将块缩放因子$S_{`torch.float32`}^b$转换为fp8格式下可表示的缩放值$S_{ue8m0}^b$；
            - 从块的浮点缩放因子$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$；
            - 为保证量化时不溢出，对指数进行向上取整，且在fp8可表示的范围内：

                $$
                E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
                $$

            - 计算块缩放因子：

                $S_{ue8m0}^b=2^{E_{int}^b}$

            - 计算块转换因子：

                $R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$

            - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^n)$，最终输出的量化结果是$(S^b, [d^i]_{i=1}^{k})$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^{k}$代表块内量化后的数据。

## 函数原型

```python
torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, *, beta=None, epsilon=1e-06, scale_alg=0, round_mode="rint", dst_type=torch.float4_e2m1fn_x2) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示进行归一化计算的输入，即公式中的$x$。shape支持1-7维，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式要求为$ND$，支持空Tensor，支持非连续的Tensor。当输出`y`的类型为`torch.float4_e2m1fn_x2`和`torch.float4_e1m2fn_x2`，即dst_type为296和297时，`x`的最后一维必须是偶数。
- **gamma**（`Tensor`）：必选参数，表示RmsNorm的缩放因子，即公式中的$gamma$。shape支持1维，shape需要与`x`的最后一维保持一致，数据类型需要与`x`保持一致或者为`torch.float32`，数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **beta**（`Tensor`）：可选参数，表示添加到RmsNorm结果上的偏置张量，即公式中的$beta$。若存在，shape、数据类型和数据格式需要与`gamma`保持一致，支持空Tensor，支持非连续Tensor。
- **epsilon**（`float`）：可选参数，表示添加到分母中的值，以确保数值稳定，即公式中的$epsilon$。数据类型为`torch.double`，默认值为1e-06。
- **scale_alg**（`int`）：可选参数，表示`mxscale_out`的计算方法。int类型，支持取0（默认值）和1。当`dst_type`为296(`torch.float4_e2m1fn_x2`)、297(`torch.float4_e1m2fn_x2`)时仅支持取0，当`dst_type`为291(`torch.float8_e5m2`)、292(`torch.float8_e4m3fn`)时支持取值为0和1，默认值为0。
  - 取值0：表示场景1，表示OCP（Open Compute Project）实现。
  - 取值1：表示场景2，表示cuBLAS实现。

- **round_mode**（`str`）：可选参数，表示指定量化结果cast到输出`y`的数据类型模式。当dst_type为296(`torch.float4_e2m1fn_x2`)、297(`torch.float4_e1m2fn_x2`)时，模式支持{"rint", "floor", "round"}；当dst_type为291(`torch.float8_e5m2`)、292(`torch.float8_e4m3fn`)时，模式仅支持"rint"，默认值为"rint"。
- **dst_type**（`int`）：可选参数，指定输出y的数据类型，输入范围为291（`torch.float8_e5m2`）、292（`torch.float8_e4m3fn`）、296（`torch.float4_e2m1fn_x2`，默认值）、297（`torch.float4_e1m2fn_x2`）。

## 返回值说明

- **y**（`Tensor`）：表示RmsNorm归一化后与`beta`相加，再进行DynamicMxQuant量化后的输出结果，支持空Tensor，shape和数据格式与输入`x`保持一致，数据类型由`dst_type`指定，当`dst_type`为296(`torch.float4_e2m1fn_x2`)、297(`torch.float4_e1m2fn_x2`)时，y的数据类型实际为`torch.uint8`。
- **mxscale_out**（`Tensor`）：表示每个分组对应的量化尺度，数据类型为`torch_npu.float8_e8m0`，数据格式要求为$ND$，实际数据类型为`torch.uint8`，shape支持2-8维。shape在尾轴上为`x`对应值除以32向上取整，并对其进行偶数pad，pad填充值为0，具体计算过程见约束说明。
- **rstd_out**（`Tensor`）：表示RmsNorm归一化后的标准差的倒数，用于归一化操作，对应公式中的$Rms(x)$的倒数。shape与`x`的前几维保持一致，前几维表示不需要norm的维度，数据格式要求为$ND$。`rstd_out`shape与`x`shape、`gamma`shape关系举例：若`x`shape=(2,3,4,8)，`gamma`shape=(8,)，rstd_out.shape=(2,3,4,1)。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输出`y`的类型为`torch_npu.float4_e2m1fn_x2`和`torch_npu.float4_e1m2fn_x2`，即dst_type为296和297时，`x1`的最后一维必须是偶数。
- 量化轴和norm轴都是输入x的最后1维。
- 输出`mxscale_out`和输入`x1`的shape约束关系：
  - rank(mxscale_out)  = rank(x1) + 1
  - mxscale_out.shape[-2] = (ceil(x1.shape[-1] / 32) + 2 - 1) / 2
  - mxscale_out.shape[-1]  = 2
  - 其他维度与输入x1一致
  - 例如，输入x1.shape = (100, 8, 64)，mxscale_out.shape = (100, 8, 1, 2)

- **单算子模式下**，算子通过读取输入Tensor(x)的requires_grad属性来决定是否输出有效的rstd。requires_grad是Pytorch Tensor的标准属性，默认值为False，当requires_grad=False时，算子不写出rstd，接口返回shape[0]的空Tensor，此时rstd为无效占位输出。
- **图模式通路**不支持反向传播，rstd固定为值为0的标量Tensor(shape[])，不含有效数据。

- **边界值场景说明**

  - 当输入是Inf时：
    - 输出y为0;
    - 输出mxscale_out为255，偶数pad填充值为0;
    - 输出rstd_out为0。

  - 当输入是NaN时：
    - 输出y为0;
    - 输出mxscale_out为255，偶数pad填充值为0;
    - 输出rstd_out为NaN。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([8, 64], dtype=torch.float16, requires_grad=True).npu()
    gamma = torch.randn([64], dtype=torch.float16).npu()
    beta = torch.randn([64], dtype=torch.float16).npu()
    y_npu, mxscale_npu, rstd_npu = torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, beta=beta, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float8_e5m2)
    y = y_npu.cpu()
    mxscale = mxscale_npu.cpu()
    rstd = rstd_npu.cpu()
    print(f"y dtype = {y.dtype}, y shape = {y.shape}")
    print(f"mxscale dtype = {mxscale.dtype}, mxscale shape = {mxscale.shape}")
    print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, gamma, beta):
            return torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, beta=beta, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float8_e5m2)

    x = torch.randn([8, 64], dtype=torch.float16).npu()
    gamma = torch.randn([64], dtype=torch.float16).npu()
    beta = torch.randn([64], dtype=torch.float16).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y_npu, mxscale_npu, rstd_npu = model(x, gamma, beta)
    y = y_npu.cpu()
    mxscale = mxscale_npu.cpu()
    rstd = rstd_npu.cpu()
    print(f"y dtype = {y.dtype}, y shape = {y.shape}")
    print(f"mxscale dtype = {mxscale.dtype}, mxscale shape = {mxscale.shape}")
    print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
    ```
