# torch\_npu.npu\_dynamic\_mx\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：目的数据类型为float4、float8的动态MX量化。在给定的轴`axis`上，根据每`block_size`个数，计算出这组数对应的量化尺度`mxscale`，然后对这组数每一个除以`mxscale`，根据`round_mode`转换到对应的`dst_type`，得到量化结果`y`。在`dst_type`为`torch.float8_e5m2`、`torch.float8_e4m3fn`时，根据`scale_alg`的取值来指定计算`mxscale`的不同算法。

- 计算公式：

  - 场景1，当`scale_alg`为0时：

    - 将输入`input`在`axis`维度上按k = `block_size`个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = `block_size`
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size\\
    $$

    - 量化后的$P_i$按对应的$V_i$的位置组成输出`y`，`mxscale`按对应的`axis`维度上的分组组成输出`mxscale`。

    - emax：对应数据类型的最大正则数的指数位。

      | dst_type | emax |
      | :---: | :---: |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - 场景2，当`scale_alg`为1时，仅适用于`torch.float8_e4m3fn`、`torch.float8_e5m2`类型：

    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值：
    $$
    Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
    - 当`max_low_bound`大于0时，对最大绝对值进行下界钳位：
    $$
    Amax(D_{`torch.float32`}^b)=max(Amax(D_{`torch.float32`}^b), max\_low\_bound)
    $$
    - 将`torch.float32`映射到目标数据类型可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：
    $$
    S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
    $$
    - 将块缩放因子$S_{`torch.float32`}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
    - 从块的浮点缩放因子$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$），$[d^i]_{i=1}^k$代表块内量化后的数据。

  - 场景3，当`scale_alg`为2时，仅适用于`torch_npu.float4_e2m1fn_x2`类型：

    - 当`dst_type_max`为0.0、6.0或7.0时：
      - 将输入`input`在`axis`维度上按k = `block_size`个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = `block_size`：
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前一/两位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases}
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size
      $$
      - 量化后的$P_i$按对应的$V_i$的位置组成输出`y`，`mxscale`按对应的`axis`维度上的分组组成输出`mxscale`。
    - 当`dst_type_max`不为0.0、6.0或7.0时：
      - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
      - 找到该块中数值的最大绝对值：
      $$
      Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - 将`torch.float32`映射到目标数据类型可表示的范围内，其中当`dst_type_max`为0时$Amax(DType)$为目标精度能表示的最大值，当`dst_type_max`不为0时$Amax(DType)$为`dst_type_max`传入值：
      $$
      S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
      $$
      - 将块缩放因子$S_{`torch.float32`}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
      - 从块的浮点缩放因子$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
      - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
      $$
      - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
      - 计算块转换因子：$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$
      - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$），$[d^i]_{i=1}^k$代表块内量化后的数据。

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant(input, *, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0, dst_type_max=0.0, max_low_bound=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape支持1-7维度，支持非连续的Tensor，数据格式支持$ND$。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **axis** (`int`)：可选参数，量化发生的轴，取值范围为[-D, D-1]，D为x的维数，默认值为-1。
- **round_mode** (`str`)：可选参数，数据转换的模式，默认值为`"rint"`。
  - 当`dst_type`为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`时，支持取值`"rint"`、`"floor"`、`"round"`。
  - 当`dst_type`为`torch.float8_e4m3fn`或`torch.float8_e5m2`时，仅支持取值`"rint"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。支持的类型为`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`，默认值为`torch_npu.float4_e2m1fn_x2`。
- **block_size** (`int`)：可选参数，指定每次量化的元素个数，必须是32的倍数，取值范围为(0, 1024]，默认值为32。当`scale_alg`为2时，仅支持取值32。当`input`的数据类型为`torch.float32`时，仅支持取值32，且`axis`对应的维度大小不能小于32。
- **scale_alg** (`int`)：可选参数，指定计算`mxscale`的算法，默认值为0。取值范围：0代表场景1、1代表场景2、2代表场景3。
  - 当`dst_type`为`torch_npu.float4_e1m2fn_x2`时仅支持取值为0。
  - 当`dst_type`为`torch_npu.float4_e2m1fn_x2`时仅支持取值为0和2。
  - 当`dst_type`为`torch.float8_e4m3fn`或`torch.float8_e5m2`时仅支持取值为0和1。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值。支持取值0.0和6.0-12.0，取值为0.0时使用目标精度能表示的最大值，取值为6.0-12.0时使用传入值，默认值为0.0。仅支持在`dst_type`为`torch_npu.float4_e2m1fn_x2`且`block_size`为32时设置该值。
- **max_low_bound** (`float`)：可选参数，每个block计算出的最大绝对值的下界钳位值，默认值为0.0表示不进行钳位。仅当`scale_alg`为1时生效，`scale_alg`不为1时必须为0.0。取值为非负数，当取值大于0时，每个block的最大绝对值将与`max_low_bound`取较大值后用于scale计算。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量。当`dst_type`为`torch.float8_e4m3fn`或`torch.float8_e5m2`时，y的数据类型与`dst_type`对应，shape与输入`input`一致。当`dst_type`为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`时，y的实际数据类型为`torch.uint8`，shape的最后一维为`input`最后一维的一半（每两个FP4数据打包为一个`torch.uint8`），查看具体值需自行解包。
- **mxscale** (`Tensor`)：每个分组对应的量化尺度，数据类型为`torch_npu.float8_e8m0fnu`，实际返回的数据类型为`torch.uint8`。`mxscale`的shape比输入`input`多一维，最后一维为2，`axis`轴的大小为`input`对应轴的值除以`block_size`向上取整并偶数pad（pad填充值为0），当`axis`为非尾轴时，`mxscale`输出需要对每两行数据进行交织处理。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 当`dst_type`为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`时，`input`的最后一维必须是偶数。
- 输入`input`和输出`mxscale`的shape约束关系：
  - rank(mxscale) = rank(x) + 1
  - axis_change = axis if axis >= 0 else axis + rank(x)
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / block_size) + 2 - 1) / 2
  - mxscale.shape[-1] = 2
  - 其他维度与输入x一致

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def dynamic_mx_quant_test(x_dtype, dst_type):
        # 构造x tensor
        x = torch.randn((16, 256), dtype=x_dtype).npu()
        y_tmp, mxscale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=dst_type, block_size=32, scale_alg=0)
        y = y_tmp.cpu()
        mxscale = mxscale.cpu()
        print("DynamicMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("mxscale:\n", mxscale)
    if __name__ == "__main__":
        dynamic_mx_quant_test(torch.bfloat16, torch.float8_e4m3fn)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, axis=-1, round_mode='rint', dst_type=torch_npu.float8_e5m2, block_size=32, scale_alg=0):
            return torch_npu.npu_dynamic_mx_quant(x, axis=axis, round_mode=round_mode, dst_type=dst_type, block_size=block_size, scale_alg=scale_alg)

    def dynamic_mx_quant_test(x_dtype, dst_type):
        # 构造x tensor
        x = torch.randn((16, 256), dtype=x_dtype).npu()
        model = DynamicMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y_tmp, mxscale = model(x, axis=-1, dst_type=dst_type, block_size=32, scale_alg=0)
        y = y_tmp.cpu()
        mxscale = mxscale.cpu()
        print("DynamicMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("mxscale:\n", mxscale)
    if __name__ == "__main__":
        dynamic_mx_quant_test(torch.float16, torch.float8_e5m2)
    ```
