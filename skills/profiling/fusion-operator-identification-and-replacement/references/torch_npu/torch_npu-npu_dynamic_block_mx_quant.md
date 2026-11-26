# torch\_npu.npu\_dynamic\_block\_mx\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：对输入变量，以数据块（32\*32）为基本块进行MX量化并转换为目标数据类型。

    在每个基本块中，根据`scale_alg`的取值采取不同的scale算法计算出当前块对应的量化参数`scale`（1\*1），将其广播为`scale1`（32\*1）和`scale2`（1\*32）输出。同时对基本块中的每一个数除以`scale`，根据`round_mode`转换到对应的`dst_type`，得到量化结果`y`。

- 计算公式：
  - 场景1，当`scale_alg`为0时：
    - 将输入input以数据块（32\*32）为基本块进行分组，一个数据块的数 $\{\{V_i\}_{i=1}^{32*32}\}$ 量化为 $\{scale, \{P_i\}_{i=1}^{32*32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      scale = 2^{shared\_exp} \\
      P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
      $$

    - 同时将scale（1\*1）广播为scale1（32\*1）和scale2（1\*32）作为输出scale1和scale2，量化后的$P_i$按对应的$V_i$的位置组成输出y。

    - emax：对应数据类型的最大正则数的指数位。

      | dst_type | emax |
      | :---: | :---: |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - 场景2，当`scale_alg`为2时，只涉及`torch_npu.float4_e2m1fn_x2`类型：
    - 将输入按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型`torch_npu.float4_e2m1fn_x2`，scale存储类型为`torch_npu.float8_e8m0fnu`。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值：

      $$
      Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - 当`dst_type_max`不为0时，按照传入的数值计算scale；当`dst_type_max`为0时，使用目标数据类型的最大值。
    - 将`torch.float32`映射到目标数据类型可表示的范围内：

      $$
      S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{dst\_type\_max}
      $$

    - 将块缩放因子$S_{`torch.float32`}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
    - 从块的浮点缩放因子$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{其余情况} \end{cases}
      $$

    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，即$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

## 函数原型

```python
torch_npu.npu_dynamic_block_mx_quant(input, *, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，表示需要量化的数据，数据类型支持`torch.float16`、`torch.bfloat16`，shape支持2-3维度，支持非连续的Tensor，数据格式支持$ND$。当`dst_type`为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`时，`input`的最后一维必须是偶数。不支持空Tensor。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，指定量化结果cast到输出y的数据类型模式，默认值为`"rint"`。
  - 当`dst_type`为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`时，支持取值`"rint"`、`"floor"`、`"round"`。
  - 当`dst_type`为`torch.float8_e5m2`、`torch.float8_e4m3fn`时，仅支持取值`"rint"`。
- **dst_type** (`int`)：可选参数，表示输出y的数据类型，支持的类型为`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`，默认类型为`torch_npu.float4_e2m1fn_x2`。
- **scale_alg** (`int`)：可选参数，表示scale的计算方法，默认值为0。取值范围：0（代表场景1）、2（代表场景2）。当`dst_type`为`torch_npu.float4_e1m2fn_x2`、`torch.float8_e5m2`、`torch.float8_e4m3fn`时仅支持取值为0。
- **dst_type_max** (`float`)：可选参数，表示目标数据类型的最大值，默认值为0.0。只支持在`scale_alg`=2且`dst_type`为`torch_npu.float4_e2m1fn_x2`场景设置该值，当前仅支持取值0.0、6.0、7.0。

## 返回值说明

- **y** (`Tensor`)：表示input量化后的对应结果，数据类型由参数`dst_type`指定，shape与输入`input`一致。当`dst_type`为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`时，y的数据类型实际为`torch.uint8`，查看具体值需自行解包。
- **scale1_out** (`Tensor`)：表示-1轴每个分组对应的量化尺度。数据类型为`torch_npu.float8_e8m0fnu`，实际数据类型为`torch.uint8`，查看具体值需要自行转换。Shape为input的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。
- **scale2_out** (`Tensor`)：表示-2轴每个分组对应的量化尺度。数据类型为`torch_npu.float8_e8m0fnu`，实际数据类型为`torch.uint8`，查看具体值需要自行转换。Shape在input的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0，scale2_out输出需要对每两行数据交织处理。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输出的目标类型为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`时，`input`的最后一维必须是偶数。
- 输入`input`和输出`scale1_out`、`scale2_out`的shape约束关系：
  - rank(scale1_out) = rank(input) + 1
  - rank(scale2_out) = rank(input) + 1
  - scale1_out.shape[-2] = (ceil(input.shape[-1] / 32) + 2 - 1) / 2
  - scale2_out.shape[-3] = (ceil(input.shape[-2] / 32) + 2 - 1) / 2
  - scale1_out.shape[-1] = 2
  - scale2_out.shape[-1] = 2
  - 其他维度与输入input一致
  - 举例：输入input的shape为[B, M, N]，对应的y的shape为[B, M, N]，scale1_out的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，scale2_out的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def dynamic_block_mx_quant_test(x_dtype, round_mode, dst_type, scale_alg, dst_type_max):
        # 构造x tensor
        x = torch.randn((1, 4), dtype=x_dtype).npu()
        y_tmp, scale1_tmp, scale2_tmp = torch_npu.npu_dynamic_block_mx_quant(x, round_mode=round_mode,
        dst_type=dst_type,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max)
        y = y_tmp.cpu()
        scale1 = scale1_tmp.cpu()
        scale2 = scale2_tmp.cpu()
        print("DynamicBlockMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale1:\n", scale1)
        print("scale2:\n", scale2)

    if __name__ == "__main__":
        dynamic_block_mx_quant_test(torch.bfloat16, 'rint', torch_npu.float4_e2m1fn_x2, 0, 0.0)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicBlockMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, round_mode='rint', dst_type=torch_npu.float4_e1m2fn_x2, scale_alg=0, dst_type_max=0.0):
            return torch_npu.npu_dynamic_block_mx_quant(x, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)

    def dynamic_block_mx_quant_test(x_dtype, round_mode, dst_type, scale_alg, dst_type_max):
        # 构造x tensor
        x = torch.randn((1, 4), dtype=x_dtype).npu()
        model = DynamicBlockMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y_tmp, scale1_tmp, scale2_tmp = model(x, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)
        y = y_tmp.cpu()
        scale1 = scale1_tmp.cpu()
        scale2 = scale2_tmp.cpu()
        print("DynamicBlockMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale1:\n", scale1)
        print("scale2:\n", scale2)

    if __name__ == "__main__":
        dynamic_block_mx_quant_test(torch.bfloat16, "rint", torch_npu.float4_e2m1fn_x2, 0, 0.0)
    ```
