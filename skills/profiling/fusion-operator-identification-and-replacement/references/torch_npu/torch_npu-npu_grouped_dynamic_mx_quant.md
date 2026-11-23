# torch_npu.npu_grouped_dynamic_mx_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：根据传入的分组索引的起始值（`group_index`）对输入x按组进行分组，以基本块（`blocksize`）为粒度，对数据执行目标数据类型为`float8`（`torch.float8_e4m3fn`，`torch.float8_e5m2`）或`float4`（`torch_npu.float4_e2m1fn_x2`，`torch_npu.float4_e1m2fn_x2`）的动态MX量化，并输出量化尺度`mxscale`（`torch.float8_e8m0fnu`）。

- 计算公式：
  - 场景1，当scale\_alg为0时：
    - 将输入x在第0维上先按照group\_index进行分组，每个group内按k = blocksize个数分组，一组k个数$\{\{V_i\}_{i=1}^{k}\}$计算出这组数对应的量化尺度mxscale\_pre，$\{mxscale\_pre, \{P_i\}_{i=1}^{k}\}$，计算公式如下：

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale\_pre = 2^{shared\_exp}
    $$

    - 这组数每一个除以mxscale\_pre，根据round\_mode转换到对应的dst\_type，得到量化结果y，计算公式如下：

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale\_pre, round\_mode), \space i\space from\space 1\space to\space blocksize
    $$

    - 量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale\_pre按对应的group\_index分组，分组内第一个维度pad为偶数，组成输出mxscale。
    - emax：对应数据类型的最大正则数的指数位，对应关系如下表：

      | dst_type | emax |
      | --- | --- |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - 场景2，当scale\_alg为1时（只涉及float8）：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型float8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值：

    $$
    Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

    - 将`torch.float32`映射到目标数据类型float8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：

    $$
    S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
    $$

    - 将块缩放因子$S_{`torch.float32`}^b$转换为fp8格式下可表示的缩放值$S_{ue8m0}^b$，即从$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，为保证量化时不溢出，对指数进行向上取整，且在fp8可表示的范围内：

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$

    - 计算块缩放因子$S_{ue8m0}^b=2^{E_{int}^b}$，块转换因子$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$，对每个块内元素$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^n)$。最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$），$[d^i]_{i=1}^k$代表块内量化后的数据。

  - 场景3，当scale\_alg为2时（只涉及float4类型）：
    - 当dst\_type\_max为0.0/6.0/7.0（`torch_npu.float4_e2m1fn_x2`）或1.875（`torch_npu.float4_e1m2fn_x2`）时：
      - 将输入x在第0维上按k = blocksize个数分组，一组k个数$\{\{V_i\}_{i=1}^{k}\}$动态量化为$\{mxscale, \{P_i\}_{i=1}^{k}\}$，k = blocksize：

      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前1/2/3位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
      $$

      - 量化后的$P_i$按对应的$V_i$的位置组成输出y，按对应维度上的分组组成输出mxscale。
    - 当dst\_type\_max不为上述特殊取值时（`torch_npu.float4_e2m1fn_x2`为6.0-12.0中的其它取值，`torch_npu.float4_e1m2fn_x2`为1.75-3.5中的其它取值）：
      - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{`torch.float32`}^b$，再把块内所有元素用同一个$S_{`torch.float32`}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
      - 找到该块中数值的最大绝对值：

      $$
      Amax(D_{`torch.float32`}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

      - 将`torch.float32`映射到目标数据类型可表示的范围内，其中当dst\_type\_max为0时，$Amax(DType)$是目标精度能表示的最大值；当dst\_type\_max不为0时，$Amax(DType)$是dst\_type\_max传入值：

      $$
      S_{`torch.float32`}^b = \frac{Amax(D_{`torch.float32`}^b)}{Amax(DType)}
      $$

      - 将块缩放因子$S_{`torch.float32`}^b$转换为fp8格式下可表示的缩放值$S_{ue8m0}^b$，从$S_{`torch.float32`}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，为保证量化时不溢出，对指数进行向上取整，且在fp8可表示的范围内：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{`torch.float32`}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

      - 计算块缩放因子$S_{ue8m0}^b=2^{E_{int}^b}$，块转换因子$R_{`torch.float32`}^b=\frac{1}{`torch.float32`(S_{ue8m0}^b)}$，对每个块内元素$d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b)$。最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$），$[d^i]_{i=1}^k$代表块内量化后的数据。

## 函数原型

```python
torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, *, round_mode="rint", dst_type=23, blocksize=32,
                                       scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示算子输入的Tensor，公式中的输入$x$。维度仅支持2维，数据格式支持ND，数据类型支持`torch.float16`、`torch.bfloat16`。支持非连续Tensor，支持空Tensor。
- **group\_index**（`Tensor`）：必选参数，表示量化分组的起始索引。维度仅支持1维，数据格式支持ND，数据类型支持`torch.int32`。支持非连续Tensor，不支持空Tensor。索引要求大于等于0、非递减，且最后一个数与x的第0维大小相等。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round\_mode**（`str`）：可选参数，表示数据转换的模式，公式中的$round\_mode$。默认值为"rint"。
  - 当dst\_type为`torch.float8_e5m2`、`torch.float8_e4m3fn`时，仅支持"rint"。
  - 当dst\_type为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`时，支持"rint"、"round"、"floor"。
- **dst\_type**（`int`）：可选参数，表示数据转换后y的数据类型。默认值为23，支持取值：
  - 23：`torch.float8_e5m2`。
  - 24：`torch.float8_e4m3fn`。
  - 296：`torch_npu.float4_e2m1fn_x2`。
  - 297：`torch_npu.float4_e1m2fn_x2`。
- **blocksize**（`int`）：可选参数，表示每次量化的元素个数，公式中的$blocksize$。当前取值仅支持32，默认值为32。
- **scale\_alg**（`int`）：可选参数，表示mxscale计算时采用的算法。支持取值为0、1、2，默认值为0。
  - 0：场景1（共享指数MX量化），float8、float4均支持。
  - 1：场景2（逐块缩放），仅float8支持。
  - 2：场景3（fp4动态量化），仅float4支持。
- **dst\_type\_max**（`float`）：可选参数，表示maxType的取值，对应公式中的$Amax(DType)$，仅dst\_type为float4时生效。默认值为0.0。
  - 0.0：Amax(DType)为量化结果数据类型的最大值。
  - `torch_npu.float4_e2m1fn_x2`：支持0.0和6.0-12.0。
  - `torch_npu.float4_e1m2fn_x2`：支持0.0和1.75-3.5。

## 返回值说明

- **y**（`Tensor`）：表示量化后的输出Tensor，公式中的$y$。shape与输入x保持一致。当dst\_type为float8时，数据类型为`torch.float8_e5m2`或`torch.float8_e4m3fn`；当dst\_type为float4时，实际返回的数据类型为`torch.uint8`，最后一维为输入x最后一维的一半（每个`torch.uint8`元素打包2个float4数据），查看具体值需自行解包，如`y.view(torch_npu.float4_e2m1fn_x2)`。支持空Tensor。float8支持非连续Tensor，float4不支持非连续Tensor。
- **mxscale**（`Tensor`）：表示每个分组对应的量化尺度，公式中的$mxscale$。数据类型为`torch.float8_e8m0fnu`，实际返回的数据类型为`torch.uint8`，查看具体值需自行转换，如`mxscale.view(torch.float8_e8m0fnu)`。假设x的shape为[m, n]，`group_index`的shape为[g]，则`mxscale`的shape为[(m/(blocksize∗2)+g), n, 2]。支持空Tensor，不支持非连续Tensor。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 确定性计算：默认确定性实现。
- 关于x、group\_index、y、mxscale的shape约束说明如下（假设x的shape为[m, n]，group\_index的shape为[g]）：
  - rank(mxscale) = rank(x) + 1。
  - mxscale.shape[0] = m / (blocksize ∗ 2) + g。
  - mxscale.shape[-1] = 2，其它维度与输入x一致。
  - 输出y的shape和x保持一致（float4输出时y的实际最后一维为x最后一维的一半）。
- dst\_type、scale\_alg、round\_mode、dst\_type\_max的参数组合约束如下表：

| dst_type | scale_alg | round_mode | dst_type_max |
| --- | --- | --- | --- |
| `torch.float8_e5m2` | 0、1 | "rint" | 不支持设置（仅默认值0.0） |
| `torch.float8_e4m3fn` | 0、1 | "rint" | 不支持设置（仅默认值0.0） |
| `torch_npu.float4_e2m1fn_x2` | 0、2 | "rint"/"round"/"floor" | 0.0和6.0-12.0 |
| `torch_npu.float4_e1m2fn_x2` | 0、2 | "rint"/"round"/"floor" | 0.0和1.75-3.5 |

## 调用示例

- 单算子模式调用，float8量化（`scale_alg`使用默认值0）

    ```python
    import torch
    import torch_npu
    
    # x的shape为[8, 1]，取值为(0, 8, 64, 512)重复两次；group_index=[4, 8]表示[0:4]和[4:8]两个分组
    x = torch.tensor([[0], [8], [64], [512], [0], [8], [64], [512]], dtype=torch.bfloat16).npu()
    group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    y, mxscale = torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, dst_type=torch.float8_e4m3fn)
    print(y)
    print(mxscale)
    print(mxscale.cpu().view(torch.float8_e8m0fnu))
    ```

    输出如下所示

    ```text
    tensor([[  0.],
            [  4.],
            [ 32.],
            [256.],
            [  0.],
            [  4.],
            [ 32.],
            [256.]], device='npu:0', dtype=torch.float8_e4m3fn)
    tensor([[[128,   0]],
    
            [[128,   0]]], device='npu:0', dtype=torch.uint8)
    tensor([[[2., 0.]],
    
            [[2., 0.]]], dtype=torch.float8_e8m0fnu)
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

        def forward(self, x, group_index):
            return torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, dst_type=torch.float8_e4m3fn)

    model = Model().npu()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    x = torch.randn((64, 256), dtype=torch.float16).npu()
    group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
    y, mxscale = model(x, group_index)
    print(y)
    print(mxscale)
    ```

    输出如下所示

    ```text
    tensor([[-128.0000,  208.0000, -160.0000,  ...,  -32.0000, -112.0000,
              128.0000],
            [-144.0000,  104.0000, -208.0000,  ...,  112.0000,  288.0000,
              120.0000],
            [  -1.7500,  -80.0000,  240.0000,  ..., -192.0000, -160.0000,
               48.0000],
            ...,
            [-160.0000,   40.0000, -160.0000,  ...,  160.0000,  288.0000,
             -384.0000],
            [-144.0000, -352.0000,    3.5000,  ...,  176.0000,  128.0000,
             -384.0000],
            [ -20.0000,   22.0000,  320.0000,  ...,  256.0000,   -4.5000,
              -96.0000]], device='npu:0', dtype=torch.float8_e4m3fn)
    tensor([[[120,   0],
             [120,   0],
             [119,   0],
             ...,
             [120,   0],
             [120,   0],
             [120,   0]],
    
            [[120,   0],
             [120,   0],
             [119,   0],
             ...,
             [120,   0],
             [120,   0],
             [119,   0]],
    
            [[  0,   0],
             [  0,   0],
             [  0,   0],
             ...,
             [  0,   0],
             [  0,   0],
             [  0,   0]]], device='npu:0', dtype=torch.uint8)
    ```
