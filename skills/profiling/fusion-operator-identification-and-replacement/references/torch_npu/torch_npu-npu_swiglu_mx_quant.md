# torch_npu.npu_swiglu_mx_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：在Swish门控线性单元（swiglu）激活函数后接DynamicMxQuant（动态MX量化）操作，对输入`x`完成swiglu激活后量化输出。接口支持MoE分组场景与多种swiglu计算模式。
- 计算公式：

  设输入为$x$，首先对输入做swiglu激活，得到的结果为$swigluOut$：

  - 当`swiglu_mode=0`（传统swiglu）时，以`activate_left=True`为例，将$x$沿`activate_dim`轴等分为前半部分$A$与后半部分$B$：

    $$
    swigluOut = Swish(A) \cdot B
    $$

    其中：

    $$
    Swish(z) = z \cdot sigmoid(z)
    $$

  - 当`swiglu_mode=1`（变种swiglu，GPT-OSS变体）时，将$x$按奇偶索引拆分为$x_{glu}$（偶数索引）与$x_{linear}$（奇数索引），并使用`clamp_limit`、`glu_alpha`、`glu_bias`进行截断与缩放：

    $$
    x_{glu} = clamp(x_{glu},\ max=clamp\_limit)
    $$

    $$
    x_{linear} = clamp(x_{linear},\ -clamp\_limit,\ clamp\_limit)
    $$

    $$
    out_{glu} = x_{glu} \cdot sigmoid(glu\_alpha \cdot x_{glu})
    $$

    $$
    swigluOut = out_{glu} \cdot (x_{linear} + glu\_bias)
    $$

  - 当`swiglu_mode=2`时，计算方式与`swiglu_mode=1`相同（同为变种swiglu，支持`clamp_limit`、`glu_alpha`、`glu_bias`），区别在于$x_{glu}$与$x_{linear}$采用前后切分（与`swiglu_mode=0`的切分方式一致），而非奇偶交错切分。

  - 当`swiglu_mode=3`（变种swiglu）时，将$x$沿`activate_dim`轴前后切分为$x_{glu}$与$x_{linear}$（切分方式与`swiglu_mode=0`一致），先激活后截断，且不使用`glu_alpha`、`glu_bias`：

    $$
    x_{glu} = x_{glu} \cdot sigmoid(x_{glu})
    $$

    $$
    x_{glu} = clamp(x_{glu},\ max=clamp\_limit)
    $$

    $$
    x_{linear} = clamp(x_{linear},\ -clamp\_limit,\ clamp\_limit)
    $$

    $$
    swigluOut = x_{glu} \cdot x_{linear}
    $$

  随后对$swigluOut$进行DynamicMxQuant量化：

  - 当`scale_alg=0`时，将$swigluOut$在`axis`维度上按$k=blocksize=32$个数分组，每组$\{V_i\}_{i=1}^{k}$动态量化为$\{mxscale,\ \{P_i\}_{i=1}^{k}\}$：

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i / mxscale,\ round\_mode)
    $$

    其中$emax$为目标数据类型最大正则数的指数位：

    |   数据类型    | emax |
    |:------------:|:----:|
    | float4_e2m1  |  2   |
    | float4_e1m2  |  0   |
    | `torch.float8_e4m3fn`|  8   |
    | `torch.float8_e5m2`  |  15  |

  - 当`scale_alg=1`时（仅涉及两种float8类型），按块计算缩放因子，找到该块中数值的最大绝对值：

    $$
    Amax(D^b) = max(\{|d_i|\}_{i=1}^{k})
    $$

    $$
    S_{`torch.float32`}^b = \frac{Amax(D^b)}{Amax(DType)}
    $$

    将$S_{`torch.float32`}^b$转换为FP8可表示的$S_{ue8m0}^b = 2^{E_{int}^b}$，并对块内元素量化：

    $$
    d^i = DType(d_{`torch.float32`}^i \cdot R_{`torch.float32`}^b),\quad R_{`torch.float32`}^b = \frac{1}{`torch.float32`(S_{ue8m0}^b)}
    $$

## 函数原型

```python
torch_npu.npu_swiglu_mx_quant(x, *, group_index=None, activate_dim=-1, activate_left=False, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0, group_mode=0, axis=-1, dst_type=296, round_mode="rint", scale_alg=0, max_dtype_value=0) -> (Tensor, Tensor)
```

## 参数说明

- **x**(`Tensor`)：必选参数，输入待处理的数据。shape为$[X_1,X_2,\dots,X_n,2H]$，维数2-7维，对应`activate_dim`轴的维度需为2的倍数。数据类型支持`torch.float16`、`torch.bfloat16`，数据格式为$ND$。
- **\***：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量为可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **group_index**(`Tensor`)：可选参数，MoE分组所需的`group_index`。要求为1维张量，shape为$[groupNum]$（$groupNum$取值范围$[1,256]$），数据类型支持`torch.int64`，数据格式为$ND$。默认值为`None`，表示不进行分组。
- **activate_dim**(`int`)：可选参数，Swish计算时选择的切分轴。取值范围为$[-1,-2,x.dim()-2,x.dim()-1]$。默认值为`-1`。
- **activate_left**(`bool`)：可选参数，是否对切分后的左半部分做Swish激活。取`True`时对左半部分做激活；取`False`时对右半部分做激活；当`swiglu_mode=1`时默认对偶数块做激活。默认值为`False`。
- **swiglu_mode**(`int`)：可选参数，swiglu计算模式。取值范围为$[0,3]$：`0`表示传统swiglu；`1`表示变种swiglu（GPT-OSS变体），使用奇偶分块，并支持`clamp_limit`、`glu_alpha`、`glu_bias`；`2`计算方式同`1`但采用前后切分（同`0`）；`3`表示变种swiglu，切分方式同`2`，区别在于先激活（sigmoid系数固定为1）再截断，不支持`glu_alpha`、`glu_bias`。默认值为`0`。当`activate_dim`为非尾轴时，`swiglu_mode`必须为`0`。
- **clamp_limit**(`float`)：可选参数，变种swiglu输入门限，用于对输入裁剪。需大于0且有限。默认值为`7.0`。
- **glu_alpha**(`float`)：可选参数，GLU激活函数系数。默认值为`1.702`。
- **glu_bias**(`float`)：可选参数，swiglu计算中的偏差。默认值为`1.0`。
- **group_mode**(`int`)：可选参数，`group_index`对应的模式。取值范围为$[0,1]$：`0`表示count模式，`1`表示cumsum模式。默认值为`0`。
- **axis**(`int`)：可选参数，DynamicMxQuant量化发生的轴。取值范围为$[-1,-2,x.dim()-2,x.dim()-1]$。默认值为`-1`。
- **dst_type**(`int`)：可选参数，输出`y`的数据类型。`296`表示`torch_npu.float4_e2m1fn_x2`，`297`表示`torch_npu.float4_e1m2fn_x2`，`292`表示`torch.float8_e4m3fn`，`291`表示`torch.float8_e5m2`。默认值为`296`。
- **round_mode**(`str`)：可选参数，输出`y`的舍入模式。取值为`"rint"`、`"round"`、`"floor"`。当`dst_type`为`torch.float8_e4m3fn`或`torch.float8_e5m2`时仅支持`"rint"`。默认值为`"rint"`。
- **scale_alg**(`int`)：可选参数，`mxscale`的计算方法。取值范围为$[0,2]$：`0`代表OCP算法，对应上述`scale_alg=0`场景；`1`代表cuBLAS算法，对应上述`scale_alg=1`场景；`2`代表RNE算法，为预留取值。当前仅支持取值`0`和`1`。当`dst_type`为`float4_e2m1`或`float4_e1m2`时仅支持`0`。默认值为`0`。
- **max_dtype_value**(`float`)：可选参数，预留参数，表示DynamicMxQuant过程中指定的目标数据类型最大值。取值不小于0，仅当`scale_alg=2`且`dst_type`为`float4_e2m1`或`float4_e1m2`时生效。默认值为`0`。

## 返回值说明

- **y**(`Tensor`)：量化后的输出，对应公式中的$P_i$或$d^i$。数据类型由`dst_type`决定，数据格式为$ND$。当`activate_dim`为尾轴时，shape为$[X_1,\dots,X_n,H]$（即在`activate_dim`轴上为输入的一半）；当`activate_dim`非尾轴时，`activate_dim`轴为输入对应轴的一半。
- **mxscale**(`Tensor`)：每个分组对应的量化尺度，对应公式中的$mxscale$或$S^b$。数据类型为`torch_npu.float8_e8m0`，数据格式为$ND$。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持Eager模式和图模式。
- 输入`x`对应`activate_dim`轴的维度需为2的倍数，且`x`的维数必须大于1维。
- 当`activate_dim`为非尾轴时，`swiglu_mode`必须为`0`。
- 当`swiglu_mode`为`2`或`3`时，`axis`必须为`-1`。
- 当`dst_type`为`float4_e2m1`或`float4_e1m2`时，`y`的最后一维需为2的倍数，且`scale_alg`必须为`0`。
- 当`dst_type`为`torch.float8_e4m3fn`或`torch.float8_e5m2`时，`round_mode`必须为`"rint"`。
- `group_index`所有元素之和不能大于输入`x`除尾轴外剩余轴的乘积，每个元素需大于0。
- 当`activate_dim`或`axis`为非尾轴且`group_index`存在时，`x`必须为2维。
- 输出`y`和`mxscale`超出`group_index`所有元素之和的部分未进行清理，该部分内存为垃圾数据。

## 调用示例

```python
import torch
import torch_npu

x = torch.randn(2, 64, dtype=torch.bfloat16).npu()
group_index = torch.tensor([2], dtype=torch.int64).npu()

y, mxscale = torch_npu.npu_swiglu_mx_quant(
    x,
    group_index=group_index,
    activate_dim=-1,
    activate_left=True,
    swiglu_mode=1,
    clamp_limit=7.0,
    glu_alpha=1.0,
    glu_bias=1.702,
    group_mode=0,
    axis=-1,
    dst_type=torch_npu.float4_e2m1fn_x2,
    round_mode="rint",
    scale_alg=0,
    max_dtype_value=0,
)

print("y:", y)
print("mxscale:", mxscale)
```
