# torch_npu.npu_clipped_swiglu

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：带截断的Swish门控线性单元激活函数，实现`x`的变体SwiGLU计算。本接口相较于torch_npu.npu_swiglu，新增了部分输入参数：`group_index`、`alpha`、`limit`、`bias`、`interleaved`，用于支持GPT-OSS模型使用的变体SwiGLU以及MoE模型使用的分组场景。

- 计算公式：

    对给定的输入张量`x`，其维度为[a,b,c,d,e,f,g…]，进行以下计算：

    1. 将`x`基于输入参数`dim`进行合轴，合轴后维度为[pre, cut, after]。其中cut轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；pre、after可以等于1。例如当`dim`为3时，合轴后`x`的维度为[a \* b \* c, d, e \* f \* g \*…]。此外，由于after轴的元素为连续存放，且计算操作为逐元素的，因此将cut轴与after轴合并，得到`x`的维度为[pre, cut* after]。

    2. 根据输入参数`group_index`，对`x`的pre轴进行过滤处理，公式如下：
        $$
        sum = \text{Sum}(group\_index)
        $$

        $$
        x = x[ : sum, : ]
        $$
        其中sum表示`group_index`的所有元素之和。当不输入`group_index`时，跳过该步骤。

    3. 根据输入参数`interleaved`，对`x`进行切分，公式如下：

        当`interleaved`为True时，表示奇偶切分：
        $$
        A = x[ : , : : 2]
        $$

        $$
        B = x[ : , 1 : : 2]
        $$

        当`interleaved`为False时，表示前后切分：
        $$
        h = x.shape[1] // 2
        $$

        $$
        A = x[ : , : h]
        $$

        $$
        B = x[ : , h : ]
        $$

    4. 根据输入参数`alpha`、`limit`、`bias`进行变体SwiGLU计算，公式如下：
        $$
        A = A.clamp(min=None, max=limit)
        $$
        
        $$
        B = B.clamp(min=-limit, max=limit)
        $$
        
        $$
        y\_glu = A * sigmoid(alpha * A)
        $$
        
        $$
        y = y\_glu * (B + bias)
        $$
    
    5. 重塑输出张量y的维度数量与合轴前的x的维度数量一致，第`dim`轴上的大小为`x`的一半，其他维度与`x`相同。
    
## 函数原型

```python
torch_npu.npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，表示目标张量。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，不支持非连续的`Tensor`，数据格式为$ND$，`x`的维数必须大于1维且第`dim`轴的元素数量为偶数。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **group_index** (`Tensor`)：可选参数，表示对`x`进行分组的情况。要求为1维张量，第i个元素代表第i组需要处理的`x`合轴后的token数量，数据类型支持`torch.int64`，数据格式$ND$。默认值为None，表示不对`x`进行分组处理。
- **dim** (`int`)：可选参数，表示需要对`x`进行切分的维度序号，取值范围为[-x.dim(), x.dim()-1]，默认值为-1。
- **alpha** (`float`)：可选参数，表示GLU激活函数系数，默认值为1.702。
- **limit** (`float`)：可选参数，表示变体SwiGLU输入门限，默认值为7.0。
- **bias** (`float`)：可选参数，表示变体SwiGLU计算中的偏差，默认值为1.0。
- **interleaved** (`bool`)：可选参数，表示输入`x`是否按奇偶方式切分，True表示为奇偶方式切分，False表示为前后方式切分，默认值为True。

## 返回值说明

`Tensor`

代表公式中的`y`，表示激活函数的输出，数据类型同输入`x`，在维度上，第`dim`维是输入`x`的`1/2`，其余维度与输入`x`相同，数据格式为$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    tokens_num = 4608
    hidden_size = 2048
    x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
    group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
    y = torch_npu.npu_clipped_swiglu(
        x.npu(),
        group_index=group_index.npu(),
        dim=-1,
        alpha=1.702,
        limit=7.0,
        bias=1.0,
        interleaved=True
    )
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

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class ClippedSwigluModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, group_index, dim, alpha, limit, bias, interleaved):
            y = torch_npu.npu_clipped_swiglu(
                x.npu(),
                group_index=group_index.npu(),
                dim=dim,
                alpha=alpha,
                limit=limit,
                bias=bias,
                interleaved=interleaved
            )
            return y
    
    tokens_num = 4608
    hidden_size = 2048
    x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
    group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
    clipped_swiglu_model = ClippedSwigluModel().npu()
    clipped_swiglu_model = torch.compile(clipped_swiglu_model, backend=npu_backend, dynamic=True)
    y = clipped_swiglu_model(x, group_index, -1, 1.702, 7.0, 1.0, True)
    ```
