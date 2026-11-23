# torch_npu.npu_incre_flash_attention

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id1 -->
<!-- npu="310p" id11 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id11 -->

## 功能说明

- API功能：增量FA实现。
- 计算公式：
    $$
    atten\_out = \text{softmax}(scale\_value * (query \mathbin{@} key^\top) + atten\_mask) \mathbin{@} value
    $$

## 函数原型

```python
torch_npu.npu_incre_flash_attention(query, key, value, *, padding_mask=None, pse_shift=None, atten_mask=None, actual_seq_lengths=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0, block_size=0, inner_precise=1) -> Tensor
```

## 参数说明

- **query** (`Tensor`)：必选参数。attention结构的Query输入，数据格式支持$ND$。其中$ND$表示N-Dimensional Tensor（任意维张量）。
    <!-- npu="310p" id12 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id12 -->

    <!-- npu="910b" id2 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id2 -->

- **key** (`Tensor`)：必选参数。attention结构的Key输入，数据格式支持$ND$。
    <!-- npu="310p" id13 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id13 -->

    <!-- npu="910b" id3 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id3 -->

- **value** (`Tensor`)：必选参数。attention结构的Value输入，数据格式支持$ND$。
    <!-- npu="310p" id14 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.int8`。
    <!-- end id14 -->

    <!-- npu="910b" id4 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id4 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **padding_mask** (`Tensor`)：预留参数，暂未使用，默认值为`None`。
- **pse_shift** (`Tensor`)：可选参数。表示在attention结构内部的位置编码参数，数据格式支持$ND$。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id15 -->
    - <term>Atlas推理系列产品</term>：仅支持`None`。
    <!-- end id15 -->

    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id5 -->

- **atten_mask** (`Tensor`)：可选参数。取值为`1`代表该位不参与计算（不生效），为`0`代表该位参与计算，默认值为`None`，即全部参与计算；数据类型支持`torch.bool`、`torch.int8`、`torch.uint8`，数据格式支持$ND$。
- **actual_seq_lengths** (`List[int]`)：可选参数。其`shape`为$(B,)$或$(1,)$，形如$[1, 2, 3]$，代表`key`、`value`中有效的$S$序列长度，默认值为`None`，即全部有效，类型为`List[int]`；数据类型为`torch.int64`，数据格式支持$ND$。
- **dequant_scale1** (`Tensor`)：可选参数。数据类型支持`torch.float32`，数据格式支持$ND$，表示BMM1后面反量化的量化因子，支持pertensor（scalar）。该参数用于BMM1后反量化场景，通常在BMM1输出为量化结果、后续计算需要恢复为浮点表示时使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id21 -->
    - <term>Atlas推理系列产品</term>暂不使用该参数。
    <!-- end id21 -->
- **quant_scale1** (`Tensor`)：可选参数。数据类型支持`torch.float32`，数据格式支持$ND$，表示BMM2前面量化的量化因子，支持pertensor（scalar）。该参数用于BMM2前量化场景，通常在进入BMM2前需要将输入转换为量化表示时使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id22 -->
    - <term>Atlas推理系列产品</term>暂不使用该参数。
    <!-- end id22 -->
- **dequant_scale2** (`Tensor`)：可选参数。数据类型支持`torch.float32`，数据格式支持$ND$，表示BMM2后面反量化的量化因子，支持pertensor（scalar）。该参数用于BMM2后反量化场景，通常在BMM2输出为量化结果、后续需要恢复为浮点表示时使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id23 -->
    - <term>Atlas推理系列产品</term>暂不使用该参数。
    <!-- end id23 -->
- **quant_scale2** (`Tensor`)：可选参数。数据格式支持$ND$，表示输出量化的量化因子，支持pertensor（scalar）和perchannel（`list`）。该参数用于最终输出量化场景；当输出需要量化时通常需要与`quant_offset2`配合使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id16 -->
    - <term>Atlas推理系列产品</term>：当前版本不支持。
    <!-- end id16 -->

    <!-- npu="910b" id6 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。
    <!-- end id6 -->

- **quant_offset2** (`Tensor`)：可选参数。数据格式支持$ND$，表示输出量化的量化偏移，支持pertensor（`scalar`）和perchannel（`list`）。该参数用于最终输出量化场景，通常与`quant_scale2`配合使用；如不使用输出量化功能时可不传或传入`None`。
    <!-- npu="310p" id17 -->
    - <term>Atlas推理系列产品</term>：当前版本不支持。
    <!-- end id17 -->

    <!-- npu="910b" id7 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。
    <!-- end id7 -->

- **antiquant_scale** (`Tensor`)：可选参数。数据格式支持$ND$，表示量化因子，支持perchannel（`list`），由`shape`决定，$BNSD$场景下`shape`为$(2, N, 1, D)$，$BSH$场景下`shape`为$(2, H)$，$BSND$场景下`shape`为$(2, N, D)$。该参数用于输入/权重反量化场景，通常用于将低比特数据恢复为计算所需格式；一般与`antiquant_offset`配合使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id18 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id18 -->

    <!-- npu="910b" id8 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id8 -->

- **antiquant_offset** (`Tensor`)：可选参数。数据格式支持$ND$，表示量化偏移，支持perchannel（`list`），由`shape`决定，$BNSD$场景下`shape`为$(2, N, 1, D)$，$BSH$场景下`shape`为$(2, H)$，$BSND$场景下`shape`为$(2, N, D)$。该参数用于输入/权重反量化场景，一般与`antiquant_scale`配合使用。如不使用该功能时可不传或传入`None`。
    <!-- npu="310p" id19 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id19 -->

    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id9 -->

- **block_table** (`Tensor`)：可选参数。数据类型支持`torch.int32`，数据格式支持$ND$。`block_table`为2维`Tensor`，表示page attention中KV存储使用的block映射表，具体约束和使用方法可见[约束说明](#zh-cn_topic_0000001711274864_section12345537164214)。如不使用该功能时可不传或传入`None`。
- **kv_padding_size** (`Tensor`)：可选参数。数据类型支持`torch.int64`，数据格式支持$ND$，表示KV左`padding`场景开启时，最后一个有效`token`到$S$的距离。如不使用该功能时可传入`None`。
- **num_heads** (`int`)：可选参数。代表`query`的头数，即`query`的$N$，默认值为`1`；数据类型为`torch.int64`。
- **scale_value** (`float`)：可选参数。代表缩放系数，用来约束梯度，默认值为`1.0`，典型值为$\frac{1}{\sqrt{D}}$；数据类型为`torch.float32`。
- **input_layout** (`str`)：可选参数。代表`query`、`key`、`value`的布局，根据输入的`query`、`key`、`value`的`shape`确定，三维`Tensor`是$BSH$，四维`Tensor`是$BNSD$或$BSND$，默认值为$BSH$，不支持其他值；数据类型为`str`。

    > [!NOTE]  
    > `query`、`key`、`value`支持多种排布格式，其中$B$（Batch Size）表示输入样本批量大小、$S$（Seq-Length）表示输入样本序列长度、$H$（Hidden Size）表示隐藏层总维度、$N$（Head-Num）表示多头数、$D$（Head-Dim）表示每个Head的隐藏维度。
    > 当`input_layout`为$BSH$时，`query`、`key`、`value`的`shape`为$(B, S, H)$；当为$BNSD$时，`shape`为$(B, N, S, D)$；当为$BSND$时，`shape`为$(B, S, N, D)$。

- **num_key_value_heads** (`int`)：可选参数。表示`key`、`value`使用的头数，用于支持 **GQA**（Grouped-Query Attention，分组查询注意力）场景。默认值为`0`，表示`key`、`value`的头数与`query`的头数相同；否则表示`key`、`value`的头数为`num_key_value_heads`。在GQA中，多个查询头会共享同一组键/值头，以减少KV缓存占用和计算开销。此时`query`的头数为`num_heads`，`key`、`value`的头数为`num_key_value_heads`，并满足`num_heads`可以被`num_key_value_heads`整除；另外，`num_heads`与`num_key_value_heads`的比值不能大于`64`。数据类型为`torch.int64`。
- **block_size** (`int`)：可选参数。page attention中KV存储每个block中最大的token个数，默认值为`0`，通常为128、256等值，数据类型支持`torch.int64`。
- **inner_precise** (`int`)：可选参数。代表高精度/高性能选择，`0`代表高精度，`1`代表高性能，默认值为`1`（高性能），数据类型支持`torch.int64`。

## 返回值说明

`Tensor`

计算的最终结果，对应公式中的$atten\_out$，`shape`与`query`保持一致。

- 非量化场景下，输出数据类型与`query`的数据类型保持一致。
- 量化场景下，若传入`quant_scale2`，则输出数据类型为`torch.int8`。

## 约束说明<a name="zh-cn_topic_0000001711274864_section12345537164214"></a>

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- `query`、`key`、`value`的维度必须保持一致，`key`、`value`的`shape`必须保持一致。
- `num_heads`的值要等于`query`的$N$。
- `input_layout`的值与`query`的`shape`相关，三维是$BSH$，四维是$BNSD$或$BSND$。
- `num_key_value_heads`的值要等于`key`、`value`的$N$，且`num_heads`需要能被`num_key_value_heads`整除。
- `query`，`key`，`value`输入，功能使用限制如下：

    <!-- npu="910b" id10 -->
    - <term>Atlas A2系列产品</term>支持$B$轴小于等于65535，支持$N$轴小于等于256，支持$S$轴小于等于262144，支持$D$轴小于等于512。
    <!-- end id10 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas推理系列产品</term>支持$B$轴小于等于256，支持$N$轴小于等于256，支持$S$轴小于等于65536，支持$D$轴小于等于512。
    <!-- end id20 -->
    - `query`、`key`、`value`输入均为`torch.int8`的场景暂不支持。

- `torch.int8`量化相关入参数量与输入、输出数据格式的综合限制：

    `query`、`key`、`value`输入为`torch.float16`，输出为`torch.int8`的场景：入参`quant_scale2`必填，`quant_offset2`可选，不能传入`dequant_scale1`、`quant_scale1`、`dequant_scale2`（即为`None`）参数。

- `pse_shift`功能使用限制如下：
    - `pse_shift`数据类型需与`query`数据类型保持一致。
    - 仅支持$D$轴对齐，即$D$轴可以被16整除。

- page attention使用限制：
    - page attention开启必要条件是`block_table`存在且有效，且传入每个batch对应的`actual_seq_lengths`。page attention场景下，`key`、`value`按照`block_table`中的索引映射到一片连续内存中的kv cache，支持`key`、`value`数据类型为`torch.float16`、`torch.bfloat16`、`torch.int8`。
    - page attention场景下，输入kv cache排布格式为$（blocknum, numKvHeads, blocksize, headDims）$或$（blocknum, blocksize, H）$，其中$blocknum$表示kv cache中物理block总数，不应小于每个batch所需block个数的总和。通常情况下，kv cache排布格式为$（blocknum, numKvHeads, blocksize, headDims）$时，性能比kv cache排布格式为$（blocknum, blocksize, H）$时更好。
    - page attention场景下，支持kv cache排布格式为$（blocknum, numKvHeads, blocksize, headDims）$，但此时`query layout`仅支持$BNSD$。
    - page attention场景下，当输入kv cache排布格式为$（blocknum, blocksize, H）$，且$H（H=numKvHeads * headDims）$超过64k时，受硬件指令约束，会被拦截报错。
    - page attention场景下，必须传入输入`actual_seq_lengths`，每个batch的`actual_seq_length`表示每个batch的`sequence`真实长度，该值除以属性输入`block_size`即表示每个batch所需block数量。
    - page attention场景下，`block_table`必须为二维`Tensor`，shape为`(B, maxBlockNumPerSeq)`，其中`B`表示batch数，`maxBlockNumPerSeq`表示各batch中所需block数量的最大值。`block_table[i][j]`表示第`i`个batch的第`j`个逻辑block映射到kv cache中的物理block索引。
    - page attention场景下，`block_table`第一维长度需等于batch数，第二维长度不能小于`maxBlockNumPerSeq`（`maxBlockNumPerSeq`为每个batch中最大`actual_seq_lengths`对应的block数量）。例如，batch数为2，属性`block_size=128`，当每个batch的`actual_seq_length`为512时，表明每个batch至少需要4个block，因此`block_table`的排布可以为`(2, 4)`。
    - page attention场景下，`block_table`中的有效值表示物理block编号，取值范围应在`[0, blocknum - 1]`内。
    - page attention场景下，`block_size`是用户自定义的参数，该参数的取值会影响page attention的性能，通常为128或256。`key`、`value`输入类型为`torch.float16`、`torch.bfloat16`时，`block_size`需要16对齐；`key`、`value`输入类型为`torch.int8`时，`block_size`需要32对齐。通常情况下，page attention可以提高吞吐量，但可能会增加单个token的解码时延。

- `quant_scale2`、`quant_offset2`为一组参数，其中`quant_offset2`可选，传入该组参数后算子输出数据类型会推导为`torch.int8`，若不期望`torch.int8`输出，请勿传入该组参数。
- KV左`padding`场景使用限制：
    - `kvCache`的搬运起点计算公式为：`Smax-kv_padding_size-actual_seq_lengths`。`kvCache`的搬运终点计算公式为：`Smax-kv_padding_size`。其中`kvCache`的搬运起点或终点小于0时，返回数据结果为全0。
    - KV左`padding`场景`kv_padding_size`小于0时将被置为0。
    - KV左`padding`场景需要同时存在`kv_padding_size`和`actual_seq_lengths`参数，否则默认为KV右`padding`场景。

## 调用示例

- 单算子调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # 生成随机数据，并发送到npu
    >>> q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> k = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> v = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>>
    >>> # 调用IFA算子
    >>> out = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale)                                        
    >>> print(out)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[[-1.4863,  0.1667,  0.7256,  ...,  0.3052, -0.3630, -0.1936]],

            [[-1.5840, -2.2305, -0.3462,  ..., -2.1055,  0.4392, -1.2842]]],
        device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import math
    
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    import os
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    
    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # 数据生成
    q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    k = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
    v = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
    atten = torch.randn(2, 1, 1, 2048).bool().npu()
    scale_value = 1/math.sqrt(128.0)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
            
        single_op = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)
        print("single op output with mask:", single_op, single_op.shape)
        print("graph output with mask:", graph_output, graph_output.shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
            [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
           device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
    graph output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
            [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
           device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
    ```
