# torch_npu-npu_lightning_indexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。

- 计算公式：
     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$
     对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

## 函数原型

```python
torch_npu.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=2**63-1, next_tokens=2**63-1, return_value=False) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]   
>
> - query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - S1表示query shape中的S，S2表示key shape中的S，T1表示query shape中的T，T2表示key shape中的T，N1表示query shape中的N，N2表示key shape中的N。

- **query**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`torch.bfloat16`和`torch.float16`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]，N1仅支持小于等于64。
    
- **key**（`Tensor`）：必选参数，在`layout_key`为PA_BSND时支持0轴非连续，数据格式支持$ND$，数据类型支持`torch.bfloat16`和`torch.float16`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2, D]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的整数倍，最大支持到1024。`layout_key`为BSND时shape为[B, S2, N2, D]，`layout_key`为TND时shape为[T2, N2, D]，N2仅支持1。
    
- **weights**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`和`torch.float32`，支持输入shape[B,S1,N1]、[T,N1]。
    
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`torch.int32`。如果不指定actual\_seq\_lengths\_query可传入None，表示和`query`的shape的S长度相同。
    - 该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中`key`的有效token数，数据类型支持`torch.int32`。如果不指定actual\_seq\_lengths\_key可传入None，表示和key的shape的S长度相同。
    - 该参数中每个Batch的有效token数不超过`key`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_key`为TND或PA_BSND时，该入参必须传入，`layout_key`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持$ND$，数据类型支持`torch.int32`。
    - PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量）

- **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

- **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前支持PA_BSND、BSND、TND，默认值"BSND"，在非PageAttention场景下，该参数值应与**layout\_query**值保持一致。

- **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持[1, 2048]以及3072、4096、5120、6144、7168、8192，数据类型支持`torch.int32`。

- **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`torch.int32`。
    
    - sparse\_mode为0时，代表defaultMask模式。
    - sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

- **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`torch.int64`。仅支持默认值2^63-1。

- **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`torch.int64`。仅支持默认值2^63-1。

- **return\_value**（`bool`）：可选参数，表示是否输出`sparse_values`。True表示输出，False表示不输出；默认值为False。该参数仅在训练且`layout_key`不为PA_BSND场景下支持。

## 返回值说明

- **sparse\_indices**（`Tensor`）：公式中的Indices输出，数据类型支持`torch.int32`，数据格式支持$ND$，当`layout_query`为"BSND"时输出shape为[B, S1, N2, sparse\_count]，当layout\_query为"TND"时输出shape为[T1, N2, sparse\_count]。

- **sparse\_values**（`Tensor`）：公式中的Indices输出对应的value值，数据类型支持`torch.bfloat16`、`torch.float16`，数据格式支持$ND$，输出shape与`sparse_indices`保持一致。

## 约束说明

- 该接口支持图模式。
- 参数key中的N仅支持为1。
- 参数query中的D和参数key中的D值相等为128。
- 参数query、key的数据类型应保持一致。
- 参数weights不为`torch.float32`时，参数query、key、weights的数据类型应保持一致。

<!-- npu="950" id4 -->
- return_value仅在<term>Ascend 950DT系列产品</term>上支持图模式。
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>：
    - query中的N仅支持8、16、24、32、64。
    - 参数weights不支持`torch.float32`类型。
    - 仅在layout_key为PA_BSND时，key支持0轴非连续。
<!-- end id5 -->
<!-- npu="A3,910b" id6 -->
- <term>Atlas A2推理系列产品</term>、<term>Atlas A3推理系列产品</term>：
    - query中的N必须小于等于64。
    - key不支持非连续。
<!-- end id6 -->
- 确定性说明：
    - 正向默认确定性实现。
    - 反向梯度默认非确定性实现，不支持开启确定性。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    # 生成随机数据, 并发送到npu
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    
    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16).npu()
    key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16).npu()
    weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16).npu()
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32).npu()
    block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1).npu()
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    # 调用lightning_indexer算子
    sparse_indices, sparse_values = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

    print(sparse_indices)
    # 执行上述代码的输出sparse_indices类似如下
    tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import math
    import torchair as tng

    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # 数据生成
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    
    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16).npu()
    key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16).npu()
    weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16).npu()
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32).npu()
    block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1).npu()
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        print("single op output:", single_op[0], single_op[0].shape)
        print("graph output:", graph_output[0], graph_output[0].shape)
    if __name__ == "__main__":
        MetaInfershape()

    # 执行上述代码的输出类似如下
    single op output: tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32) torch.Size([1, 1, 1, 2048])

    graph output: tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32) torch.Size([1, 1, 1, 2048])
    ```
