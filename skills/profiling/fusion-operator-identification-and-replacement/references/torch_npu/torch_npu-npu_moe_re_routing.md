# torch\_npu.npu\_moe\_re\_routing

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。
    - MoE 网络：混合专家（Mixture of Experts）网络，每层包含多个并行“专家”子网络，由路由（Routing）模块决定每个token由哪个或哪些专家处理。
    - token：模型处理的最小数据单元，通常为词或子词经过嵌入（embedding）后得到的特征向量。
    - 专家（Expert）：MoE网络中的子网络。在多卡部署时，专家被切分到不同卡上，每张卡承载若干个专家。
    - AlltoAll操作：集合通信原语。每张卡将本地数据按目的卡切分后发送给所有其他卡，同时接收来自所有其他卡的数据。

- 计算公式：

    $$SrcOffset = \sum_{i=0}^{cur\_rank} \left( \sum_{j=0}^{cur\_expert} expert\_token\_num\_per\_rank(i,j) \right)$$

    $$DstOffset = \sum_{j=0}^{cur\_expert} \left( \sum_{i=0}^{cur\_rank} expert\_token\_num\_per\_rank(i,j) \right)$$

    - SrcOffset指当前需要移动的token源偏移，根据输入`expert_token_num_per_rank`的值进行计算。
    - DstOffset指当前需要移动的token目的偏移。
    - cur\_rank是`expert_token_num_per_rank`的纵轴索引，表示该token原本在的卡。
    - cur\_expert是`expert_token_num_per_rank`的横轴索引，表示该token由卡上专家cur\_expert计算。

- 流程说明

    该算子位于MoE（Mixture of Experts）并行推理流程的**通信与计算衔接阶段**，具体位置如下：

    ```text
    Gating/TopK Routing
        ↓
    npu_moe_distribute_dispatch_v2  （AlltoAll 分发 token 到各卡）
        ↓
    npu_moe_re_routing  ← 本算子：按专家顺序重排 token
        ↓
    Expert FFN 计算
        ↓
    npu_moe_distribute_combine_v2  （结果聚合与 AlltoAll 返回）
    ```

  - 作用说明：

    1. **前置阶段**：`npu_moe_distribute_dispatch_v2` 通过AlltoAll通信将token从各卡收集到本地。此时token在本地内存中是**按源卡（rank）顺序**排列的。

    2. **本算子作用**：由于每个专家需要连续、批量地处理属于自己的token，本算子将token从“按源卡排列”转换为“**按专家（expert）排列**”，即同一专家的token在内存中连续存放。同时输出 `permute_token_idx`，用于后续Expert计算完成后恢复原始token顺序。

    3. **后置阶段**：Expert FFN层根据 `expert_token_num` 中记录的专家token数量，依次读取 `permute_tokens` 中连续的数据块进行计算。

    4. **数据恢复**：Expert计算结束后，利用本算子输出的 `permute_token_idx`，通过 `npu_moe_distribute_combine_v2` 将计算结果按原路聚合并返回。

## 函数原型

```python
torch_npu.npu_moe_re_routing(tokens, expert_token_num_per_rank, *, per_token_scales=None, expert_token_num_type=1, idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

> [!NOTE]  
> Tensor中shape使用的变量说明：
>
> - A：表示token个数，取值要求Sum\(expert\_token\_num\_per\_rank\)=A。
> - H：表示token长度，取值要求0<H<16384。
> - N：表示卡数，取值无限制。
> - E：表示卡上的专家数，取值无限制。

- **tokens** (`Tensor`)：必选参数，表示待重新排布的token。要求为2维，shape为\[A, H\]，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`，数据格式为$ND$。
- **expert\_token\_num\_per\_rank** (`Tensor`)：必选参数，二维矩阵，矩阵中元素[i, j]表示当前卡上从卡i获取到的专家j处理的token数。要求为2维，shape为\[N, E\]，数据类型支持`torch.int32`、`torch.int64`，数据格式为$ND$。取值必须大于0。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **per\_token\_scales** (`Tensor`)：可选参数，表示每个token对应的scale，需要随token同样进行重新排布。要求为1维，shape为\[A\]，数据类型支持`torch.float32`，数据格式为$ND$。
- **expert\_token\_num\_type** (`int`)：可选参数，表示输出`expert_token_num`的模式。0为cumsum模式，1为count模式，默认值为1。当前只支持为1。
- **idx\_type** (`int`)：可选参数，表示输出`permute_token_idx`的索引类型。0为gather索引，1为scatter索引，默认值为0。当前只支持为0。

## 返回值说明

- **permute\_tokens** (`Tensor`)：表示重新排布后的token。要求为2维，shape为\[A, H\]，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`，数据格式为$ND$。
- **permute\_per\_token\_scales** (`Tensor`)：表示重新排布后的`per_token_scales`，输入不携带`per_token_scales`的情况下，该输出无效。要求为1维，shape为\[A\]，数据类型支持`torch.float32`，数据格式为$ND$。
- **permute\_token\_idx** (`Tensor`)：表示每个token在原排布方式的索引。要求为1维，shape为\[A\]，数据类型支持`torch.int32`，数据格式为$ND$。
- **expert\_token\_num** (`Tensor`)：表示每个专家处理的token数。要求为1维，shape为\[E\]，数据类型支持`torch.int32`、`torch.int64`，数据格式为$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import random
    import copy
    import math
    
    tokens_num = 16384
    tokens_length = 7168
    rank_num = 16
    expert_num = 16
    tokens = torch.randint(low=-10, high = 20, size=(tokens_num, tokens_length), dtype=torch.int8)
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int32)
    tokens_sum = 0
    for i in range(rank_num):
        for j in range(expert_num):
            if i == rank_num - 1 and j == expert_num - 1:
                expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                break
            if tokens_num >= rank_num * expert_num :
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
            elif tokens_sum >= tokens_num:
                rand_num = 0
            else:
                rand_int = tokens_num - tokens_sum
                rand_num = random.randint(0, rand_int)
            expert_token_num_per_rank[i][j] = rand_num
            tokens_sum += rand_num
    per_token_scales = torch.randn(tokens_num, dtype = torch.float32)
    expert_token_num_type = 1
    idx_type = 0
    tokens_npu = copy.deepcopy(tokens).npu()
    per_token_scales_npu = copy.deepcopy(per_token_scales).npu()
    expert_token_num_per_rank_npu = copy.deepcopy(expert_token_num_per_rank).npu()
    permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = torch_npu.npu_moe_re_routing(tokens_npu, expert_token_num_per_rank_npu, per_token_scales=per_token_scales_npu, expert_token_num_type=expert_token_num_type, idx_type=idx_type)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import random
    import copy
    import math
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    tokens_num = 16384
    tokens_length = 7168
    rank_num = 16
    expert_num = 16
    tokens = torch.randint(low=-10, high = 20, size=(tokens_num, tokens_length), dtype=torch.int8)
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int32)
    tokens_sum = 0
    for i in range(rank_num):
        for j in range(expert_num):
            if i == rank_num - 1 and j == expert_num - 1:
                expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                break
            if tokens_num >= rank_num * expert_num :
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
            elif tokens_sum >= tokens_num:
                rand_num = 0
            else:
                rand_int = tokens_num - tokens_sum
                rand_num = random.randint(0, rand_int)
            expert_token_num_per_rank[i][j] = rand_num
            tokens_sum += rand_num
    per_token_scales = torch.randn(tokens_num, dtype = torch.float32)
    expert_token_num_type = 1
    idx_type = 0
    tokens_npu = copy.deepcopy(tokens).npu()
    per_token_scales_npu = copy.deepcopy(per_token_scales).npu()
    expert_token_num_per_rank_npu = copy.deepcopy(expert_token_num_per_rank).npu()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, tokens, expert_token_num_per_rank, per_token_scales=None, expert_token_num_type=1, idx_type=0):
            permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = torch_npu.npu_moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales, expert_token_num_type=expert_token_num_type, idx_type=idx_type)
            return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = model(tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu, expert_token_num_type, idx_type)
    ```
