# torch_npu.npu_moe_init_routing

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

- API功能：MoE的routing计算，根据[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的计算结果做routing处理。
- 计算公式：
    $$
    expanded\_expert\_idx, sorted\_row\_idx = keyValueSort(expert\_idx, row\_idx)\\
    expanded\_row\_idx[sorted\_row\_idx[i]] = i\\
    expanded\_x[i] = x[sorted\_row\_idx[i] \% num\_rows]
    $$

- 等价计算逻辑

    ```python
    import numpy as np

    def cpu_op_exec( x, row_idx, expert_idx, active_num):
        num_rows = x.shape[0]
        hidden_size = x.shape[-1]
        k = expert_idx.shape[-1]
        sort_expert_for_source_row = np.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable")
        expanded_expert_idx = np.sort(
            expert_idx.reshape((-1,)), axis=-1)

        expanded_dst_to_src_row = np.take_along_axis(
            row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
        expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
        expanded_row_idx[expanded_dst_to_src_row] = np.arange(
            expanded_dst_to_src_row.shape[-1])
        active_num = min(active_num, num_rows) * k
        expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return expanded_x, expanded_row_idx, expanded_expert_idx


    if __name__ == "__main__":
        n = 10
        col = 200
        k = 2
        dtype = np.float32 
        x = np.random.uniform(-1, 1, size=(n, col)).astype(dtype)
        row_idx = np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32)
        expert_idx = np.random.randint(0, 100, size=(n, k)).astype(np.int32)
        active_num = n
        expanded_x, expanded_row_idx, expanded_expert_idx = cpu_op_exec( x, row_idx, expert_idx, active_num)
        print(f"expanded_x:{expanded_x}")
        print(f"expanded_row_idx:{expanded_row_idx}")
        print(f"expanded_expert_idx:{expanded_expert_idx}")
    ``` 
    
## 函数原型

```python
torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，MoE的输入（即token特征输入），要求为一个2维张量，shape为(NUM_ROWS, H)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式要求为$ND$。shape大小需要小于2^24。
- **row_idx** (`Tensor`)：必选参数，指示每个位置对应的原始行位置，shape要求与`expert_idx`一致。数据类型支持`torch.int32`，数据格式要求为$ND$。
- **expert_idx** (`Tensor`)：必选参数，[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的输出，即每一行特征对应的K个处理专家，要求是2维，shape为(NUM_ROWS, K)，数据类型支持`torch.int32`，数据格式为$ND$。
- **active_num** (`int`)：必选参数，表示总的最大处理row数，输出`expanded_x`仅支持此处设置的最大处理行数。

## 返回值说明

- **expanded_x** (`Tensor`)：根据`expert_idx`进行扩展过的特征，要求是一个2维张量，shape为(min(NUM_ROWS, active_num) \* K, H)。数据类型同`x`，数据格式要求为$ND$。
- **expanded_row_idx** (`Tensor`)：`expanded_x`和`x`的映射关系，要求是一个1维张量，shape为(NUM_ROWS\*K, )，数据类型支持`torch.int32`，数据格式要求为$ND$。
- **expanded_expert_idx** (`Tensor`)：输出`expert_idx`排序后的结果。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num)
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
    
    device=torch.device(f'npu:0')
    
    torch_npu.npu.set_device(device)
    
    class MoeInitRoutingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, row_idx, expert_idx, active_num):
            expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num=active_num)
            return expanded_x, expanded_row_idx, expanded_expert_idx
    
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    
    moe_init_routing_model = MoeInitRoutingModel().npu()
    moe_init_routing_model = torch.compile(moe_init_routing_model, backend=npu_backend, dynamic=True)
    expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing_model(x, row_idx, expert_idx, active_num=active_num)
    print(expanded_x)
    print(expanded_row_idx)
    print(expanded_expert_idx)
    ```
    