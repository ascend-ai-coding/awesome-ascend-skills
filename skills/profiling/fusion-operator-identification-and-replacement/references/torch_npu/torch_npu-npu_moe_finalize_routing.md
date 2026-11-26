# torch_npu.npu_moe_finalize_routing

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：在MoE计算的最后，合并MoE FFN(Feedforward Neural Network)的输出结果。

- 计算公式：

    $$
    expertid = expertForSourceRow[i,k] \\
    out(i,j) = skip1_{i,j} + skip2Optional_{i,j} + \sum_{k=0}^{K}(scales_{i,k} * (expandPermutedRows_{expandedSrcToDstRow_{i+k*num\_rows},j} + bias_{expertid,j}))
    $$

- 等价计算逻辑

    ```python
    import numpy as np
    from copy import deepcopy   
        
    def generate_input_data(expert_num=16, token_len=10, top_k=4, num_rows=50):
        expanded_permuted_rows = np.random.randn(num_rows * top_k, token_len).astype(np.float32)
        skip1 = np.random.randn(num_rows, token_len).astype(np.float32)
        skip2_optional = np.random.randn(num_rows, token_len).astype(np.float32)
        bias = np.random.randn(expert_num, token_len).astype(np.float32)
        scales = np.random.randn(num_rows, top_k).astype(np.float32)
        expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row)
        expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(num_rows, top_k)).astype(np.int32)
        return expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row
        
    def generate_input_data_drop_pad(expert_num=16, token_len=10, c=20, top_k=4, num_rows=50):
        expanded_permuted_rows = np.random.randn(expert_num, c, token_len).astype(np.float32)
        skip1 = np.random.randn(num_rows, token_len).astype(np.float32)
        skip2_optional = np.random.randn(num_rows, token_len).astype(np.float32)
        bias = np.random.randn(expert_num, token_len).astype(np.float32)
        scales = None
        expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row)
        expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(num_rows, top_k)).astype(np.int32) 
        return expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row
        
            
    def moe_finalize_routing_np(expanded_permuted_rows: np.array,skip1: np.array,skip2_optional: np.array ,bias: np.array ,scales: np.array ,expanded_src_to_dst_row: np.array ,expert_for_source_row: np.array):
        NK = expanded_src_to_dst_row.shape[0]
        K = 1
        if scales is not None:
            K = scales.shape[1] 
        num_rows = NK // K
        H = expanded_permuted_rows.shape[-1]
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        if (skip1 is not None) and (skip2_optional is not None):
            out = skip1 + skip2_optional
        elif (skip2_optional is not None) and (skip1 is None):
            out = deepcopy(skip2_optional)
        elif (skip2_optional is None) and (skip1 is not None):
            out = deepcopy(skip1)
        else:
            out = np.zeros([num_rows, H])
        for i in range(num_rows):
            for k in range(K):
                value = expanded_src_to_dst_row[k * num_rows + i]
                if value == -1:
                    dst_row = 0
                else:
                    dst_row = expanded_permuted_rows[value, :]
                expert_id = expert_for_source_row[i, k]
                    
                scalesV = 1.0
                if scales is not None:
                    scalesV = scales[i, k]
                if bias is not None:
                    out[i, :] += scalesV * (dst_row + bias[expert_id, :])
                else:
                    out[i, :] += scalesV * dst_row
        return out

    if __name__ == "__main__":
        #test_moe_finalize_routing
        expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row = generate_input_data(expert_num=16, token_len=5, top_k=4, num_rows=5)
        out1 = moe_finalize_routing_np(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row)
        print(f"\nout with moe finalize routing:{out1}")
        #moe_finalize_routing_np_drop_pad
        expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row = generate_input_data_drop_pad(expert_num=16, token_len=5, top_k=1, num_rows=20)
        out2 = moe_finalize_routing_np(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row)
        print(f"\nout with moe finalize routing drop pad:{out2}")  
    ```

## 函数原型

```python
torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode=0, k=1) -> Tensor
```

## 参数说明
>
> [!NOTE]  
> shape中的符号说明：
>
> - $NUM\_ROWS$：为行数。
> - $K$：表示从总的专家$E$中选出$K$个专家。
> - $H$：表示每个token序列长度，为列数。
> - $E$: 表示专家数，$E$需要大于等于$K$。
> - $C$: 表示专家处理token数量的能力阈值。

- **expanded_permuted_rows** (`Tensor`)：必选参数，对应公式中的$expandPermutedRows$，经过专家处理过的结果，要求为一个2维张量，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式要求为$ND$。`drop_pad_mode`参数为0或2时，shape为$(NUM\_ROWS * K, H)$，`drop_pad_mode`参数为1或3时，shape为$(E, C, H)$。
- **skip1** (`Tensor`)：必选参数，允许为None，对应公式中的$skip1$，求和的输入参数1，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape要求与输出`out`的shape一致。
- **skip2** (`Tensor`)：必选参数，允许为None，对应公式中的$skip2$，求和的输入参数2，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape要求与输出`out`的shape一致。`skip1`参数为`None`时，`skip2`参数必须也为`None`。
- **bias** (`Tensor`)：必选参数，允许为None，对应公式中的$bias$，专家的偏差，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致。shape支持$(E，H)$。
- **scales** (`Tensor`)：必选参数，允许为None，对应公式中的$scales$，专家的权重，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape支持($NUM\_ROWS，K)$。当`scales`为None时，`K`通过参数`k`获取。
- **expanded_src_to_dst_row** (`Tensor`)：必选参数，对应公式中的$expandedSrcToDstRow$，保存每个专家处理结果的索引，要求为一个1维张量，数据类型支持`torch.int32`。shape支持$(NUM\_ROWS * K)$，`drop_pad_mode`参数为0或2时，Tensor的取值范围是$[0, NUM\_ROWS * K-1]$，`drop_pad_mode`参数为1或3时，Tensor的取值范围是$[-1, E*C - 1]$。
- **expert_for_source_row** (`Tensor`)：必选参数，允许为None，公式中的$expertForSourceRow$，每行处理的专家号，要求为一个2维张量，数据类型支持`torch.int32`。shape支持$(NUM\_ROWS，K)$，Tensor的取值范围是[0,E-1]。
- **drop_pad_mode** (`int`)：可选参数，表示是否为丢弃模式（丢弃模式即drop pad模式，非丢弃模式即drop less模式）和`expanded_src_to_dst_row`的排列方式（行排列或列排列），取值范围为[0, 3]，默认值为`0`。
     - 0表示非丢弃模式，`expanded_src_to_dst_row`按列排列。
     - 1表示丢弃模式，`expanded_src_to_dst_row`按列排列。
     - 2表示非丢弃模式场景，`expanded_src_to_dst_row`按行排列。
     - 3表示丢弃场景，`expanded_src_to_dst_row`按行排列。
- **k** (`int`)：可选参数，表示top-K专家数，默认值为`1`。`scales`为None时，需要显式传入`k`值。

## 返回值说明

`Tensor`

输出参数`out`，代表最后MoE FFN合并的输出结果。数据维度支持2维，shape支持$(NUM\_ROWS, H)$，dtype与expanded_permuted_rows保持一致。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。

## 调用示例

- 单算子模式调用
    - drop less模式调用示例

        ```python
        import torch
        import torch_npu
        expert_num = 16
        token_len = 10
        top_k = 4
        num_rows = 50
        device = torch.device('npu')
        dtype = torch.float32
        expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
        skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
        drop_pad_mode = 0
        output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        ```

    - drop pad模式调用示例

        ```python
        import torch
        import torch_npu
        expert_num = 16
        token_len = 10
        top_k = 4
        num_rows = 50
        expert_capacity = 8
        device = torch.device('npu')
        dtype = torch.float32
        expanded_permuted_rows = torch.randn((expert_num, expert_capacity, token_len), device=device, dtype=dtype)
        skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        expanded_src_to_dst_row = torch.randint(low=-1, high=expert_num * expert_capacity - 1, size=(num_rows * top_k,), device=device, dtype=torch.int32)
        drop_pad_mode = 1
        output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        ```

- 图模式调用

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode):
            return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)

    def main():
        expert_num = 16
        token_len = 10
        top_k = 4
        num_rows = 50
        device =torch.device('npu')
        dtype = torch.float32

        expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
        skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
        drop_pad_mode = 0

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        custom_output = model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        print(custom_output.shape, custom_output.dtype)

    if __name__ == '__main__':
        main()
    
    # 执行上述代码的输出类似如下
    torch.Size([50, 10]) torch.float32
    ```
