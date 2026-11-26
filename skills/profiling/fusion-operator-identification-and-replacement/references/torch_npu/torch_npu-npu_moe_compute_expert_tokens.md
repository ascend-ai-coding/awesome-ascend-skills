# torch_npu.npu_moe_compute_expert_tokens

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：MoE（Mixture of Experts，混合专家模型）计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。
- 计算公式：

    $$
    expertTokens_i = BinarySearch(sortedExpertForSourceRow,i), \quad i \in [0,numExpert)
    $$

## 函数原型

```python
torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert) -> Tensor
```

## 参数说明

- **sorted_expert_for_source_row** (`Tensor`)：必选参数，每个source row经过排序后对应的专家索引，对应公式中的$sortedExpertForSourceRow$，要求是一个1维变量，数据类型支持`torch.int32`，数据格式要求为$ND$。输入值取值范围为[0, `num_expert`-1]，且需按非递减顺序排列；Tensor元素个数需小于$2^{24}$。

- **num_expert** (`int`)：必选参数，表示总专家数。对应公式中的$numExpert$，取值范围为(0, 2048]。

## 返回值说明

`Tensor`

 对应公式中的$expertTokens$，要求的是一个1维张量，shape为[`num_expert`]，数据类型与`sorted_expert_for_source_row`保持一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    sorted_experts = torch.tensor([0,0,0,1,1,2], dtype=torch.int32)
    num_experts = 3
    output = torch_npu.npu_moe_compute_expert_tokens(sorted_experts.npu(), num_experts)
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
        
        def forward(self, sorted_experts, num_experts):
            return torch_npu.npu_moe_compute_expert_tokens(sorted_experts, num_experts)
    def main():
        sorted_experts = torch.tensor([0,0,0,1,1,2], dtype=torch.int32)
        num_experts = 3
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(sorted_experts, num_experts)
    if __name__ == '__main__':
        main()
    ```
