# Algorithm-Level Insights Index

Domain-specific algorithmic optimization knowledge for common operator families.
These insights go beyond instruction-level optimization — they reduce the amount of work itself.

## Key Principle

> When instruction-level optimization hits a "balance wall" (cube ≈ vector utilization),
> the only way forward is to reduce computation on BOTH sides simultaneously.
> This requires algorithm-level thinking, not pipeline tuning.

## Operator Families

| File | Operator Family | Key Insight |
|------|----------------|-------------|
| `attention_family.md` | FlashAttention, MHA, GQA, MQA | Block-sparse skip, online softmax, KV-cache |
| `reduction_ops.md` | LayerNorm, RMSNorm, Softmax, Mean/Var | Two-pass vs one-pass, Welford, tree reduction |
| `elementwise_fusion.md` | GELU, SiLU, Add+Mul chains | Operator fusion, in-place computation |

## How to Use

1. **During Init**: Read the relevant family file to understand algorithmic optimization potential
2. **During open_exploration**: Use as inspiration source for novel approaches
3. **During stagnation reflection**: Check if an untried algorithmic approach exists

## 触发条件

| 条件 | 应读文件 |
|------|---------|
| 算子含 QK^T, softmax, attention | `attention_family.md` |
| 算子含 ReduceSum/Max, Norm, Mean/Var | `reduction_ops.md` |
| 算子是多个 elementwise 操作链 | `elementwise_fusion.md` |
| 停滞 + best_score 处于平衡态 | 全部读取，寻找算法突破口 |
