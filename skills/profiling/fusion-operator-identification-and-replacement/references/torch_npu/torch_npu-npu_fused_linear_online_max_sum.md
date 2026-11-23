# torch\_npu.npu\_fused\_linear\_online\_max\_sum

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：词汇表并行场景下融合矩阵乘与交叉熵前处理算子。功能等价于Megatron的MatMul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切分融合MatMul与CELoss。本接口为通信前部分，需与npu\_fused\_cross\_entropy\_loss\_with\_max\_sum配合使用。

- 计算公式：

  1. $input$与$weight^T$做矩阵乘得到：

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
  2. 计算$vocabParallelLogitsOutOptional$每行的最大值：

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
  3. 计算$vocabParallelLogitsOutOptional$与$logitsMaxLocalOut$的差值：

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$
  4. 计算$subRes$经过指数运算后每行的和

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$
  5. 计算$target$小于$vocabStartIndex$或$target$大于$vocabEndIndex$的mask

     $$
     targetMask = (target < vocabStartIndex) | (target > vocabEndIndex)
     $$
  6. 计算$maskedTargetOut$

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$
  7. 计算$predictedLogitsLocalOut$

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$
  8. 计算$targetMaskOut$

     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     `torch.uint8`(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  其中$0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, 0 \le k \lt alignNum / 8$。

## 函数原型

```python
torch_npu.npu_fused_linear_online_max_sum(input, weight, target, vocab_start_index, vocab_end_index, return_logits=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，MatMul计算的左矩阵，2维Tensor。数据类型支持`torch.bfloat16`、`torch.float16`，数据格式支持ND。`input.size(1)`需小于等于65534。支持空Tensor。
- **weight**（`Tensor`）：必选参数，MatMul计算的右矩阵，2维Tensor。数据类型需与`input`一致，数据格式支持ND。`weight.size(1)`需与`input.size(1)`一致。支持空Tensor。
- **target**（`Tensor`）：必选参数，目标索引，1维Tensor。数据类型支持`torch.int32`、`torch.int64`，数据格式支持ND。`target.size(0)`需与`input.size(0)`一致。支持空Tensor。
- **vocab\_start\_index**（`int`）：必选参数，本卡分配的词汇表起始索引。取值范围`[0, max(target) - 1]`。
- **vocab\_end\_index**（`int`）：必选参数，本卡分配的词汇表结束索引。取值范围`[vocab_start_index, min(vocab_start_index + weight.size(0) - 1, max(target) - 1)]`。
- **return\_logits**（`bool`）：可选参数，是否返回MatMul结果`vocabParallelLogits`，默认值为`False`。`True`时走高性能分支，`False`时走省显存分支。

## 返回值说明

- **logits\_max**（`Tensor`）：MatMul计算后各行的最大值，数据类型支持`torch.float32`，shape为`[input.size(0)]`。
- **sum\_exp\_logits**（`Tensor`）：`subRes`经exp后各行累加结果，数据类型支持`torch.float32`，shape为`[input.size(0)]`。
- **predicted\_logits**（`Tensor`）：`subRes`经`maskedTarget`筛选后的结果，数据类型支持`torch.float32`，shape为`[input.size(0)]`。
- **target\_mask**（`Tensor`）：词汇表mask的packed bit表示，数据类型支持`torch.uint8`，shape为`[(input.size(0) + 7) // 8]`。
- **masked\_target**（`Tensor`）：target经mask过滤后的结果，数据类型与`target`一致，shape为`[input.size(0)]`。
- **vocab\_parallel\_logits**（`Tensor`）：MatMul计算结果。`return_logits`为`True`时，数据类型与`input`一致，shape为`[input.size(0), weight.size(0)]`；`return_logits`为`False`时，返回空Tensor。

## 约束说明

- `input`与`weight`数据类型必须一致。
- `target`与`masked_target`数据类型必须一致。
- `vocabParallelLogits`（`return_logits`为`True`时）与`input`数据类型必须一致。
- `vocab_start_index`不可小于0。
- `vocab_end_index`不可小于`vocab_start_index`。

## 调用示例

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()

# 高性能模式（return_logits=True）
logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
    torch_npu.npu_fused_linear_online_max_sum(
        input_tensor, weight_tensor, target_tensor,
        vocab_start_index=0, vocab_end_index=64, return_logits=True
    )

# 省显存模式（return_logits=False）
logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, _ = \
    torch_npu.npu_fused_linear_online_max_sum(
        input_tensor, weight_tensor, target_tensor,
        vocab_start_index=0, vocab_end_index=64, return_logits=False
    )
```
