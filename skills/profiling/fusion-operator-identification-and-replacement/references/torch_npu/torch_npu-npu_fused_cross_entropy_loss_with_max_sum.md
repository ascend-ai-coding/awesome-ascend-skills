# torch\_npu.npu\_fused\_cross\_entropy\_loss\_with\_max\_sum

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题。本接口为计算Loss与Softmax结果的部分，需配合npu\_fused\_linear\_online\_max\_sum使用。在多卡场景下，需先对`logits_max`和`sum_exp_logits`执行全局通信（all-reduce），再调用本接口。

- 计算公式：

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) / sum\_exp\_logits.unsqueeze(dim = -1)
    $$

## 函数原型

```python
torch_npu.npu_fused_cross_entropy_loss_with_max_sum(logits_max, sum_exp_logits, predicted_logits, *, label_smoothing=0.0, input=None, weight=None, vocab_parallel_logits=None) -> (Tensor, Tensor)
```

## 参数说明

- **logits\_max**（`Tensor`）：必选参数，全局通信后的MatMul结果各行最大值，1维Tensor。数据类型支持`torch.float32`，数据格式支持ND。
- **sum\_exp\_logits**（`Tensor`）：必选参数，全局通信后的exp累加结果，1维Tensor。数据类型支持`torch.float32`，数据格式支持ND。shape与`logits_max`一致。
- **predicted\_logits**（`Tensor`）：必选参数，全局通信后的预测logits，1维Tensor。数据类型支持`torch.float32`，数据格式支持ND。shape与`logits_max`一致。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **label\_smoothing**（`float`）：可选参数，标签平滑系数，当前仅支持0.0，默认值为0.0。
- **input**（`Tensor`）：可选参数，MatMul输入左矩阵，当前仅支持None，默认值为None。
- **weight**（`Tensor`）：可选参数，MatMul输入右矩阵，当前仅支持None，默认值为None。
- **vocab\_parallel\_logits**（`Tensor`）：可选参数，MatMul计算结果。传入时计算Softmax输出，不传入时Softmax输出为空Tensor。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`，2维Tensor，shape第0维需与`logits_max`一致。默认值为None。

## 返回值说明

- **loss**（`Tensor`）：交叉熵Loss，数据类型支持`torch.float32`，shape与`logits_max`一致。
- **softmax**（`Tensor`）：Softmax结果。`vocab_parallel_logits`非None时，数据类型支持`torch.float32`，shape与`vocab_parallel_logits`一致；`vocab_parallel_logits`为None时，返回空Tensor。

## 约束说明

- `logits_max`、`sum_exp_logits`、`predicted_logits`的shape需一致。
- `label_smoothing`当前仅支持0。

## 调用示例

```python
import torch
import torch_npu

# 假设已通过npu_fused_linear_online_max_sum获取输出并完成全局通信
batch = 128
vocab_size = 256

logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
predicted_logits = torch.randn(batch, dtype=torch.float32).npu()

# 不返回softmax
loss, _ = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0
)

# 返回softmax（传入vocab_parallel_logits）
vocab_parallel_logits = torch.randn(batch, vocab_size, dtype=torch.float16).npu()
loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0, vocab_parallel_logits=vocab_parallel_logits
)
```
