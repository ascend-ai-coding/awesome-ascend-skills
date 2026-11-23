# torch_npu.npu_apply_adam_w

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：实现AdamW优化器的参数更新计算。AdamW在Adam的基础上将权重衰减（weight decay）与梯度更新解耦，直接作用于待优化参数，常用于大模型训练场景下的参数优化。接口在更新一阶矩`m`、二阶矩`v`的同时原地更新待优化参数`var`，并可选支持AMSGrad修正与maximize（对梯度取反，用于最大化目标）。
- 计算公式：

  设待优化参数为$var$、一阶矩为$m$、二阶矩为$v$、当前梯度为$grad$，则中间梯度$g_t$为：

  $$
  g_t = \begin{cases} -grad, & maximize = True \\ grad, & maximize = False \end{cases}
  $$

  一阶矩、二阶矩更新：

  $$
  m_{out} = \beta_1 \cdot m + (1 - \beta_1) \cdot g_t
  $$

  $$
  v_{out} = \beta_2 \cdot v + (1 - \beta_2) \cdot g_t^2
  $$

  分母项$denom$（由`amsgrad`决定是否引入`max_grad_norm`修正）：

  $$
  denom = \begin{cases} \sqrt{\dfrac{\max(max\_grad\_norm,\ v_{out})}{1 - \beta_2^{power} \cdot \beta_2}} + \epsilon, & amsgrad = True \\[2ex] \sqrt{\dfrac{v_{out}}{1 - \beta_2^{power} \cdot \beta_2}} + \epsilon, & amsgrad = False \end{cases}
  $$

  待优化参数更新：

  $$
  var_{out} = var \cdot (1 - lr \cdot weight\_decay) - lr \cdot \dfrac{m_{out} / (1 - \beta_1^{power} \cdot \beta_1)}{denom}
  $$

## 函数原型

```python
torch_npu.npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm=None, amsgrad=None, maximize=None, out=(var, m, v)) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **beta1_power**(`Scalar`)：必选参数，一阶矩指数衰减率`beta1`的幂，对应公式中的$\beta_1^{power}$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **beta2_power**(`Scalar`)：必选参数，二阶矩指数衰减率`beta2`的幂，对应公式中的$\beta_2^{power}$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **lr**(`Scalar`)：必选参数，学习率，对应公式中的$lr$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **weight_decay**(`Scalar`)：必选参数，权重衰减系数，对应公式中的$weight\_decay$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **beta1**(`Scalar`)：必选参数，一阶矩估计的指数衰减率，对应公式中的$\beta_1$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **beta2**(`Scalar`)：必选参数，二阶矩估计的指数衰减率，对应公式中的$\beta_2$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **epsilon**(`Scalar`)：必选参数，添加到分母中以提高数值稳定性的项，对应公式中的$\epsilon$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **grad**(`Tensor`)：必选参数，当前梯度，对应公式中的$grad$。数据格式支持$ND$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
- **max_grad_norm**(`Tensor`)：**可选参数**，AMSGrad修正所需的历史二阶矩最大值，对应公式中的$max\_grad\_norm$。数据格式支持$ND$，shape与数据类型与`grad`保持一致，支持`torch.float32`、`torch.float16`、`torch.bfloat16`。默认值为`None`；当`amsgrad`为`True`时**必须传入**。
- **amsgrad**(`bool`)：**可选参数**，是否使用AMSGrad修正。默认值为`None`（等效于`False`）。当前版本仅支持`False`。
- **maximize**(`bool`)：**可选参数**，是否对优化目标进行最大化（等效于对梯度`grad`取反）。默认值为`None`（等效于`False`）。
- **out**(`tuple(Tensor, Tensor, Tensor)`)：**必选参数**，指定输出张量`(var, m, v)`，计算结果原地写回该组张量：
  - **var**(`Tensor`)：待优化参数，对应公式中的$var$。数据格式支持$ND$，数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
  - **m**(`Tensor`)：一阶矩，对应公式中的$m$。Shape与数据类型与`var`一致。
  - **v**(`Tensor`)：二阶矩，对应公式中的$v$。Shape与数据类型与`var`一致。

## 返回值说明

- **var**(`Tensor`)：更新后的待优化参数，对应公式中的$var_{out}$。数据格式支持$ND$，Shape与数据类型与输入`var`一致。
- **m**(`Tensor`)：更新后的一阶矩，对应公式中的$m_{out}$。Shape与数据类型与输入`m`一致。
- **v**(`Tensor`)：更新后的二阶矩，对应公式中的$v_{out}$。Shape与数据类型与输入`v`一致。

> [!NOTICE]
> 返回的`(var, m, v)`与`out`中传入的三个张量为同一对象，计算结果为原地更新。

## 约束说明

- `var`、`m`、`v`、`grad`的Shape需保持一致。
- 当`amsgrad`为`True`时，必须传入`max_grad_norm`，否则接口报错；当前版本`amsgrad`仅支持`False`。
- 该接口仅支持训练场景。
- 该接口仅支持Eager模式调用。

## 调用示例

```python
import torch
import torch_npu

shape = (21130, 512)
var = torch.rand(shape, dtype=torch.float32).npu() * 10 + 10
m = torch.rand(shape, dtype=torch.float32).npu() * 5 + 5
v = torch.rand(shape, dtype=torch.float32).npu() * 5 + 0.1
grad = (torch.rand(shape, dtype=torch.float32).npu() - 0.5) * 10

beta1_power = 0.9
beta2_power = 0.999
lr = 0.01
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
max_grad_norm = None
amsgrad = False
maximize = False

var_out, m_out, v_out = torch_npu.npu_apply_adam_w(
    beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon,
    grad, max_grad_norm, amsgrad, maximize,
    out=(var, m, v),
)

print("var_out:", var_out)
print("m_out:", m_out)
print("v_out:", v_out)
```
