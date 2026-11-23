# torch_npu.npu_cross_entropy_loss

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：计算输入`input`和标签`target`之间的交叉熵损失。此API将原生`CrossEntropyLoss`中的log_softmax和nll_loss融合，降低计算时使用的内存。
- 计算公式：

    公式中*x*是输入`input`，*y* 是标签`target`，*weight*是权重，*C* 是标签数，*N* 是批处理大小。

    交叉熵损失`loss`的计算公式：
    $$
    loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reduction = ‘mean’} \\\sum_{n=1}^Nl_n,&\text {if reduction = ‘sum’ }\\\{l_0,l_1,...,l_n\},&\text{if reduction = ‘none’ }\end{cases}
    $$
    
    其中$l_n$的计算公式为：
    $$
    l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
    $$
  
    第*n*个样本对第*c*个类别的对数概率`log_prob`计算公式为：
    $$
    lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
    $$

    $$
    logProb_{n,c} = x_{n,c} - lse_n
    $$

## 函数原型

```python
torch_npu.npu_cross_entropy_loss(input, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0, lse_square_scale_for_zloss=0.0, return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）: 必选参数，表示输入，对应公式中*x*；数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`；shape为[N, C]，N为批处理大小，C为标签数，必须大于0。
- **target**（`Tensor`）: 必选参数，表示标签，对应公式中*y*；数据类型支持`torch.int64`；shape为[N]，与`input`第零维相同，取值范围[0, C)。
- **weight**（`Tensor`）: 可选参数，表示每个类别指定的缩放权重；数据类型支持`torch.float32`；shape为[C]，与`input`第二维相同，取值范围(0, 1]，不指定值时默认为全一。
- **reduction**（`str`）: 可选参数，表示loss的归约方式；支持范围["mean", "sum", "none"]，`mean`表示平均归约，`sum`表示求和归约，`none`表示无归约，默认为`mean`。
- **ignore_index**（`int`）: 可选参数，表示指定忽略的标签；数值必须小于C，当小于0时表示不指定忽略标签；默认值为-100。
- **label_smoothing**（`float`）: 可选参数，表示计算loss时的平滑量；取值范围[0.0, 1.0)；默认值为0.0。
- **lse_square_scale_for_zloss**（`float`）: 可选参数，表示计算zloss所需要的scale；取值范围[0.0, 1.0)；默认值为0.0；当前暂不支持。
- **return_zloss**（`bool`）: 可选参数，控制是否返回zloss；设置为`True`时返回zloss，设置为`False`时不返回zloss；默认值为`False`；当前暂不支持。

## 返回值说明

- **loss**（`Tensor`）：表示输出损失；数据类型与`input`相同；`reduction`为`none`时shape为[N]，与`input`第零维一致，否则shape为[1]。
- **log_prob**（`Tensor`）: 用于反向计算的输出；数据类型与`input`相同；shape为[N, C]，与`input`一致。
- **zloss**（`Tensor`）: 表示辅助损失；数据类型与`input`相同；shape与`loss`一致；当`return_zloss`为`True`时输出zloss，否则将返回空tensor；当前暂不支持。
- **lse_for_zloss**（`Tensor`）:在zloss场景下给反向计算的输出；数据类型与`input`相同；shape为[N]，与`input`第零维一致；`lse_square_scale_for_zloss`不为`0.0`时将返回该输出，否则将返回空tensor；当前暂不支持。

## 约束说明

- 输入shape中N取值范围(0, 200000]。
- 当input.requires_grad=True时，`sum`/`none`模式下不支持修改`label_smoothing`的默认值；`mean`模式下只支持传入可选参数的默认值，包括`weight`，`ignore_index`和`label_smoothing`。
- 输入`lse_square_scale_for_zloss`与`return_zloss`暂不支持。
- 输出`zloss`与`lse_for_zloss`暂不支持。
- 输出中仅`loss`支持梯度计算。
- 该接口仅支持训练场景下使用。
- 该接口支持Eager模式和图模式。

## 调用示例

- 当reduction设置为`mean`时，示例如下：

    ```python
    import torch
    import torch_npu

    N = 4096 # 批处理大小
    C = 8080 # 标签数

    # 构造输入和标签
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # 调用NPU交叉熵损失函数，当input.requires_grad=True时，默认的mean模式下只支持传入可选参数的默认值
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target)

    loss.backward()
    ```
    
- 当reduction设置为`sum`时，示例如下：

    ```python
    import torch
    import torch_npu

    N = 4096 # 批处理大小
    C = 8080 # 标签数

    # 构造输入和标签
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # 调用NPU交叉熵损失函数，当input.requires_grad=True时，sum模式下不支持修改label_smoothing的默认值
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target, reduction="sum", ignore_index=100)

    loss.backward()
    ```
