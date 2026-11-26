# torch\_npu.npu\_top\_k\_top\_p\_sample

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：
  根据输入词频`logits`、`top_k`/`top_p`/`min_ps`采样参数、随机采样权重分布`q`，进行topK-topP-minP-Sample采样计算，输出每个batch的最大词频`logits_select_idx`，以及topK-topP采样后的词频分布`logits_top_kp_select`。

  算子包含4个可单独开启，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：topK采样、topP采样、minP显著性采样、不采样 / 指数采样 / 多项式随机采样 。目前支持以下12种计算场景。如下表所示：

  | 计算场景 | topK采样 | topP采样 | minP采样 | 后继处理 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|
  |Argmax采样|×|×|×|None|对输入`logits`每个batch取最大logits和对应索引，结果作为logits_select_idx[batch,1]|
  |topK采样|√|×|×|None|无|
  |topP采样|×|√|×|None|无|
  |qSample采样|×|×|×|qSample|对输入`logits`每个batch使用`q[i]`进行指数采样，从结果中取最大值和索引，作为logits_select_idx[batch,1]|
  |topK-topP采样|√|√|×|None|无|
  |topK-qSample采样|√|×|×|qSample|无|
  |topK-multiNomial采样|√|×|×|multiNomial|无|
  |topK-minP-multiNomial采样|√|×|√|multiNomial|无|
  |topP-qSample采样|×|√|×|qSample|无|
  |topK-topP-qSample采样|√|√|×|qSample|vLLM框架标准完整功能|
  |topK-topP-multiNomial采样|√|√|×|multiNomial|min_ps为无效值，但仍执行多项式采样|
  |topK-topP-minP-multiNomial采样|√|√|√|multiNomial|Sglang框架标准完整功能|

- 计算公式：
  输入`logits`为大小是[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
  `logits`中的每一行$logits[batch][:]$根据相应的top_k[batch]、top_p[batch]、q[batch, :]、min_ps[batch]，执行不同的计算场景。<br>
  下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  topK采样
  1. 按分段长度v采用分段topK归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. top_k[batch]对应当前batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，如果top_k[batch]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会跳过当前batch的排序，将输入logits[batch]直接传入下一模块。<br>

  * 具体计算流程如下所示：

  * 根据输入`top_k[b]`与`ks_max`的关系，判断是否进行topK采样：

  | 参数类型 | ≤ | 有效域 | 无效域 |
  | :-------:| :------:|:-------:|:-------:|
  |`top_k[b]`|跳过topK采样|1≤topK≤min(voc_size,ks_max),执行topK采样|top_k>min(voc_size,ks_max),跳过topK采样|

  * 对当前batch分割为若干子段，滚动计算top_k_value[b]：

    $$
    top\_k\_value[b] = {Max(top\_k[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ top\_k\_value[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge top\_k\_min[b][s-1] \right \} \right \}\\
    Card(top\_k\_value[b])=top\_k[b]
    $$
    其中：
    $$
    top\_k\_min[b][s] = Min(top\_k\_value[b]\left \{  s \right \})
    $$
    v表示预设的滚动topK时的固定分段长度：

    $$
    v=8*ks\_max
    $$
    `ks_max`有效取值范围[1,1024]，默认为1024，并且需要向上对齐到8的整数倍。

  * 生成需要过滤的mask：

    $$
    top\_k\_mask = sorted\_value>top\_k\_value
    $$

  * 将小于阈值的部分通过mask置为默认无效值defLogit:

    $$
    sorted\_value[b][v]=
    \begin{cases}
    -inf & \text{top\_k\_mask[b][v]=true} \\
    sorted\_value[b][v] & \text{top\_k\_mask[b][v]=false} &
    \end{cases}
    $$

  * 其中defLogits取决于入参属性Attr.optional.Bool.input_is_logits，该属性控制输入logits和输出`logits_top_kp_select`的归一化：
    $$
    \text{defLogit} =
      \begin{cases}
      -inf, & \text{inputIsLogits} = \text{True} \\
      0, & \text{inputIsLogits} = \text{False}
      \end{cases}
    $$

  topP采样

  * 根据入参约束属性Attr.optional.Bool.input_is_logits(false)，如果该属性为True，则对排序后结果进行归一化：

    $$
    \text{logit\_sortProb} =
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * 根据输入`top_p[b]`的数值，本模块的处理策略如下：

  | 参数类型 | ≤ | 有效域 | 无效域 |
  | :-------:| :------:|:-------:|:-------:|
  |`top_p[b]`|保留1个最大词频token|0<top_p<1,执行topP采样|top_p≥1,跳过topP采样|

  * 如果执行常规topP采样，且如果前序topK环节已有排序输出结果，则根据topK采样输出计算累积词频，并根据top_p截断采样：

    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$

  * 如果执行常规topP采样，但前序topK环节被跳过，则计算top-p的mask:

    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$

  * 将需要过滤的位置设置为默认无效值defLogit，得到logits_sort，记为$sortedValue[b][v]$:

    $$
    sortedValue[b][v] =
    \begin{cases}
    defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
    logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
    \end{cases}
    $$

  * 取过滤后$sortedValue[b][v]$每行中前topK个元素，查找这些元素在输入中的原始索引，整合为logits_idx:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
    $$

  * 从输入logits中按logitsIdx顺序遍历取出元素，其余位置填入defLogit，作为logitsSortMasked：

    $$
    \text{logitsSortMasked}[b, \text{:Len}(\text{logitsIdx}[b][:])] = \text{Logits}[b, \text{logitsIdx}[b][:]]
    $$
    $$
    \text{logitsSortMasked}[b, \text{Len}(\text{logitsIdx}[b][:])\text{:}] = \text{defLogit}
    $$

  * （sglang框架支持更新）直接使用截断后的sortedValue作为logitsSortMasked：

    $$
    logitsSortMasked[b,:] = sortedValue[b]
    $$

  min_p采样

  * 如果min_ps[b]∈(0, 1)，则执行min_p采样：

    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$
    $$
    \text{minPMask}[b] =
    \begin{cases}
    0, & \text{logitsSortMasked}[b] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b] \geq \text{minPThd}
    \end{cases}
    $$
    $$
    \text{logitsSortMasked}[b,:] =
    \begin{cases}
    \text{defLogit}, & \text{minPMask}[b] = 0 \\
    \text{logitsSortMasked}[b,:], & \text{minPMask}[b] = 1
    \end{cases}
    $$

  * 其他情况：

    $$
    \text{logitsSortMasked}[b, :] =
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    min_ps[b]≥1时，每个batch仅取1个最大token，其余位置填充defLogit。

  可选输出

  * 如果​入参属性Attr.Bool.is_need_logits=True，则使用topK-topP-minP联合采样后的logitsIndexMasked，进行`logits_top_kp_select`输出。

    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    其中，topK、topP、minP采样环节如果被跳过，则相应mask为全1。

  * 接下来使用logitsIndexMasked对输入logits进行Select，过滤输入logits中的高频token作为`logits_top_kp_select`输出：

    $$
    \text{logitsTopKpSelect}[b][v] =
    \begin{cases}
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  后继处理

  * 此阶段输入为前序对前序topK-topP-minP采样的联合结果logitsSortMasked。

  * 此处输入需要确保logitsSortMasked∈(0,1)，根据输入logits的实际情况，配置入参约束属性Attr.optional.Bool.input_is_logits，即：

    $$
    \text{inputIsLogits} =
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    使得
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    接下来有三种模式：None，qSample，multiNomial，通过入参约束属性attr.optional.Str.post_sample加以控制。

  * None

  * 直接对每个batch通过Argmax取最大元素和索引，并通过gatherOut输出。
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$

  * qSample

  * 先对probs进行指数分布采样：

    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$

  * 再进行Argmax-GatherOut输出结果：

    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$

  * multiNomial

  * 使用多项式随机采样，根据logitsSortMasked中的概率值，执行无放回的多项式采样，对每个batch取1个样本，将采样结果作为当前batch的输出：

    $$
    \text{sampleIdx}[b] = \text{multiNomial}(\text{logitsSortMasked}[b,:], \text{numSamples}=1, \text{seed}[b], \text{offset}[b])
    $$
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{sampleIdx}[b]]
    $$

  * 对于采样种子，当attr.optional.Str.post_sample="multiNomial"时，q约束为`torch.int64`，分别从第一列和第二列获取multiNomial采样的seed和offset：

    $$
    \text{seed}[b] =
    \begin{cases}
    q[b, 0], & b < qRows \\
    q[-1, 0], & b \ge qRows
    \end{cases}
    $$

    $$
    \text{offset}[b] =
    \begin{cases}
    q[b, 1], & b < qRows \\
    q[-1, 1], & b \ge qRows
    \end{cases}
    $$

  * 该采样过程以aclnn.Multinomial为基准，可参见<https://gitcode.com/cann/ops-math/blob/master/random/stateless_sample_multinomial/docs/aclnnMultinomial.md>。

  * TorchNPU调用时，采样种子和偏移默认使用内建值，可参见<https://gitcode.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MultinomialKernelNpuOpApi.cpp>。

## 函数原型

```python
torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=None, eps=1e-8, is_need_logits=False, top_k_guess=32, min_ps=None, ks_max=1024, input_is_logits=True, post_sample='qSample', generator=None) -> (Tensor, Tensor)
```

## 参数说明

- **logits**（`Tensor`）：必选参数，表示待采样的输入词频，目前支持2维，词频索引固定为最后一维。数据类型支持`torch.float16`、`torch.bfloat16`和`torch.float32`，数据格式支持$ND$，支持非连续Tensor。
- **top_k**（`Tensor`）：必选参数，表示每个batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，无效范围则跳过topK，目前支持1维。数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续Tensor。
- **top_p**（`Tensor`）：必选参数，表示每个batch采样的p值，有效范围为0<top\_p[batch]<1，目前支持1维。数据类型和数据格式与`logits`保持一致，支持非连续Tensor。
    - 在任何情况下，topP对每个batch的输出都会保留至少1个token。
    - top_p[batch] ≤0时，对当前batch仅保留概率最大的1个token。
    - top_p[batch]处于合法值范围(0,1)时，对当前batch执行标准topP采样。
    - p>=1时跳过相应batch的topP步骤，提取整个batch信息并生成ones掩模作为输出。
- **q**（`Tensor`）：可选参数，topK-topP采样输出的随机采样权重分布矩阵，数据类型支持`torch.float32`，数据格式支持$ND$，支持非连续Tensor，默认值为None, 此时跳过后继采样，从probs计算logits_select_idx。
    - 根据post_sample的模式不同，该参数约束如下：
    - post_sample = qSample时, 尺寸约束为[batch, voc_size], 数据类型必须为`torch.float32`，指数分布采样矩阵，维度需与logits的一致。
    - post_sample = multiNomial时, multiNomial随机采样参数矩阵，数据类型必须为`torch.int64`，用于为aclnnMultinomial采样提供控制参数。合法的尺寸为[q_row, 2]，其中q_row≥1：
        - 第1列对应aclnnMultinomial.seed参数：对应当前batch的随机数种子。
        - 第2列对应aclnnMultinomial.offset参数：随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
        - 如果q_row \< batch，则默认使用最后一个batch的采样参数作为后续batch的multiNomial采样参数。
- **eps**（`float`）：可选参数，在softmax和权重采样中防止除零，默认值为1e-8。
- **is_need_logits**（`bool`）：可选参数，控制`logits_top_kp_select`的输出条件，默认值为False。
- **top_k_guess**（`int`）：可选参数，仅在当前batch的top_k为无效值时开启，适用于跳过topK的top_k_guess-TopP加速采样。有效值范围top_k_guess>0，默认为32，用于TopP加速采样中基于top_k_guess的直接索引过滤。如果传入非正数，视为跳过top_k_guess环节，直接使用基于cumsum的标准topP实现，对当前batch做topP全排序采样，保持基准性能。
- **ks_max**（`int`）：可选参数，约束topK采样中允许的topk[batch]合法值上限，影响跳过topK采样的条件，允许传入任意非零正整数。有效值范围[1,1024]之间的整数，传入超过1024的值会自动设为1024。
- **input_is_logits**（`bool`）：可选参数，该参数控制输入logits在topP及后续步骤之前，是否进行归一化处理，并决定可选输出logits_top_kp_select中的无效logits默认值类型。logits表示“未经归一化的原始值”，而相对地已经过归一化的则定义为“probs”。该参数的取值影响如下：
    - 若该参数取值为True，输入的logits中的数值不能确保在[0,1]区间内。由于logits未进行归一化，在进行top_p采样等后续步骤之前，先对输入进行softmax处理。logits_top_kp_select中的无效logits默认值defLogit=-inf。
    - 若该参数取值为False，输入logits中的所有元素都确保在[0,1]区间内。输入logits已经归一化，为避免梯度平滑化，top_p采样等后续步骤直接使用前级处理的结果。logits_top_kp_select中的无效logits默认值defLogit=0。
- **post_sample**（`str`）：可选参数，该参数控制topk-topp采样之后的后继处理策略。第一优先级：判断q是否为None，如果q=None，则无视参数提供的post_sample内容，强制后继处理模式一概设为None。参数合法值允许：
    - qSample(默认值)：倾向于使用qSample采样。
    - multiNomial：使用multiNomial采样（多项式随机抽样），此时入参中的q矩阵将被解析为随机种子，执行multiNomial-gather。
    - None：显式强调不使用任何后继处理，此时传入任何q!=None都被无视。
- **generator**（`Generator`）：可选参数，multiNomial使用的随机数生成器，必须指定seed才能传入。

## 返回值说明

- **logits_select_idx**（`Tensor`）：表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probs_opt[batch, :])在输入`logits`中的位置索引。数据类型支持`torch.int64`，数据格式支持$ND$。
- **logits_top_kp_select**（`Tensor`）：表示经过topK-topP-minP采样获得mask，对原输入`logits`中高频token的过滤结果。仅在`is_need_logits=true`时输出计算和搬运，否则直接输出相应尺寸的空tensor。数据类型支持`torch.float32`，数据格式支持$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口目前不支持图模式。
- `logits`、`q`、`logits_top_kp_select`的尺寸和维度必须完全一致。
- `logits`、`top_k`、`top_p`、`logits_select_idx`除最后一维以外的所有维度必须顺序和大小完全一致。目前`logits`只能是2维，`top_k`、`top_p`、`logits_select_idx`必须是1维非空Tensor。`logits`、`top_k`、`top_p`不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
- 如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
- 如果1024<$top\_k[batch]<voc\_size[batch]$，则视为选择当前batch的全部有效元素并跳过topK环节。
- 如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1或≤0。
- 如果需要单独跳过Sample模块，使用其默认值或设置`q`为None；如需使用Sample模块，则必须传入对应尺寸的Tensor。

## 调用示例

```python
import numpy as np
import torch
import torch_npu
logits = torch.from_numpy(np.random.uniform(-2, 2, size=[2, 4])).type(torch.float16).npu()
top_ks = torch.from_numpy(np.random.uniform(1, 2, size=[2, ])).type(torch.int32).npu()
top_ps = torch.from_numpy(np.random.uniform(0.4, 0.5, size=[2, ])).type(torch.float16).npu()
q = None
min_ps = torch.from_numpy(np.random.uniform(0.1, 0.5, size=[2, ])).type(torch.float16).npu()
post_sample = 'multiNomial'
if post_sample == "multiNomial":
    generator_npu = torch.Generator(device="npu")
    generator_npu.manual_seed(1)
else:
    generator_npu = None
npu_out_index, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits, top_ks, top_ps, q=q, eps=1e-8, is_need_logits=True, top_k_guess=32, min_ps=min_ps, ks_max=1024, input_is_logits=True, post_sample=post_sample, generator=generator_npu)

print(npu_out_index)
print(logits_top_kp_select)
```
