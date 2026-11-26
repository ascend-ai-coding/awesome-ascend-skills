# torch_npu.npu_recurrent_gated_delta_rule

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

- API功能：该接口实现了变步长Recurrent Gated Delta Rule（RGDR）的计算逻辑，是Transformer线性注意力机制的关键算子之一。通过引入门控机制与递归更新策略，RGDR能够在保持线性时间复杂度的同时，有效捕捉长距离依赖关系，显著降低模型对序列长度的敏感性。

- 计算公式：

  在每个时间步 $t$，网络根据当前的输入 $q_t$、$k_t$、$v_t$ 和上一个隐藏状态 $S_{t-1}$，计算当前的输出 $o_t$ 和新的隐藏状态 $S_t$。
  在这个过程中，门控单元会决定有多少新信息存入隐藏状态，以及有多少旧信息需要被遗忘。
  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt}) (I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt}) S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt}) S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := \frac{S_t q_t}{\sqrt{D_k}}
  $$

  其中，$S_{t-1},S_t \in R^{D_v \times D_k}$，$q_t, k_t \in R^{D_k}$，$v_t \in R^{D_v}$，$\alpha_t \in R$，$\alpha_{kt} \in R^{D_k}$，$\beta_t \in R$，$o_t \in R^{D_v}$。

## 函数原型

```python
torch_npu.npu_recurrent_gated_delta_rule(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None) -> Tensor
```

## 参数说明

- **query** (`Tensor`)：必选输入，对应公式中的$q$，数据类型支持`torch.bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。不支持空tensor。

- **key** (`Tensor`)：必选输入，对应公式中的$k$，数据类型支持`torch.bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。不支持空tensor。

- **value** (`Tensor`)：必选输入，对应公式中的$v$，数据类型支持`torch.bfloat16`，数据格式支持ND，shape为（$T$, $N_v$, $D_v$）。不支持空tensor。

- **state** (`Tensor`)：必选输入&输出，对应公式中的状态矩阵$S$，数据类型支持`torch.bfloat16`、`torch.float32`，数据格式支持ND，shape为（$BlockNum$, $N_v$, $D_v$, $D_k$）。不支持空tensor。

- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **beta** (`Tensor`)：可选输入。对应公式中的$β$，数据类型支持`torch.bfloat16`，数据格式支持ND，shape为（$T$, $N_v$）。不支持空tensor。

- **scale** (`float`)：可选输入。表示query的缩放因子，对应公式中的 $1/\sqrt{D_k}$。数据类型支持`torch.float32`。

- **actual_seq_lengths** (`Tensor`)：可选输入。表示各batch的输入序列长度。数据类型支持`torch.int32`，数据格式支持ND，shape为（$B$）。不支持空tensor。要求$1 \le L_i \le 8$，$L_i$ 表示第i个actual_seq_lengths的值。

- **ssm_state_indices** (`Tensor`)：可选输入。表示输入序列到状态矩阵的映射索引。`state[ssm_state_indices[i]]`表示第i个token的状态矩阵。`ssm_state_indices[i]`要求取值大于等于0且小于$BlockNum$。数据类型支持`torch.int32`，数据格式支持ND，shape为（$T$）。不支持空tensor。

- **num_accepted_tokens** (`Tensor`)：可选输入，`num_accepted_tokens[i]`表示投机推理第i个batch接受的token数量，要求取值大于等于1且不大于$L_i$。默认为None，表示每个batch接受的token数为1。数据类型支持`torch.int32`，数据格式支持ND，shape为（$B$,）。不支持空tensor。

- **g** (`Tensor`)：可选输入，衰减系数，对应公式中的$α=e^g$。默认为None，表示全0。数据类型支持`torch.float32`，数据格式支持ND，shape为（$T$, $N_v$）。不支持空tensor。

- **gk** (`Tensor`)：可选输入，衰减系数，对应公式中的$α=e^{gk}$。默认为None，表示全0。数据类型支持`torch.float32`，数据格式支持ND，shape为（$T$, $N_v$, $D_k$）。不支持空tensor。

## 返回值说明

**out** (`Tensor`)：表示计算结果，公式中的$o$，注意力计算结果。输出的数据类型为`torch.bfloat16`，数据格式为ND，shape为($T$, $N_v$, $D_v$)。

## 约束说明

- 参数里Shape使用的变量如下：
    - $T=\sum_i^B L_i$ 表示累积序列长度。
    - $B$ 表示batch size。
    - $L_i$ 表示第i个序列的长度，由actual_seq_lengths传入，其取值范围为$1 \le L_i \le 8$。
    - $N_k$ 表示key的头数，其取值范围为$1 \le N_k \le 256$。
    - $N_v$ 表示value的头数，其取值范围为$1 \le N_v \le 256$，$N_v \bmod N_k = 0$。
    - $D_k$ 表示key向量的维度，其取值范围为$1 \le D_k \le 512$。
    - $D_v$ 表示value向量的维度，其取值范围为$1 \le D_v \le 512$。
    - $BlockNum$为状态矩阵分块数量，其取值范围为$T \le BlockNum$。
- 该接口仅支持在推理场景下使用，当前TND场景，beta、scale、actual_seq_lengths、ssm_state_indices必传。
- 该接口仅支持单算子模式和静态图模式调用。
- state仅支持0轴、1轴非连续Tensor。
- 以下约束由于算子无法获取tensor中具体数值，故需用户保证，算子不校验：
    - $ssm\_state\_indices[i] \lt BlockNum$
    - $1 \le actual\_seq\_lengths[i] \le 8$，且actual_seq_lengths[i]累加和等于$T$
    - $1 \le num\_accepted\_tokens[i] \le actual\_seq\_lengths[i]$
    - $-1 \le query[i][j][k] \le 1$
    - $-1 \le key[i][j][k] \le 1$
    - $g[i][j] \lt 0$
    - $gk[i][j][k] \lt 0$
    - $0 \lt beta[i][j] \lt 1$

## 调用示例

- 单算子调用

    ```python
    import torch
    import torch_npu

    # 构造输入
    bs, mtp, nk, nv, dk, dv = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1, mtp + 1, (bs,)).npu()).to(torch.int32)

    # 调用算子
    o = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state, beta=beta, scale=scale,
        actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=None)
    print(o)
    ```

- 静态图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger

    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    # "ENABLE_ACLNN"是否走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
 
        def forward(self, query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g):
            return torch_npu.npu_recurrent_gated_delta_rule(
                query, key, value, state, beta=beta, scale=scale,
                actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens, g=g, gk=None)


    # 构造输入
    bs, mtp, nk, nv, dk, dv = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1, mtp + 1, (bs,)).npu()).to(torch.int32)

    #调用
    model = MyModel()
    model = model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    o = model(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g)
    print(o)
    ```
