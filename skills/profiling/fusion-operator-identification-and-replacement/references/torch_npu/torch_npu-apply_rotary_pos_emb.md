# torch_npu.npu_apply_rotary_pos_emb

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

为提升推理网络性能，将query和key两路算子融合为单路，在旋转位置编码计算中直接对结果执行原地更新。

## 函数原型

```python
torch_npu.npu_apply_rotary_pos_emb(query, key,  cos, sin, *, layout='BSND', rotary_mode='half') -> (Tensor, Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，待执行旋转位置编码的第一个张量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$。`layout`为TND时，shape为3维，其他`layout`场景下shape为4维。

  <!-- npu="A3,910b,310p" id5 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：不支持空Tensor，shape最后一维（D）必须等于128或者64。
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - Ascend 950DT系列产品：支持空Tensor，shape最后一维（D）小于等于1024。
  <!-- end id6 -->

- **key**（`Tensor`）：必选参数，待执行旋转位置编码的第二个张量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$。`layout`为TND时，shape为3维，其他`layout`场景下shape为4维。

  <!-- npu="A3,910b,310p" id7 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：不支持空Tensor，shape最后一维（D）必须等于128或者64。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - Ascend 950DT系列产品：支持空Tensor，shape最后一维（D）小于等于1024。
  <!-- end id8 -->

- **cos**（`Tensor`）：必选参数，旋转位置编码余弦值张量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$。`layout`为TND时，shape为3维，其他`layout`场景下shape为4维。

  <!-- npu="A3,910b,310p" id9 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：不支持空Tensor，shape中B维度与`query`、`key`的B维度一致，shape第3维（N）必须等于1，shape最后一维（D）必须等于128或者64。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - Ascend 950DT系列产品：支持空Tensor，shape中B维度与`query`、`key`的B维度一致，或者等于1，shape中N维度必须等于1，shape最后一维（D）小于等于1024。
  <!-- end id10 -->

- **sin**（`Tensor`）：必选参数，旋转位置编码正弦值张量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$。`layout`为TND时，shape为3维，其他`layout`场景下shape为4维。

  <!-- npu="A3,910b,310p" id11 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：不支持空Tensor，shape中B维度与`query`、`key`的B维度一致，shape第3维（N）必须等于1，shape最后一维（D）必须等于128或者64。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - Ascend 950DT系列产品：支持空Tensor，shape中B维度与`query`、`key`的B维度一致，或者等于1，shape最后一维（D）小于等于1024。
  <!-- end id12 -->

- **layout**（`str`）：可选参数，张量布局格式，支持"BSND"、"SBND"、"BNSD"、"TND"。默认值为"BSND"。

  <!-- npu="A3,910b,310p" id13 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：支持BSND的4维Tensor、TND的3维Tensor。
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - Ascend 950DT系列产品：支持BSND、SBND、BNSD的4维Tensor，TND的3维Tensor。
  <!-- end id14 -->

- **rotary_mode**（`str`）：可选参数，旋转编码模式，支持"half"、"quarter"、"interleave"，默认值为"half"。

  <!-- npu="A3,910b,310p" id15 -->
  - Atlas推理系列产品、Atlas A2系列产品、Atlas A3系列产品：支持"half"模式。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - Ascend 950DT系列产品：支持"half"、"interleave"、"quarter"模式。
  <!-- end id16 -->

## 返回值说明

- **query_out**（`Tensor`）：表示原地更新后的query张量。
- **key_out**（`Tensor`）：表示原地更新后的key张量。

## 约束说明

<!-- npu="A3,910b,310p" id17 -->
- <term>Atlas推理系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - `layout`为"BSND"，`query`、`key`、`cos`、`sin`输入shape的前2维（B、S）必须相等；`layout`为"TND"时，第1维（T）必须相等。
  - `query`、`key`输入shape的最后一维（D）必须相等，`cos`、`sin`输入shape的最后一维（D）必须相等。
  - 输入张量`query`、`key`、`cos`、`sin`的数据类型必须相同。
  - `layout`为"BSND"时，输入`query`的shape用（q_b, q_s, q_n, q_d）表示，`key_out`的shape用（q_b, q_s, k_n, q_d）表示，`cos`和`sin`的shape用（q_b, q_s, 1, cos_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。`layout`为"TND"时，输入query的shape用（q_t, q_n, q_d）表示，`key`的shape用（q_t, k_n, q_d）表示，`cos`和`sin`的shape用（q_t, 1, cos_d）表示。其中，t表示b和s合轴，n表示head_num，d表示head_dim。
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：
  - 对于任意`layout`，`query`与`key`除N维度外其他维度必须相同；`query`、`key`输入shape的最后一维（D）必须相等，`cos`、`sin`输入shape的最后一维（D）必须相等，且小于等于`query`、`key`输入shape的最后一维（D）。
  - 输入张量`query`、`key`、`cos`、`sin`的数据类型必须相同。
  - `rotary_mode`为"half"和"interleave"时，输入shape最后一维必须被2整除；`rotary_mode`为"quarter"时，输入shape最后一维必须被4整除。
<!-- end id18 -->
<!-- npu="310p" id19 -->
- Atlas推理系列产品不支持`torch.bfloat16`。
<!-- end id19 -->

## 调用示例

```python
import torch
import torch_npu

def test_npu_apply_rotary_pos_emb():

    # 固定参数
    batch = 1
    seq_len = 64
    num_heads = 8
    head_dim = 64

    # 创建输入数据
    query = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    cos = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()
    sin = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()

    # 调用 npu_apply_rotary_pos_emb API
    q_out, k_out = torch_npu.npu_apply_rotary_pos_emb(
        query, key, cos, sin,
        layout="BSND",
        rotary_mode="half"
    )

    print("API: npu_apply_rotary_pos_emb test passed!")
    print(f"Output query: {q_out}")
    print(f"Output key: {k_out}")

if __name__ == "__main__":
    test_npu_apply_rotary_pos_emb()
```
