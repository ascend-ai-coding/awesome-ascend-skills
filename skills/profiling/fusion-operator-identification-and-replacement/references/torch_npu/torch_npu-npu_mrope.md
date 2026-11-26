# torch_npu.npu_mrope

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

## 功能说明

- API功能：推理场景下将sin、cos以预计算cache形式传入，对`query`、`key`做旋转位置编码（RoPE）。支持普通**RoPE**（一维`positions`）与多模态**MRoPE**（二维`positions`，按`mrope_section`整合多段位置编码），`cache_mode='default'`为分段式拼接，`cache_mode='interleave'`为交错式拼接。

- 计算公式：

  **MRoPE模式**：
  
  positions的shape输入是[m, num_tokens]，m为mropeSection的元素数，支持3或4：
    
  $$
  cosSin[i] = cosSinCache[positions[i]]
  $$

  $$
  cos, sin = cosSin.chunk(2, dim=-1)
  $$

    - `cache_mode`为`default`：
        - mropeSection的元素数为3：

        $$
        cos0 = cos[0, :, :mropeSection[0]]
        $$

        $$
        cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
        $$

        $$
        cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
        $$

        $$
        cos = torch.cat((cos0, cos1, cos2), dim=-1)
        $$

        $$
        sin0 = sin[0, :, :mropeSection[0]]
        $$

        $$
        sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
        $$

        $$
        sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
        $$

        $$
        sin= torch.cat((sin0, sin1, sin2), dim=-1)
        $$

        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

        - mropeSection的元素数为4：

        $$
        cos = torch.cat([m[i]\ for\ i, m\ in\ enumerate(cos.split(mropeSection, dim=-1))], dim=-1)
        $$

        $$
        sin = torch.cat([m[i]\ for\ i, m\ in\ enumerate(sin.split(mropeSection, dim=-1))], dim=-1)
        $$
        
        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

    - `cache_mode`为`interleave`：

        $$
        cosTmp = cos
        $$

        $$
        cos [..., 1:mropeSection[1] * 3:3] = cosTmp[1, ..., 1:mropeSection[1] * 3:3]
        $$

        $$
        cos[..., 2:mropeSection[1] * 3:3] = cosTmp[2, ..., 2:mropeSection[1] * 3:3]
        $$

        $$
        sinTmp = sin
        $$

        $$
        sin[..., 1:mropeSection[1] * 3:3] = sinTmp [1, ..., 1:mropeSection[1] * 3:3]
        $$

        $$
        sin[..., 2:mropeSection[1] * 3:3] = sinTmp [2, ..., 2:mropeSection[1] * 3:3]
        $$

        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

    - `rotary_mode`为`half`（GPT-NeoX style）计算模式：

      $$
      x1, x2 = torch.chunk(queryRot, 2, dim=-1)
      $$

      $$
      o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
      $$

      $$
      o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
      $$

      $$
      queryRot = torch.cat((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

    - `rotary_mode`为`interleaved`（GPT-J style）计算模式：

      $$
      x1 = queryRot[..., ::2]
      $$

      $$
      x2 = queryRot[..., 1::2]
      $$

      $$
      o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
      $$

      $$
      o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
      $$

      $$
      queryRot = torch.stack((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

  **RoPE模式**：
  
  positions的shape输入是[num_tokens]：

  $$
  cosSin[i] = cosSinCache[positions[i]]
  $$

  $$
  cos, sin = cosSin.chunk(2, dim=-1)
  $$

  $$
  queryRot = query[..., :rotaryDim]
  $$

  $$
  queryPass = query[..., rotaryDim:]
  $$

  - `rotary_mode`为`half`（GPT-NeoX style）计算模式：

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

  - `rotary_mode`为`interleaved`（GPT-J style）计算模式：

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    queryOut = torch.cat((queryRot, queryPass), dim=-1)
    $$

  $key$的计算过程与$query$相同。

## 函数原型

```python
torch_npu.npu_mrope(positions, query, key, cos_sin_cache, head_size, *, mrope_section=None, rotary_mode='half', cache_mode='default') -> (Tensor, Tensor)
```

## 参数说明

- **positions**（`Tensor`）：必选参数，位置索引，用于从cache中选取位置编码，对应公式中的$positions$。  
  - RoPE模式：1维张量，shape`(num_tokens)`，数据类型为`torch.int32`、`torch.int64`，格式$ND$，支持非连续张量。
  - MRoPE模式：2维张量，shape`(3, num_tokens)`或`(4, num_tokens)`，要求与`mrope_section`分段数一致。  
  - 通过`positions`从`cos_sin_cache`中选取位置编码时，每个取值作为cache行索引，应均小于第0维长度`max_seq_len`。
- **query**（`Tensor`）：必选参数，待施加RoPE的Query，对应公式中的$query$。2维张量，shape`(num_tokens, num_q_heads * head_size)`，类型`torch.float16`、`torch.bfloat16`、`torch.float32`，格式$ND$，支持非连续张量。
- **key**（`Tensor`）：必选参数，待施加RoPE的Key。2维张量，shape`(num_tokens, num_k_heads * head_size)`，类型与`query`一致，格式$ND$，支持非连续张量。
- **cos_sin_cache**（`Tensor`）：必选参数，预计算的位置编码cache，对应公式中的$cosSinCache$。2维张量，shape`(max_seq_len, rotary_dim)`。最后一维经`chunk(2, dim=-1)`拆成cos、sin两半。`query`、`key`、`cos_sin_cache`的浮点类型须一致。
- **head_size**（`int`）：必选参数，单头维度大小，即每个注意力头的特征维长度。
- **mrope_section**（`int[]`）：可选参数，MRoPE各段在half旋转维度上的长度配置，对应公式中的$mropeSection$。不传或`[0, 0, 0]`表示RoPE模式；传`[16, 24, 24]`、`[24, 20, 20]`、`[8, 12, 12]`、`[16, 16, 16, 16]`表示MRoPE模式，MRoPE时需与`positions`的行数一致。
- **rotary_mode**（`str`）：可选参数，`half`表示NeoX风格，`interleaved`表示GPT-J风格。默认值为`half`。
- **cache_mode**（`str`）：可选参数，`default`对应分段式cos/sin拼接；`interleave`对应交错式拼接。默认值为`default`。RoPE下仅支持`default`；MRoPE下支持`default`与`interleave`。

## 返回值说明

- **query_out**（`Tensor`）：对`query`施加RoPE/MRoPE后的Query，对应公式中的$queryOut$，2维张量，shape`(num_tokens, num_q_heads * head_size)`，类型`torch.float16`、`torch.bfloat16`、`torch.float32`，格式$ND$，支持非连续张量，与输入`query`的shape与数据类型一致。
- **key_out**（`Tensor`）：对`key`施加RoPE/MRoPE后的Key，2维张量，shape`(num_tokens, num_k_heads * head_size)`，类型与输入`key`一致，格式$ND$，支持非连续张量，与输入`key`的shape与数据类型一致。

## 约束说明

- **维度**：`rotary_dim`为`cos_sin_cache`最后一维大小，且满足`rotary_dim <= head_size`。
- **对齐与倍数**：
  - `head_size`：`torch.float16`/`torch.bfloat16`时为**32**的倍数；`torch.float32`时为**16**的倍数。
  - `rotary_dim`：`torch.float16`/`torch.bfloat16`时为**32**的倍数；`torch.float32`时为**16**的倍数。
- **MRoPE**：`mrope_section`各元素之和应等于**`rotary_dim / 2`**（与half维度的cos/sin分段一致）。
- **cache_mode**：当`mrope_section`为**`[16, 16, 16, 16]`时，仅支持`cache_mode='default'`（即不支持交错拼接）。

## 调用示例

- MRoPE示例（三段`mrope_section`）：

  `positions`为二维`(3, num_tokens)`，与`[16, 24, 24]`等三段配置对应；`mrope_section`三数之和须等于`rotary_dim / 2`。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_3 = [16, 24, 24]
query_out, key_out = torch_npu.npu_mrope(
  positions_mrope,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=mrope_section_3,
  rotary_mode="half",
  cache_mode="default",
)
```

- MRoPE示例（四段`mrope_section`）：

  `positions`为`(4, num_tokens)`，与`[16, 16, 16, 16]`等四段配置对应；该配置下**`cache_mode`仅支持`'default'`**。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope4 = torch.arange(num_tokens, dtype=torch.int64).repeat(4, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_4 = [16, 16, 16, 16]
query_out, key_out = torch_npu.npu_mrope(
  positions_mrope4,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=mrope_section_4,
  rotary_mode="half",
  cache_mode="default",
)
```

- RoPE示例：

  使用一维`positions`，shape为`(num_tokens,)`；`mrope_section`置为`[0, 0, 0]`表示不开启MRoPE（与当前`torch_npu`默认行为一致）。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_1d = torch.arange(num_tokens, dtype=torch.int64).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

query_out, key_out = torch_npu.npu_mrope(
  positions_1d,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=[0, 0, 0],
  rotary_mode="half",
  cache_mode="default",
)
```
