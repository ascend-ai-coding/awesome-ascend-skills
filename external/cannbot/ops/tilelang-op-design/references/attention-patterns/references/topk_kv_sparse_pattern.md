# Attention Topk KV Sparse Pattern

一句话：topk sparse 在 AscendC 算子里定义“每个 query/head-group 只访问一小批 selected KV token”；实现上要把原本连续 KV tile 改成 token-level sparse gather tile，再接回 flash attention 主循环。

## 先读这个

命中信号：输入含 `indices/topk`，K/V 访问不是连续 `token range`，而是每轮取一批离散 token 行。

本模式只负责：`selected logical token -> selected KV row`。实现可以是直接从连续 KV 按 token 行 gather，也可以先经过 page 映射再 gather；不要把 topk sparse 简化成“只是多一个 mask”。

## 1. 识别条件

- 输入含 `indices/topk`。
- `indices[..., i]` 表示第 `i` 个被选中的 KV token，不是 block id，不是 page id。
- 每个 query 只消费 `topk` 个 selected token，而不是完整 `seqlen_kv`。
- 若有 GQA/MQA，topk sparse 不改变 `q_head -> kv_head` 映射；它只改变这个 `kv_head/group` 访问哪些 token。
- 只有连续 dense KV tile、只有 block sparse、只有 page 映射但没有离散 token 选择时，不套 topk sparse。

## 2. 核心规则

topk sparse 有两层实现语义，但 kernel 责任只有一层：把离散 token 选择变成可计算的 dense tile。

### Continuous KV sparse gather

适用：K/V 本身是连续 token layout，如 `[B, Skv, Hkv, D]`；只是每轮不是取连续 `[t:t+BLOCK_N)`，而是取 `indices` 指定的 token 行。

```text
for sparse tile i:
  VG:
    load indices tile
    for each selected token:
      clamp / validate token index
      gather K/V row from KV[b, token, kv_head, :]
      write workspace_k / workspace_v
  C1/MM1: Q @ gathered K -> ws_s
  V1:     scale / online softmax -> ws_p + ws_meta
  C2/MM2: P @ gathered V -> ws_o
  V2:     merge acc_o, normalize, write output
```

### Sparse-on-PA gather

适用：selected token 还要先经过 page/block_table 映射；topk sparse 仍只负责“选哪些 logical token”，PA 负责“这些 logical token 落到哪个 physical row”。

```text
for sparse tile i:
  VG:
    load indices tile
    for each selected logical token:
      validate token / causal / actual length
      physical row = page_mapping(batch, logical token)
      gather K/V row into workspace_k / workspace_v
  C1/MM1: Q @ gathered K -> ws_s
  V1:     scale / online softmax -> ws_p + ws_meta
  C2/MM2: P @ gathered V -> ws_o
  V2:     merge acc_o, normalize, write output
```

新增 pipeline 点只有一个：相对普通 FA，多了 `VG`，并且 `VG` 产出的 K/V tile 生命周期必须覆盖 `C1` 和 `C2` 两段。

## 3. 地址/形状公式

连续 KV:

```text
Q:       [B, Sq, Hq, D]
KV:      [B, Skv, Hkv, D]
indices: [B, Sq, Hsel, topk] 或 [B, Sq, topk]

kv_head = q_head / n_groups
n_groups = Hq / Hkv

selected_token = indices[b, q, kv_group, topk_col]   或 indices[b, q, topk_col]
valid = selected_token >= 0
valid = valid && selected_token < Skv

cache_offset(token, kv_head, d) =
  (((b * Skv) + selected_token) * Hkv + kv_head) * D + d

workspace_k[row, :] = KV[b, selected_token, kv_head, :]
workspace_v[row, :] = KV[b, selected_token, kv_head, :]
```

若结合 page attention:

```text
selected_token = indices[...]
logical_block  = selected_token // block_size
block_offset   = selected_token % block_size
physical_block = block_table[b, logical_block]

cache_offset(block, row, kv_head, d) =
  ((physical_block * block_size + block_offset) * Hkv + kv_head) * D + d
```

invalid selected row 必须 zero-fill 保护 GEMM；若该无效项理论上不可见，score 侧还必须 mask 成 `-inf`。

## 4. 与其它模式的边界

```text
head mapping -> topk token selection -> optional page mapping -> gather K/V tile -> softmax/PV -> output
```

- GQA/MQA：决定 `kv_head`；topk sparse 用它决定 gather 哪个 head 的行。
- PA：决定 `logical token -> physical row`；topk sparse 不负责 page/block 寻址。
- Block Sparse：决定哪些连续 block 可见；topk sparse 决定哪些离散 token 被取出。
- Mask/Causal：决定 token 是否有效；topk sparse 只负责把有效 token gather 进 tile。
- Sink/Local/Dense Hybrid：决定哪些 token 来源属于稠密段或特殊段；topk sparse 只处理“这一段 sparse token 列表”。

## 5. 检查点

- `indices` 第一个语义必须是 token，不是 block/page。
- 一个 sparse tile 的 gather 粒度是 token row，不是连续 range。
- 若一个 `kv_group` 被多个 query heads 共享，gather 只做一次。
- `VG` 必须显式进入 pipeline，不能只写“先 gather 再算”。
- `VG -> C1` 有依赖；`VG` 产出的 `V tile` 生命周期至少持续到 `C2` 结束。
- 若做 ring slot，`VG(t+1)` 不能覆盖 `t` 仍在用的 `workspace_k/workspace_v`。
- `topk` 非整除 tile size 时，要有 tail gather / tail mask / tail MM 的一致裁剪。
- 若 token 有 sentinel/negative index，只 zero-fill 不够；需要和 score 可见性规则一致。
