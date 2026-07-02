# Attention TND Pattern
一句话：TND 模式在 AscendC 算子里定义“不同 batch 的 Q/K/V 以变长方式拼接到同一条 `T` 维上，batch 边界靠前缀长度恢复”；它只改 Q/K/V 的逻辑布局和 batch 边界寻址，不改 attention 主数学。

## 先读这个

命中信号：`q/k/v` 的 shape 是 `(T, H, D)`，并额外带 `actual_q_len/actual_kv_len` 这类前缀和长度张量，而不是常规 `[B, H, S, D]`。
本模式只负责：从拼接后的 `T` 维布局恢复“当前 q block 属于哪个 batch、对应的 kv 起止范围在哪里”。不要把 TND 简化成“只是换了个 tensor 维度顺序”，也不要把它误写成“每个 batch 等长”的特殊实现。

## 1. 识别条件

- `Q/K/V` 是 `(T, H, D)` 而不是 `BHSD/BSHD`。
- `actual_q_len/actual_kv_len` 表示每个 batch 的累计长度或等价边界信息。
- 不同 batch 的 `q_len/kv_len` 可以不同；这不是特例，而是 TND 语义本体。
- 每个 batch 的 query/token 数据在 `T` 维上连续拼接，batch 之间不交错。
- 若没有前缀长度信息，就无法从 TND 恢复 batch 边界，不属于这张卡。

## 2. 核心规则

TND 的标准语义是“变长 batch 拼接”。AscendC 实现时有主体路径和一种常见特化。

### General TND：variable-length batch decode

适用：每个 batch 的 `q_len/kv_len` 可能不同，需要从 `actual_q_len/actual_kv_len` 或 host 预计算结果恢复边界。

```text
host or kernel:
  for each batch b:
    q_start[b], q_len[b]
    kv_start[b], kv_len[b]
  for each logical q_block:
    map q_block -> (batch_id, local_q_block)

kernel task:
  use batch_id/local_q_block to recover q_start/kv_start
  C0 LoadQ from Q[q_start + local_q_row]
  for t in range(kv_loops_this_batch + prelaunch):
    C1/MM1: load K[kv_start + t * BLOCK_N] -> QK -> ws_s
    V1:     online softmax -> ws_p + ws_meta
    C2/MM2: load V[kv_start + nowK * BLOCK_N] -> PV -> ws_o
    V2:     merge acc_o
```

关键点：

- 变化不在 `C1/V1/C2/V2`，而在“任务到 batch 边界”的恢复。
- `kv_loops`、`tail_valid`、以及与其它模式组合时的局部长度，都是 per-batch 量。
- 当前仓库的 AscendC 路径本质上也是先恢复或预计算每个 batch 的边界，再进入标准 flash-attention 主循环。

新增 pipeline 点：

- 相对 baseline，TND **不新增 stage**。
- 它只在进入主循环前新增一层 batch-boundary decode：
  - 当前 task 属于哪个 batch
  - 当前 batch 的 `q_start/q_len`
  - 当前 batch 的 `kv_start/kv_len`
- 后续 `C1/C2` 只是在 T 维上用 `start + local_offset` 读 Q/K/V。

## 3. 地址/形状公式

输入布局：

```text
Q/K/V: (T, H, D)
actual_q_len:  [len0, len0+len1, ...]
actual_kv_len: [kv0, kv0+kv1, ...]
```

从前缀和恢复单 batch 长度：

```text
q_len[b]  = actual_q_len[b]  - (b == 0 ? 0 : actual_q_len[b-1])
kv_len[b] = actual_kv_len[b] - (b == 0 ? 0 : actual_kv_len[b-1])

q_start[b]  = (b == 0 ? 0 : actual_q_len[b-1])
kv_start[b] = (b == 0 ? 0 : actual_kv_len[b-1])
```

TND 地址：

```text
q_offset(t_row, head, d)  = ((t_row * H) + head) * D + d
kv_offset(t_row, head, d) = ((t_row * H) + head) * D + d
```

当前 batch 内第 `local_t` 行映射为：

```text
global_q_row  = q_start[batch_id]  + local_q_row
global_kv_row = kv_start[batch_id] + local_kv_row
```

任务恢复：

```text
q_block_idx = logical block index over concatenated Q
batch_id, local_q_block = map_q_block(q_block_idx)
```

uniform specialization 才可以简化成：

```text
q_blocks_per_batch = ceil(q_len_per_batch / BLOCK_M)
batch_id           = q_block_idx // q_blocks_per_batch
local_q_block      = q_block_idx %  q_blocks_per_batch
```

## 4. 与其它模式的边界

```text
TND batch-boundary decode -> KV/source selection -> score visibility -> online softmax -> PV -> output
```

- Head Sharing：TND 只改 `(T, H, D)` 地址；不改 `q_head -> kv_head`。
- PA：PA 解决 page/block 到 physical row；TND 解决 flattened `T` 维到 batch 范围。
- Mask/Causal：若叠加 causal，判断仍应使用当前 batch 内的 logical row/col，而不是跨 batch 的全局 `T` 行号。
- Sparse/TopK：若再叠 sparse，sparse 只在当前 batch 的 token 范围内选行；TND 不负责 token 选择。
- Sink：sink 是额外 source；TND 只定义 local Q/K/V 的拼接布局。

## 5. 检查点

- TND 的核心不是 `(T, H, D)` 这个形状本身，而是如何恢复 batch 边界。
- `q_start/kv_start` 必须在当前 batch 范围内单独计算；不能把全局 `T` 维直接当单序列扫。
- `C1` 和 `C2` 都要使用同一组 `kv_start/kv_len`；不能一边按 batch 内偏移，一边按全局偏移。
- `tail_valid` 应该是当前 batch 的局部量，不是全局 `T_kv % BLOCK_N`。
- 若和 causal 组合，causal 的 row/col 比较应在 batch 内逻辑位置上做，不能跨 batch 互相可见。
