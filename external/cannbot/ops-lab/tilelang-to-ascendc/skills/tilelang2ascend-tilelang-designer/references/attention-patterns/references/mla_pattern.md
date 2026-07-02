# Attention MLA Pattern
一句话：MLA 模式在 AscendC 算子里定义“Q/K 的打分维度和 V 的输出维度分离，KV 以 latent/compressed 形式存储”；实现上要把 `QK` 路和 `PV` 路按不同维度、不同数据段组织回同一条 flash attention 主循环。

## 先读这个

命中信号：输入不再是单一 `q/k/v`，而是 `q_nope/q_rope`、`k_nope/k_rope`、`v`，或单头压缩 `kv_cache` + `headdim_qk/headdim_v`。
本模式只负责：把 `QK` 的有效维度定义成 `Dqk`，把 `PV` 的有效输出维度定义成 `Dv`，并说明 latent KV 如何供这两条路径分别消费。不要把 MLA 简化成“只是 head sharing”或“只是 paged attention”。

## 1. 识别条件

- Q/K 的打分维度与输出 V 维度不同：`Dqk != Dv`。
- K 常被拆成两段：
  - `k_nope` / `q_nope`
  - `k_rope` / `q_rope`
- 或者 KV cache 已压成单头 latent 表示：
  - `kv_cache[..., 1, Dqk]`
  - K 打分用完整 `Dqk`
  - V 输出只用前 `Dv` 列
- 若有多 query heads，KV 常是单头或少头，需要广播/共享给所有 query heads。

## 2. 核心规则

MLA 有两条 AscendC 实现路径。

### Dense MLA：split-QK + narrow-V

适用：`q_nope/q_rope/k_nope/k_rope/v` 都是连续布局，不需要 page/block_table。

```text
C0 LoadQ_nope + Q_rope
for kv tile t with prelaunch:
  C1/MM1:
    load K_nope[t]
    Q_nope @ K_nope^T -> partial_s
    load K_rope[t]
    Q_rope @ K_rope^T -> accumulate into same S
    write ws_s
  V1:
    online softmax -> ws_p + ws_meta
  C2/MM2:
    load V[t] with width Dv
    P @ V -> ws_o
  V2:
    merge acc_o
```

关键点：

- MLA 不是新增一条独立 pipeline，而是把 `C1` 内部拆成两个 score contribution：
  - `Q_nope @ K_nope^T`
  - `Q_rope @ K_rope^T`
- 这两个 contribution 在 softmax 前合成同一个 `S`。
- `C2` 不再消费 `Dqk` 全宽，而只消费 `Dv`。

### Paged MLA：latent cache + page gather

适用：KV 已压成 paged latent cache，常见 shape 为 `[num_blocks, block_size, 1, Dqk]`。

```text
LoadQ
for kv tile t with prelaunch:
  GatherK(latent rows from page cache, full Dqk)
  C1/MM1: Q @ K^T -> ws_s
  V1:     online softmax (+ optional causal) -> ws_p + ws_meta
  GatherV(latent rows from page cache, first Dv cols)
  C2/MM2: P @ V -> ws_o
  V2:     merge acc_o
```

关键点：

- paged MLA 的 local source 是单头 latent cache，不是普通 `[Hkv, D]` 多头 cache。
- `GatherK` 取整行 `Dqk`。
- `GatherV` 可以复用同一 latent row，但 `C2` 只消费前 `Dv` 列。
- 若是 decode 形态，还可叠 causal；但 causal 只是 score 可见性，MLA 主体仍是 `Dqk/Dv` 分离。

新增 pipeline 点：

- 相对 baseline，MLA 的新增点主要在 `C1` 和 `C2`：
  - `C1` 的 K-dim 可能是 `nope + rope` 两段累加
  - `C2` 的 V-dim 只取 `Dv`
- 若是 paged MLA，还会额外引入 `GatherK/GatherV`，但这是和 PA 组合后的实现路径，不是 MLA 单独的新 softmax 语义。

## 3. 地址/形状公式

dense MLA:

```text
Q_nope: [B, Hq, Sq, Dnope]
Q_rope: [B, Hq, Sq, Drope]
K_nope: [B, Hkv, Skv, Dnope]
K_rope: [B, Hkv, Skv, Drope]
V:      [B, Hkv, Skv, Dv]

Dqk = Dnope + Drope
smScale = 1 / sqrt(Dqk)
```

head sharing:

```text
group_size = Hq / Hkv
kv_head    = q_head / group_size
```

连续地址：

```text
q_nope_offset(b, q_head, q_row, d) =
  (((b * Hq) + q_head) * Sq + q_row) * Dnope + d

k_nope_offset(b, kv_head, kv_row, d) =
  (((b * Hkv) + kv_head) * Skv + kv_row) * Dnope + d

v_offset(b, kv_head, kv_row, d) =
  (((b * Hkv) + kv_head) * Skv + kv_row) * Dv + d
```

paged MLA:

```text
kv_cache: [block_num, block_size, 1, Dqk]

logical_block  = kv_row // block_size
block_offset   = kv_row % block_size
physical_block = block_table[b, logical_block]

latent_offset(block, row, d) =
  ((block * block_size + row) * Dqk) + d
```

消费规则：

```text
K path uses latent[..., 0:Dqk]
V path uses latent[..., 0:Dv]
```

若采用 `workspace_kv`：

```text
workspace_kv[..., 0, row, 0:Dqk] = gathered K row
workspace_kv[..., 1, row, 0:Dqk] = gathered latent row for V path
```

即使 `workspace_kv` 为 `Dqk` 宽，`C2` 也只读前 `Dv` 列。

## 4. 与其它模式的边界

```text
q_head -> kv_head mapping -> latent/page source selection -> QK split(Dqk) -> online softmax -> PV on Dv -> output
```

- Head Sharing：若 `Hkv < Hq`，先算 `kv_head`；MLA 不取代 head sharing。
- PA：只负责 latent cache 的 page/block 映射；MLA 负责 `Dqk/Dv` 消费规则。
- Sparse/TopK：若再叠 sparse，sparse 只决定取哪些 latent rows；MLA 仍决定 K 路和 V 路读哪些列。
- Mask/Causal：只作用在 `ws_s` 之后的 score 可见性；不改 MLA 的维度分离。
- Sink：若存在 sink latent source，也必须遵守同一套 `Dqk` 打分、`Dv` 输出规则。

## 5. 检查点

- `smScale` 必须按 `Dqk` 算，不是按 `Dv`。
- `C1` 的两个 K 段贡献必须在 softmax 前累加到同一个 `S`。
- `C2` 只读 `Dv` 宽度；不能把 `Dqk` 全宽当成输出 V 维。
- 若和 head sharing 组合，`kv_head` 同时作用于 `k_nope/k_rope/v` 三条 load 路。
- 若和 page cache 组合，`GatherK`/`GatherV` 读的是同一 latent row，但 K 路消费 `Dqk`，V 路只消费 `Dv`。
- 不要把 host 侧 concat/repeat/reconstruct 当成 kernel 规则；kernel 规则是：分段 load、分段累加、按不同维度消费。