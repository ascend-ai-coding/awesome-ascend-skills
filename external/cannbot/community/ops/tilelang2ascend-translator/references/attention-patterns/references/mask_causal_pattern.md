# Attention Mask Causal Pattern
一句话：mask/causal 模式在 AscendC 算子里定义“哪些 score 列可见、哪些必须在 softmax 前变成 `-inf`”；它只改 score 可见性，不改 K/V source、本体 head 映射或 PV 路径。

## 先读这个

命中信号：输入显式带 `mask`，或语义上要求 `q` 只能看到满足时序约束的 `kv`。
本模式只负责：在 `QK -> softmax` 之间插入 score 可见性约束。实现可以是 causal 公式生成，也可以是显式 mask 读取；不要把它写成新的 KV source 或新的 attention 分支。

## 1. 识别条件

- causal：可见性由 `(global_kv <= global_q + offset)` 这类坐标关系决定，不需要额外 mask tensor。
- dense mask：输入显式带 `mask`，形状通常是 `[B, H, Sq, Skv]`。
- 该模式不改变：
  - `q_head -> kv_head`
  - local/page/sparse 的 KV 读取路径
  - `P @ V` 的 value 聚合公式
- 若 local/window/block sparse 已先限制候选列，mask/causal 仍只负责“这些候选列里谁可见”。

## 2. 核心规则

mask/causal 有两条 AscendC 实现路径。

### Causal：公式可见性 + tile skip

适用：可见性完全由 query / key 的相对位置决定，不需要额外 mask tensor。

```text
C0 LoadQ
for kv tile t with prelaunch:
  C1/MM1:
    if whole tile is strictly above diagonal:
      skip tile
    else:
      load K[t] -> QK -> ws_s
  V1:
    load ws_s
    if tile crosses diagonal:
      build row-wise visibleCols from global_q/global_kv
      apply -inf mask on invisible cols
    apply scale + online softmax -> ws_p + ws_meta
  C2/MM2:
    load P + V[t] -> PV -> ws_o
  V2:
    merge acc_o
```

关键点：

- causal 的新增逻辑主要在两处：
  - `C1` 前的整 tile skip
  - `V1` 内的跨对角线局部 mask
- 完全可见 tile 不需要额外 mask，只沿用 baseline softmax。
- 尾块 padding mask 和 causal mask 是两层约束，二者都在 `V1` 的 score 侧处理。

### Dense Mask：显式 mask tile

适用：可见性来自输入 `mask[b, h, q_row, k_col]`，不是简单坐标公式。

```text
C0 LoadQ
for kv tile t with prelaunch:
  C1/MM1: load K[t] -> QK -> ws_s
  V1:
    load ws_s
    load mask rows for current q half-tile and kv tile
    uint8 mask -> half -> float
    select(score, -inf) before online softmax
    write ws_p + ws_meta
  C2/MM2: load P + V[t] -> PV -> ws_o
  V2:     merge acc_o
```

关键点：

- dense mask 不改 `C1/C2` 数据源，只在 `V1` 给 score 注入可见性。
- 当前仓库实现里，显式 mask 路径是 `uint8 -> half -> float` 再转成 `-inf` 选择，不是直接拿 byte 比较后软掩码。
- mask 读取粒度是“当前 q half-tile 的逐行 mask + 当前 kv tile 的列区间”。

新增 pipeline 点：

- 相对普通 FA，没有新增新的 KV source；只是 `V1` 增加了“score mask 注入”子阶段。
- causal 还额外允许在 `C1` 前做 whole-tile skip；这是 mask/causal 相对 baseline 唯一会改变 `C1` 调度的地方。

## 3. 地址/形状公式

causal:

```text
q_block_start  = bx * BLOCK_M
kv_block_start = t  * BLOCK_N

whole_tile_skip if kv_block_start > q_block_start + BLOCK_M - 1

global_q  = bx * BLOCK_M + local_q_row
global_kv = t  * BLOCK_N + local_k_col

visible iff global_kv <= global_q
```

decode / unequal lengths 时，causal 可见性可带偏移：

```text
visible iff global_kv <= global_q + (actual_kv_len - actual_q_len)
```

显式 mask:

```text
Mask: [B, H, Sq, Skv]

mask_offset(b, h, q_row, k_col) =
  (((b * H + h) * Sq + q_row) * Skv + k_col)
```

V1 当前子块读的 mask 区域：

```text
q_row = bx * BLOCK_M + rowStart + local_row
k_col = t  * BLOCK_N + local_col
```

尾块：

```text
tail_valid = Skv % BLOCK_N
```

尾块无效列也必须在 score 侧加 `-inf`；不要只靠 K/V zero-fill。

## 4. 与其它模式的边界

```text
head mapping -> KV/source selection -> score visibility(mask/causal) -> online softmax -> PV -> output
```

- Head Sharing / GQA / MQA：只决定 `kv_head`；mask/causal 不改 head 映射。
- PA / Sparse / Sink：先决定“读哪些 K/V 行”；mask/causal 再决定这些列在 score 上是否可见。
- Layout：只影响 score/mask 的 GM offset，不改 output layout。
- Bias/position term：若也在 softmax 前加到 score，属于同一阶段的组合，但不是 mask/causal 自己定义的新 source。

## 5. 检查点

- mask/causal 只作用在 score 侧；不要改写 K/V source 路径。
- causal 至少要明确两件事：
  - 哪些 whole tile 可以在 `C1` 前直接跳过
  - 跨对角线 tile 在 `V1` 用什么坐标公式裁剪
- dense mask 要明确 mask tile 的读取索引：当前 q 子块行、当前 kv tile 列。
- 尾块 padding 和 mask/causal 是两层约束；二者都要能把无效列送成 `-inf`。
- 如果实现里有 ring slots，mask 注入仍然跟随当前 `ws_s/ws_p/ws_meta` slot；不能跨 slot 读错状态。
- 只做 zero-fill 不等于 masked-out；softmax 前必须真的注入 `-inf` 语义。