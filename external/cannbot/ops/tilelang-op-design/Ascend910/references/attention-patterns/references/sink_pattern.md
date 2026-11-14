# Attention Sink Pattern
一句话：sink 模式在 AscendC 算子里定义“一小段固定前缀 KV source 始终参与 attention”；实现上把 sink tile 插到普通 local KV tile 之前，共享同一套 online softmax / PV 累加。

## 先读这个

命中信号：输入除 `q/k/v` 或 `k_cache/v_cache` 外，还额外带 `sink_k/sink_v`，shape 类似 `[B, H, sink_size, D]`。
本模式只负责：在 local KV 之外再引入一段固定 sink KV source，并保证 sink score 与 local score 在同一 softmax 里归一化。实现可以是 sink direct load + local direct load，也可以是 sink direct load + local paged load/gather；不要把 sink 简化成“单独算一支再后加 bias”。

## 1. 识别条件

- 输入含 `sink_k`、`sink_v`。
- `sink_k/sink_v` 是单独的 KV source，不和 local `k/v` 或 `k_cache/v_cache` 共用同一存储布局。
- 输出语义不是两次 attention 相加，而是：
- 先拼 `sink_score` 和 `local_score`
- 再做一次统一 softmax
- 再分别对 `sink_v` 和 `local_v` 做 PV 并累加
- 若 local KV 来自 page cache，sink 模式仍成立；page 只决定 local 段怎么取，不改变 sink 的职责。

## 2. 核心规则

sink 有两条 AscendC 实现路径。

### Dense Sink：sink + continuous local

适用：local KV 本身就是连续布局，不需要 page/block 映射，也不需要 token-level sparse gather。

```text
C0 LoadQ
for t in range(sink_loops + kv_loops + prelaunch):
  t < sink_loops:
    C1/MM1: load sink_k[t]  -> QK -> ws_s
    V1:     online softmax  -> ws_p + ws_meta
  t >= sink_loops:
    C1/MM1: load local_k[t - sink_loops] -> QK -> ws_s
    V1:     online softmax  -> ws_p + ws_meta

  t >= prelaunch:
    now = t - prelaunch
    now < sink_loops:
      C2/MM2: load sink_v[now] -> PV -> ws_o
    now >= sink_loops:
      C2/MM2: load local_v[now - sink_loops] -> PV -> ws_o
    V2: merge acc_o with alpha/sumexp
```

关键点：

- sink tile 和 local tile 共用同一条 `C1 -> V1 -> C2 -> V2` 主循环。
- sink 不是单独 softmax；sink 只是把 unified loop 的前几轮数据源切到 `sink_k/sink_v`。
- `sink_loops = ceil(sink_size / BLOCK_N)`，`total_loops = sink_loops + kv_loops`。

### Sink + PA：sink direct load + local paged path

适用：sink 段是连续 `sink_k/sink_v`，但 local 段来自 paged KV cache。

```text
LoadQ
for t in range(total_loops + prelaunch):
  if t < sink_loops:
    C1/MM1: direct load sink_k[t] -> QK -> ws_s
    V1:     online softmax -> ws_p + ws_meta
  else:
    GatherK(local t) -> C1/MM1 -> ws_s
    V1:              online softmax -> ws_p + ws_meta

  if t >= prelaunch:
    now = t - prelaunch
    if now < sink_loops:
      C2/MM2: direct load sink_v[now] -> PV -> ws_o
    else:
      GatherV(local now) -> C2/MM2 -> ws_o
    V2: merge acc_o with alpha/sumexp
```

关键点：

- sink 段不走 page/block_table。
- local paged 段仍按 PA 规则做 `logical token -> physical cache row`。
- sink 和 paged local 最终仍在同一条 online softmax 状态里合并。

新增 pipeline 点：

- 相对普通 FA，sink 没有引入新的 softmax 语义，而是引入了**第二个 KV source**。
- 相对普通 local loop，新增的是 `is_sink / local_t = t - sink_loops` 这层 source dispatch。
- 若 local 段是 paged/gather path，则 `GatherK/GatherV` 只发生在 local 分支，sink 分支保持 direct load。

## 3. 地址/形状公式

sink source:

```text
sink_k/sink_v: [B, H, sink_size, D]

sink_offset(b, h, sink_token, d) =
  (((b * H) + h) * sink_size + sink_token) * D + d
```

continuous local source:

```text
k/v: [B, H, Skv, D]

local_offset(b, h, local_token, d) =
  (((b * H) + h) * Skv + local_token) * D + d
```

paged local source:

```text
k_cache/v_cache: [block_num, Hkv, block_size, D]
block_table:     [B, block_table_len]

logical_block  = local_token // block_size
block_offset   = local_token % block_size
physical_block = block_table[b, logical_block]

cache_offset(block, kv_head, row, d) =
  (((block * Hkv) + kv_head) * block_size + row) * D + d
```

统一循环索引：

```text
sink_loops  = ceil(sink_size / BLOCK_N)
kv_loops    = ceil(local_kv_len / BLOCK_N)
total_loops = sink_loops + kv_loops

if t < sink_loops:
  source = sink
  token_base = t * BLOCK_N
else:
  source = local
  local_t = t - sink_loops
  token_base = local_t * BLOCK_N
```

尾块：

```text
sink_tail_valid = sink_size % BLOCK_N
kv_tail_valid   = local_kv_len % BLOCK_N
```

尾块无效列要在 score 侧 mask；只做 zero-fill 不够。

## 4. 与其它模式的边界

```text
head mapping -> sink/local source dispatch -> optional page mapping for local -> softmax/PV -> output
```

- Head Sharing / GQA / MQA：决定 `kv_head`；sink 模式不改 head 映射。
- PA：只负责 local cache 段的 physical row 寻址；sink 段不走 page table。
- Sparse / TopK：若 local 段还是离散 token 选择，则 sparse 只作用于 local 段；sink 段仍是固定前缀 source。
- Mask / Causal：决定 local token 是否可见；sink 是否永远可见由该组合模式决定，但 sink 模式本身不定义新的 mask 语义。
- Layout：sink/source dispatch 不改 output layout，只改 KV source 读取路径。

## 5. 检查点

- `sink_k/sink_v` 是独立 source，不应并进 local cache/page 映射里解释。
- sink 和 local 必须进入同一套 online softmax 状态；不能先各自 softmax 再合并。
- unified loop 必须明确：
  - `t < sink_loops` 走 sink
  - `t >= sink_loops` 走 local
  - `now = t - prelaunch` 的消费阶段要复用同样的 source dispatch
- 若 local 是 paged path，只有 local 分支需要 `block_table`/gather；sink 分支不查 page table。
- `sink_v` 生命周期必须覆盖到对应的 `C2/MM2`；不能只在 score 端接入 sink 而 value 端漏掉。
- `sink_tail_valid/kv_tail_valid` 要分别处理；不要把 sink 尾块和 local 尾块混成一套规则。
- ring/prelaunch 下，某个 slot 的 sink/local tile 在 `V2` 之前都不能被下一轮覆盖。
