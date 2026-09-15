# Attention Head Sharing Pattern
一句话：head sharing 模式在 AscendC 算子里定义“多个 query heads 共享同一个 KV head”；它只改 `q_head -> kv_head` 的映射和 K/V 读取地址，不改 score/mask/softmax/PV 主流程。

## 先读这个

命中信号：`Q` 的 head 数大于 `K/V` 的 head 数，常见字段是 `n_heads` / `n_kv_heads`、`nheads_q` / `nheads_kv`，并满足整除或退化到单一 KV head。
本模式只负责：把当前 q head 或 q head group 映射到应该读取的 kv head。实现可以是 GQA 的分组共享，也可以是 MQA 的单头共享；不要把它写成 host 侧先 `repeat/expand` 出完整 K/V 再照常算。

## 1. 识别条件

- `Hq > Hkv`，且 `Hq % Hkv == 0`；或 MQA 特例 `Hkv = 1`。
- `Q` layout 仍按 query heads 展开，`K/V` layout 按更少的 kv heads 存储。
- 输出 head 数仍是 `Hq`；head sharing 不减少输出 head，只减少 K/V head source 数。
- 若和 page/sparse/sink 组合，head sharing 仍只先解决“这个 q head 应该取哪个 kv head”，别的模式再决定 token/source。

## 2. 核心规则

head sharing 有两条 AscendC 实现路径。

### GQA：group-to-kv-head 映射

适用：`Hq > Hkv > 1`，多个 q heads 按固定 group 共享一个 kv head。

```text
task = (batch, q_head, q_block)
kv_head = q_head / n_groups
n_groups = Hq / Hkv

C0 LoadQ(q_head)
for kv tile t with prelaunch:
  C1/MM1: load K[b, kv_head, t] -> QK -> ws_s
  V1:     online softmax -> ws_p + ws_meta
  C2/MM2: load V[b, kv_head, t] -> PV -> ws_o
  V2:     merge acc_o
```

关键点：

- pipeline 和 baseline 一样，没有新增 stage。
- 新增逻辑只有 `kv_head = q_head / n_groups` 这一步。
- 同一个 group 里的所有 q heads，各自输出不同，但都从同一 `kv_head` 读 K/V。

### MQA：all-q-heads share head 0

适用：`Hkv = 1`，所有 q heads 共用同一个 K/V head。

```text
task = (batch, q_head, q_block)
kv_head = 0

C0 LoadQ(q_head)
for kv tile t with prelaunch:
  C1/MM1: load K[b, 0, t] -> QK -> ws_s
  V1:     online softmax -> ws_p + ws_meta
  C2/MM2: load V[b, 0, t] -> PV -> ws_o
  V2:     merge acc_o
```

关键点：

- MQA 是 GQA 的极端特例：`n_groups = Hq`，但 kernel 实现通常直接把 `kv_head` 写死成 `0`。
- 不要在 host 侧先 `expand` K/V 到 `Hq` 个 heads；kernel 直接按单头地址读更准确。

新增 pipeline 点：

- 相对 baseline，head sharing **不新增 stage**。
- 它只改 `C1/MM1` 和 `C2/MM2` 读取 K/V 时的 head 地址。
- 所以它的实现重点不是 workspace 或新信号，而是 head 映射公式必须在所有 K/V load 点一致使用。

## 3. 地址/形状公式

baseline:

```text
Q: [B, Hq, Sq, D]
K/V: [B, Hq, Skv, D]

baseline kv_head = q_head
```

GQA:

```text
Q:   [B, Hq, Sq, D]
K/V: [B, Hkv, Skv, D]

n_groups = Hq / Hkv
kv_head  = q_head / n_groups
```

MQA:

```text
Q:   [B, Hq, Sq, D]
K/V: [B, 1, Skv, D]

kv_head = 0
```

连续 KV 地址：

```text
q_offset(b, q_head, q_row, d) =
  (((b * Hq) + q_head) * Sq + q_row) * D + d

kv_offset(b, kv_head, kv_row, d) =
  (((b * Hkv) + kv_head) * Skv + kv_row) * D + d
```

若和 page attention 组合，head sharing 先产出 `kv_head`，再参与 cache 地址：

```text
cache_offset(block, kv_head, row, d) =
  (((block * Hkv) + kv_head) * block_size + row) * D + d
```

## 4. 与其它模式的边界

```text
q_head -> kv_head mapping -> KV/source selection -> score visibility -> online softmax -> PV -> output
```

- PA：head sharing 先给出 `kv_head`，PA 再用它做 cache row 地址。
- Sparse / TopK：先按 `kv_head` 确定 head source，再按 sparse 规则选 token。
- Sink：sink/local 两个 source 都要使用同一个 `kv_head` 映射规则；sink 不会覆盖 head sharing。
- Mask / Causal：只改 score 可见性；不会改 `kv_head`。
- MLA / head-dim split：若 K/V 维度分段，head sharing 只管 head 索引，不管分段维度。

## 5. 检查点

- `kv_head` 映射必须同时用于 K load 和 V load；不能只改一边。
- GQA 下 `n_groups = Hq / Hkv` 必须是整数关系；否则不是这个模式。
- 不要把 host 侧 `repeat_interleave/expand` 当成 kernel 规则；kernel 规则是“少头存储，按映射取址”。
- pipeline 不新增 stage；如果文档主体讲成 workspace 或新队列扩展，说明抽取跑偏了。
- 若与 page/sparse/sink 组合，先算 `kv_head`，再进入 token/source/path 规则；不要反过来。

## 6. 常见错误

| 错误 | 后果 | 正确做法 |
| --- | --- | --- |
| 把 head sharing 写成“先把 K/V 复制到每个 q head” | 丢掉模式本身的存储语义 | kernel 内按 `kv_head` 公式直接取 K/V |
| K 用 `kv_head`，V 仍用 `q_head` | QK 和 PV 对不上 | K/V 两条 load 路都共用同一 `kv_head` |
| 先做 page/sparse，再决定 `kv_head` | 地址解释错位 | 先确定 `q_head -> kv_head`，再交给别的模式 |
| 把 MQA 讲成新 pipeline | 误导实现 | MQA 只是 `kv_head = 0` 的地址特例 |
| `Hq % Hkv != 0` 还按 GQA 写公式 | head 分组错误 | 只有整除共享才属于这张卡 |
