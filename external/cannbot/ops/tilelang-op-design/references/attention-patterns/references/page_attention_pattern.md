# Attention Page Attention Pattern

一句话：PA 在 AscendC 算子里定义 logical KV token 到 physical cache row 的映射；dense 连续 tile 可 direct page load，复杂 sparse PA 要 device-side gather 成 dense workspace tile。

## 先读这个

命中信号：K/V 是 `k_cache/v_cache`，layout 类似 `[block_num, block_size, Hkv, D]`，并带 `block_table/page_table`。

本模式只负责：所有 K/V 读取必须经过 `logical token -> physical cache row`。实现可以是 AIC direct load，也可以是 V 侧 gather workspace；

## 1. 识别条件

- 输入含 `k_cache`、`v_cache`、`block_table/page_table`。
- cache 第二维是 `block_size/page_block_size`，不是 batch 维。
- `block_table[b, logical_block] = physical_block`。
- Q/O layout、GQA/MQA、mask/causal 是组合模式；PA 只处理 cache row 寻址。

## 2. 核心规则

PA 有两条 AscendC 实现路径，如果可以实现direct page load，必须实现direct page load。

### Dense PA：direct page load

适用：logical KV tile 连续、`block_size % BLOCK_N == 0`、不需要 topk/sparse 重排，K/V 能表达为 `rows x cols + src_stride`。

```text
C0 LoadQ
for kv tile t with prelaunch:
  C1/MM1: page load K -> QK -> ws_s
  V1:     scale/mask/online softmax -> ws_p + ws_meta
  C2/MM2: load P + page load V -> PV -> ws_o
  V2:     merge acc_o with softmax exp/sum -> final output
```

```text
logical_token  = t * BLOCK_N + local_col
logical_block  = logical_token // block_size
block_offset   = logical_token % block_size
physical_block = block_table[b, logical_block]

base = ((physical_block * block_size + block_offset) * Hkv + kv_head) * D

K: LoadNdGmToNzL1(K_cache + base, rows=BLOCK_N, cols=D, src_stride=Hkv*D)
V: LoadNdGmToNzL1(V_cache + base + ni*BASE_K, rows=BLOCK_N, cols=BASE_K, src_stride=Hkv*D)
```

### Complex PA：gather workspace

适用：从topk/sparse indices 当中选 token 等复杂情况无法直接load的情况。
```text
for sparse tile i:
  V gather:
    load indices tile
    for each selected logical token:
      check sentinel / causal / actual length
      physical row = page_mapping(batch, logical token)
      copy K prefix/tail into workspace_k / workspace_k_tail
      invalid row zero-fill and record/apply mask
  C1: wait gather; load workspace K/tail; QK GEMM + tail contribution -> ws_s
  V2: apply sparse/causal mask; online softmax -> ws_p
  C2: P @ gathered V/value-prefix -> ws_o
  V3/V4: merge accumulator, normalize, write output
```

## 3. 地址/形状公式

```text
K_cache/V_cache: [block_num, block_size, Hkv, D]
block_table:     [B, block_table_len]

cache_offset(block, row, kv_head, d) =
  ((block * block_size + row) * Hkv + kv_head) * D + d

Q/O BHSD: ((b * Hq + h) * Sq + q) * D + d
Q/O BSHD: ((b * Sq + q) * Hq + h) * D + d
```

GQA/MQA:

```text
kv_head = q_head / n_groups
n_groups = Hq / Hkv
```

有效长度:

```text
actual_kv_len[b] = cache_seqlens[b] 或 context_lens[b]
valid_col = logical_token < actual_kv_len[b]
```

sparse/gather PA:

```text
selected_token = indices[q, kv_group, topk_col]
valid = selected_token >= 0
valid = valid && selected_token < actual_kv_len[b]
valid = valid && causal_visible(q, selected_token)

logical_block  = selected_token // block_size
block_offset   = selected_token % block_size
physical_block = block_table[b, logical_block]

workspace_k[row, :]      = cache[physical_block, block_offset, kv_head, prefix_dim]
workspace_k_tail[row, :] = cache[physical_block, block_offset, kv_head, tail_dim]
```

invalid gather row 要 zero-fill，但 score 侧仍必须 mask 成 `-inf`。

## 4. 与其它模式的边界

```text
head mapping -> PA K/V load or gather -> score scale/mask/causal -> online softmax -> PV -> output
```

- Head/KV Sharing：决定 `kv_head`；PA 用它参与 cache offset。
- Mask/Causal：按 logical KV position 判断；不要用 physical block/order。
- Sink：sink K/V 不走 page table；PA 只处理 cache/local 段。
- TopK Sparse：logical indices 走 PA gather；physical flat offset 不再套 PA。
- Layout：Q/O 和 cache 分开写 offset。

## 5. 检查点

- `block_size` 来自 cache/page 维，不是 `seq_len` 或 `BLOCK_N`。
- `block_table` 第二维索引是 logical block，不是 token，也不是 physical row。
- K 和 V 使用同一套 logical token -> physical cache row 映射。
- `src_stride = Hkv * D`；GM cache offset 用实际 `D`，workspace 可用 `dimAlign`。
- GQA 场景用 `kv_head` 读 cache，不能用 `q_head`。
- dense direct-load PA 要求 tile 不跨 page；复杂 sparse PA 可以逐 token/逐行 gather 跨 page。
- `block_table_len >= ceil_div(actual_kv_len, block_size)`。
- online softmax 的 `max/sum/exp` 状态按 ring slot 和 row 保存，PA 不另起归一化。
- decode causal 用 `actual_kv_len - Sq` 偏移，仍按 logical token 判断。
- value-prefix / rope-tail 分段时，K 路可 gather 到多个 workspace；PV 只使用 V/value-prefix 输出维度。
