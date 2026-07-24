---
name: triton-ascend-case-sort-topk
description: "TopK 优化：目标维 permute 到末维后按行处理，每行切分为 <=4096 的 2 幂 tile，tile 内用 ext.sort 排序取 top-K，再用固定 next_pow2(2*K) 缓冲区合并，避免全局排序和选择排序"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# @architecture_decision("ub-aware-row-block", reason="TopK 每行计算量小，需通过 ROWS_PER_BLOCK 提升每 program 计算密度；行数受 UB 容量约束")
# @architecture_decision("temp-buffer-merge", reason="Vector Sort 对连续 2K 缓冲排序更高效，避免 in-register gather 的额外比较/选择指令")
# @llm_hint("compiler_hint", "multibuffer=False, unit_flag=False")
# TopK tile-wise partial sort + merge

## 任务特征
- 沿某一维度取前 K 个值
- 排序维度需先 permute 到末维，按 (rows, n) 二维处理

## 优化 1：tile-wise 排序而非全局排序/选择排序

错误：
- 全局 `ext.sort` 整行（n 大时开销高）
- `for step in range(k): for i in range(n):` 选择排序

正确：
```python
TILE = min(4096, next_pow2(N))
COMBINED = next_pow2(2 * K)

for c in range(0, N, TILE):
    tile = load(X[row, c:c+TILE])
    tile = where(col_idx < N, tile, PAD_VAL)
    sorted_tile = ext.sort(tile, dim=1, descending=DESCENDING)
    chunk_topk = sorted_tile[:, :K]          # 或 ext.gather

    if c == 0:
        best = chunk_topk
    else:
        merged = alloc([1, COMBINED])
        merged = fill(merged, PAD_VAL)
        merged[:, 0:K] = best
        merged[:, K:2*K] = chunk_topk
        sorted_merged = ext.sort(merged, dim=1, descending=DESCENDING)
        best = sorted_merged[:, :K]
```

## 优化 2：UB-aware 多行并行

TopK 每行计算量小（只取前 K），必须提升每个 program 处理的行数，否则 kernel launch 和调度开销会主导延迟。

设计约束：
- `ROWS_PER_BLOCK` 不是固定常数，而是由 UB 预算动态决定。
- 同时驻留 UB 的 buffer 估算：输入 tile + best + chunk_topk + merged + 输出。
- 目标 grid：`min(cdiv(n_rows, ROWS_PER_BLOCK), num_cores)`。

草图标注：
```python
# @llm_hint("parallel", "coreidx")
for core_idx in range(grid_size):
    # @llm_hint("vectorize", "simd")
    for row_block in range(0, n_rows, ROWS_PER_BLOCK):
        rows = row_block + tl.arange(0, ROWS_PER_BLOCK)
```

## 优化 3：temp buffer merge

合并当前全局 best 与新 chunk 的 top-k 时，禁止在寄存器内用 `gather + modulo + where` 拼接；应使用显式 workspace：

1. 分配 workspace `[ROWS_PER_BLOCK, COMBINED]`。
2. 用极值填充整个 workspace。
3. 把 `best` 写入 `[0, K)`，`chunk_topk` 写入 `[K, 2K)`。
4. 对 workspace 做 `tl.sort`，再取前 K。

Why：Ascend Vector Sort 对连续内存布局更友好；`idx % K` 和多层 `tl.where` 会引入额外的标量/向量比较指令。

## 优化 4：top-k 提取用 gather 而非标量 mask-sum

错误：
```python
for rank in range(K):
    idx_mask = (tl.arange(0, BLOCK_SIZE) == rank)
    val = tl.sum(tl.where(idx_mask, sorted_vals, 0.0))
```

正确：
```python
topk_idx = tl.arange(0, K)
chunk_topk = tl.gather(sorted_tile, topk_idx, axis=1)
```

## 优化 5：合并缓冲区固定为 next_pow2(2*K)

- 合并长度 <= 256（K <= 128 时），`ext.sort` 走硬件排序高效。
- 禁止按 `num_chunks * k` 动态扩展后使用选择排序兜底。

## 关键参数
- `TILE <= 4096` 且为 2 的幂
- `COMBINED = next_pow2(2 * K)`
- `PAD_VAL = -inf`（降序）或 `inf`（升序）
