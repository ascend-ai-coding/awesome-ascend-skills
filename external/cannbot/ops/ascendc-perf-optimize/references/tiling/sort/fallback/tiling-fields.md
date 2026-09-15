# Sort 族 — TilingData 字段语义

> Sort 族各 Pattern（A/B/C）的 TilingData 字段并集。

---

## 1. 通用基础字段

| 字段 | 类型 | 含义 | 推导出处 |
|------|------|------|---------|
| `N` | int64 | 总元素数 | 输入shape |
| `K` | int64 | Top-K 值 | 输入参数 |
| `dtype` | str | 数据类型 | 输入参数 |
| `dtype_bytes` | int | 单元素字节数 | dtype映射 |
| `pattern` | str | "A" / "B" / "C" | N vs tileSize × coreNum判定 |
| `tile_size` | int64 | 单tile元素数 | UB容量推导, 工程值4096 |
| `total_tiles` | int64 | 总tile数 | `CeilDiv(N, tileSize)` |
| `core_num` | int64 | 实际使用核数 | tiling计算 |
| `ub_size` | int | UB总容量 | 平台获取 |

## 2. 每核参数 (Pattern B/C)

| 字段 | 含义 | 出现条件 | 推导 |
|------|------|---------|------|
| `tiles_per_core` (S_c) | 每核tile数 | B/C | `CeilDiv(totalTiles, coreNum)` |
| `elements_per_core` (E_c) | 单核处理元素数 | B/C | `S_c × tileSize` |
| `front_core_tiles` | 前coreNum-1核的tile数 | C | `CeilDiv(totalTiles, coreNum)` |
| `last_core_tiles` | 尾核tile数 | C | `totalTiles - (usedCore-1)*frontCoreTiles` |
| `last_tile_size` | 末tile元素数(≤tileSize) | A/B/C | `N - tileSize × (totalTiles-1)` |

## 3. 归并参数 (Pattern C)

| 字段 | 含义 | 推导 |
|------|------|------|
| `merge_rounds_phase2` | 核内归并轮次 | `ceil(log_M(S_c))` |
| `merge_rounds_phase3` | 跨核归并轮次 | `ceil(log_M(coreNum)) - 1` |
| `once_max_merge` | Phase 2/3单次归并最大元素数 | `(ubSize / 64B) / 32 × 32` |
| `once_max_output` | Phase 4单次输出最大元素数 | `(ubSize / outputBPE) / 32 × 32` |

## 4. UB 预算字段

| 字段 | 含义 | 出现于 |
|------|------|--------|
| `sort_bytes_per_elem` | Phase 1每元素UB占用 | 所有Pattern |
| `merge_bytes_per_elem` | Phase 2/3每元素UB占用 (=64B) | B/C |
| `output_bytes_per_elem` | Phase 4每元素UB占用 | B/C |

## 5. Workspace字段

| 字段 | 含义 | 出现条件 |
|------|------|---------|
| `workspace_bytes` | 跨核workspace总字节数 | Pattern C |

**公式**: `usedCore × ((E_c × 8B × 2 + 31) / 32 × 32)`

## 6. 分支—字段交叉表

| 字段 | Pattern A | Pattern B | Pattern C |
|------|:---:|:---:|:---:|
| N / K / dtype / tile_size | ✅ | ✅ | ✅ |
| total_tiles / core_num | ✅ | ✅ | ✅ |
| tiles_per_core (S_c) | 1 | 1 | ✅ |
| elements_per_core (E_c) | N | tile_size | ✅ |
| front/last_core_tiles | ➖ | ➖ | ✅ |
| last_tile_size | N | ✅ | ✅ |
| merge_rounds_phase2/3 | ➖ | 1轮 (Phase 3 only) | ✅ |
| once_max_merge/output | ➖ | ✅ | ✅ |
| workspace_bytes | ➖ | ➖ | ✅ |

## 7. 约束检查清单

- [ ] N > tileSize × coreNum? (Pattern C前提)
- [ ] tileSize = 4096?
- [ ] Phase 2循环: `while listNum != 1`
- [ ] Phase 3循环: `while listNum > M`
- [ ] GetCoreWsOffset(i) = i × E_c × 2 (直接乘法)
- [ ] truncationFlag在归并执行后设置
- [ ] workspace按核数分配 (非总元素数)
