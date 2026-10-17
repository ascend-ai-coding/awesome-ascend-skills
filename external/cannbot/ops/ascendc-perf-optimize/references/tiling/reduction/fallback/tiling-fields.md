# Reduction 族 — TilingData 字段语义

> 五模板（AR-SmallR / AR-FullLoad / AR-Recompute / ARA-FullLoad / ARA-Recompute）的 TilingData 字段并集。

---

## 1. 通用基础字段

所有模板均包含：

| 字段 | 含义 | 来源 |
|------|------|------|
| `template` | 模板名称（五选一） | 决策树输出 |
| `mode` | `"AR"` 或 `"ARA"` | A0==1 为 AR，否则 ARA |
| `A1` | 外层保留轴长度 | 合轴 |
| `R` | 归约轴长度 | 合轴 |
| `A0` | 内层保留轴长度 | 合轴（AR 模式下为 1） |
| `dtype` / `dtype_bytes` | 数据类型 | 输入参数 |
| `core_num` | 可用核数 | 平台参数 |
| `used_core_num` | 实际使用核数 | 多核分配 |
| `tiles_per_core` | 每核处理的 tile 数 | `ceil(total_tiles / core_num)` |

---

## 2. 切分粒度字段

| 字段 | 含义 | 出现模板 |
|------|------|---------|
| `a1_tile_len` | AR-SmallR 中 A1 方向的 tile 长度 | AR-SmallR |
| `a1_tile_tail` | A1 方向尾块长度 | AR-SmallR |
| `rows_per_ub` | AR-FullLoad 中 UB 一次处理的 A1 行数 | AR-FullLoad |
| `rows_per_core` | 每核处理的 A1 行数 | AR-FullLoad, AR-Recompute |
| `r_align` | R 按 32B 对齐后的长度 | AR-FullLoad |
| `r_chunk` | AR-Recompute 中 R 方向分块大小 | AR-Recompute |
| `r_chunk_tail` | R 方向尾块大小 | AR-Recompute |
| `r_loop_count` | R 分块轮数 | AR-Recompute |
| `a0_tile_len` | ARA 模式中 A0 方向的 tile 长度 | ARA-FullLoad, ARA-Recompute |
| `a0_tile_tail` | A0 方向尾块长度 | ARA-FullLoad, ARA-Recompute |
| `a0_outer` | A0 方向 tile 总数 | ARA-FullLoad, ARA-Recompute |
| `total_tiles` | 总 tile 数（A1 × a0_outer） | AR-SmallR, ARA 族 |
| `r_bin_size` | ARA-Recompute 中 R 方向分块粒度 | ARA-Recompute |
| `tmp_buf_size` | Reduce API 临时缓冲字节数 | 所有模板 |

---

## 3. 二分累加字段

AR-Recompute 和 ARA 族在块间合并时使用二分累加：

| 字段 | 含义 | 出现模板 |
|------|------|---------|
| `fold_base` | 小于循环轮数的最大 2 的幂 | AR-Recompute, ARA-Recompute |
| `fold_remain` | 超出 fold_base 的剩余轮数 | AR-Recompute, ARA-Recompute |
| `cache_layers` | 跨 bin 累加所需的 cache 层数 | ARA-Recompute |

ARA-FullLoad 在 R > 8 时使用行级 BinaryAdd，需额外记录分组参数（quotient、remainder 等），由 kernel 侧消费，host tiling 按 R 和 a0_tile_len 推导。

---

## 3.5 多轴交错模板字段（交错-tail-R / 交错-tail-A）

> 仅交错连续载入模板使用（见 [tiling-flow.md §11–13](tiling-flow.md#11-多轴交错连续载入模板细节)）。
> 五模板不含这些字段；此时 `axisShape[]` / `axisStride[]` 存严格交替后的全轴布局（偶数≥6 tail-R / 奇数≥7 tail-A）。

| 字段 | 含义 | 出现模板 |
|------|------|---------|
| `axisShape[i]` / `axisStride[i]` | 严格交替后各轴长度/步长（连续性校验用） | 交错-tail-R, 交错-tail-A |
| `fastInnerStart` | 内块起始轴索引（tail-R=axisNum-4，tail-A=axisNum-4） | 交错-tail-R, 交错-tail-A |
| `fastGroupCnt` | group 数 = ∏(外层 A 轴) | 交错-tail-R |
| `fastCoresPerGroup` | 每 group 切 rows 轴的核数 | 交错-tail-R |
| `fastRowsAxisIdx` | 被切分+连续载入的 rows 轴索引（=fastInnerStart-1） | 交错-tail-R |
| `fastRowsPerLoad` | 一次载入的 rows 行数（增大 burst，≤8） | 交错-tail-R |
| `fastInnerChunkA0` | 内块 A0 分块粒度（命中要求 == szA0） | 交错-tail-R |
| `raraSlabCnt` | slab 数 = ∏(内块外 A 轴 0,2) | 交错-tail-A |
| `raraChunkR0` | 内块最外 R0 分块粒度（≤ min(R0,8)） | 交错-tail-A |
| `workspace_size` | tail-R = groupCnt×coresPerGroup×outLanes×4；**tail-A = 0** | 交错-tail-R |

---

## 4. 增强标志字段

| 字段 | 含义 | 触发条件 |
|------|------|---------|
| `enable_group_reduce` | R 方向跨核归约 | R 超大且 A 维过小 |
| `enable_welford` | Welford 在线算法 | Norm 类 + Recompute |
| `enable_dichotomy` | 二分累加（精度） | Sum 精度敏感 |
| `with_index` | 跟踪极值索引 | ArgMax / ArgMin |
| `workspace_size` | 跨核 workspace 字节数 | Group Reduce 启用 |

---

## 5. 模板—字段交叉表

| 字段 | AR-SmallR | AR-FullLoad | AR-Recompute | ARA-FullLoad | ARA-Recompute |
|------|:---:|:---:|:---:|:---:|:---:|
| A1 / R / A0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| a1_tile_len / tail | ✅ | ➖ | ➖ | ➖ | ➖ |
| rows_per_ub / rows_per_core | ➖ | ✅ | ✅ | ➖ | ➖ |
| r_align | ✅ | ✅ | ➖ | ➖ | ➖ |
| r_chunk / r_chunk_tail | ➖ | ➖ | ✅ | ➖ | ➖ |
| a0_tile_len / tail / outer | ➖ | ➖ | ➖ | ✅ | ✅ |
| total_tiles | ✅ | ➖ | ➖ | ✅ | ✅ |
| r_bin_size | ➖ | ➖ | ➖ | ➖ | ✅ |
| fold_base / fold_remain | ➖ | ➖ | ✅ | ➖ | ✅ |
| tmp_buf_size | ➖ | ✅ | ➖ | ➖ | ➖ |

---

## 6. 对齐规则

| 场景 | 对齐粒度 |
|------|---------|
| A0 切分 | 64 元素(FP32) / 128 元素(FP16) |
| R 方向 | 32B 对应的元素数 |
| UB Buffer | 32B |
| tmp_buf_size | ≥ 4096B |
