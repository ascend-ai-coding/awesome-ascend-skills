# Conversion 族 — TilingData 字段语义

> Small-channel transpose 的 TilingData 字段说明。

---

## 1. 字段定义

| 字段 | 类型 | 含义 | 推导出处 |
|------|------|------|---------|
| `C` | int | 小通道维大小 | 合轴: 被转置的通道轴乘积 |
| `N` | int | 展平后的长轴长度 | 合轴: 其余轴乘积 |
| `tile_N` | int | 每个tile处理的有效元素数 | UB容量约束 + repeats约束 |
| `tile_NA` | int | tile_N对齐到32后的宽度 | `AlignUp(tileN, 32)` |
| `repeats` | int | TransDataTo5HD的repeat次数 | `tile_NA / 16` |
| `total_tiles` | int | 总tile数 | `CeilDiv(N, tileN)` |
| `block_dim` | int | 实际使用的vector core数 | `min(coreNum, totalTiles)` |
| `tiles_per_core` | int | 每核处理的tile数 | `CeilDiv(totalTiles, blockDim)` |
| `core_num` | int | 实际使用核数 | 多核分配 |
| `ub_bytes_per_tile` | int | 单tile的UB占用(字节) | `tileNA * (16*C + 32)` |
| `ub_size` | int | UB总容量 | 平台获取 |
| `reserved_bytes` | int | UB预留字节 | 用户指定 |

## 2. 约束检查

| 约束 | 条件 | 检查方式 |
|------|------|---------|
| 通道维 | C ≤ 16 | small-channel假设 |
| repeats上限 | repeats ≤ 255 | `tileNA ≤ 4080` |
| UB容量 | ub_bytes_per_tile ≤ ubSize - reserved | UB预算公式 |
| 32元素对齐 | tile_N % 32 == 0 | AlignUp |
| 16-block对齐 | tile_NA % 16 == 0 | AlignUp |

## 3. 设计检查清单

- [ ] 是否已将问题统一建模为 [C, N] → [N, C]
- [ ] tileN是否按32对齐
- [ ] tileNA/16是否满足repeats ≤ 255
- [ ] UB预算是否使用 `tileNA * (16*C + 32)` 公式
- [ ] totalTiles是否足够铺满核
- [ ] blockDim是否受totalTiles限制
- [ ] offset table是否按tileNA构建
- [ ] 是否为尾块保留了curN和tileNA的区分
