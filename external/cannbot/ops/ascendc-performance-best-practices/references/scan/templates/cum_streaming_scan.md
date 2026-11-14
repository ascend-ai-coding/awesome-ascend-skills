# 流式前缀扫描（含索引）—— Streaming Scan

代码：[dav310/cum_streaming_scan.template](dav310/cum_streaming_scan.template)

## 解决的问题

带位置输出的极值扫描（**Cummin / Cummax**）要同时输出 `values` 与 `indices`（argmin/argmax）。Sklansky 的"整行相加/取极值"语义不直接适配索引传播（层内组合会丢失"哪一步取得极值"）。流式扫描沿 R **逐步串行更新常驻 UB 的 running-(value, idx)**，用 `Compare`+`Select` 两路同 mask 同时更新值与索引，天然支持索引、NaN 传播与任意 `combine`。

## 适用条件

- **并行度充足**：融合后 `M×N` 较大，切行（work-item）即可用满核。Cummin/Cummax 的常见形态（多行 + 中等 R）均命中。
- R 可不必全量入 UB（每步只驻留当前 r 的 `[mBlk,nTileLen]` 切片 + 常驻 running 状态）。
- 不适用：`M×N` 极小 + R 超长（1D 大 R，work-item 数 ≪ 核数）→ 改走 [cum_core_partition_scan](cum_core_partition_scan.md)。

## 执行流程

1. 融合 `(M, R, N)`；work-item = `mBlk` 行 × `nTileLen` 列，向量宽 `W = mBlk*nTileLen`。
2. `InitRunning`：running-value = identity（min→+inf / max→-inf），running-idx = 0。
3. 沿 `r = 0..R-1`：
   - `CopyInSlice`：strided `DataCopyPad`（`PaddingMode::Compact`）把该 r 的 `[mBlk,nTileLen]` 切片打包成连续 `W`；
   - `ComputeStep`：fp16/bf16 升 fp32；`Compare<GT>`（min）得 mask；`Select` 两路更新 value 与 idx；
   - `CopyOutSlice`：写出该 r 的 values（降回原精度）与 indices。
4. 流水事件：`MTE2→V`（载入后计算）、`V→MTE3`（计算后写出）、`MTE3→MTE2`（写出后下一载入）。

## 关键技术

| 技术 | 要点 |
|------|------|
| **Compare + Select 两路同 mask** | `gtMask=(x>curMin)`；`curMin=Select(curMin, x, gtMask)`、`curIdx=Select(curIdx, r, gtMask)`。同一 mask 保证值与索引一致 |
| **并列语义（对齐 torch）** | torch.cummin/cummax 为 **last occurrence**：并列时取更晚位置。用 `GT`（而非 `GE`）→ `x<=curMin` 时更新索引为当前 r。Cummax 用 `LT` |
| **NaN 传播** | `Compare` 对 NaN 返回 false → NaN 被取用并向后传播，对齐 torch |
| **fp16/bf16 升 fp32** | `DIST_UNPACK_B16` + Cast 升 fp32 比较（min/max 无误差），输出 `DIST_PACK_B32` 降回 |
| **★ MTE3→V WAR 屏障** | 一核处理多 work-item 时，本 item 末次 CopyOut(MTE3) 读 curMin/curIdx，下一 item InitRunning(V) 覆写它们；`MTE3_MTE2` 不约束 MTE3→V。缺屏障会使每个 work-item **最后一个 r 的 idx 被覆写成 0**（实测踩坑） |

## 关键代码索引

- `ProcessWorkItem`：R 循环 + 四个流水屏障 + 末尾 `MTE3→V` WAR 屏障。
- `ComputeStep`：`Compare<CMPMODE::GT>` + 两个 `Select` + fp16/bf16 Cast 升降。
- `CopyInSlice` / `CopyOutSlice`：strided `DataCopyPad<PaddingMode::Compact>`，`srcStride/dstStride = (R*N - nLen)`。

## 适配

- **Cummax**：`CMPMODE::GT → LT`；identity `+INFINITY → -INFINITY`（int `INT_MAX → INT_MIN`）。
- **Cumsum/Cumprod（无索引）**：`Compare/Select` → `Add`/`Mul`；去掉 idx 缓冲与 CopyOut 的 indices 分支。
