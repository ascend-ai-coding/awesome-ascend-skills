# 技术卡片 vec-14：对齐拆分搬运（让各段 UB 起始 32B 对齐，VF 走对齐微指令）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 访存
- **一句话**：一行要拆成多段分别 load 进寄存器时，搬运阶段就把每段起始 pad 到 32B 对齐，别为省一次 DataCopy 把整行连续搬入——后段非对齐会让 VF load/store 退化成非对齐微指令，反而更慢。

## 何时用（诊断特征）

- VF 把一行按段（前后半 / 偶奇 / 四分段）用不同 offset 分别 load 进寄存器（如 RoPE half：`off` 与 `halfOff`）。
- 段长 `segLen*sizeof(T)` 不是 32B 整数倍；若整行连续搬入 UB，第二段起始落在非 32B 对齐地址。
- trace 上 VF 的 load/store 微指令出现非对齐形态（RVECLD/RVECST 额外拍数），或访存段耗时高于同长度对齐搬运。

## 原理

RVEC load/store 对 32B 对齐地址走单拍对齐微指令，非对齐地址要拆成跨边界的多拍访问、更慢。让 VF 各段都对齐的代价只是：搬运时**把前一段 pad 到 32B 边界**（多占一点 UB）、用**两次（多次）DataCopyPad** 分别落到对齐偏移。对访存密集算子，这点搬运/UB 开销远小于非对齐访存惩罚。**搬运的对齐口径必须与 VF 里的段 offset 完全一致**。

## 怎么改（before / after）

```cpp
// before：整行 D 一次性连续搬入，后半落偏移 D/2（(D/2)*sizeof(T) 非 32B 整数倍 -> 非对齐）
DataCopyPad(inUb, gm[off], oneRowParams, pad);   // 后半在 inUb[D/2]，VF 后半 load 非对齐

// after：分两次搬，前半 pad 到 alignedHalf，后半从对齐偏移起（segLen = D/2）
int64_t alignedHalf = Ops::Base::CeilAlign<int64_t>(segLen, BLOCK_TYPE_SIZE / sizeof(T)); // = VF 的 halfDAlign
cp.srcStride = (D - segLen) * sizeof(T);                       // GM 侧跳过另一段（多行批量）
cp.dstStride = alignedHalf * sizeof(T) / BLOCK_TYPE_SIZE;      // UB 段间对齐步进
DataCopyPad(inUb,               gm[off],           cp, pad);   // 前半 -> UB 偏移 0
DataCopyPad(inUb[alignedHalf],  gm[off + segLen],  cp, pad);   // 后半 -> 对齐偏移
// VF 内：halfOff = off + alignedHalf; 两段 load/store 均对齐
// CopyOut 对称：srcStride = alignedHalf(对齐 UB)，dstStride = D - segLen(紧凑 GM)
```

## 性能数据

msprof `aiv_time` / 访存段：对齐微指令替代非对齐访问。**待实测**（预期：段长非 32B 对齐、访存密集的算子，收益抵掉并超过多一次 DataCopyPad + pad UB 的开销）。

## 红线（Stop rules）

- **搬运与计算对齐口径必须一致**：CopyIn 的 `dstStride`、VF 里的段 offset（`halfDAlign`）、CopyOut 的 `srcStride` 必须是同一个 `CeilAlign(segLen, 32/sizeof(T))`，不一致会读错数据。
- 仅当段长确实非对齐才拆；段长本就 32B 对齐时，整行连续搬入更省（别无谓多搬一次）。
- pad 会增大 UB 占用，进 tile 预算模型时按对齐后的 pitch 计算，别用 valid 长度。
- 权衡：段极短、行极多时多一次 DataCopyPad 的固定开销可能吃掉收益，实测 `aiv_time` 对比确认。

## 关联

- 源：RoPE arch35 `CopyInHalfMode`/`CopyOutHalfMode` 分两次搬运对齐前后半；详见 `ascendc-performance-best-practices` 技能 rope 族 `vector_fused`（对齐搬运小节）。
- 相邻卡片：vec-05（对齐后各段寄存器内直算）、vec-13（VF 尾块统一 mask）、vec-01（UB 层融合与布局）。
