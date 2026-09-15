# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 6353
- **PR作者**: hellokitty911
- **代码文件**: 6 个文件
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 10 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 10 | 5 | 100% |

---

## 发现问题

### 文件: gmm/quant_grouped_matmul_dequant/op_host/quant_grouped_matmul_dequant_tiling.cpp（Tiling侧）


#### [1] 人工检视意见

- **提出人**: weixin_44156099
- **作者**: hellokitty911
- **文件**: gmm/quant_grouped_matmul_dequant/op_host/quant_grouped_matmul_dequant_tiling.cpp
- **行号**: 553
- **评论时间**: 2026-07-02
- **Commit**: 9f796b34565c
- **问题描述**:

  > 【review】Host 侧未拒绝 K=0，新增优化 tiling 会除以 optKb。Host 侧只校验 originK % 16 != 0，originK == 0 会通过。进入 perToken && dynamicQuant 新优化路径后，kBytes 为 0，optKb 计算为 0，随后 _Params.optKPasses 使用 _Params.optKb 作为除数。

- **代码片段**（行553）:
```cpp
 543 | 
 544 |         uint32_t bestMb = 16;
 545 |         uint32_t bestNb = 16;
 546 |         uint64_t bestScore = static_cast<uint64_t>(-1);
 547 |         for (uint32_t nbI = 0; nbI < AscendC::NB_CANDIDATE_COUNT; ++nbI) {
 548 |             uint32_t nb = AscendC::NB_CANDIDATES[nbI];
 549 |             if (nb > _Params.originN) continue;
 550 |             for (uint32_t mbI = 0; mbI < AscendC::MB_CANDIDATE_COUNT; ++mbI) {
 551 |                 uint32_t mb = AscendC::MB_CANDIDATES[mbI];
 552 |                 if (mb < 32U) continue;
 553 |                 uint64_t kbRaw = std::min({kBytes, l0aTotal / mb, l0bTotal / nb,
 554 |                                            static_cast<uint64_t>(AscendC::MMAD_K_MAX)});
 555 |                 uint32_t kb = static_cast<uint32_t>((kbRaw / K_FRACTAL_INT8) * K_FRACTAL_INT8);
 556 |                 if (kb == 0) continue;
 557 |                 if (!l0Fits(mb, nb, kb)) continue;
 558 |                 if (!l1Fits(mb, nb, kb)) continue;
 559 |                 if (!ubFits(mb, nb)) continue;
 560 | 
 561 |                 const uint64_t weightBytes = static_cast<uint64_t>(_Params.originN) * kBytes;
 562 |                 const uint64_t iters = (_Params.originM + mb - 1) / mb;
```

---

#### [2] 人工检视意见

- **提出人**: weixin_44156099
- **作者**: hellokitty911
- **文件**: gmm/quant_grouped_matmul_dequant/op_host/quant_grouped_matmul_dequant_tiling.cpp
- **行号**: 647
- **评论时间**: 2026-07-02
- **Commit**: 9f796b34565c
- **问题描述**:

  > 【review】Host 侧未拒绝 N=0，候选搜索失败后仍下发非法 optNBlockSize。建议：增加 originN > 0 校验；候选循环增加 found 检查，没有满足 UB/L1/L0 预算的候选时返回 tiling 失败或回退旧路径；下发前校验 optNBlockSize > 0。

- **代码片段**（行647）:
```cpp
 637 |                 }
 638 |             }
 639 |             _Params.optNCoreNum = 1u << chosen;
 640 |             _Params.optMCoreNum = static_cast<uint32_t>(NUM_OF_AICORE) / _Params.optNCoreNum;
 641 |         }
 642 | 
 643 |         uint64_t totalIters = (_Params.originM + bestMb - 1) / bestMb;
 644 |         _Params.optPerCoreIters = static_cast<uint32_t>((totalIters + _Params.CoreNum - 1) / _Params.CoreNum);
 645 |         _Params.optRemainderIters = static_cast<uint32_t>(totalIters % _Params.CoreNum);
 646 |         _Params.optWeightBlockN1 = 0;
 647 |         const uint32_t totalColBatch = static_cast<uint32_t>((_Params.originN + bestNb - 1) / bestNb);
 648 |         const uint64_t cbBytes = static_cast<uint64_t>(bestNb) * _Params.originK;
 649 |         const uint64_t l2Eff = static_cast<uint64_t>(AscendC::L2_HEADROOM_BYTES) *
 650 |                                AscendC::L2_WEIGHT_SHARE_PCT / 100;
 651 |         uint32_t nBlockSize = cbBytes > 0 ? static_cast<uint32_t>(l2Eff / cbBytes) : 1u;
 652 |         if (nBlockSize == 0) nBlockSize = 1;
 653 |         if (nBlockSize > totalColBatch) nBlockSize = totalColBatch;
 654 |         _Params.optNBlockSize = nBlockSize;
 655 | 
 656 |         const uint64_t xRawL1Bytes = mbAlignedXL1 * kBytes * sizeof(uint16_t);
```

---

### 文件: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_init.h（Kernel侧）

#### [3] 人工检视意见

- **提出人**: wanyukang
- **作者**: hellokitty911
- **文件**: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_init.h
- **行号**: 238
- **评论时间**: 2026-06-04
- **Commit**: 9f796b34565c
- **问题描述**:

  > 【review】[MED] UB容量校验完全依赖Host侧Tiling，Kernel无自检机制
  > 
  > - 文件：`gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_init.h` L238-246, 767(注释)
  > - 问题：H1: InitUbLayout 缓冲区布局可能超出 UB 容量，但无 kernel 侧校验
  > 
  > InitUbLayout 通过 arena 偏移量手动布局所有 UB 缓冲区。Phase X 工作区(phaseXStart 起)和 Phase D 区(dOff=phaseXStart 起)在同一 UB 区域上做了别名复用。
  > 
  > dequantScratchLT_ 需要 `MbAligned * Nb * sizeof(half)` 字节。Phase X 端对应偏移为 phaseXStart，其起始的 xRawPingLT_ 仅需 `chunkHalves * sizeof(half)` 字节。当 `MbAligned * Nb > 2 * chunkHalves` 时，dequantScratch 会溢出到 Phase X 的后续缓冲区(absLT_/reduceWorkLT_/maxLT_ 等)，导致 Phase D 执行时篡改 Phase X 期望的数据结构（如果 Phase D 和后续 Phase X 之间有未完成的写操作）。
  > 
  > ASCEND_ASSERT 已被注释，注释说明"budget enforced host-side"——完全依赖 Host Tiling 计算的 UB 预算不会出错。
  > - 当前代码：
  > ```
  > 238:     dequantScratchLT_ = arena[dOff].ReinterpretCast<half>();
  > 239:     dOff += MbAligned * Nb * sizeof(half) + UB_BANK_PAD;
  > 243:     }
  > 245:     (void)phaseXStart;
  > 246:     // ASCEND_ASSERT(max(off, dOff) <= UB_BYTES);  // budget enforced host-side.
  > 247: }
  > ```
  > - 改进方案：
  > ```
  > 建议：(1) 启用 ASCEND_ASSERT 或等效的运行时检查，在开发/调试阶段捕获UB溢出；(2) 在代码注释中增加 host 侧对应的校验函数名称和文件路径，便于追溯。
  > ```

- **代码片段**（行238）:
```cpp
 228 |     // UB hog that would force Mb=16 on K>=4096 shapes.
 229 |     int8ScratchLT_ = arena[off].ReinterpretCast<int8_t>();
 230 |     off += MbAligned * chunkHalves + UB_BANK_PAD;
 231 | 
 232 |     (void)kHalves;  // legacy reference; chunkHalves is the sizing axis now.
 233 |     (void)kBytes;   // legacy; int8Scratch sized by chunkHalves now.
 234 | 
 235 |     // Phase D aliases the Phase X region: single dequantScratch (NZ output of
 236 |     // VDEQ16). The cb-alternating output uses persistent outPing/Pong.
 237 |     uint32_t dOff = phaseXStart;
 238 |     dequantScratchLT_ = arena[dOff].ReinterpretCast<half>();
 239 |     dOff += MbAligned * Nb * sizeof(half) + UB_BANK_PAD;
 240 |     if constexpr (!ScaleIsU64) {
 241 |         scaleFp32StagingLT_ = arena[dOff].ReinterpretCast<float>();
 242 |         dOff += Nb * sizeof(float) + UB_BANK_PAD;
 243 |     }
 244 | 
 245 |     (void)phaseXStart;
 246 |     // ASCEND_ASSERT(max(off, dOff) <= UB_BYTES);  // budget enforced host-side.
 247 | }
```

---


### 文件: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_staged_normal.h（Kernel侧）

#### [4] 人工检视意见

- **提出人**: weixin_44156099
- **作者**: hellokitty911
- **文件**: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_staged_normal.h
- **行号**: 310
- **评论时间**: 2026-07-02
- **Commit**: 9f796b34565c
- **问题描述**:

  > 【review】Staged 动态量化全零行会写出 0 scale，后续 pertoken 倒数除零。建议：写入 scale 前或取倒数前统一做 max(abs(scale), eps) / 零值分支保护；这里可复用新优化路径中已有的 DQ_MAX_EPS 语义。

- **代码片段**（行310）:
```cpp
 300 |         if(fracMIdx_ != (int)fracMIdx) {
 301 |           WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS[0]);
 302 |         }
 303 |         for(int32_t k = 0; k < NM_FRACTAL_INT8; k++){
 304 |           if(fracMIdx_ != (int)fracMIdx) {
 305 |             if(tilingData->dynamicQuant) {
 306 |               pertokenScale[k] = FLOAT_1 / ubXScaleFloat.GetValue(k * FLOAT_PERBLOCK);
 307 |             } else {
 308 |               pertokenScale[k] = FLOAT_1 / ubXScaleFloat.GetValue(k);
 309 |             }
 310 |           }
 311 |           Muls<float, false>(ubXFloatND[mulOffset], ubXFloatND[mulOffset], pertokenScale[k], MASK_PLACEHOLDER, 1, unaryParams);
 312 |           mulOffset += processXKBaseN;
 313 |         }
 314 |         fracMIdx_ = fracMIdx;
 315 |       } else {
 316 |         SetVectorMask<half, MaskMode::COUNTER>(NM_FRACTAL_INT8 * processXKBaseN);
 317 |         Muls<float, false>(ubXFloatND, ubXFloatND, x_scale_quant, MASK_PLACEHOLDER, 1, unaryParams);
 318 |       }
 319 | 
```

---


### 文件: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_utils.h（Kernel侧）


#### [5] 人工检视意见

- **提出人**: weixin_44156099
- **作者**: hellokitty911
- **文件**: gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_utils.h
- **行号**: 157
- **评论时间**: 2026-07-02
- **Commit**: 9f796b34565c
- **问题描述**:

  > 【review】Weight 预取使用 uint32 中间值和 uint16 DataCopyParams，缺少大 N/K 上界保护。建议：offset 相关乘加在参与运算前提升到 uint64_t；Host tiling 侧补充 K1q/blockLen/srcStride <= UINT16_MAX 的约束，或 Kernel 对超限场景分段搬运；同时显式校验 effN1 >= ngCount。

- **代码片段**（行157）:
```cpp
 147 |     const uint32_t ngStart  = baseNG;
 148 | 
 149 |     uint64_t wGmOff;
 150 |     uint32_t effN1;
 151 | 
 152 |     if constexpr (!BlockedZN) {
 153 |         // Flat ZN: rows of K1 each hold N1 fractals of K0_N0 bytes.
 154 |         // K-pass kp starts at K1-row `k1Start`; per-row stride = N1 * K0_N0.
 155 |         // Compute uint32 fractal offsets first, widen to uint64 only for the
 156 |         // final base sum. (310P scalar unit can't lower uint64 * uint64.)
 157 |         const uint32_t kpFracOff = k1Start * N1;           // uint32
 158 |         const uint32_t fracTotal = kpFracOff + ngStart;     // uint32
 159 |         wGmOff = expertBaseOff
 160 |                + static_cast<uint64_t>(fracTotal) * K0_N0;
 161 |         effN1  = N1;
 162 |         (void)K;
 163 |         (void)weightBlockN1;
 164 |     } else {
 165 |         const uint32_t blockNG       = weightBlockN1;
 166 |         const uint32_t blockIdx      = ngStart / blockNG;
```

---

## 被检视代码

> 本报告基于 PR 6353 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `gmm/quant_grouped_matmul_dequant/op_host/quant_grouped_matmul_dequant_tiling.cpp`
- `gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_init.h`
- `gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_phase_x.h`
- `gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_staged_normal.h`
- `gmm/quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_staged_run.h`
- ... 共 6 个文件
