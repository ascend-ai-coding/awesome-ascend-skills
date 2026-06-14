# 案例：MLA / PagedAttention Kernel 与 Golden 对齐

本文档为 **算子级案例**，不属于 Golden 技能正文。正文仅保留通用原则；涉及具体 `.cce` 路径、测试文件名时一律在此维护。

## 背景

- 关联：Issue #288
- 场景：CSV golden 与 AscendC kernel 实际行为不一致时的对齐记录

## 典型陷阱：SWA Mask

golden 曾假设 kernel 对 `mask_type=6` (SWA_NORM) 应用滑动窗口 mask，但 kernel 的 `SoftmaxStage1` 实际只处理部分 mask type：

```cpp
// mla.cce:2566（路径相对 ATB 源码树）
if (mask_type == 3) { /* LOOK_AHEAD */ }
else if (need_mask && mask_type == 4) { /* MASK_FREE */ }
// mask_type == 6 (SWA_NORM) 在此路径下不触发预期 mask 行为
```

Golden 若错误增加 -10000 mask，可导致大面积精度超标；需按 kernel 实际分支调整 golden，而非按文档臆测。

## 对齐步骤（本案例）

1. **阅读 kernel 源码**：在 ATB 仓库中定位 AscendC 实现，确认关键分支  
   - 本案例：`src/kernels/mixkernels/multi_latent_attention/op_kernel/mla.cce`
2. **参考 kernel 级测试**：  
   - `tests/apitest/kernelstest/mix/test_paged_attention_mla_mtp_split.py`  
   - 其中 `ref_masked_attention()` 可作为已验证参考，与 CSV golden 对比差异。
3. **对比 mask 处理**：确认 kernel **实际**处理的 mask type，不要假设所有枚举均有对应实现。
4. **注意 tiling 差异**：decode 与 prefill 的 tiling 参数不同，kernel 行为可能不同，golden 需与当前 case 路径一致。

## DataGen 参考类（勿依赖固定行号）

PagedAttention / MultiLatentAttention 在 `data_generation.py` 中的 `customize` / `golden` / `case_preprocess` 会随需求迭代增减。**不要使用历史文档或聊天记录里的行号**定位；应在 ATB 仓库内对 `data_generation.py` 按类名检索，例如：

```bash
rg -n "^class PagedAttentionOperation|^class MultiLatentAttentionOperation" tests/apitest/opstest/python/data_generation.py
```

路径以前缀 `tests/apitest/opstest/python/` 为准；若目录调整，用 `find <ATB_REPO_PATH> -name data_generation.py` 确认。

## 维护说明

- ATB 路径随版本可能变更；若搬迁文件，请同步更新本页路径。
- 新增其他算子的 kernel 对齐案例时，建议新增独立 `references/case-study-<op>-*.md`，勿堆叠进技能正文。
