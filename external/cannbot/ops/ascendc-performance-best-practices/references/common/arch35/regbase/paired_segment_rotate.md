# `paired-segment-rotate`

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 解决的问题

把两个对齐分段以不同 offset 载入寄存器，在寄存器内完成 sin/cos 乘加和 B16/F32 Cast，避免中间结果反复落 UB。

## 为什么这样写

| 代码段 | 原因 | 不能拆开的约束 |
|---|---|---|
| `CopyInHalfMode` / `CopyOutHalfMode` | 前后半分别放到 32B 对齐的 UB 区域，VF 可对两段使用规则 load/store | 搬运的 `dstStride/srcStride` 必须与 VF 的 `halfDAlign` 完全一致 |
| `LoadTwoTensorForDtypeT` | 一次把前后半装入两个 FP32 寄存器，B16→FP32 Cast 与 load 融合 | `DIST_UNPACK_B16`、CastTrait、mask 和源 dtype 必须一起核对 |
| `HalfAlignVF` 六条乘加 | 直接完成前后半旋转乘加，不产生旋转后的 UB 临时张量 | sin/cos 的前后半布局、正负号及配对关系不能互换 |
| `UpdateMask` | 每拍消费剩余元素数，尾拍自动缩窄 | `updateCnt` 是递减状态，不能在循环内反复重置 |
| `StoreOneTensorForDtypeT` | FP32 热链结束后才按目标 dtype 舍入和 Pack | FP16/BF16 的 `CAST_RINT` 属于精度语义，不是随意的性能参数 |

核心收益不是“用了 Reg API”，而是旋转、乘加和 Cast 的中间值不落 UB；GM↔UB 搬运仍由 `LocalTensor`/`DataCopyPad` 完成。

## 怎么使用

- 代码：[`code/paired_segment_rotate_vf.h`](code/paired_segment_rotate_vf.h)
- 最小编译探针：[`probes/paired_segment_rotate_compile_probe.cpp`](probes/paired_segment_rotate_compile_probe.cpp)
- 设备数值回归：[`paired_segment_rotate_npu_check.asc`](../../../../tests/paired_segment_rotate_npu_check.asc)，覆盖三种 dtype、对齐与尾块维度。
- 源构建契约：`device_only`（最终命令含 `--cce-aicore-only`）；移植到 `xasc_combined` 时必须重新验证声明作用域和入口
- 已实际收录：`HalfAlignVF`、通用 load/store Cast 封装、成对分段的对齐搬运。

按以下顺序接入：

1. 源码/算子名出现 `rope`、`rotary`、`rotary_pos_emb`，或公式为 `x*cos + rotate(x)*sin`、前后半交换乘加时，先把本条目标为候选；不能因基线没有 Reg/MicroAPI 而跳过。
2. 确认目标确实是 rotate-half 布局。若是 interleave、quarter 或 batch 变体，本骨架不覆盖，不能硬套。
3. 同时迁移成对搬运、load/store Cast 封装和 `HalfAlignVF`，再把 `dLen/dAlign/currSNum/currDNum` 接到真实 TilingData。
4. 先用探针对目标 SDK 做单 dtype、单 VF 编译，再进完整 kernel；源/目标构建契约不一致时按 FAQ 调整声明可见域，不修改 SDK。
5. 用前后半边界、非 32B 对齐 D、尾拍以及 FP16/BF16/FP32 分别做精度验证，再比较性能。

## 前置条件

- A5 / DAV_3510 纯 AIV 路径。
- half 模式的 GM→UB 搬运步长和 VF 内 `halfDAlign` 必须使用同一 `CeilAlign(D/2, 32/sizeof(T))`。
- 尾块由 `UpdateMask` 处理；FP16/BF16 回写舍入保持 `CAST_RINT`。

这是 `integration_skeleton`，只覆盖 rotate-half 布局。当前文件没有收录 Interleave、Quarter、DeepSeek 和 Batch 版本；遇到这些模式时记录范例缺口，不能根据注释补写。
