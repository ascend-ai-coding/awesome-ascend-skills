---
title: Tiling 审查笔记
purpose: 设计和审查 RegBase tiling 时使用，覆盖 tiling packet、有效长度/对齐长度、tail、dtype route 和 UB 大小。
read_when:
  - Host/tiling 还没有把全局 shape 拆成 block、tile 和 tail。
  - 设计串讲或 review 中需要检查 tiling。
  - 出现 tail、block split 或 UB 大小问题。
keywords:
  - tiling
  - review
  - tail
next_reads:
  - regbase_programming_notes.md
  - ../pitfalls/precision_guide.md
  - ../pitfalls/common_traps.md
depth: practice
topic_type: experience
---

# Tiling 审查笔记

RegBase 的 VF body 只能处理当前 tile 的寄存器通道。Host/tiling 层必须先把全局 shape 拆成可解释的 block、tile 和 tail，并把 kernel 需要的 route 信息传清楚。

## 1. tiling packet 至少包含

- 总元素数或关键维度。
- 每核处理元素数。
- 每个 tile 的有效长度和对齐长度。
- tile 循环次数。
- tail 元素数。
- dtype route 或 `TILING_KEY`。
- UB buffer 大小或推导所需参数。

## 2. 有效长度与对齐长度

| 用途 | 使用有效长度 | 使用对齐长度 |
|---|---|---|
| mask 生成 | 是 | 否 |
| 数学 API count | 是 | 视 API 而定，必须验证 |
| UB buffer 大小 | 否 | 是 |
| UB offset | 否 | 通常是 |
| GM 写回 | 是 | 不能越界 |

## 3. tail 策略

- 每个 tile 都应能得到当前有效长度 `validCount`。
- VF 内用 `MaskReg` 表示有效通道。
- store 回 UB 和 GM 写回都不能写出有效范围。
- 如果 copy 层使用 padding，必须说明 padding 不参与最终输出语义。

## 4. 多 route 场景

- dtype route：fp16、bf16、fp32 可能需要不同 VF chain。
- shape route：小 shape 可简化 pipeline，大 shape 需要多 tile。
- alignment route：对齐路径和非对齐路径可以不同，但要保持数学一致。

## 5. 审查必查项

- blockDim 是否来自 shape、device 或 route 推导，不能无依据写死。
- 每核工作量是否覆盖全部元素且无重叠。
- tile 有效长度和对齐长度是否区分。
- tail count 是否传入 kernel 或可由 kernel 推导。
- UB 总使用量是否包含双缓冲和 scratch。
- `TILING_KEY` 是否与 dtype route 匹配。

## 6. 常见问题

| 问题 | 影响 |
|---|---|
| 用对齐长度参与数学 count | tail 结果错误 |
| 用有效长度计算 UB offset | 下一 tile 地址错 |
| 忽略最后一个 block 的剩余量 | 丢数据或越界 |
| 多 route 使用同一 tiling 字段但语义不同 | 开发者难以实现 |
| repeat count 超过 API 限制 | 构建失败或运行行为不稳定 |

如果 VF 或普通 vector API 的 repeat count 使用 `uint8_t`，单次 repeat 不能超过 255；当 row count、repeat count 或 tile count 可能超过单次接口限制时，应在 Host tiling 或 kernel loop 中拆分。

## 7. 审查输出建议

问题要指向 DESIGN.md 的具体章节，并说明它会影响 API 可行性、内存规划、多核策略、伪代码可实现性还是精度风险。

## 相关文档

- [[regbase_programming_notes]]
- [[../pitfalls/precision_guide]]
- [[../pitfalls/common_traps]]
