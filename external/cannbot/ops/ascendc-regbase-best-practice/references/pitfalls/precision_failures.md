---
title: 精度失败
purpose: 从输出异常、tail 错误、NaN/Inf 或 dtype 误差定位 RegBase 精度问题。
read_when:
  - 测试输出不匹配。
  - 只有部分 shape 或 tail case 失败。
keywords:
  - precision
  - failure
  - tail
next_reads:
  - precision_guide.md
  - symptom_to_cause.md
  - ../api/regbase_api_reference.md
depth: foundation
topic_type: pitfall
---

# 精度失败

## 1. 症状与可能原因

| 症状 | 优先检查 |
|---|---|
| 只有 tail case 错 | `MaskReg`、store 长度、padding 值 |
| 全量偏差 | dtype cast、近似公式、scale 或 epsilon |
| 输出全 0 | store 路径、select mask、output UB 是否写入 |
| 随机错误值 | UB ownership、未初始化 register、越界 load/store |
| NaN/Inf 增多 | 除零保护、sqrt/rsqrt 输入、overflow |
| fp16/bf16 误差大 | 是否需要 fp32 中间计算 |

## 2. 定位顺序

1. 先确认输入输出和参考 golden 一致。
2. 查 Host tiling 的 count、offset、tail。
3. 查 `CopyIn` / `CopyOut` 有效长度。
4. 查 VF mask、cast 和 store。
5. 查数学公式和近似实现。

## 3. 修复原则

- 先修数据边界，再修数学公式。
- 先让 tail case 正确，再做性能优化。
- 修改精度策略后同步更新 DESIGN.md。
- 不要通过放宽 atol/rtol 掩盖明显实现错误。

## 相关文档

- [[precision_guide]]
- [[symptom_to_cause]]
- [[../api/regbase_api_reference]]
