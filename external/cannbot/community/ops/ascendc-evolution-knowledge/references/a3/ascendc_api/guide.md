# API 避坑快速参考

## #0 标量陷阱（性能杀手，编译可过但跑得慢）

| 陷阱 | 症状 | 快速修复 |
|------|------|---------|
| **标量 for 循环替代 Vector API** | 算子功能正确但性能极差（加速比 0.01x-0.5x） | 逐项检查 kernel 中的 `for` 循环，用 ReduceSum/Sort32/Adds/Muls 替代 |

**这是生成代码中最常见的性能问题**。TileLang → AscendC 转译后常残留 raw pointer + 标量 for 循环，完全绕过了 Vector 单元。详见 `optimization_patterns/scalar_to_vector.md`。

## Top 5 致命陷阱（编译可过但结果错）

| # | 陷阱 | 症状 | 快速修复 |
|---|------|------|---------|
| 1 | 32 字节对齐违规 | 输出部分正确部分乱码 | FP16 count 必须是 16 的倍数 |
| 2 | 忘记 FreeTensor | 多 tile 后 UB 耗尽死锁 | 每个 DeQue 配一个 FreeTensor |
| 3 | PipeBarrier 缺失 | 输出全 0 或随机值 | VECTOR 写后、MTE3 读前加 PipeBarrier |
| 4 | Cast 舍入模式错误 | FP16 精度偏差大 | FP32→FP16 用 CAST_ROUND |
| 5 | 尾块越界 | 最后一批数据错误 | 最后一个 tile 用 tailLen |

## 完整列表

15 个常见陷阱详见 → [common_pitfalls.md](common_pitfalls.md)

## 何时必读

- **子 agent 写内核代码前**：必读本 guide.md，先做标量检查再写代码
- 遇到编译通过但精度不对 → 读 common_pitfalls.md 逐项排查
- 遇到精度对但性能极差 → 先读 `optimization_patterns/scalar_to_vector.md`
