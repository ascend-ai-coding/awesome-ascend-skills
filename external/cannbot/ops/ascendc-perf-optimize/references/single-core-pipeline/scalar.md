# Scalar Bound / 小 case 优化策略

## 判定条件

- 总计算量极小（如 element count < 阈值）
- Scalar 指令占比高，Vector/Cube 单元空闲
- 指令发射率低，IPC 偏低

## 仿真图分析要点

- 定位 Scalar 指令占比较大的时间窗口
- 识别 Scalar 与 Vector 间不必要的同步等待

## 优化策略

| 策略 | 操作 | 效果 |
|------|------|------|
| **Scalar 优化** | 减少冗余 scalar 计算，合并条件分支 | 降低 scalar 指令数 |
| **循环展开** | 展开小循环减少分支代价 | 提升 IPC |
| **减少循环轴** | 根据tiling最简化循环轴 | 降低scalar |
| **指令选择** | 使用高效 scalar 指令替代低效序列 | 缩短关键路径 |
| **减少标量-向量转换** | 避免不必要的 Scalar ↔ Vector 数据搬移 | 减少搬移开销 |
| **使用性能友好的API** | 使用set_flag和wait_flag代替Queue，使用LocalTensor代替Tbuffer，去除Tpipe | 减少封装带来的scalar |

## Tiling 修正建议

- 适当增大单次处理粒度，减少循环次数
- 考虑与其他 kernel 融合以减少 launch 开销
