# 抽象硬件架构

## ascend910b/ascend910_93 (A2/A3) 硬件说明

AI Core 内部包含三类核心组件：

**计算单元**：
- **Scalar**：执行标量计算、指令发射和控制
- **Vector**：执行向量运算，适合元素级操作，910B1/910B2 48个vector core，910B3/910B4 40个vector core
- **Cube**：执行矩阵运算，适合矩阵乘法等密集计算，910B1/910B2 24个cube core，910B3/910B4 20个vector core

**存储单元**：
- **Local Memory**：AI Core 内部存储，包括多级缓存
  - L1 Buffer：512KB
  - L0A/L0B Buffer：各 64KB
  - L0C Buffer：128KB
  - Unified Buffer (UB)：192KB（示例值，实际编码时通过接口获取）

**搬运单元**：
- **DMA**：负责数据在不同存储单元之间的高效传输

**异步执行机制**：
- **指令流**：Scalar 计算单元发射指令到不同单元
- **同步信号流**：保证指令执行顺序和数据一致性
- **数据流**：数据从 GM→Local Memory→计算单元→Local Memory→GM
