# Ascend NPU 关键术语

仅列出 Agent 不一定知道的 Ascend 特有概念。

## 硬件架构

### AI Core
- A2/A3 通常有 **24 个 AI Core**，每个含 **1 个 Cube 核 + 2 个 Vector 核**
- Cube Core：矩阵计算（`tl.dot`），使用 L1 Buffer
- Vector Core：向量/逐元素计算，使用 UB
- Core 内指令集不同，但均为 SIMD

### 内存层次

| 层级 | 容量 | 所属 | 用途 |
|------|------|------|------|
| **GM** (Global Memory) | GB 级 | 全局共享 | 存储 I/O 和参数，延迟高 |
| **UB** (Unified Buffer) | **192KB** (A2/A3) | Vector Core 独占 | 当前计算数据，延迟低 |
| **L1 Buffer** | ~1MB | Cube Core 独占 | Cube Core 执行数据 |

### UB 约束
- **32 字节对齐**（对齐访问性能更高）
- 单值缓冲区（均值、方差等）需分配 32B 空间（即使逻辑上只需 4B）
- 安全 BLOCK_SIZE = `(196608 - 32) / (缓冲区数 × dtype大小) × 0.8`

## Tiling 切分

- **核间切分**：任务分配到不同 AI Core，`grid = 物理核数`
- **核内切分**：单 Core 内分批次处理，适配 UB 192KB

## 内存对齐

| 层级 | 对齐要求 |
|------|---------|
| UB 缓冲区 | 32 字节 |
| GM 缓冲区 | 16 字节 |

```python
# 对齐计算
aligned_bytes = ((actual_bytes + 31) // 32) * 32
```

## 关键规则

- **归约操作必须升精度到 FP32**（FP16/BF16 输入时）
- **矩阵乘法累加器用 FP32**
- **数据搬运**：GM ↔ UB/L1，应减少 GM 访问次数、提高数据复用
- **Block 对齐**：矩阵运算 BLOCK_M/N/K 必须为 16 倍数（Cube 单元粒度）
