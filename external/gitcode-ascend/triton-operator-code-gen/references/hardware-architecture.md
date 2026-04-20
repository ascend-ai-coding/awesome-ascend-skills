# 昇腾 NPU 硬件架构

## 核心架构

### AI Core 组成

昇腾 NPU 的计算核心是 AI Core，A2/A3 芯片通常有 **24 个 AI Core**。

每个 AI Core 包含：

| 组件 | 数量 | 功能 | 专用缓存 |
|------|------|------|----------|
| **Cube Core** | 1 | 矩阵乘法计算 | L1 Buffer (1MB) |
| **Vector Core** | 2 | 向量计算（逐元素、归约等） | UB (192KB) |

### 核心类型选择

| 算子类型 | 使用核心 | 获取核数方法 |
|----------|----------|--------------|
| 纯向量计算（逐元素、归约） | Vector Core | `get_npu_vectorcore_num()` |
| 矩阵乘法（tl.dot） | AI Core | `get_npu_aicore_num()` |
| CV 混合算子 | AI Core + Vector Core | `get_npu_aicore_num()` |

```python
import torch
import triton.runtime.driver as driver


def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

---

## 存储层次

### 三级存储架构

```
┌─────────────────────────────────────────────────────────┐
│                    GM (Global Memory)                    │
│                    DDR 大容量存储                         │
│                    容量：GB 级别                          │
│                    带宽：较低                             │
└───────────────────────────┬─────────────────────────────┘
                            │ MTE2/MTE3 指令搬运
                            ▼
┌─────────────────────────────────────────────────────────┐
│              片上存储 (On-Chip Memory)                    │
├─────────────────────────┬───────────────────────────────┤
│     L1 Buffer (1MB)     │      UB (192KB)               │
│     Cube Core 专用       │      Vector Core 专用          │
│     矩阵分块缓存          │      向量计算缓存              │
└─────────────────────────┴───────────────────────────────┘
```

### GM (Global Memory)

- **位置**：DDR 内存
- **容量**：GB 级别
- **特点**：大容量但访问延迟高
- **用途**：存储输入输出张量、模型参数

### UB (Unified Buffer)

- **位置**：AI Core 内部，Vector Core 专用
- **容量**：192KB (A2/A3)
- **特点**：高速缓存，访问延迟低
- **用途**：
  - 存放向量计算的输入输出数据
  - 归约操作的中间结果
  - 激活函数计算的临时数据

### L1 Buffer

- **位置**：AI Core 内部，Cube Core 专用
- **容量**：通常 1MB (A2/A3)
- **特点**：矩阵计算专用缓存
- **用途**：
  - 存放矩阵乘法的分块数据
  - QK^T、SV 计算的中间结果

---

## 数据通路

### 完整计算通路

```
┌──────────────────────────────────────────────────────────────────┐
│                         计算流程                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   GM ──────► UB ──────► Vector Core ──────► UB ──────► GM        │
│   (输入)    MTE2搬运    向量计算          结果    MTE3搬运  (输出)  │
│                                                                  │
│   或                                                              │
│                                                                  │
│   GM ──────► L1 ──────► Cube Core ──────► L1 ──────► GM          │
│   (输入)    搬运       矩阵乘法          结果    搬运     (输出)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Cube-Vector 协同（CV 算子）

对于同时需要矩阵乘法和向量计算的算子（如 Attention）：

```
GM ──► L1 ──► Cube Core ──► L1/UB ──► Vector Core ──► UB ──► GM
        │         │            │           │           │
        │    QK^T 计算      结果转移    Softmax     最终输出
        │         │            │           │
        └─────────┴────────────┴───────────┘
                    数据通路
```

**关键点**：
- Cube 计算结果需要从 L1 转移到 UB
- Vector Core 处理 Softmax 等向量操作
- 使用 workspace 缓存中间结果

---

## 内存对齐要求

### 对齐规则

| 场景 | 对齐要求 | 说明 |
|------|----------|------|
| VV（Vector-Vector） | 32 字节 | 纯向量计算 |
| CV（Cube-Vector） | 512 字节 | 矩阵+向量混合计算 |
| UB 缓冲区 | 32 字节 | 所有 UB 分配 |
| 单值缓冲区 | 32 字节 | 均值、方差等归约结果 |

---

## Grid 分核策略

### 合并 Grid 分核

当逻辑核数大于物理核数时，使用 `TRITON_ALL_BLOCKS_PARALLEL=1` 自动优化：

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

---

## 数据类型优化

### 避免使用 INT64

Vector Core 的部分运算不支持 INT64，会退化为标量运算：

| 操作 | 不支持的数据类型 |
|------|------------------|
| Vector ADD | int64 |
| Vector CMP | int64/int32 |

**解决方案**：使用 FP32 进行比较运算

```python
# 优化前：int64 比较，退化为标量
cols = tl.arange(0, BLOCK_N)
xbar = tl.where(cols < N, x - mean, 0.0)

# 优化后：转为 FP32 比较
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

---

## 常见问题

### UB 溢出

**症状**：`ub overflow, requires xxxx bits while 1572684 bits available!`

**原因**：
- 单次循环数据量过大
- 缓冲区分配不合理
- 非对齐访存导致额外开销

**解决**：
1. 减小 BLOCK_SIZE
2. 使用核内循环切分
3. 确保访存对齐

### coreDim 超限

**症状**：`coreDim=xxxx can't be greater than UINT16_MAX`

**原因**：grid 大小超过 65535

**解决**：
1. 增大 BLOCK_SIZE
2. 设置 `TRITON_ALL_BLOCKS_PARALLEL=1`
3. 使用核内循环减少 grid 数量

### 精度损失

**症状**：FP16 输入时结果不准确

**原因**：归约操作精度不足

**解决**：
1. 归约前升精度到 FP32
2. 在 FP32 下完成所有计算
3. 最后降精度到输出类型

---