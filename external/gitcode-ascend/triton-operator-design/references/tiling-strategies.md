# Tiling 策略详解

本文档详细说明 Ascend NPU 上的 Tiling 切分策略设计方法。

## Tiling 的核心思想

### 为什么需要 Tiling

1. **UB 容量有限**：AI Core 的 UB 通常只有 192KB，无法一次性加载所有数据
2. **提高并行度**：通过核间切分，让多个 AI Core 并行处理
3. **优化内存访问**：减少 GM 访问次数，提高数据复用率

### Tiling 的两个层次

```
┌─────────────────────────────────────────┐
│         Global Memory (GM)              │
│  ┌──────┬──────┬──────┬──────┬──────┐  │
│  │Core 0│Core 1│Core 2│Core 3│ ...  │  │
│  └──────┴──────┴──────┴──────┴──────┘  │
│         核间切分 (Inter-Core)           │
└─────────────────────────────────────────┘
              ↓ 每个核独立处理
┌─────────────────────────────────────────┐
│         Unified Buffer (UB)             │
│  ┌──────┬──────┬──────┬──────┐         │
│  │Tile 1│Tile 2│Tile 3│ ...  │         │
│  └──────┴──────┴──────┴──────┘         │
│         核内切分 (Intra-Core)           │
└─────────────────────────────────────────┘
```

## 核间切分策略

### 切分原则

1. **独立性**：每个 AI Core 处理的数据相互独立，避免跨核通信
2. **负载均衡**：尽量让每个 Core 处理相同数量的数据
3. **数据局部性**：连续的数据分配给同一个 Core，提高缓存命中率

### 常见切分模式

#### 模式 1：按 Batch 维度切分

**适用场景**：
- 输入形状为 [B, D]，B 是 batch 维度，D 是特征维度
- 算子在 D 维度上独立计算（如 LayerNorm、RMSNorm）

**切分方法**：
```python
# 输入: x[B, D]
# AI Core 数量: num_cores

# 步骤1: 计算每个 Core 处理的 batch 数量
batch_per_core = ceil(B / num_cores)

# 步骤2: 计算当前 Core 的 batch 范围
core_id = get_core_id()  # 获取当前 Core ID
batch_start = core_id * batch_per_core
batch_end = min((core_id + 1) * batch_per_core, B)

# 步骤3: 当前 Core 处理的数据
x_core = x[batch_start:batch_end, :]  # 形状: [batch_per_core, D]
```

**示例**：
```
输入: x[1024, 768]
AI Core 数量: 8

切分结果:
- Core 0: x[0:128, :]    # 128 个 batch
- Core 1: x[128:256, :]  # 128 个 batch
- Core 2: x[256:384, :]  # 128 个 batch
- ...
- Core 7: x[896:1024, :] # 128 个 batch
```

#### 模式 2：按特征维度切分

**适用场景**：
- 输入形状为 [B, D]，需要在 D 维度上并行
- 算子在 B 维度上有依赖（如某些归约操作）

**切分方法**：
```python
# 输入: x[B, D]
# AI Core 数量: num_cores

# 步骤1: 计算每个 Core 处理的特征维度大小
features_per_core = ceil(D / num_cores)

# 步骤2: 计算当前 Core 的特征范围
core_id = get_core_id()
feature_start = core_id * features_per_core
feature_end = min((core_id + 1) * features_per_core, D)

# 步骤3: 当前 Core 处理的数据
x_core = x[:, feature_start:feature_end]  # 形状: [B, features_per_core]
```

#### 模式 3：按行切分（矩阵运算）

**适用场景**：
- 矩阵乘法：C = A × B
- 输入 A 形状为 [M, K]，B 形状为 [K, N]

**切分方法**：
```python
# 矩阵乘法: C[M, N] = A[M, K] × B[K, N]
# 按 M 维度切分

# 步骤1: 计算每个 Core 处理的行数
rows_per_core = ceil(M / num_cores)

# 步骤2: 计算当前 Core 的行范围
core_id = get_core_id()
row_start = core_id * rows_per_core
row_end = min((core_id + 1) * rows_per_core, M)

# 步骤3: 当前 Core 计算
A_core = A[row_start:row_end, :]  # 形状: [rows_per_core, K]
C_core = A_core @ B               # 形状: [rows_per_core, N]
```

### 负载均衡处理

当数据量不能被 Core 数量整除时：

```python
# 方法1: 向上取整（最后一个 Core 可能处理较少数据）
batch_per_core = ceil(B / num_cores)

# 方法2: 动态分配（更均衡）
base_batch = B // num_cores
remainder = B % num_cores

if core_id < remainder:
    # 前 remainder 个 Core 多处理一个 batch
    batch_start = core_id * (base_batch + 1)
    batch_end = batch_start + base_batch + 1
else:
    # 后面的 Core 处理 base_batch 个 batch
    batch_start = remainder * (base_batch + 1) + (core_id - remainder) * base_batch
    batch_end = batch_start + base_batch
```

## 核内切分策略

### UB 空间计算

#### 步骤 1：确定数据类型大小

```python
# 数据类型大小（字节）
type_sizes = {
    'FP16': 2,
    'BF16': 2,
    'FP32': 4,
    'INT8': 1,
    'INT32': 4,
}
```

#### 步骤 2：列出所有缓冲区需求

```python
# 以 RMSNorm 为例，FP16 输入升精度到 FP32 计算

# 输入缓冲区（FP16）
input_buffer_size = D * type_sizes['FP16']

# 升精度缓冲区（FP32）
upcast_buffer_size = D * type_sizes['FP32']

# 平方缓冲区（FP32）
square_buffer_size = D * type_sizes['FP32']

# 均值缓冲区（FP32，需要 32B 对齐）
mean_buffer_size = 32  # 即使逻辑上只需 4B

# Gamma 缓冲区（FP32）
gamma_buffer_size = D * type_sizes['FP32']

# RMS 值缓冲区（FP32，需要 32B 对齐）
rms_buffer_size = 32

# 输出缓冲区（FP16）
output_buffer_size = D * type_sizes['FP16']

# 总空间
total_buffer_size = (
    input_buffer_size +
    upcast_buffer_size +
    square_buffer_size +
    mean_buffer_size +
    gamma_buffer_size +
    rms_buffer_size +
    output_buffer_size
)
```

#### 步骤 3：计算单次循环处理量

```python
# UB 总大小
UB_SIZE = 192 * 1024  # 192KB

# 单次循环可处理的 batch 数量
batch_per_iteration = UB_SIZE // total_buffer_size

# 实际使用的空间
actual_ub_used = batch_per_iteration * total_buffer_size

# 检查是否超出 UB 容量
assert actual_ub_used <= UB_SIZE, f"UB 空间不足: {actual_ub_used} > {UB_SIZE}"
```

### 缓冲区分配策略

#### 策略 1：固定分配

**适用场景**：
- 缓冲区大小固定
- 所有缓冲区同时使用

**示例**：
```python
# UB 空间分配（192KB）
UB_BASE = 0  # UB 起始地址

# 输入缓冲区（64KB）
input_buffer = UB_BASE
input_buffer_size = 64 * 1024

# 中间缓冲区（96KB）
intermediate_buffer = input_buffer + input_buffer_size
intermediate_buffer_size = 96 * 1024

# 输出缓冲区（32KB）
output_buffer = intermediate_buffer + intermediate_buffer_size
output_buffer_size = 32 * 1024
```

#### 策略 2：动态分配

**适用场景**：
- 缓冲区大小可变
- 缓冲区分时复用

**示例**：
```python
# 阶段 1: 加载输入数据
input_buffer = UB_BASE
load_input(input_buffer, size=D*2)  # FP16

# 阶段 2: 升精度计算（复用输入缓冲区）
upcast_buffer = input_buffer  # 复用同一块空间
cast_to_fp32(input_buffer, upcast_buffer)

# 阶段 3: 计算平方（需要新缓冲区）
square_buffer = UB_BASE + D*4  # FP32
compute_square(upcast_buffer, square_buffer)
```

### 对齐处理

#### 32 字节对齐计算

```python
def align_to_32(size):
    """将大小对齐到 32 字节"""
    return ((size + 31) // 32) * 32

# 示例
actual_size = 4  # 逻辑上需要 4 字节
aligned_size = align_to_32(actual_size)  # 实际分配 32 字节
```

#### 单值缓冲区处理

```python
# 归约操作的单值结果（如均值、方差）
# 逻辑上只需要 4 字节（FP32）
# 但硬件要求 32 字节对齐

mean_value_size = 4  # 逻辑大小
mean_buffer_size = 32  # 实际分配大小

# 分配缓冲区
mean_buffer = allocate_ub(mean_buffer_size)  # 分配 32 字节
```

## 典型算子的 Tiling 策略

### 案例 1：LayerNorm

**算子特点**：
- 输入形状：[B, D]
- 在 D 维度上计算均值和方差
- 每个 batch 独立计算

**Tiling 策略**：

1. **核间切分**：按 batch 维度切分
   ```python
   batch_per_core = ceil(B / num_cores)
   ```

2. **核内切分**：
   - 单次处理 1 个 batch（因为需要完整的 D 维度计算均值和方差）
   - UB 需求：
     ```
     输入缓冲区(FP16): D × 2B
     升精度缓冲区(FP32): D × 4B
     均值缓冲区(FP32): 32B
     方差缓冲区(FP32): 32B
     Gamma 缓冲区(FP32): D × 4B
     Beta 缓冲区(FP32): D × 4B
     输出缓冲区(FP16): D × 2B
     ```

### 案例 2：Softmax

**算子特点**：
- 输入形状：[B, D]
- 在 D 维度上计算 exp 和归约
- 每个 batch 独立计算

**Tiling 策略**：

1. **核间切分**：按 batch 维度切分

2. **核内切分**：
   - 如果 D 较小（< 4096），单次处理多个 batch
   - 如果 D 较大，需要分块计算（复杂）
   - UB 需求：
     ```
     输入缓冲区(FP16): D × 2B
     升精度缓冲区(FP32): D × 4B
     Max 值缓冲区(FP32): 32B
     Sum 值缓冲区(FP32): 32B
     Exp 缓冲区(FP32): D × 4B
     输出缓冲区(FP16): D × 2B
     ```

### 案例 3：矩阵乘法

**算子特点**：
- 输入形状：A[M, K], B[K, N]
- 输出形状：C[M, N]
- 需要多次访问 B 矩阵

**Tiling 策略**：

1. **核间切分**：按 M 维度切分（每个 Core 计算输出的若干行）

2. **核内切分**：
   - 将 K 维度分块，每次加载部分 A 和 B
   - 累加部分结果
   - UB 需求：
     ```
     A 块缓冲区: tile_m × tile_k × 2B
     B 块缓冲区: tile_k × tile_n × 2B
     C 累加缓冲区: tile_m × tile_n × 4B
     ```

## Tiling 策略设计检查清单

### 核间切分检查

- [ ] 切分维度选择合理（考虑数据独立性）
- [ ] 负载均衡（每个 Core 处理的数据量相近）
- [ ] 无跨核通信（每个 Core 独立完成任务）
- [ ] 边界处理正确（最后一个 Core 的数据范围）

### 核内切分检查

- [ ] 所有缓冲区都已列出
- [ ] 缓冲区总大小 < UB 总大小
- [ ] 单值缓冲区分配 32B 空间
- [ ] 精度转换策略明确（是否需要升/降精度）
- [ ] 单次循环处理量计算正确

### 对齐检查

- [ ] UB 缓冲区地址 32 字节对齐
- [ ] 单值缓冲区分配 32B 空间
- [ ] 所有缓冲区大小都考虑了对齐

### 性能优化检查

- [ ] 减少 GM 访问次数
- [ ] 提高数据复用率
- [ ] 充分利用向量计算
- [ ] 避免不必要的精度转换

## 常见错误和解决方案

### 错误 1：UB 空间不足

**症状**：运行时错误，提示 UB 溢出

**原因**：
- 缓冲区总大小超过 UB 容量
- 忘记考虑对齐开销

**解决**：
1. 重新计算所有缓冲区大小
2. 减少单次循环处理的数据量
3. 考虑缓冲区复用

### 错误 2：对齐错误

**症状**：硬件错误或性能下降

**原因**：
- 缓冲区地址未对齐
- 单值缓冲区分配空间不足

**解决**：
1. 使用对齐函数计算地址
2. 单值缓冲区统一分配 32B

### 错误 3：负载不均衡

**症状**：部分 Core 提前完成，整体性能下降

**原因**：
- 数据量不能被 Core 数整除
- 切分策略不合理

**解决**：
1. 使用动态分配策略
2. 调整切分维度

### 错误 4：精度损失

**症状**：FP16 输入时结果不准确

**原因**：
- 归约操作未升精度
- 累加次数过多

**解决**：
1. 归约前升精度到 FP32
2. 使用 Kahan 求和等算法

## 参考资源
- Triton-ascend 编程优化指南（https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/migration_guide/performance_guidelines.md）