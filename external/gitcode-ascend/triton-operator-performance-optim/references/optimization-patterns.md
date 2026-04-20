# Ascend Triton 性能优化模式与代码样例

本文档包含 Triton 算子在 Ascend NPU 上的具体优化代码模式，供优化工作流中按需参考。

## 基础调优四板斧

### 1. Block Size 与 Grid Size

```python
# BLOCK_SIZE 必须匹配 UB 容量（192KB）
# FP16 数据：BLOCK_SIZE 建议 1024-2048
# 矩阵乘法：BLOCK_M/N/K 必须为 16 的倍数（Cube 单元粒度）

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
```

**检查点**：
- [ ] BLOCK_SIZE 是否适配 UB 容量？
- [ ] 矩阵运算 BLOCK 是否为 16 的倍数？
- [ ] Grid 维度是否映射到 AI Core 物理布局（推荐 1D Grid，且不超过物理核数）？

### 2. 强制向量化内存访问

```python
# 连续内存访问
offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 好
# offsets = block_start + tl.arange(0, BLOCK_SIZE) * 2  # 差：非连续

# Mask 防止越界（Ascend 对越界访问零容错）
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
```

**检查点**：
- [ ] 内存访问是否连续？
- [ ] 是否添加了 Mask？
- [ ] 地址是否 32 字节对齐？

### 3. UB 缓存与数据复用（核内再分块）

```python
# 当 BLOCK_SIZE 过大时，在核内再分块以适配 UB 容量
for sub_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = start + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
    x_chunk = tl.load(x_ptr + offsets, mask=mask)
    # ... 处理
    tl.store(y_ptr + offsets, y_chunk, mask=mask)
```

### 4. 编译时常量与循环展开

```python
# tl.constexpr 强制编译期确定
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    for i in tl.static_range(0, BLOCK_SIZE):  # 静态展开
        ...
```

## Ascend 硬件特化优化

### Cube 单元适配（矩阵乘法）

```python
# Cube 单元仅支持 16x16 基础粒度
BLOCK_M: tl.constexpr = 128  # 必须为 16 的倍数
BLOCK_N: tl.constexpr = 256
BLOCK_K: tl.constexpr = 64

# 精度策略：累加器用 FP32，写回 FP16
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)  # FP16
    b = tl.load(b_ptr + ...)  # FP16
    acc += tl.dot(a, b)       # FP32 累加
tl.store(c_ptr + ..., acc.to(tl.float16))  # 写回 FP16
```

### 数值稳定性（归约操作）

```python
# 归约前必须升精度到 FP32
x_fp32 = x.to(tl.float32)
mean = tl.sum(x_fp32, axis=-1) / D
var = tl.sum((x_fp32 - mean[:, None])**2, axis=-1) / D
```

### UB 空间管理

```python
# 计算 UB 需求
UB_SIZE = 192 * 1024  # 192KB
input_buffer = D * 2      # FP16
upcast_buffer = D * 4     # FP32
output_buffer = D * 2     # FP16
total = input_buffer + upcast_buffer + output_buffer

# 单值缓冲区必须 32B 对齐
mean_buffer = 32  # 不是 4B
```

## 高级优化技术

### 算子融合

将 Memory-Bound 操作转化为 Compute-Bound 操作：

```python
# 融合前：多次 GM 访问
x = load(x_ptr)      # GM → UB
y = relu(x)          # UB 计算
store(y_ptr, y)      # UB → GM
z = load(y_ptr)      # GM → UB（冗余！）
w = softmax(z)       # UB 计算
store(w_ptr, w)      # UB → GM

# 融合后：减少 GM 访问
x = load(x_ptr)
y = relu(x)
w = softmax(y)       # 直接复用 UB 中的 y
store(w_ptr, w)
```

### Double Buffer

```python
# 乒乓加载，隐藏访存延迟
# Buffer A 加载时，Buffer B 计算
# Buffer B 加载时，Buffer A 计算
```

## 精度保护模式

任何优化都不能破坏数值精度。以下是必须遵守的精度保护模式：

### 归约操作升精度

```python
# 所有归约操作（sum/mean/max/var）必须在 FP32 下进行
x_fp32 = x.to(tl.float32)
result = tl.sum(x_fp32, axis=-1)  # FP32 归约
```

### 矩阵乘法混合精度

```python
# 存储 FP16，累加 FP32，写回 FP16
a = tl.load(a_ptr)           # FP16 加载
b = tl.load(b_ptr)           # FP16 加载
acc = tl.dot(a, b)           # FP32 累加
tl.store(c_ptr, acc.to(tl.float16))  # FP16 写回
```

### 精度验证方法

```python
# 优化后必须对比 PyTorch-NPU 原生实现
torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-3)

# 测试多种输入规模，包括非对齐边界
for size in [127, 128, 255, 256, 1023, 1024, 4096]:
    verify_correctness(size)
```
