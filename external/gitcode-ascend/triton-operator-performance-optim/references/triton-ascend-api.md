# Triton-Ascend 特有 API 与高性能实现

本文档总结 Triton-Ascend 后端特有的 API 和高性能实现模式。

## Ascend 特化的内存管理

### UB（Unified Buffer）约束

```python
# UB 容量：192KB（A2/A3）
UB_SIZE = 192 * 1024

# 计算缓冲区需求时必须考虑：
# 1. 数据类型大小
# 2. 32 字节对齐
# 3. 所有中间缓冲区

def calculate_ub_requirement(D, dtype='fp16'):
    type_sizes = {'fp16': 2, 'fp32': 4, 'bf16': 2}
    element_size = type_sizes[dtype]
    
    # 输入缓冲区
    input_size = D * element_size
    
    # 升精度缓冲区（归约操作需要）
    upcast_size = D * 4  # 总是 FP32
    
    # 输出缓冲区
    output_size = D * element_size
    
    # 单值缓冲区（32B 对齐）
    scalar_size = 32
    
    total = input_size + upcast_size + output_size + scalar_size
    return total
```

### L1 Buffer（Cube Core 专用）

```python
# L1 容量：1MB（比 UB 大）
# 用于矩阵乘法中的数据缓存

# 在矩阵乘法中：
# - A 矩阵块：存储在 L1
# - B 矩阵块：存储在 L1
# - C 累加器：存储在 UB 或 L1
```

## Ascend 特化的编译提示

### num_warps 和 num_stages

- **作用**：在 Ascend 上，这两者无效

## 高性能实现模式

### 模式 1：矩阵乘法（GEMM）

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Ascend 优化的矩阵乘法：
    1. BLOCK_M/N/K 必须为 16 的倍数（Cube 单元粒度）
    2. 累加器使用 FP32（数值稳定性）
    3. 写回使用 FP16（存储效率）
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的位置
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    
    # 初始化累加器（FP32）
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # K 维度分块累加
    for k in range(0, K, BLOCK_K):
        # 加载 A 块（FP16）
        a = tl.load(a_ptr + (rm + tl.arange(0, BLOCK_M)[:, None]) * stride_am +
                          (k + tl.arange(0, BLOCK_K)[None, :]) * stride_ak)
        
        # 加载 B 块（FP16）
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K)[:, None]) * stride_bk +
                          (rn + tl.arange(0, BLOCK_N)[None, :]) * stride_bn)
        
        # 矩阵乘法（触发 Cube 单元）
        acc += tl.dot(a, b)
    
    # 写回结果（FP16）
    tl.store(c_ptr + (rm + tl.arange(0, BLOCK_M)[:, None]) * stride_cm +
                    (rn + tl.arange(0, BLOCK_N)[None, :]) * stride_cn,
             acc.to(tl.float16))
```

### 模式 2：LayerNorm

```python
@triton.jit
def layernorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ascend 优化的 LayerNorm：
    1. 归约前升精度到 FP32
    2. 单次遍历计算均值和方差（减少 GM 访问）
    3. 使用 Welford 算法提高数值稳定性
    """
    pid = tl.program_id(0)
    
    # 计算当前行
    row_start = pid * N
    
    # 加载一行数据
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_start + offsets, mask=mask).to(tl.float32)
    
    # 计算均值（Welford 算法）
    mean = tl.sum(x, axis=0) / N
    
    # 计算方差
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # 归一化
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # 应用 gamma 和 beta
    gamma = tl.load(gamma_ptr + offsets, mask=mask).to(tl.float32)
    beta = tl.load(beta_ptr + offsets, mask=mask).to(tl.float32)
    y = x_norm * gamma + beta
    
    # 写回
    tl.store(y_ptr + row_start + offsets, y.to(tl.float16), mask=mask)
```

### 模式 3：Softmax（Online Softmax）

```python
@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ascend 优化的 Softmax（Online 算法）：
    1. 单次遍历计算 max 和 sum
    2. 避免多次 GM 访问
    3. 数值稳定
    """
    pid = tl.program_id(0)
    row_start = pid * N
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 加载一行
    x = tl.load(x_ptr + row_start + offsets, mask=mask).to(tl.float32)
    
    # Online Softmax: 单次遍历
    # max_val = max(x)
    max_val = tl.max(x, axis=0)
    
    # exp(x - max)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    
    # sum(exp)
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 归一化
    y = exp_x / sum_exp
    
    # 写回
    tl.store(y_ptr + row_start + offsets, y.to(tl.float16), mask=mask)
```

### 模式 4：Flash Attention（简化版）

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Ascend 优化的 Flash Attention：
    1. 分块计算 QK^T
    2. 在线 Softmax
    3. 减少中间结果存储
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 当前 Q 块的位置
    q_start = pid_b * H * S * D + pid_h * S * D + pid_m * BLOCK_M * D
    
    # 加载 Q 块
    q = tl.load(q_ptr + q_start + 
                tl.arange(0, BLOCK_M)[:, None] * D +
                tl.arange(0, BLOCK_D)[None, :])
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    sum_exp = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 遍历 K、V 块
    for n in range(0, S, BLOCK_N):
        # 加载 K、V 块
        k = tl.load(k_ptr + pid_b * H * S * D + pid_h * S * D +
                    n + tl.arange(0, BLOCK_N))
        v = tl.load(v_ptr + ...)
        
        # 计算 QK^T
        qk = tl.dot(q, k.T)
        
        # Online Softmax 更新
        new_max = tl.maximum(max_score, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - new_max[:, None])
        new_sum = sum_exp * tl.exp(max_score - new_max) + tl.sum(exp_qk, axis=1)
        
        # 更新累加器
        acc = acc * (sum_exp / new_sum)[:, None] + \
              tl.dot(exp_qk / new_sum[:, None], v)
        
        max_score = new_max
        sum_exp = new_sum
    
    # 写回
    tl.store(o_ptr + ..., acc.to(tl.float16))
```

## 性能调优技巧

### 技巧 1：避免 CPU-NPU 同步

```python
# 差：在热路径中使用 item()
for tensor in tensors:
    value = tensor.item()  # 每次触发同步！

# 好：批量操作
values = [t.item() for t in tensors]  # 单次批量同步

# 更好：保持在设备上
max_value = torch.max(tensor)  # 无同步
if max_value > threshold:      # 设备上比较
    ...
```

### 技巧 2：内存对齐优化

```python
# Ascend 要求 32 字节对齐
def align_to_32(size):
    return ((size + 31) // 32) * 32

# 分配缓冲区时
buffer_size = align_to_32(actual_size)
```

### 技巧 3：数据布局优化

```python
# Ascend Cube 单元偏好特定布局
# 矩阵乘法：推荐使用 ND（NCHW/NHWC）布局

# 避免频繁的转置操作
# 差：在 kernel 中转置
x_t = tl.trans(x)  # 额外开销

# 好：预先转置数据
x_t = x.T.contiguous()  # 在 kernel 外完成
```

### 技巧 4：精度策略

```python
# 原则：
# 1. 存储用 FP16（节省内存和带宽）
# 2. 计算用 FP32（数值稳定）
# 3. 累加用 FP32（避免精度损失）

# 示例：矩阵乘法
a_fp16 = tl.load(a_ptr)           # FP16 加载
b_fp16 = tl.load(b_ptr)           # FP16 加载
acc_fp32 = tl.dot(a_fp16, b_fp16) # FP32 累加
tl.store(c_ptr, acc_fp32.to(tl.float16))  # FP16 写回
```

## 常见陷阱与解决方案

### 陷阱 1：UB 溢出

```python
# 问题：BLOCK_SIZE 过大
BLOCK_SIZE = 8192  # 太大！

# 解决：核内再分块
BLOCK_SIZE = 2048
SUB_BLOCK_SIZE = 512

for sub in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    # 处理 SUB_BLOCK_SIZE 个元素
    ...
```

### 陷阱 2：Cube 单元未命中

```python
# 解决：使用dot进行矩阵乘计算
c = tl.dot(a, b) # 调用Cube单元进行计算
```

### 陷阱 3：数值溢出

```python
# 问题：FP16 范围有限
x = tl.load(x_ptr)  # FP16
y = x * 1000        # 可能溢出！

# 解决：升精度计算
x_fp32 = tl.load(x_ptr).to(tl.float32)
y_fp32 = x_fp32 * 1000
y = y_fp32.to(tl.float16)
```

### 陷阱 4：非连续访存

```python
# 问题：地址跳跃访问
offsets = tl.arange(0, BLOCK_SIZE) * 2  # 非连续

# 解决：使用连续内存布局
offsets = tl.arange(0, BLOCK_SIZE)  # 好
```

### 陷阱 5：不支持python风格的slice操作

```python
# 问题：python风格的slice操作不支持
subx = x[1:3] # 编译错误
y[2:4] = suby # 编译错误

# 解决：使用triton.language.extra.cann.extension中的操作
subx = extension.extract_slice(
    x,
    offsets=(1,),
    sizes=(2,),
    strides=(1,),  # 只能是包含1的tuple
)  # 好
y = extension.insert_slice(
    y,
    suby,
    offsets=(2,),
    sizes=(2,),
    strides=(1,),  # 只能是包含1的tuple
)  # 好
```

### 陷阱 6：多维以及超过物理核数的grid对性能无益

```python
# 问题：多维以及超过物理核数的grid对性能无益
grid=(2,2,2) # 性能与2*2*2=8相同
grid=(1024,) # 超过物理核数，总时延引入了host调度开销

# 解决：使用小于等于物理核数的grid
core_num = driver.active.utils.get_device_properties("npu")["num_aicore"] # 如果是包含tl dot的算子
core_num = driver.active.utils.get_device_properties("npu")["num_vectorcore"] # 其余类型的算子
grid=(core_num,) # 好
```

### 陷阱 7：tl.where的mask参数计算过程包含整形的比较操作

```python
# 问题：tl.where的mask参数计算过程包含int的比较操作
x = tl.load(x_ptr) # 假设int32
mask = x > 0 # 坏
res = tl.where(mask, a, b)

# 解决：先cast为浮点类型
x = tl.load(x_ptr) # 假设int32
mask = x.to(tl.float32) > 0 # 好
res = tl.where(mask, a, b)

## 参考资源

- [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)
- [Triton-Ascend 教程](https://gitcode.com/Ascend/triton-ascend/tree/main/python/tutorials)
- [Triton-Ascend 性能指南](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/migration_guide/performance_guidelines.md)
