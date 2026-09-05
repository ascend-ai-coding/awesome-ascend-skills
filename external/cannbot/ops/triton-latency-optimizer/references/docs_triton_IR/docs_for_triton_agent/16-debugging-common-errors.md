# 常见编译/运行错误速查

## 触发条件

当你在以下场景中遇到问题时，查阅本文档：

- kernel 编译报 `MLIRCompilationError`
- 编译报 `ub overflow` 或 `local memory limit exceeded`
- 编译报 `coreDim > UINT16_MAX`
- 运行时报 `illegal memory access` 或 `misaligned address`
- 运行时报 `deadlock detected` 或 `barrier timeout`
- NPU 结果与参考结果差异过大
- kernel 执行超时或性能远低于预期
- 需要快速定位错误类型和解决方案

## 核心知识

### 错误分类速查表

| 错误类别 | 典型错误信息 | 发生阶段 | 严重程度 |
|---------|------------|---------|---------|
| UB 溢出 | `ub overflow, requires xxxx bits while 1572864 bits available!` | 编译/运行 | 阻断 |
| coreDim 超限 | `coreDim=xxxx can't be greater than UINT16_MAX` | 编译 | 阻断 |
| 数据类型不支持 | `unsupported type: uint8/fp64` | 编译 | 阻断 |
| 操作不支持 | `failed to legalize operation 'tt.dot'` | 编译 | 阻断 |
| 对齐错误 | `misaligned memory access` | 编译/运行 | 阻断 |
| 内存访问越界 | `illegal memory access` / `segfault` | 运行 | 阻断 |
| 死锁 | `deadlock detected` | 运行 | 阻断 |
| 同步超时 | `barrier timeout` / `event timeout` | 运行 | 阻断 |
| 精度异常 | 结果与参考值差异过大 | 运行 | 功能 |
| Scalar 退化 | Vector 流水利用率 <10% | 运行 | 性能 |
| 离散访存 | MTE2 搬运时间过长 | 运行 | 性能 |

### 编译错误定位方法

**第一步：确定错误发生阶段**

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

检查 `~/.triton/dump/` 目录：

| 文件状态 | 含义 | 下一步 |
|---------|------|--------|
| 无 `kernel.ttir.mlir` | Python Kernel -> TTIR 阶段失败 | 检查 kernel 语法 |
| 有 `ttir.mlir` 但无 `ttadapter.mlir` | TTIR -> TTAdapter 转换失败 | 检查操作是否受 NPU 支持 |
| 有 `ttadapter.mlir` | TTAdapter -> 二进制阶段失败 | 使用 `MLIR_ENABLE_DUMP=1` 定位失败 Pass |

**第二步：查看 Pass 详情**

```bash
export MLIR_ENABLE_DUMP=1
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py 2>mlir_dump.log
```

**第三步：手动编译 TTAdapter IR**

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

### 运行时错误定位方法

| 方法 | 环境变量/命令 | 适用场景 |
|------|-------------|---------|
| 解释器模式 | `TRITON_INTERPRET=1` | 精度问题、逻辑错误 |
| 设备打印 | `TRITON_DEVICE_PRINT=1` + `tl.device_print` | NPU 上的中间结果 |
| UB 占用分析 | `ENABLE_PRINT_UB_BITS=1` | UB 溢出排查 |
| 内存分析 | `TRITON_MEMORY_DISPLAY=1` | 内存使用详情 |
| msProf 上板 | `msprof op --kernel-name=xxx python3 xxx.py` | 性能瓶颈 |
| msProf 仿真 | `msprof op simulator --soc-version=Ascend910B3 ...` | 指令级分析 |

### 环境变量速查

| 场景 | 推荐环境变量组合 |
|------|----------------|
| 编译错误排查 | `TRITON_DEBUG=1` + `MLIR_ENABLE_DUMP=1` + `TRITON_DISABLE_CACHE=1` |
| 精度调试 | `TRITON_INTERPRET=1` + `TRITON_DISABLE_CACHE=1` |
| 运行时打印 | `TRITON_DEVICE_PRINT=1` + `TRITON_DISABLE_CACHE=1` |
| UB 溢出排查 | `ENABLE_PRINT_UB_BITS=1` + `TRITON_DEBUG=1` + `TRITON_DISABLE_CACHE=1` |
| coreDim 超限 | `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 深度编译调试 | `TRITON_DEBUG=1` + `MLIR_ENABLE_DUMP=1` + `TRITON_ENABLE_LLVM_DEBUG=1` + `LLVM_DEBUG_ONLY="isel"` |
| 性能分析 | `msprof op --kernel-name=xxx python3 xxx.py` |
| 编译加速 | `TRITON_ASCEND_COMPILE_SPEED_OPT=1` |

## 代码模式

### 错误 1：UB 溢出

**错误信息**：
```
ub overflow, requires 3072256 bits while 1572864 bits available!
(possible reason: large or block number is more than what user expect
due to multi-buffer feature is enabled and some ops need extra local buffer.)
```

**原因**：单个 AI Core 一次处理的数据量超过 UB 容量。

**UB 容量参考**：

| 硬件型号 | 单 buffer | double buffer | 说明 |
|---------|----------|--------------|------|
| A2/A3 系列 | 192 KB | 96 KB x 2 | 最常见 |
| 910_95 系列 | 256 KB | 128 KB x 2 | 较新型号 |

**解决方案**：

```python
# 方案 1：减小 BLOCK_SIZE
BLOCK_SIZE = 1024  # 从 32768 减小

# 方案 2：增加核内 Tiling（BLOCK_SIZE_SUB）
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_idx in range(num_sub_blocks):
        offsets = base_offset + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)

# 方案 3：关闭 multibuffer
triton.Config({'XS': 128, 'multibuffer': False})

# 方案 4：查看 UB 占用量
# export ENABLE_PRINT_UB_BITS=1
```

**注意**：int8 类型会占用更大的片上空间（约为其他类型的 2 倍），考虑使用 int32 替代。

### 错误 2：coreDim 超限

**错误信息**：
```
coreDim=524288 can't be greater than UINT16_MAX
```

**原因**：Grid 维度超过 NPU 硬件限制（65535）。`coreDim = ceil(N / BLOCK_SIZE)`。

**解决方案**：

```python
# 方案 1：环境变量（kernel 逻辑必须对执行顺序不敏感）
# export TRITON_ALL_BLOCKS_PARALLEL=1

# 方案 2：动态计算 BLOCK_SIZE
N = x.numel()
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
```

**警告**：`TRITON_ALL_BLOCKS_PARALLEL=1` 要求 kernel 逻辑对执行顺序不敏感，有原子操作或跨核同步时不能开启，否则会死锁。

### 错误 3：数据类型不支持

**错误信息**：
```
error: unsupported type: uint8 / uint64 / fp64
```

**原因**：NPU 不支持 uint8/uint16/uint32/uint64/fp64 等数据类型。

**解决方案**：

```python
# uint8 -> int32
x_int32 = x.to(tl.int32)

# uint64 -> int64
x_int64 = x.to(tl.int64)

# fp64 -> fp32
x_fp32 = x.to(tl.float32)
```

**NPU 支持的数据类型**：int8/16/32/64, fp16/32, bf16, bool（910_95 额外支持 fp8）

**NPU 不支持的数据类型**：uint8/16/32/64, fp64

### 错误 4：对齐错误

**错误信息**：
```
error: misaligned memory access
```

**原因**：Vector 算子要求 32 字节访存对齐，Cube-Vector 融合算子要求 512 字节对齐。

**解决方案**：

```python
# 方案 1：确保 BLOCK_SIZE * element_size 满足对齐要求
# fp32: BLOCK_SIZE 应为 8 的倍数（8 * 4B = 32B）
# fp16/bf16: BLOCK_SIZE 应为 16 的倍数（16 * 2B = 32B）
# Cube-Vector 融合: BLOCK_SIZE 应为 512B 对齐

# 方案 2：使用 tl.multiple_of 提示编译器对齐信息
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
offsets = tl.multiple_of(offsets, 128)  # 提示 offsets 是 128 的倍数
```

### 错误 5：dot 操作降级失败

**错误信息**：
```
error: failed to legalize operation 'tt.dot'
```

**原因**：dot 操作的输入数据类型不被支持，或 BLOCK_SIZE 不满足对齐要求。

**解决方案**：

```python
import triton.language.extra.cann.extension as al

# 方案 1：确保输入为 fp16/fp32/bf16/int8
a = tl.load(a_ptr + offsets).to(tl.float16)
b = tl.load(b_ptr + offsets).to(tl.float16)

# 方案 2：添加 compile_hint
a = al.compile_hint(a, "dot_pad_only_k")
b = al.compile_hint(b, "dot_pad_only_k")
result = tl.dot(a, b)

# 方案 3：检查 BLOCK_M 和 BLOCK_N 是否为合法值
# 推荐 512B 对齐的 BLOCK_SIZE：BLOCK_M=128, BLOCK_N=256, BLOCK_K=256
```

### 错误 6：Block Pointer 与复杂控制流搭配编译失败

**错误信息**：
```
error: failed to legalize operation 'tt.advance'
```

**原因**：`tl.advance` 与复杂循环/分支语句搭配时可能出现编译问题。

**解决方案**：

```python
# 错误写法：tl.advance 与复杂循环搭配
block_ptr = tl.make_block_ptr(...)
for i in range(N):
    x = tl.load(block_ptr)
    block_ptr = tl.advance(block_ptr, [BLOCK_SIZE])  # 可能编译失败

# 正确写法：重新创建 block_ptr 替代 advance
for i in range(N):
    offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_ptr = tl.make_block_ptr(base=base_ptr, shape=shape,
                                   strides=strides, offsets=[i * BLOCK_SIZE],
                                   block_shape=[BLOCK_SIZE], order=[0])
    x = tl.load(block_ptr)
```

### 错误 7：stride 顺序不支持

**错误信息**：
```
error: unsupported stride pattern for transpose
```

**原因**：NPU 不支持通过调整 stride 参数顺序实现转置语义。

**解决方案**：

```python
# 错误写法：通过 stride 实现转置
block_ptr = tl.make_block_ptr(base=ptr, shape=[M, N],
                               strides=[1, M],  # 转置 stride
                               offsets=[0, 0],
                               block_shape=[BLOCK_M, BLOCK_N],
                               order=[0, 1])

# 正确写法：使用 order 参数表达转置，stride 反映真实内存布局
block_ptr = tl.make_block_ptr(base=ptr, shape=[N, M],
                               strides=[M, 1],  # 真实内存布局
                               offsets=[0, 0],
                               block_shape=[BLOCK_N, BLOCK_M],
                               order=[1, 0])    # order 表达转置
```

### 错误 8：精度异常

**典型表现**：NPU 运行结果与 PyTorch/CPU/GPU 参考结果差异过大。

**排查步骤**：

```python
# 第一步：解释器模式验证逻辑
# export TRITON_INTERPRET=1
# export TRITON_DISABLE_CACHE=1

# 第二步：在 NPU 上打印中间结果
# export TRITON_DEVICE_PRINT=1
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.device_print("x after load: ", x)
    result = tl.sqrt(x)
    tl.device_print("result after sqrt: ", result)
    tl.store(out_ptr + offsets, result, mask=mask)
```

**常见精度问题原因**：

| 原因 | 表现 | 解决方案 |
|------|------|---------|
| 浮点计算顺序差异 | 微小差异（1e-6 量级） | 正常现象，使用 `torch.allclose` |
| 数据类型退化 | 较大差异 | 检查 int64/int32 是否导致 Scalar 退化 |
| bf16/fp16 精度损失 | 中等差异 | 关键计算在 fp32 下进行 |
| mask 处理差异 | 特定位置差异 | 检查 boundary_check 和 mask 逻辑 |
| tl.exp 近似 | Softmax 差异 | NPU 上 exp 是近似计算，差异 1e-7 量级属正常 |

**fp16 精度不够的通用解决方案**：

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    # kernel 内部使用 fp32 精度计算
    x_fp32 = x.to(tl.float32)
    result = tl.sqrt(x_fp32)
    # 写回时转换
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
```

### 错误 9：Scalar 退化

**典型表现**：Vector 流水利用率极低（<10%），Scalar 流水成为瓶颈。

**常见 Scalar 退化原因**：

| 操作 | 不支持的数据类型 | 退化行为 |
|------|----------------|---------|
| Vector ADD | int64 | 退化为 Scalar 加法 |
| Vector CMP | int64, int32 | 退化为 Scalar 比较 |

**解决方案**：

```python
# 优化前：cols < N 导致 Scalar CMP（cols 为 int64）
cols = tl.arange(0, BLOCK_N)
xbar = tl.where(cols < N, x - mean, 0.0)

# 优化后：转换为 fp32 使用 Vector CMP
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

### 错误 10：device_print 缺少 prefix

**错误信息**：
```
error: device_print requires a prefix string
```

**解决方案**：

```python
# 错误写法
tl.device_print(x)

# 正确写法
tl.device_print("x: ", x)
```

### 错误 11：Tensor 形状为空

**错误信息**：
```
error: all shapes must have size >= 1
```

**原因**：NPU 不支持 shape 中某个维度 size 小于 1 的 tensor。

**解决方案**：

```python
# 在调用 kernel 前检查数据规模
if x.numel() == 0:
    return  # 跳过空 tensor
kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
```

### 错误 12：bool 类型不支持

**错误信息**：
```
error: unsupported type: bool
```

**原因**：部分操作不支持 bool 类型。

**解决方案**：Triton 内部会将 bool 转为 int8 进行运算，通常无需手动处理。如果特定操作报错，显式转换：

```python
mask_int8 = mask.to(tl.int8)
```

### 错误 13：atomic_add 多核累加不支持

**错误信息**：
```
error: atomic_add not supported for multi-core accumulation
```

**解决方案**：

```python
# 方案 1：使用 atomic_cas 自旋锁保护共享缓冲区
while True:
    old = tl.atomic_cas(lock_ptr, 0, 1)
    if old == 0:
        break
# 临界区：普通 add
tl.store(out_ptr + offsets, current_val + new_val)
# 释放锁
tl.atomic_xchg(lock_ptr, 0)

# 方案 2：两阶段归约策略
# 阶段 1：每个 Block 写入自己的部分结果
# 阶段 2：单独的 kernel 做最终归约
```

### 错误 14：MLIR Pass 失败

**错误信息**：
```
error: Failed to run BishengHIR pipeline
```

**解决方案**：

```bash
# 使用 MLIR_ENABLE_DUMP=1 查看具体哪个 Pass 失败
export MLIR_ENABLE_DUMP=1
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py 2>mlir_dump.log

# 手动编译 TTAdapter IR 进行调试
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true
```

### 错误 15：运行时死锁

**错误信息**：
```
deadlock detected
```

**常见原因与解决方案**：

| 原因 | 解决方案 |
|------|---------|
| `atomic_cas` 自旋锁未释放 | 确保锁的获取和释放在同一执行路径上，避免条件分支导致锁未释放 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` 与同步操作冲突 | 有同步操作时不能开启此选项 |
| `sync_block_set/wait` 不配对 | 确保每个 set 有对应 wait，且参数一致 |
| 条件分支中获取锁 | 避免在 if/else 分支中获取锁，确保所有路径都能释放锁 |

### 错误 16：NPU 执行超时

**可能原因**：

1. Grid 分核数过多，分批调度开销大
2. 死锁（kernel 逻辑对执行顺序敏感）
3. 数据规模异常大

**解决方案**：

```python
# 方案 1：Grid 对齐物理核数
num_aicore = triton.runtime.get_npu_properties()["num_aicore"]
grid = (min(num_aicore, triton.cdiv(N, BLOCK_SIZE)),)

# 方案 2：检查 kernel 是否存在跨核依赖
# 方案 3：减小数据规模进行测试
```

### 错误 17：结果全零或全 NaN

**排查方法**：

```python
# 使用 device_print 逐步排查
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.device_print("x: ", x)  # 检查数据是否正确加载
    result = tl.sqrt(x)
    tl.device_print("result: ", result)  # 检查是否有 NaN
    tl.store(out_ptr + offsets, result, mask=mask)
```

**常见原因**：
- 数据未正确加载（mask 错误）
- 数据类型转换错误
- 除零操作或 0 的负数次幂
- 使用 `tl.where` 保护危险操作：

```python
# 安全的 sqrt
result = tl.where(x > 0, tl.sqrt(x), 0.0)

# 安全的除法
result = tl.where(denom != 0, numer / denom, 0.0)
```

## 910_95 特别注意

### UB 容量差异

| 硬件型号 | 单 buffer | double buffer |
|---------|----------|--------------|
| A2/A3 | 192 KB | 96 KB x 2 |
| 910_95 | 256 KB | 128 KB x 2 |

910_95 的 UB 容量更大，UB 溢出的阈值更高。但 int8 类型在 910_95 上同样会占用更大的片上空间。

### multibuffer 默认关闭

910_95 上 multibuffer 默认关闭（A2/A3 默认开启）。如果遇到 UB 溢出，检查是否手动开启了 multibuffer：

```python
# 910_95 上默认不需要 multibuffer，除非有特殊需求
triton.Config({'XS': 128, 'multibuffer': False})  # 默认行为
```

### L0C -> UB 直通（FixPipe）

910_95 支持 L0C -> UB 直通通路（通过 FixPipe），A2/A3 不支持。这意味着 Cube 计算结果可以直接搬入 UB，减少了中间搬运步骤。但也改变了同步的 Pipe 组合，可能影响同步行为。

### fp8 数据类型

910_95 额外支持 fp8 数据类型（A2/A3 不支持）。使用 fp8 时注意：
- fp8 的动态范围更小，更容易溢出
- 确保计算在 fp32 下进行，只在存储时转换为 fp8

### SIMT 模式

910_95 支持 SIMT 模式（A2/A3 不支持）。SIMT 模式下每个 Scalar Core 可以独立执行不同的指令路径，减少了分支发散的影响。

### 同步机制差异

910_95 采用 Reg-based 同步架构（SetFlag/WaitFlag），而非 A2/A3 的 FFTS 机制。在 910_95 上：
- 不需要设置 `ffts_base_addr`
- 跨核同步更轻量，延迟更低
- 同步语义与 FFTS 一致，Triton 层 API 无需修改

### 编译目标差异

手动编译 TTAdapter IR 时，注意 `--target` 参数：

```bash
# A2/A3
bishengir-compile kernel.ttadapter.mlir --target=Ascend910B3 ...

# 910_95
bishengir-compile kernel.ttadapter.mlir --target=Ascend910_95 ...
```

## 相关文档

- [01-debug-overview.md](../docs_triton_ascend/07-Debugging/01-debug-overview.md) - 调试方法总览
- [02-interpreter-mode.md](../docs_triton_ascend/07-Debugging/02-interpreter-mode.md) - 解释器模式调试
- [03-compile-errors.md](../docs_triton_ascend/07-Debugging/03-compile-errors.md) - 编译错误排查
- [04-runtime-errors.md](../docs_triton_ascend/07-Debugging/04-runtime-errors.md) - 运行时错误排查
- [05-environment-variables.md](../docs_triton_ascend/07-Debugging/05-environment-variables.md) - 调试相关环境变量
- [03-error-codes.md](../docs_triton_ascend/09-Reference/03-error-codes.md) - 错误码参考
- [04-env-variables.md](../docs_triton_ascend/09-Reference/04-env-variables.md) - 环境变量完整参考
- [05-faq.md](../docs_triton_ascend/09-Reference/05-faq.md) - 常见问题速查
