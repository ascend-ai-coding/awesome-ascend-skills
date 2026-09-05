# 迁移常见问题

## 概述

将 Triton 算子从 GPU 迁移到 NPU 后，常遇到两类核心问题：UB 溢出和 coreDim 超限。此外，精度差异、数据类型不支持、编译错误和运行时错误也是常见问题。本文以 Q&A 格式系统梳理迁移过程中的常见问题及解决方案。

## 关键概念

| 问题类型 | 典型错误信息 | 根本原因 |
|---------|-------------|---------|
| UB 溢出 | `ub overflow, requires xxxx bits while 1572864 bits available!` | 单次处理数据量超过 UB 容量（A2/A3: 192 KB，910_95: 256 KB） |
| coreDim 超限 | `coreDim=xxxx can't be greater than UINT16_MAX` | Grid 维度超过 NPU 硬件限制（65535） |
| 精度差异 | NPU 结果与 GPU/CPU 参考结果不一致 | 浮点计算顺序差异、数据类型退化 |
| 数据类型不支持 | 编译错误或运行时错误 | NPU 不支持 uint8/uint16/uint32/uint64/fp64 |
| Scalar 退化 | 性能大幅下降 | Vector 操作不支持的数据类型退化为标量计算 |
| 离散访存 | 性能下降或 UB 溢出 | 非连续内存访问模式 |

## 详细内容

### Q1：什么是 UB 溢出？如何解决？

**问题描述**：

NPU 的 UB（Unified Buffer）是 AI Core 的片上缓存，A2/A3 系列容量为 192 KB（1,572,864 bits），910_95 系列容量为 256 KB。当单个 AI Core 一次处理的数据量超过 UB 容量时，编译器会报错。

**典型错误信息**：

```
triton.compiler.errors.MLIRCompilationError:
[ConvertLinalgRToBinary] encounters error:
loc("kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
loc("kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available!
```

**解决方案**：

1. **使用 for 循环进行核内 Tiling**：将大数据块拆分为多个小块，每次处理一小块数据

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x * 2.0
        tl.store(out_ptr + offsets, out, mask=mask)
```

2. **减小 BLOCK_SIZE**：降低单次处理的数据量

3. **关闭 multibuffer**：multibuffer 需要额外 UB 空间，关闭后可释放部分空间

4. **注意 UB 总量限制**：A2/A3 系列开启 double buffer 时所有 tensor 总和不能超过 96 KB，关闭 double buffer 时不能超过 192 KB；910_95 系列分别为 128 KB 和 256 KB

### Q2：什么是 coreDim 超限？如何解决？

**问题描述**：

NPU 的 coreDim 参数不能超过 UINT16_MAX（65535）。当处理大规模数据且 BLOCK_SIZE 较小时，Grid 维度可能超过此限制。

**典型错误信息**：

```
coreDim=524288 can't be greater than UINT16_MAX
```

**计算公式**：

```
coreDim = ceil(N / BLOCK_SIZE)
需要满足：ceil(N / BLOCK_SIZE) <= 65535
即：BLOCK_SIZE >= ceil(N / 65535)
```

**解决方案**：

1. **设置环境变量 TRITON_ALL_BLOCKS_PARALLEL=1**：编译器自动调整逻辑核数量为物理核数

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

2. **增大 BLOCK_SIZE**：减少所需的核心数量

```python
N = x.numel()
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
```

**注意**：`TRITON_ALL_BLOCKS_PARALLEL` 要求 triton kernel 的逻辑对执行顺序不敏感，否则可能导致死锁。

### Q3：如何处理 coreDim 超限和 UB 溢出的复合问题？

**问题描述**：

增大 BLOCK_SIZE 解决 coreDim 超限后，可能导致 UB 溢出。例如：N = 1073741824，BLOCK_SIZE 从 4096 调整到 32768 后，coreDim 合规但 UB 溢出。

**解决方案**：

引入 `BLOCK_SIZE_SUB` 参数，将大块进一步细分：

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

MAIN_BLOCK_SIZE = 32768   # 确保 coreDim 合规
SUB_BLOCK_SIZE = 1024     # 控制 UB 使用量
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
masked_fill_kernel[grid](inp, expand_mask, value, out, N, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
```

### Q4：为什么会出现精度差异？如何排查？

**问题描述**：

NPU 运行结果与 GPU/CPU 参考结果存在数值差异。

**可能原因**：

1. 浮点计算顺序差异：SIMD 和 SIMT 的计算顺序不同，导致浮点累加结果有微小差异
2. 数据类型退化：int64 在 Vector ADD/CMP 中退化为 scalar，可能影响中间计算精度
3. bf16/fp16 精度差异：不同硬件的浮点实现可能有微小差异
4. tl.load 的 other 默认值填充行为差异

**排查方法**：

1. 使用解释器模式获取 CPU 基准结果：

```bash
export TRITON_INTERPRET=1
python your_triton_program.py
```

2. 使用 `tl.device_print` 打印中间结果：

```python
tl.device_print("intermediate value:", tmp)
```

3. 逐步对比：先对比输入，再对比每一步的中间结果

4. 检查数据类型：确保没有意外的类型转换

### Q5：NPU 不支持的数据类型如何处理？

**问题描述**：

NPU 不支持 uint8、uint16、uint32、uint64、fp64 等数据类型。

**解决方案**：

| 不支持类型 | 替换方案 | 注意事项 |
|-----------|---------|---------|
| uint8 | int8 | 注意符号位，范围 0-255 变为 -128-127 |
| uint16 | int16 | 注意符号位和范围 |
| uint32 | int32 | 注意符号位和范围 |
| uint64 | int64 | 注意符号位和范围 |
| fp64 | fp32 | 精度降低，需评估影响 |

**特别注意**：int64 在 Vector ADD 和 Vector CMP 操作中会退化为 scalar 运算，严重影响性能。建议在不影响精度的前提下使用 int32 或 fp32。

### Q6：为什么会出现离散访存导致的性能问题？

**问题描述**：

GPU 上正常的访存模式在 NPU 上性能大幅下降。

**根本原因**：

NPU 的 Vector Unit 对连续访存有强偏好。离散访存会导致：
1. 多次非对齐访存，增加搬运开销
2. 运算退化为 scalar 模式
3. UB 空间利用效率低

**典型场景**：

对 (64, 32) 二维数据搬运，对应 stride (12832, 128) 是非对齐访存。对齐数据的访存 stride 应为 (32, 1)。

**解决方案**：

1. 调整数据布局，使最内轴连续
2. 在最内轴新增大小为 1 的轴，变为 (64, 32, 4)，stride 为 (12832, 128, 1)
3. 使用 `tl.make_block_ptr` 时，确保 stride 反映真实的连续内存布局

### Q7：如何排查编译错误？

**问题描述**：

迁移后出现 `MLIRCompileError` 等编译错误。

**排查步骤**：

1. 启用调试转储：

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

2. 查看转储文件（`~/.triton/dump/` 目录下）：
   - `kernel.ttir.mlir`：Triton IR（编译输入）
   - `kernel.ttadapter.mlir`：适配器 IR（转换输出）

3. 启用 MLIR IR dump：

```bash
export MLIR_ENABLE_DUMP=1
python your_program.py
```

4. 使用解释器模式验证逻辑正确性：

```bash
export TRITON_INTERPRET=1
python your_program.py
```

### Q8：如何排查运行时错误？

**问题描述**：

编译通过但运行时出错。

**排查步骤**：

1. 检查数据类型是否受 NPU 支持
2. 检查 Grid 配置是否超过 coreDim 限制
3. 检查 UB 使用量是否超限
4. 使用 msProf 工具采集性能数据：

```bash
msprof op --kernel-name=target_kernel_name python3 your_program.py
```

5. 检查运行时日志中的错误信息

### Q9：为什么 UB 溢出报错中提到的 bits 和实际数据量不匹配？

**问题描述**：

报错信息中的 bits 需求远大于预期的数据量。

**可能原因**：

1. multibuffer 功能开启后会额外占用 UB 空间
2. 某些操作需要额外的临时 buffer
3. int8 类型由于特殊处理会占用更大的片上空间
4. 非对齐访存需要额外的 padding 空间

**解决方案**：

1. 关闭 multibuffer：`triton.Config({'XS': 128, 'multibuffer': False})`
2. 减小 BLOCK_SIZE_SUB
3. 确保访存对齐
4. 设置 `ENABLE_PRINT_UB_BITS=1` 获取当前 UB 占用量

### Q10：如何判断算子是否应该使用 Vector Core 数量还是 AI Core 数量？

**判断规则**：

| 算子类型 | 并发任务数 | 说明 |
|---------|-----------|------|
| 纯 Vector 算子（无 tl.dot） | Vector Core 数量 | 不涉及矩阵乘法 |
| CV 融合算子（含 tl.dot） | AI Core 数量（= Cube Core 数量） | Cube 和 Vector 协同工作 |

**获取核数的方法**：

```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties(0)
num_aicore = props["num_aicore"]
# 对于 A2/A3 系列，Vector Core 数量 = AI Core 数量 * 2
```

### Q11：为什么 NPU 上 tl.where 的性能比 GPU 差？

**问题描述**：

使用 `tl.where` 进行条件选择时，NPU 性能明显低于 GPU。

**可能原因**：

`tl.where` 中的比较操作如果使用了 int64/int32 类型的索引，Vector CMP 不支持这些类型，会退化为 scalar 运算。

**解决方案**：

将比较操作中的整数索引转换为 fp32：

```python
# 优化前
xbar = tl.where(cols < N, x - mean, 0.0)  # cols 是 int64，CMP 退化为 scalar

# 优化后
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # CMP 使用 Vector 单元
```

### Q12：如何获取当前 UB 占用量？

**方法**：

设置环境变量 `ENABLE_PRINT_UB_BITS=1`，编译时会输出当前 UB 占用量信息，供 inductor 等工具使用。

```bash
export ENABLE_PRINT_UB_BITS=1
python your_program.py
```

## NPU 适配要点

1. **UB 溢出是 NPU 最常见的编译错误**：A2/A3 系列 UB 为 192 KB，910_95 系列为 256 KB，需严格控制数据量
2. **coreDim 超限是第二大常见问题**：可通过增大 BLOCK_SIZE 或设置 TRITON_ALL_BLOCKS_PARALLEL=1 解决
3. **复合问题需要双重优化**：同时考虑 coreDim 和 UB 限制
4. **数据类型选择直接影响性能**：避免使用导致 scalar 退化的类型
5. **访存模式影响巨大**：优先连续访存，避免离散访存

## 相关文档

- [01-架构差异](./01-architecture-differences.md)
- [02-代码迁移模式](./02-code-migration-patterns.md)
- [04-Block-Pointer-迁移注意事项](./04-block-pointer-migration.md)
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md)
- [FAQ.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/FAQ.md)
