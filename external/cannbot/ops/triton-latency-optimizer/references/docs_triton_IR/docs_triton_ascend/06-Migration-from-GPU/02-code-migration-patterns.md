# 常见代码迁移模式

## 概述

将 Triton 算子从 GPU 迁移到 NPU 时，需要系统性地替换设备标识、调整 Grid 配置、修改参数设置和适配数据类型。本文总结了常见的代码迁移模式，每个模式都提供具体的 diff 示例，帮助开发者快速完成迁移工作。

## 关键概念

| 迁移项 | GPU 写法 | NPU 写法 | 说明 |
|--------|---------|---------|------|
| 设备标识 | `device='cuda'` | `device='npu'` | 全局替换 |
| 设备库导入 | `import torch` | `import torch_npu` | 需额外导入 NPU 适配库 |
| 设备获取 | `triton.runtime.driver.active.get_active_torch_device()` | 不需要 | NPU 无需显式获取设备 |
| Grid 配置 | `lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)` | 同左，但需对齐物理核数 | Grid 大小应 <= 物理核数 |
| num_warps | `num_warps=4/8/16` | 不使用 | NPU 无 Warp 概念 |
| num_stages | `num_stages=2/3/4` | 不使用 | NPU 无软件流水线概念 |
| 数据类型 | uint8/uint16/uint32/uint64/fp64 | 不支持 | 需替换为支持的类型 |
| Block Pointer stride | 可通过 stride 顺序实现转置 | 只能通过 order 参数实现转置 | 硬件限制 |
| tl.load 对齐 | 无严格对齐要求 | Vector: 32字节对齐; CV融合: 512字节对齐 | 硬件要求 |

## 详细内容

### 迁移模式 1：设备标识与导入替换

最基本的迁移步骤，将所有 GPU 相关的设备标识替换为 NPU。

```diff
 import torch
+ import torch_npu
 import triton
 import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()
- x = torch.rand(size, device='cuda')
- y = torch.rand(size, device='cuda')
+ x = torch.rand(size, device='npu')
+ y = torch.rand(size, device='npu')
```

**要点**：
- 必须导入 `torch_npu` 库，提供 NPU 设备支持
- 所有 `device='cuda'` 替换为 `device='npu'`
- 移除 `get_active_torch_device()` 调用，NPU 无需此逻辑
- 移除 GPU 设备一致性校验断言（`assert x.device == DEVICE`）

### 迁移模式 2：Grid 配置调整

GPU 上 Grid 可以自由定义，但 NPU 上 Grid 大小应与物理核数对齐。

```diff
 def add(x: torch.Tensor, y: torch.Tensor):
     output = torch.empty_like(x)
     n_elements = output.numel()
-    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
-    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
+    # 获取物理核数
+    from triton.runtime import driver
+    props = driver.active.utils.get_device_properties(0)
+    num_cores = props["num_aicore"]
+    grid = lambda meta: (num_cores,)
+    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=triton.cdiv(n_elements, num_cores))
     return output
```

**要点**：
- NPU Grid 优先使用 1D，2D 会被合并为 1D
- Grid 大小建议等于物理核数，避免分批调度开销
- 纯 Vector 算子：Grid = Vector Core 数量
- 含 tl.dot 算子：Grid = AI Core 数量
- 也可使用 `kernel[n, 1, 1](...)` 直接指定核数

### 迁移模式 3：num_warps 和 num_stages 参数移除

GPU 上 `num_warps` 和 `num_stages` 是重要的性能调优参数，但在 NPU 上无效。

```diff
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 128}),
+        triton.Config({'BLOCK_SIZE': 256}),
+        triton.Config({'BLOCK_SIZE': 512}),
     ],
     key=['n_elements'],
 )
 @triton.jit
 def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     ...
```

**要点**：
- `num_warps`：NPU 没有 Warp 概念，此参数无效
- `num_stages`：NPU 没有软件流水线概念，此参数无效
- NPU 特有的 autotune 参数：`multibuffer`（流水并行数据搬运）
- NPU autotune 示例：`triton.Config({'XS': 128, 'multibuffer': True})`

### 迁移模式 4：数据类型替换

NPU 不支持部分 GPU 数据类型，需要替换为等价的受支持类型。

```diff
 @triton.jit
 def compute_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < N
-    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float64)
-    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float64)
+    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
+    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
     output = x + y
     tl.store(output_ptr + offsets, output, mask=mask)

- x = torch.randn(size, dtype=torch.float64, device='cuda')
+ x = torch.randn(size, dtype=torch.float32, device='npu')
```

**NPU 不支持的数据类型及替换建议**：

| 不支持的类型 | 替换建议 | 说明 |
|-------------|---------|------|
| fp64 (float64) | fp32 (float32) | 精度降低，需评估影响 |
| uint8 | int8 | 范围不同，需注意符号位 |
| uint16 | int16 | 范围不同，需注意符号位 |
| uint32 | int32 | 范围不同，需注意符号位 |
| uint64 | int64 | 范围不同，需注意符号位 |

**性能相关数据类型优化**：

| 操作 | 不推荐类型 | 推荐类型 | 原因 |
|------|-----------|---------|------|
| Vector ADD | int64 | int32 | int64 退化为 scalar 运算 |
| Vector CMP | int64/int32 | fp32 | int64/int32 退化为 scalar 运算 |

### 迁移模式 5：tl.load/tl.store 参数调整

NPU 上 `tl.load`/`tl.store` 的部分参数行为与 GPU 不同。

```diff
 @triton.jit
 def kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
     idx = tl.arange(0, BLOCK_SIZE)
     mask = idx < M
-    data = tl.load(x_ptr + idx, mask=mask)
+    data = tl.load(x_ptr + idx, mask=mask, care_padding=False)
     output = data * 2.0
     tl.store(output_ptr + idx, output, mask=mask)
```

**关键参数差异**：

| 参数 | GPU | NPU | 说明 |
|------|-----|-----|------|
| `cache_modifier` | 支持（"ca"/"cg"） | 不支持 | NPU 无 PTX cache 语义 |
| `eviction_policy` | 支持 | 不支持 | NPU 无 PTX eviction 语义 |
| `volatile` | 支持 | 不支持 | NPU 无 PTX volatile 语义 |
| `care_padding` | 不存在 | 新增 | 设为 False 可提升并行度 |
| `other` | 默认填充 0 | 默认填充 0 | NPU 会先用 Vector 置零再 MTE2 搬运 |

**care_padding 优化说明**：

当 `tl.load` 加载的数据只能部分填充目标内存时，NPU 默认会先用 Vector 核将全部内存置为指定值（未指定 other 则置 0），再用 MTE2 指令搬运数据。这导致 MTE2 和 Vector 产生依赖，无法高效并行。如果未填充部分不影响后续计算结果，添加 `care_padding=False` 可去掉默认值填充，提升并行度。

### 迁移模式 6：核内 Tiling 分块（避免 UB 溢出）

GPU 上可以一次性加载大量数据到 Shared Memory，但 NPU 的 UB 容量有限（A2/A3 为 192 KB，910_95 为 256 KB），需要使用 for 循环进行核内 Tiling。

```diff
 @triton.jit
-def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
+def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
     pid = tl.program_id(axis=0)
-    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
-    mask = offsets < N
-    x = tl.load(x_ptr + offsets, mask=mask)
-    out = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
-    tl.store(out_ptr + offsets, out, mask=mask)
+    base_offset = pid * BLOCK_SIZE
+    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
+    for sub_idx in range(num_sub_blocks):
+        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
+        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
+        mask = offsets < N
+        x = tl.load(x_ptr + offsets, mask=mask)
+        out = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
+        tl.store(out_ptr + offsets, out, mask=mask)
```

**要点**：
- `BLOCK_SIZE`：核间切分大小，控制 coreDim
- `BLOCK_SIZE_SUB`：核内切分大小，控制 UB 使用量
- for 循环增加 Tiling 可使"数据搬入/计算/数据搬出"并行起来
- 需确保分块后数学等价性

### 迁移模式 7：完整向量加法迁移示例

综合以上所有迁移模式的完整示例：

```diff
 import torch
+ import torch_npu
 import triton
 import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()

 @triton.jit
 def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     block_start = pid * BLOCK_SIZE
     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(x_ptr + offsets, mask=mask)
     y = tl.load(y_ptr + offsets, mask=mask)
     output = x + y
     tl.store(output_ptr + offsets, output, mask=mask)

 def add(x: torch.Tensor, y: torch.Tensor):
     output = torch.empty_like(x)
-    assert x.device == DEVICE and y.device == DEVICE
     n_elements = output.numel()
     grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
     add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
     return output

 torch.manual_seed(0)
 size = 98432
- x = torch.rand(size, device='cuda')
+ x = torch.rand(size, device='npu')
- y = torch.rand(size, device='cuda')
+ y = torch.rand(size, device='npu')
 output_torch = x + y
 output_triton = add(x, y)
 print(f'Max diff: {torch.max(torch.abs(output_torch - output_triton))}')
```

### 迁移模式 8：Vector CMP 类型转换（性能优化）

NPU 的 Vector CMP 不支持 int64/int32，会导致 scalar 退化。需要手动转换为 fp32。

```diff
 @triton.jit
 def layernorm_kernel(X, Out, Mean, Rstd, M, N, eps, BLOCK_N: tl.constexpr):
     cols = tl.arange(0, BLOCK_N)
     x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
     mean = tl.sum(x, axis=0) / N
-    xbar = tl.where(cols < N, x - mean, 0.0)
+    cols_cmp = cols.to(tl.float32)
+    xbar = tl.where(cols_cmp < N, x - mean, 0.0)
     var = tl.sum(xbar * xbar, axis=0) / N
     rstd = 1 / tl.sqrt(var + eps)
     out = (x - mean) * rstd
     tl.store(Out + cols, out, mask=cols < N)
```

**要点**：
- `tl.load`/`tl.store` 中的 mask 使用 CMP 时，编译器通常可以自动优化为 Vector 操作
- `tl.where` 中的 CMP 需要手动转换类型
- 将 `cols`（int64）转换为 `cols_cmp`（fp32）后，CMP 操作可以使用 Vector 单元

### 迁移模式 9：离散访存优化为连续访存

GPU 上利用线程绑定最低维度的写法，在 NPU 上可能导致离散访存和 scalar 退化。

```diff
 # 离散访存（GPU 风格）
- block_ptr = tl.make_block_ptr(
-     base=input_ptr,
-     shape=(1024,),
-     strides=(32,),
-     offsets=(i_t * 16,),
-     block_shape=(BT,),
-     order=(0,)
- )

 # 连续访存（NPU 优化）
+ block_ptr = tl.make_block_ptr(
+     base=input_ptr,
+     shape=(1024, 32),
+     strides=(32, 1),
+     offsets=(i_t * BT, 0),
+     block_shape=(BT, 32),
+     order=(1, 0)
+ )
```

**要点**：
- 将一维 `(1024,)` 看成二维 `(1024, 32)`，最低维度 32 是连续的
- stride 从 `(32,)` 改为 `(32, 1)`，确保每个线程块访问连续的 32 个元素
- order 从 `(0,)` 改为 `(1, 0)`，先行后列，保证访存连续性

## NPU 适配要点

1. 迁移第一步始终是设备标识替换：`cuda` → `npu`，添加 `import torch_npu`
2. 移除所有 GPU 专属参数：`num_warps`、`num_stages`、`cache_modifier`、`eviction_policy`
3. Grid 配置应与物理核数对齐，避免分批调度
4. 注意 UB 容量限制，必要时使用核内 Tiling
5. 数据类型优先使用 fp32/int32，避免 scalar 退化
6. 访存模式优先连续，避免离散访存

## 常见问题（Q&A）

**Q1：迁移后算子能直接运行吗？**

A：简单的算子（如向量加法）通常只需替换设备标识即可运行。但涉及 Grid 分核数较大、UB 容量敏感、特殊数据类型的算子，需要额外调整。

**Q2：autotune 的 Config 需要修改吗？**

A：需要。移除 `num_warps` 和 `num_stages` 参数，可以添加 NPU 特有的 `multibuffer` 参数。`triton.Config({'XS': 128, 'multibuffer': True})`。

**Q3：为什么 care_padding=False 能提升性能？**

A：NPU 在 `tl.load` 时默认会先填充未覆盖的内存区域，这引入了 Vector 和 MTE2 之间的依赖。如果未覆盖区域不影响计算结果，设置 `care_padding=False` 可以消除这个依赖，提升指令并行度。

## 相关文档

- [01-架构差异](./01-architecture-differences.md)
- [03-迁移常见问题](./03-common-issues.md)
- [04-Block-Pointer-迁移注意事项](./04-block-pointer-migration.md)
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md)
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md)
