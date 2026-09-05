# GPU Triton → Ascend 910_95 NPU 迁移概览

## 触发条件

当 Agent 收到将 GPU Triton 算子迁移到 NPU 的任务时，按本文档执行迁移。典型场景：

- 用户提供了在 `device='cuda'` 上运行的 Triton kernel，要求适配到 Ascend NPU
- 用户要求将已有 GPU Triton 算子移植到 910_95 平台
- 用户遇到 GPU→NPU 迁移后的编译/运行错误，需要排查

---

## 核心知识：GPU vs NPU 架构差异速查表

### 计算核心

| 维度 | GPU (NVIDIA) | NPU (Ascend 910_95) |
|------|-------------|---------------------|
| 计算核心 | SM (Streaming Multiprocessor) | AI Core (Cube + Vector) |
| 执行模型 | SIMT（单指令多线程） | SIMD（单指令多数据）；910_95 额外支持 SIMT 模式 |
| 矩阵计算单元 | Tensor Core | Cube Unit |
| 向量计算单元 | CUDA Cores (128/SM) | Vector Unit (1:2 配比，1 Cube + 2 Vector) |
| 最小调度单位 | Warp（32 线程） | AI Core（一个 Block 绑定一个 AI Core） |

### 内存层次

| 层级 | GPU | NPU (910_95) | 关键差异 |
|------|-----|-------------|---------|
| 全局内存 | Global Memory (HBM, 40-80 GB) | GM (32-64 GB) | 容量相近 |
| 片上共享内存 | Shared Memory (48-164 KB/SM) | UB (256 KB/AI Core) | UB 由编译器自动管理，不能手动控制 |
| 片上私有缓存 | L1 Cache（与 Shared Memory 共享） | L1 Buffer（与 UB 独立） | NPU UB/L1 分离，GPU 共享 |
| 寄存器 | Register File (256 KB/SM) | RF 128KB + DCache 32~120KB（SIMT 模式） | 910_95 SIMT 模式有独立 RF |

### 并行调度

| 维度 | GPU | NPU (910_95) |
|------|-----|-------------|
| 线程组织 | Thread → Warp → Block → Grid | 无线程概念（SIMD）；SIMT 模式下有线程概念 |
| Grid 本质 | 逻辑任务维度，与物理核解耦 | 物理核组映射，绑定 AI Core 拓扑 |
| Grid 维度限制 | 无硬限制 | Grid 大小 ≤ AI Core 总数，coreDim ≤ 65535 |
| 超核调度 | 硬件自动调度，开销小 | 分批调度，额外设备侧开销大 |
| num_warps | 有效（控制并行度） | 无效（无 Warp 概念） |
| num_stages | 有效（软件流水线） | 无效（无软件流水线概念） |

### 数据搬运流水线

```
GPU:  GM → Shared Memory → Register → 计算 → Register → GM
NPU:  GM ──MTE2──> UB ──Vector/Cube──> UB ──MTE3──> GM
                    ↑                      |
                    └── multibuffer 双缓冲 ──┘
```

- NPU 使用专用 DMA 引擎（MTE2/MTE3）搬运数据
- `multibuffer` 是 NPU 特有的流水并行优化，需额外 UB 空间（双缓冲时可用空间减半）

### 数据类型支持

| 数据类型 | GPU | NPU (910_95) | 备注 |
|---------|-----|-------------|------|
| fp16 / bf16 / fp32 / int8 / int16 / int32 | 支持 | 支持 | - |
| fp8 | 支持 | 支持 | 910_95 支持 FP8 类型转换和 dot_scaled |
| int64 | 支持 | 支持（Vector ADD/CMP 退化为 scalar） | 性能严重下降 |
| fp64 | 支持 | 不支持 | 需替换为 fp32 |
| uint8 | 支持 | 部分不支持（Block Pointer 场景不支持） | 需替换为 int8 |
| uint16/uint32/uint64 | 支持 | 不支持 | 需替换为对应 int 类型 |

---

## 迁移检查清单：必须修改的代码模式

### 1. 设备标识替换

```diff
 import torch
+ import torch_npu
 import triton
 import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()
- x = torch.rand(size, device='cuda')
+ x = torch.rand(size, device='npu')
```

- 全局替换 `device='cuda'` → `device='npu'`
- 必须导入 `import torch_npu`
- 移除 `get_active_torch_device()` 调用和设备一致性校验断言

### 2. 移除 num_warps / num_stages

```diff
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 128}),
+        triton.Config({'BLOCK_SIZE': 256}),
     ],
     key=['n_elements'],
 )
```

- `num_warps`：NPU 无 Warp 概念，此参数无效
- `num_stages`：NPU 无软件流水线概念，此参数无效
- NPU 特有 autotune 参数：`multibuffer`（流水并行数据搬运）

### 3. Grid 配置调整

```diff
 def add(x: torch.Tensor, y: torch.Tensor):
     output = torch.empty_like(x)
     n_elements = output.numel()
-    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
-    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
+    from triton.runtime import driver
+    props = driver.active.utils.get_device_properties(0)
+    num_cores = props["num_aicore"]
+    grid = lambda meta: (num_cores,)
+    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=triton.cdiv(n_elements, num_cores))
     return output
```

- Grid 优先使用 1D（2D 会被合并为 1D）
- Grid 大小建议等于物理核数，避免分批调度开销
- 纯 Vector 算子：Grid = Vector Core 数量
- 含 `tl.dot` 算子：Grid = AI Core 数量
- 也可使用 `kernel[n, 1, 1](...)` 直接指定核数

### 4. 数据类型替换

```diff
-    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float64)
+    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
```

| 不支持类型 | 替换方案 | 注意事项 |
|-----------|---------|---------|
| fp64 | fp32 | 精度降低，需评估影响 |
| uint8 | int8 | 范围 0-255 变为 -128-127 |
| uint16/uint32/uint64 | int16/int32/int64 | 注意符号位和范围 |

### 5. tl.load / tl.store 参数调整

```diff
-    data = tl.load(x_ptr + idx, mask=mask)
+    data = tl.load(x_ptr + idx, mask=mask, care_padding=False)
```

| 参数 | GPU | NPU | 说明 |
|------|-----|-----|------|
| `cache_modifier` | 支持 | 不支持 | 移除 |
| `eviction_policy` | 支持 | 不支持 | 移除 |
| `volatile` | 支持 | 不支持 | 移除 |
| `care_padding` | 不存在 | 新增 | 设为 False 可提升并行度 |

> care_padding 详细说明见 [08-data-type-precision.md](08-data-type-precision.md)。

### 6. 核内 Tiling 分块（避免 UB 溢出）

```diff
 @triton.jit
-def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
+def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
     pid = tl.program_id(axis=0)
-    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
-    mask = offsets < N
-    x = tl.load(x_ptr + offsets, mask=mask)
-    out = x * 2.0
-    tl.store(out_ptr + offsets, out, mask=mask)
+    base_offset = pid * BLOCK_SIZE
+    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
+    for sub_idx in range(num_sub_blocks):
+        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
+        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
+        mask = offsets < N
+        x = tl.load(x_ptr + offsets, mask=mask)
+        out = x * 2.0
+        tl.store(out_ptr + offsets, out, mask=mask)
```

- `BLOCK_SIZE`：核间切分大小，控制 coreDim
- `BLOCK_SIZE_SUB`：核内切分大小，控制 UB 使用量
- for 循环增加 Tiling 可使"数据搬入/计算/数据搬出"并行

### 7. Block Pointer 行为差异

```diff
 # GPU 写法：通过 stride 交换实现转置（NPU 不支持）
 block_ptr = tl.make_block_ptr(
     base=x_ptr,
     shape=(M, N),
-    strides=(1, M),      # stride 交换
+    strides=(N, 1),      # stride 反映真实内存布局
     offsets=(0, 0),
     block_shape=(BLOCK_M, BLOCK_N),
-    order=(0, 1),
+    order=(1, 0),        # 通过 order 表达转置语义
 )
```

- **stride 必须反映真实内存布局**，不能通过交换 stride 实现转置
- **转置语义只能通过 order 参数表达**
- `tl.advance` 与复杂循环/分支搭配可能编译失败，改用重新创建 `tl.make_block_ptr`
- Block Pointer 不支持 uint8/uint16/uint32/uint64/fp64

### 8. 离散访存 → 连续访存优化

```diff
 # 离散访存（GPU 风格，NPU 性能差）
 block_ptr = tl.make_block_ptr(
     base=input_ptr,
-    shape=(1024,),
-    strides=(32,),
-    offsets=(i_t * 16,),
-    block_shape=(BT,),
-    order=(0,)
+    shape=(1024, 32),
+    strides=(32, 1),
+    offsets=(i_t * BT, 0),
+    block_shape=(BT, 32),
+    order=(1, 0)
 )
```

- 将一维展平数据视为二维，确保最低维度连续
- stride 最低维度为 1 表示连续访存

### 9. Vector CMP 类型转换

```diff
-    xbar = tl.where(cols < N, x - mean, 0.0)   # cols 是 int64，CMP 退化为 scalar
+    cols_cmp = cols.to(tl.float32)
+    xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # CMP 使用 Vector 单元
```

- `tl.where` 中的比较操作如果使用 int64/int32 索引，Vector CMP 不支持，退化为 scalar
- 将整数索引转换为 fp32 后可使用 Vector 单元

---

## 910_95 特别注意

| 特性 | 910_95 值 | 与 910B 差异 |
|------|----------|-------------|
| UB 容量 | 256KB（可用 248KB） | 910B 为 192KB |
| L0C→UB 直通 | 支持（FixPipe） | 910B 不支持 |
| MultiBuffer | 默认关闭 | 910B 默认开启 |
| FP8 | 支持 dot_scaled | 910B 不支持 |
| SIMT 模式 | 支持 | 910B 不支持 |
| 同步方式 | SetFlag/WaitFlag | 910B 使用 FFTS |

> 完整硬件规格详见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 常见迁移错误速查

| 错误类型 | 典型错误信息 | 根因 | 解决方案 |
|---------|-------------|------|---------|
| UB 溢出 | `ub overflow, requires xxxx bits while 2097152 bits available!` | 单次处理数据量超过 256 KB | 核内 Tiling 分块；减小 BLOCK_SIZE_SUB；关闭 multibuffer |
| coreDim 超限 | `coreDim=xxxx can't be greater than UINT16_MAX` | Grid 维度超过 65535 | 增大 BLOCK_SIZE；设置 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 复合问题 | 增大 BLOCK_SIZE 后 UB 溢出 | coreDim 和 UB 同时受限 | 引入 BLOCK_SIZE_SUB，核间大块 + 核内小块 |
| 精度差异 | NPU 结果与 GPU/CPU 不一致 | 浮点计算顺序差异、类型退化 | 用 `TRITON_INTERPRET=1` 获取 CPU 基准；`tl.device_print` 打印中间结果 |
| Scalar 退化 | 性能大幅下降 | int64 在 Vector ADD/CMP 中退化为 scalar | 替换为 int32 或 fp32 |
| 离散访存 | 性能下降或 UB 溢出 | 非连续内存访问 | 调整数据布局使最内轴连续；改用二维 Block Pointer |
| Block Pointer 编译失败 | MLIR 编译错误 | advance 与复杂循环/分支搭配 | 改用重新创建 `tl.make_block_ptr` 或手动指针算术 |
| 数据类型不支持 | 编译/运行时错误 | 使用了 uint16/uint32/uint64/fp64 | 替换为对应 int 类型或 fp32 |

### 调试环境变量

| 环境变量 | 用途 |
|---------|------|
| `TRITON_INTERPRET=1` | 解释器模式，CPU 逐元素执行，验证逻辑正确性 |
| `TRITON_DEBUG=1` | 启用调试转储 |
| `TRITON_DISABLE_CACHE=1` | 禁用编译缓存 |
| `MLIR_ENABLE_DUMP=1` | 转储 MLIR IR 中间文件 |
| `ENABLE_PRINT_UB_BITS=1` | 编译时输出 UB 占用量 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动调整逻辑核数量为物理核数（注意：要求 kernel 对执行顺序不敏感） |

---

## 迁移工作流建议

### 阶段一：先跑通

1. **设备替换**：`cuda` → `npu`，添加 `import torch_npu`
2. **移除 GPU 专属参数**：删除 `num_warps`、`num_stages`、`cache_modifier`、`eviction_policy`
3. **数据类型替换**：fp64 → fp32，uint 系列 → int 系列
4. **Grid 调整**：Grid 大小对齐物理核数，优先 1D
5. **验证正确性**：用 `TRITON_INTERPRET=1` 对比 CPU 结果

### 阶段二：再优化

1. **UB 优化**：核内 Tiling 分块，引入 BLOCK_SIZE_SUB
2. **访存优化**：离散访存 → 连续访存，调整 Block Pointer stride/order
3. **并行优化**：启用 `multibuffer`，设置 `care_padding=False`
4. **类型优化**：Vector CMP 中 int64/int32 → fp32，避免 scalar 退化
5. **性能剖析**：使用 `msprof op` 采集性能数据，定位瓶颈

### 阶段三：910_95 专项

1. 评估是否需要 SIMT 模式（控制流密集场景）
2. 利用 FP8 支持进行混合精度优化
3. 利用 L0C→UB 直通路径减少 Cube→Vector 搬运开销

---

## 相关文档链接

- [01-architecture-differences.md](../docs_triton_ascend/06-Migration-from-GPU/01-architecture-differences.md) — GPU vs NPU 架构差异详解
- [02-code-migration-patterns.md](../docs_triton_ascend/06-Migration-from-GPU/02-code-migration-patterns.md) — 代码迁移模式与 diff 示例
- [03-common-issues.md](../docs_triton_ascend/06-Migration-from-GPU/03-common-issues.md) — 迁移常见问题 Q&A
- [04-block-pointer-migration.md](../docs_triton_ascend/06-Migration-from-GPU/04-block-pointer-migration.md) — Block Pointer 迁移注意事项
