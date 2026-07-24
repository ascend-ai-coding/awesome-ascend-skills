# NPU 编译参数速查文档

> 触发条件：Agent 需要为 kernel 配置编译参数时查阅本文档

## 1. 编译参数分类速查表

NPU 编译参数通过 `kernel[grid](...参数)` 调用时以关键字参数传入，底层映射到 `NPUOptions` 数据类的各字段。与 GPU 使用 `num_stages`/`num_warps` 控制流水线和并行度不同，NPU 使用一组专门的参数来控制 Cube-Vector 协同、多缓冲流水线和编译路径。

### 1.1 核心参数（按算子类型选择）

| 参数 | 类型 | 默认值 | 含义 | 适用算子类型 |
|------|------|--------|------|-------------|
| `enable_flatten` | bool | None | 启用 IR 展平优化，将多维循环展平为一维，提升向量化效率 | 纯 Vector: True; CV 融合: False |
| `multibuffer` | bool | 非910_95时 True | 启用 ping-pong 多缓冲流水线，在计算当前数据的同时预取下一批数据，隐藏内存延迟 | 通用 |
| `enable_auto_bind_sub_block` | bool | None | 启用自动绑定子块（sub-block），将 Vector 核心划分为多个子块并行执行 | CV 融合: True; 纯 Vector: False |
| `sync_solver` | bool | None | 启用同步求解器，自动求解 Cube-Vector 间的同步点位置 | CV 融合: True |
| `set_workspace_multibuffer` | int | None | 设置 workspace 多缓冲数量，为中间计算结果分配多份缓冲区 | CV 融合: 2 |
| `limit_auto_multi_buffer_of_local_buffer` | str | None | 限制本地缓冲（UB）的多缓冲策略。`"no-limit"` 表示不限制 | CV 融合: "no-limit" |
| `enable_mixed_cv` | bool | None | 启用混合 CV 模式，允许 Cube 和 Vector 在同一 kernel 中混合执行 | CV 融合: True |

### 1.2 辅助参数

| 参数 | 类型 | 默认值 | 含义 | 副作用 |
|------|------|--------|------|--------|
| `enable_ubuf_saving` | bool | None | 启用 UB 节省优化（A2/A3 平台），减少 UB 占用 | 可能降低计算并行度 |
| `enable_hivm_auto_cv_balance` | bool | None | 启用 HIVM 自动 CV 负载均衡 | 编译时间增加 |
| `inject_barrier_all` | bool | None | 在所有操作间注入屏障同步 | 性能可能下降，用于调试 |
| `inject_block_all` | bool | None | 在所有块间注入同步 | 性能可能下降，用于调试 |
| `disable_auto_inject_block_sync` | bool | None | 禁用自动注入块同步 | 可能导致数据竞争 |
| `enable_vf_fusion` | bool | False | 启用 VF（Vector Function）融合 | UB 占用增加 |
| `limit_auto_multi_buffer_only_for_local_buffer` | bool | None | 限制多缓冲仅用于本地缓冲（UB） | 减少全局缓冲开销 |
| `enable_cce_vf_auto_sync` | bool | None | 启用 CCE VF 自动同步 | 编译时间增加 |
| `enable_cce_vf_remove_membar` | bool | None | 启用 CCE VF 移除内存屏障 | 可能导致同步问题 |
| `disable_size_align_for_cast` | bool | None | 禁用类型转换的大小对齐要求 | 可能产生非对齐访问 |
| `tile_mix_vector_loop` | int | None | 混合 Vector 循环 tiling 大小（A2/A3） | 影响 UB 占用 |
| `tile_mix_cube_loop` | int | None | 混合 Cube 循环 tiling 大小（A2/A3） | 影响 L1 占用 |
| `unit_flag` | bool | None | 启用单元标志同步 | 调试用 |

### 1.3 编译模式参数

| 参数 | 类型 | 默认值 | 含义 | 编译路径 |
|------|------|--------|------|----------|
| `compile_mode` | str | "simd" | 编译模式选择 | 见下表 |
| `num_warps` | int | 4 | Warp 数量，SIMD 模式影响 HFusion 向量化策略，SIMT 模式控制线程数 | 通用 |
| `num_stages` | int | 1 | 流水线阶段数（NPU 上通常保持 1） | 通用 |
| `auto_blockify_size` | int | 1 | AutoBlockify 分块大小，配合 `TRITON_ALL_BLOCKS_PARALLEL` 使用 | 通用 |
| `add_auto_scheduling` | bool | False | 启用自动调度（DAG 亲和性优化） | SIMT/混合模式 |
| `enable_bishengir_simt_optimization` | int | 000 | SIMT 优化控制位（位模式） | SIMT 模式 |

**compile_mode 详解：**

| 值 | 行为 | 编译路径 |
|----|------|----------|
| `"simd"` | 默认模式，`parallel_mode` 设为 `"simd"` | Linalg → HFusion → HIVM → Binary |
| `"unstructured_in_simt"` | 非结构化转 SIMT，自动设 `force_simt_template=True` | SIMD + SIMT 混合路径 |
| `"simt_only"` | 纯 SIMT，自动设 `force_simt_only=True`、`parallel_mode="simt"` | TTIR → TTGIR → LLVM → Binary |

### 1.4 精度参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `default_dot_input_precision` | str | "ieee" | Dot 操作默认输入精度 |
| `allowed_dot_input_precisions` | tuple | ("ieee", "hf32") | 允许的 Dot 输入精度 |
| `enable_fp_fusion` | bool | True | 启用浮点融合（FMA） |
| `disable_fma` | bool | False | 禁用 FMA（提高精度但降低性能） |

## 2. 不同算子类型的参数配置示例

### 2.1 纯 Vector 算子

纯 Vector 算子只使用 Vector 核心进行逐元素运算、归约等操作，不涉及 Cube（矩阵乘法）。

**典型场景：** 向量加法、LayerNorm、Softmax、激活函数、逐元素运算

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](
    x, y, out, N,
    BLOCK_SIZE=1024,
    enable_flatten=True,
    multibuffer=True,
)
```

**参数说明：**
- `enable_flatten=True`：纯 Vector 算子无 Cube 参与，展平优化可提升向量化效率
- `multibuffer=True`：启用 ping-pong 流水线隐藏内存延迟
- 无需 `enable_auto_bind_sub_block`、`sync_solver`、`enable_mixed_cv` 等 CV 相关参数

### 2.2 CV 融合算子

CV 融合算子同时使用 Cube（矩阵计算）和 Vector（向量计算）核心，需要 Cube-Vector 协同调度和同步。

**典型场景：** Flash Attention、融合矩阵乘法+后处理、MatMul+ReLU+量化

```python
@triton.jit
def flash_attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, ...,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ...):
    ...
    for start_n in range(begin, end, BLOCK_N):
        k = tl.load(k_block_ptr, ...)
        s = tl.dot(q, k)
        s = s * scale + tl.where(mask, 0.0, -2.0**30)
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.math.exp(s - m_new[:, None])
        v = tl.load(v_block_ptr, ...)
        pv = tl.dot(p.to(dtype), v)
        ...

grid = (AICORE_NUM,)
flash_attn_fwd_kernel[grid](
    q, k, v, o, ...,
    BLOCK_M=128, BLOCK_N=32,
    enable_auto_bind_sub_block=True,
    enable_flatten=False,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_mixed_cv=True,
)
```

**参数说明：**
- `enable_auto_bind_sub_block=True`：将 Vector 核心划分为子块，与 Cube 并行执行
- `enable_flatten=False`：CV 融合算子不能展平，因为 Cube 和 Vector 需要各自独立的循环结构
- `set_workspace_multibuffer=2`：为中间结果分配双缓冲，配合 ping-pong 流水线
- `sync_solver=True`：自动求解 Cube-Vector 间的同步点，避免手动插入同步
- `limit_auto_multi_buffer_of_local_buffer="no-limit"`：不限制 UB 的多缓冲分配，避免 UB 不足导致编译失败
- `multibuffer=True`：启用多缓冲流水线
- `enable_mixed_cv=True`：启用混合 CV 模式，允许 Cube 和 Vector 交替执行

### 2.3 纯 Cube 算子（矩阵乘法前向）

```python
@triton.jit
def matmul_fwd_kernel(x_ptr, w_ptr, y_ptr, M, N, K,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    ...
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, ...)
        w = tl.load(w_ptrs, ...)
        accumulator = tl.dot(x, w, accumulator)
    ...

grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
matmul_fwd_kernel[grid](
    x, w, y, M, N, K,
    BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8,
    enable_auto_bind_sub_block=True,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_flatten=True,
)
```

**参数说明：**
- 矩阵乘法前向虽然主体是 Cube 操作，但 epilogue（类型转换、store）在 Vector 上执行
- `enable_flatten=True`：MatMul 前向的 Vector 后处理部分可以展平
- `enable_auto_bind_sub_block=True`：Cube 和 Vector 协同需要子块绑定
- `sync_solver=True`：自动求解 Cube→Vector 的同步点

### 2.4 纯 Vector 后处理算子（如 bias 梯度计算）

```python
@triton.jit
def bwd_b_kernel(dy_ptr, db_ptr, M, N,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    ...
    sum_b = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_b += dy
        dy_ptrs += BLOCK_SIZE_M * N
    tl.store(db_ptr + col_off, tl.sum(sum_b, 1), mask=col_off < N)

grid = (triton.cdiv(N, BLOCK_SIZE_N),)
bwd_b_kernel[grid](
    dy, db, M, N,
    enable_auto_bind_sub_block=False,
)
```

**参数说明：**
- `enable_auto_bind_sub_block=False`：纯 Vector 算子不需要子块绑定
- 无需 CV 相关参数

## 3. 参数传递方式

### 3.1 在 kernel[grid](...) 调用时传入

这是最常用的方式，编译参数作为关键字参数直接传入 kernel 启动调用：

```python
kernel[grid](
    arg1, arg2, ...,           # kernel 的位置参数
    BLOCK_SIZE=1024,           # constexpr 参数
    enable_flatten=True,       # 编译参数
    multibuffer=True,          # 编译参数
    sync_solver=True,          # 编译参数
)
```

编译参数与 kernel 参数混合传入，Triton 会自动区分：属于 `NPUOptions` 字段的被识别为编译参数，其余为 kernel 参数。

### 3.2 在 autotune Config 中传入

通过 `triton.Config` 的关键字参数传入，autotune 会为每个配置分别编译：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32},
            multibuffer=True,
            enable_mixed_cv=True,
            enable_auto_bind_sub_block=True,
            sync_solver=True,
            enable_flatten=False,
            set_workspace_multibuffer=2,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            multibuffer=True,
            enable_mixed_cv=True,
            enable_auto_bind_sub_block=True,
            sync_solver=True,
            enable_flatten=False,
            set_workspace_multibuffer=2,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
    ],
    key=["QK_DIM", "V_DIM"],
)
@triton.jit
def my_kernel(...):
    ...
```

### 3.3 通过 compile() 函数直接传入

```python
compiled_kernel = triton.compile(
    kernel,
    options={
        "compile_mode": "simt_only",
        "num_warps": 8,
        "enable_flatten": True,
        "multibuffer": True,
    }
)
```

## 4. 精度回退策略

当 kernel 编译失败或运行时出现精度问题时，应按照以下策略逐步回退：

### 4.1 编译失败回退

```
1. 完整 CV 融合参数（最高性能）
   enable_auto_bind_sub_block=True, enable_flatten=False,
   set_workspace_multibuffer=2, sync_solver=True,
   limit_auto_multi_buffer_of_local_buffer="no-limit",
   multibuffer=True, enable_mixed_cv=True

2. 去掉 enable_mixed_cv（禁用混合 CV）
   enable_auto_bind_sub_block=True, enable_flatten=False,
   set_workspace_multibuffer=2, sync_solver=True,
   limit_auto_multi_buffer_of_local_buffer="no-limit",
   multibuffer=True

3. 去掉 enable_auto_bind_sub_block（禁用子块绑定）
   enable_flatten=False, multibuffer=True,
   sync_solver=True

4. 去掉 sync_solver（禁用同步求解器）
   enable_flatten=False, multibuffer=True

5. 最小参数集（最大兼容性）
   multibuffer=True
```

### 4.2 精度问题回退

```
1. 启用 FMA → disable_fma=True（禁用 FMA 提高精度）
2. enable_fp_fusion=True → enable_fp_fusion=False（禁用浮点融合）
3. default_dot_input_precision="ieee"（确保 IEEE 精度）
```

### 4.3 UB 溢出回退

当出现 `UB overflow` 错误时：

```
1. 减小 BLOCK_M / BLOCK_N（减少每个 block 的数据量）
2. set_workspace_multibuffer=2 → 1（减少 workspace 缓冲）
3. limit_auto_multi_buffer_of_local_buffer="no-limit" → 限制多缓冲
4. enable_flatten=False（展平可能增加 UB 占用）
5. enable_ubuf_saving=True（A2/A3 平台启用 UB 节省）
```

## 5. 910_95 特别注意

### 5.1 平台检测

910_95 平台通过 `is_compile_on_910_95` 自动检测，`NPUOptions.compile_on_910_95` 默认值即为检测结果。

检测逻辑（[get_ascend_devices.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/tools/get_ascend_devices.py)）：
- PCI 设备 ID 包含 `0xd806`
- 或 `npu-smi info` 输出包含 `ascend910_95`/`ascend950`/`910_958b`

### 5.2 关键差异

| 特性 | A2/A3 (910B) | 910_95 |
|------|-------------|--------|
| `multibuffer` 默认值 | True | **False** |
| UB 大小 | 192 KB | **256 KB** |
| RF 大小 | 无 | **128 KB** |
| FFTS 支持 | 支持 | **不支持**（自动禁用） |
| fixpipe L0C→UB | 不支持 | **支持** |
| `copy` (UB→UB/L1) | 不支持 | **支持** |
| Vector 核心数 | = Cube 核心数 | **= Cube 核心数 x 2** |
| `shared_mem_dynamic_size`（SIMT） | 221184 | 122880 |

### 5.3 910_95 上的参数调整建议

1. **multibuffer 需显式开启**：910_95 默认 `multibuffer=False`，需要手动设为 `True` 以获得最佳性能
2. **UB 空间更大**：256KB UB 允许更大的 BLOCK_SIZE，但需配合 `limit_auto_multi_buffer_of_local_buffer="no-limit"` 使用
3. **Vector 核心数翻倍**：910_95 的 Vector 核心数是 Cube 的 2 倍，grid 可以使用更多核心
4. **fixpipe 可直达 UB**：利用 `fixpipe` 将 Cube 结果从 L0C 直接搬运到 UB，实现零拷贝融合
5. **FFTS 自动禁用**：910_95 不支持 FFTS，编译器会自动跳过 FFTS 相关 Pass

### 5.4 910_95 上的典型配置

```python
# Flash Attention (910_95)
fwd_kernel[grid](
    q, k, v, o, l, ...,
    BLOCK_M=128, BLOCK_N=32,
    multibuffer=True,                    # 910_95 默认 False，需显式开启
    enable_mixed_cv=True,
    enable_auto_bind_sub_block=True,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    enable_flatten=False,
    set_workspace_multibuffer=2,
)
```

## 6. 与 GPU 编译参数对比

| 维度 | GPU (CUDA) | NPU (Ascend) |
|------|-----------|-------------|
| 流水线控制 | `num_stages=2~4`（软件流水线阶段数） | `multibuffer=True`（ping-pong 多缓冲） |
| 并行度控制 | `num_warps=4/8/16`（warp 数量） | `num_warps=4`（影响向量化策略，非线程数概念） |
| 线程级并行 | GPU 天然 SIMT | `compile_mode="simt_only"` + `num_warps=8/16` |
| Cube-Vector 协同 | 无（GPU 无 Cube/Vector 分离） | `enable_mixed_cv=True` + `sync_solver=True` |
| 子块绑定 | 无 | `enable_auto_bind_sub_block=True` |
| 展平优化 | 无 | `enable_flatten=True/False` |
| workspace 缓冲 | 无 | `set_workspace_multibuffer=2` |
| UB 限制 | 无（使用 Shared Memory） | `limit_auto_multi_buffer_of_local_buffer="no-limit"` |
| 精度控制 | `tf32`/`ieee`/`tf32x3` | `ieee`/`hf32` |

**关键区别：**
- GPU 的 `num_stages` 控制软件流水线深度，NPU 的 `multibuffer` 控制 ping-pong 缓冲
- GPU 的 `num_warps` 直接控制线程并行度，NPU 的 `num_warps` 主要影响编译器的向量化策略
- NPU 特有的 Cube-Vector 分离架构需要 `enable_mixed_cv`、`sync_solver`、`enable_auto_bind_sub_block` 等参数来协调双核
- NPU 的 UB 空间有限，需要通过 `limit_auto_multi_buffer_of_local_buffer` 等参数精细控制

## 7. 实际代码中的参数使用参考

### 7.1 Flash Attention（CV 融合算子）

来源：`flash_attention_npu_v8.py`

**前向 kernel（CV 融合）：**
```python
fwd_kernel[grid](
    q, k, v, o, l, ...,
    QK_DIM=qk_dim, V_DIM=v_dim, ...,
    multibuffer=True,
    enable_mixed_cv=True,
    enable_auto_bind_sub_block=True,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    enable_flatten=False,
    set_workspace_multibuffer=2,
)
```

**反向预处理 kernel（纯 Vector）：**
```python
bwd_preprocess_ifmn[grid](
    o, do, d, ...,
    multibuffer=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
)
```

**反向 QKV kernel（CV 融合）：**
```python
bwd_qkv_kernel[grid](
    q, k, v, dq, dk, dv, ...,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    enable_flatten=False,
    sync_solver=True,
    enable_mixed_cv=True,
)
```

### 7.2 融合矩阵乘法（CV 融合算子）

来源：`fused_matmul_npu_v3.py`

**前向 kernel（MatMul + Bias，CV 融合）：**
```python
fused_matmul_fwd_kernel[grid](
    x, w, b, y, M, N, K,
    HAS_BIAS=has_bias,
    enable_auto_bind_sub_block=True,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_flatten=True,
)
```

**反向 bias 梯度 kernel（纯 Vector）：**
```python
fused_matmul_bwd_b_kernel[grid](
    dy, db, M, N,
    enable_auto_bind_sub_block=False,
)
```

**反向 x 梯度 kernel（纯 Vector 矩阵乘法）：**
```python
fused_matmul_bwd_x_kernel[grid](
    dy, w, dx, M, N, K,
    enable_auto_bind_sub_block=False,
)
```

**反向 w 梯度 kernel（纯 Vector 矩阵乘法）：**
```python
fused_matmul_bwd_w_kernel[grid](
    dy, x, dw, lock_w, M, N, K,
    enable_auto_bind_sub_block=False,
)
```

## 8. 快速决策流程

```
kernel 是否包含 tl.dot()？
├── 否 → 纯 Vector 算子
│   └── enable_flatten=True, multibuffer=True
│       （无需 CV 相关参数）
│
└── 是 → 是否有 Cube 后的 Vector 处理（如 softmax、mask、激活）？
    ├── 否 → 纯 Cube 算子（MatMul 前向）
    │   └── enable_auto_bind_sub_block=True, enable_flatten=True,
    │       set_workspace_multibuffer=2, sync_solver=True,
    │       limit_auto_multi_buffer_of_local_buffer="no-limit",
    │       multibuffer=True
    │
    └── 是 → CV 融合算子（Flash Attention、融合 MatMul+后处理）
        └── enable_auto_bind_sub_block=True, enable_flatten=False,
            set_workspace_multibuffer=2, sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            multibuffer=True, enable_mixed_cv=True
```

## 9. 相关文档链接

- [01-extension-overview.md](../docs_triton_ascend/03-Ascend-Extensions/01-extension-overview.md) - Ascend 扩展 API 总览
- [02-pipe-and-core.md](../docs_triton_ascend/03-Ascend-Extensions/02-pipe-and-core.md) - PIPE/CORE 枚举详解
- [07-compile-options.md](../docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md) - 编译选项完整参考
- [compiler.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py) - NPUOptions 数据类定义
- [get_ascend_devices.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/tools/get_ascend_devices.py) - 910_95 平台检测逻辑
- [runtime/utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py) - NPU 运行时参数（核心数、UB 大小等）
