# NPU Autotune 配置指南

## 触发条件

当 Agent 需要为 kernel 添加或调整 Autotune 配置时，参考本文档。典型场景包括：

- 将 GPU kernel 迁移到 NPU，需要移除 `num_stages`/`num_warps` 并重新配置 autotune
- 现有 kernel 使用固定 Tiling 参数，需要添加 autotune 以适配不同数据规模
- 已有 autotune 配置但性能不理想，需要调整候选值范围或 key 参数
- 新编写 kernel，需要为其设计 autotune 配置

---

## 核心知识：NPU Autotune 配置规则

### NPU Autotune 与 GPU 的核心差异

这是迁移时最重要的差异，必须牢记：

| 维度 | GPU Autotune | NPU Autotune |
|------|-------------|-------------|
| Config 内容 | Tiling 参数 + `num_stages` + `num_warps` | **仅 Tiling 参数** |
| `num_stages` | 控制软件流水线阶段数（1~4） | **不使用**，NPU 硬件自动管理流水线 |
| `num_warps` | 控制线程束数量（4/8/16/32） | **不使用**，NPU SIMD 模式固定 `num_warps=1` |
| 候选值约束 | 灵活，非 2 的幂也可 | **必须为 2 的幂次**（128, 256, 512, 1024, 2048...） |
| Grid 更新 | `lambda meta:` 动态计算 | 同样支持 `lambda meta:` 动态计算 |
| 缓存机制 | 按 key 值缓存 | 按 key 值缓存，相同 key 复用结果 |

**关键规则：NPU 的 `triton.Config` 中不要添加 `num_stages` 和 `num_warps` 参数。**

GPU 上的典型写法（**NPU 上错误**）：
```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4)
```

NPU 上的正确写法：
```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 64})
```

### Config 只包含 Tiling 参数

NPU 上的 `triton.Config` 仅包含 Tiling 分块参数，常见参数名：

| 参数名模式 | 含义 | 典型值 |
|-----------|------|--------|
| `BLOCK_SIZE` | 一维分块大小 | 128, 256, 512, 1024, 2048 |
| `BLOCK_M` / `BLOCK_SIZE_M` | M 维度分块大小 | 32, 64, 128, 256 |
| `BLOCK_N` / `BLOCK_SIZE_N` | N 维度分块大小 | 32, 64, 128, 256 |
| `BLOCK_K` / `BLOCK_SIZE_K` | K 维度（归约轴）分块大小 | 32, 64, 128, 256 |
| `GROUP_SIZE_M` | M 维度分组大小 | 8（通常固定） |
| `SPLIT_K` | K 维度拆分因子 | 8, 16 |

---

## 代码模式：配置示例

### 单参数 Autotune（一维 Tiling）

最简单的形式，适用于一维向量操作：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
)
@triton.jit
def vector_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

### 多参数组合 Autotune（二维/三维 Tiling）

适用于矩阵乘法、Flash Attention 等多维分块场景：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
    ],
    key=["M", "N"],
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

多参数列表推导会自动生成所有组合（3 x 3 = 9 个配置）。

### 使用函数生成 Config 列表

当 Config 列表较长或需要条件逻辑时，使用函数封装：

```python
def get_fwd_configs():
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
    ]
    return configs

@triton.autotune(
    configs=get_fwd_configs(),
    key=["QK_DIM", "V_DIM"],
)
@triton.jit
def fwd_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

### 使用 filter 筛选 Config

对生成的 Config 列表进行过滤，排除不合理的组合：

```python
def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    return m % n == 0

@triton.autotune(
    configs=list(filter(keep, get_bwd_configs())),
    key=["QK_DIM", "V_DIM"],
)
@triton.jit
def bwd_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

---

## key 参数选择原则

### 什么是 key

`key` 参数指定调优维度——当这些参数的值发生变化时，会触发重新调优；相同值会复用缓存结果。

### 选择原则

**选择代表数据规模的变量**，而非指针或 constexpr 参数：

| 场景 | 推荐 key | 说明 |
|------|---------|------|
| 一维向量操作 | `["n_elements"]` | 元素总数决定最优 BLOCK_SIZE |
| 矩阵乘法 | `["M", "N"]` 或 `["M", "N", "K"]` | 矩阵维度决定最优分块 |
| Flash Attention | `["QK_DIM", "V_DIM"]` | head_dim 影响最优 BLOCK_M/BLOCK_N |
| 变长序列 | `["max_seqlen_q", "max_seqlen_k"]` | 序列长度影响分块策略 |

### key 的两种形式

**列表形式**（简单场景，推荐）：
```python
key=["M", "N", "K"]
```
参数按顺序映射到轴名称 x, y, z, w, v, t。

**字典形式**（复杂场景，需显式指定轴信息）：
```python
key={'x': 'M', 'y': 'N', 'rx': 'K'},
hints={
    'split_params': {'x': 'BLOCK_M'},
    'tiling_params': {'rx': 'BLOCK_K'},
    'low_dim_axes': ['y'],
    'reduction_axes': ['x'],
}
```

### 常见错误

- **不要把 Tiling 参数名作为 key**：`key=["BLOCK_SIZE"]` 是错误的，BLOCK_SIZE 是 autotune 要搜索的参数，不是触发条件
- **不要把指针参数作为 key**：`key=["x_ptr"]` 没有意义
- **不要遗漏影响性能的维度**：如果 K 维度变化也影响最优配置，应加入 key

---

## 候选值范围选择

### 基本原则

1. **2 的幂次**：候选值必须是 2 的幂次（32, 64, 128, 256, 512, 1024, 2048, 4096）
2. **从小到大覆盖**：确保小规模和大规模场景都有合适的候选值
3. **小值覆盖小 shape**：小值使 Grid 分核数更多，匹配物理核数
4. **大值覆盖大 shape**：大值充分利用 UB 空间，发挥 SIMD 批量处理优势

### UB 容量估算

910_95 的 UB 为 248KB（预留 8KB 后可用约 240KB），经验公式：

```
最大 Tiling 元素数 ≈ UB_SIZE / (数据类型大小 × buffer 数量)
FP16 (2B), 3 buffers: 240KB / (2B × 3) ≈ 40960 elements
FP16 (2B), 2 buffers: 240KB / (2B × 2) ≈ 61440 elements
```

### 常见场景推荐范围

| 场景 | 推荐候选值 | 说明 |
|------|-----------|------|
| 一维 BLOCK_SIZE | `[128, 256, 512, 1024, 2048]` | 从 128 到 2048 覆盖常见规模 |
| 二维 BLOCK_M | `[32, 64, 128, 256]` | M 维度通常不需要太大 |
| 二维 BLOCK_N | `[32, 64, 128]` | N 维度受 UB 限制 |
| BLOCK_K | `[32, 64, 128, 256]` | 归约轴分块 |
| Flash Attention BLOCK_M | `[32, 64, 128]` | Attention 场景 UB 压力大 |
| Flash Attention BLOCK_N | `[32, 64, 128]` | K/V 维度分块 |

### 边界值处理

如果 autotune 总是选择候选列表的边界值（如总是选 2048），说明需要扩展范围：

```python
# 如果最优值总是 2048，扩展到 4096
configs = [
    triton.Config({"BLOCK_SIZE": BS})
    for BS in [128, 256, 512, 1024, 2048, 4096]
]
```

---

## 装饰器位置

`@triton.autotune` 必须放在 `@triton.jit` 装饰器**上方**，顺序不可颠倒：

```python
@triton.autotune(          # 必须在上
    configs=[...],
    key=["n_elements"],
)
@triton.jit                # 必须在下
def kernel(...):
    ...
```

如果使用 `@libentry()`，装饰顺序为：

```python
@libentry()                # 最外层
@triton.autotune(          # 中间层
    configs=[...],
    key=["N"],
)
@triton.jit                # 最内层
def kernel(...):
    ...
```

---

## Grid 动态更新

当 Grid 计算依赖 autotune 参数时，必须使用 `lambda meta:` 动态计算，确保 Grid 随 Tiling 参数变化而更新。

### 关联场景（Grid 依赖 Tiling）

```python
# Grid 与 BLOCK_SIZE 关联
grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
kernel[grid](x, y, output, n_elements)
```

```python
# Grid 与多个 Tiling 参数关联
grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
)
kernel[grid](x, w, y, M, N, K)
```

### 非关联场景（Grid 不依赖 Tiling）

当 Grid 由物理核数等固定值决定时，不需要 lambda：

```python
NUM_CORES = AICORE_NUM
grid = (NUM_CORES,)
kernel[grid](q, k, v, o, AICORE_NUM=NUM_CORES, ...)
```

### 实际示例：FusedMatmul 的 Grid 更新

```python
class FusedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        total_len, in_dim = x.shape
        out_dim = w.shape[1]
        y = x.new_empty(total_len, out_dim)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        fused_matmul_fwd_kernel[grid](
            x, w, b, y, total_len, out_dim, in_dim,
            HAS_BIAS=has_bias,
        )
        ...
```

---

## reset_to_zero 参数用法

### 问题场景

当 kernel 使用 `atomic_add` 等原子操作累加结果时，autotune 过程中每个 Config 都会运行一次，导致输出被多次累加，结果不正确。

### 解决方案

使用 `reset_to_zero` 参数，指定在评估每个 Config 之前需要重置为零的参数名列表：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
        for BM in [64, 128, 256]
        for BN in [32, 64, 128]
    ],
    key=["QK_DIM", "V_DIM"],
    reset_to_zero=["dq_ptr"],    # 每次评估前将 dq_ptr 指向的张量重置为零
)
@triton.jit
def bwd_qkv_kernel(
    q_ptr, k_ptr, v_ptr,
    dq_ptr, dk_ptr, dv_ptr,      # dq_ptr 使用 atomic_add 累加
    ...
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ...
    tl.atomic_add(dq_ptr + offset, dq, mask=mask)
    ...
```

### 何时需要 reset_to_zero

| 场景 | 是否需要 | 说明 |
|------|---------|------|
| kernel 使用 `tl.store` 写入 | 不需要 | 每次运行覆盖旧值 |
| kernel 使用 `tl.atomic_add` 累加 | **需要** | 多次运行会累加 |
| 输出张量在 autotune 前已初始化为零 | 视情况 | 如果初始化在循环内则不需要 |

---

## 多参数组合 Autotune

### 笛卡尔积组合

使用列表推导生成所有参数组合：

```python
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
    for BM in [32, 64, 128]
    for BN in [32, 64, 128]
]
# 生成 3 × 3 = 9 个配置
```

### 三维组合

```python
configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK})
    for BM in [64, 128]
    for BN in [64, 128]
    for BK in [32, 64, 128]
]
# 生成 2 × 2 × 3 = 12 个配置
```

### 带固定参数的组合

某些参数（如 `GROUP_SIZE_M`）通常不需要搜索，可以固定：

```python
configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8})
    for BM in [128]
    for BN in [128]
    for BK in [128]
]
```

### 控制组合数量

组合数量 = 各维度候选数之积。过多配置会显著增加首次运行时间。建议：

- 总配置数控制在 **20 个以内**
- 优先搜索对性能影响最大的维度
- 使用 `filter` 排除明显不合理的组合

---

## 910_95 特别注意

### UB 容量差异

910_95 的 UB 为 **248KB**（910B 为 192KB），Tiling 候选值上限可适当增大：

```
910_95: UB = 248KB, FP16 最大元素数 ≈ 126976 (单 buffer)
910B:   UB = 192KB, FP16 最大元素数 ≈ 98304  (单 buffer)
```

### SIMT VF 模式

910_95 支持 SIMT VF 模式，在此模式下：
- `num_warps` 固定为 1，不需要也不应该在 Config 中设置
- 编译器会自动推断 SIMD/SIMT/MIX 模式

### 候选值对齐要求

910_95 上 Tiling 值必须满足 32B 对齐：
- FP16 (2B)：元素数需为 16 的倍数 → 最小 Tiling 为 32
- FP32 (4B)：元素数需为 8 的倍数 → 最小 Tiling 为 32
- 使用 2 的幂次值天然满足对齐要求

### L0C 直通 UB

910_95 支持 L0C -> UB 直通（FixPipe），Cube 计算结果可直接送入 Vector 工作区，无需经过 GM 中转。这使得 CV 融合 kernel 的 Tiling 策略与 910B 不同，autotune 候选值可以更激进。

---

## 实际 Autotune 配置示例

### 示例 1：Flash Attention 前向 Kernel

来源：`flash_attention_npu_v8.py`

```python
def get_fwd_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)
        ]
    else:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                },
            )
            for BM in [32, 64, 128]
            for BN in [32, 64, 128]
        ]
        return configs

@triton.autotune(
    configs=get_fwd_configs(),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
)
@triton.jit
def fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        ...,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr,
        SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        AICORE_NUM: tl.constexpr,
        ...
):
    ...
```

**要点分析**：
- NPU 分支（`else`）的 Config **不含 `num_stages` 和 `num_warps`**，与 GPU 分支形成对比
- key 包含 `QK_DIM`、`V_DIM`、`MASK_FN`、`SPARSE_OPT`，涵盖影响最优 Tiling 的所有维度
- 候选值 `[32, 64, 128]` 是 Flash Attention 场景下 UB 压力较大时的典型范围
- Grid 不依赖 Tiling 参数（使用固定 `AICORE_NUM`），因此不需要 `lambda meta:`

### 示例 2：Flash Attention 反向 QKV 融合 Kernel（含 reset_to_zero）

来源：`flash_attention_npu_v8.py`

```python
def get_bwd_qkv_configs():
    return [
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ]

def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9:
        if m == 64 and config.num_warps == 8:
            return False
    return m % n == 0

@triton.autotune(
    list(filter(keep, get_bwd_qkv_configs())),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
    reset_to_zero=["dq_ptr"]
)
@triton.jit
def bwd_qkv_kernel(
        q_ptr, k_ptr, v_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        do_ptr, l_ptr, d_ptr,
        ...,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ...
    tl.atomic_add(
        dq_ptr + (dq_offs_base + row_offs[:, None] * q_head * QK_DIM + col_offs[None, :]),
        dq,
        mask=row_mask[:, None]
    )
    ...
```

**要点分析**：
- 使用 `filter(keep, ...)` 过滤不合理的组合（`BLOCK_M % BLOCK_N != 0` 的配置）
- `reset_to_zero=["dq_ptr"]` 因为 kernel 内使用 `tl.atomic_add` 累加 dq
- 手动枚举所有 Config 而非列表推导，可以更精细地控制候选组合
- BLOCK_M 从大到小排列（256 → 32），优先尝试大块

### 示例 3：Fused Matmul 前向 Kernel（含 Grid 动态更新）

来源：`fused_matmul_npu_v3.py`

```python
def fwd_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [128]
        for s in [3, 4]
        for w in [4, 8]
    ]
    return configs

@triton.autotune(
    configs=fwd_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_matmul_fwd_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, N, K,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    ...

# 调用处 - Grid 动态更新
class FusedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        total_len, in_dim = x.shape
        out_dim = w.shape[1]
        y = x.new_empty(total_len, out_dim)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        fused_matmul_fwd_kernel[grid](
            x, w, b, y, total_len, out_dim, in_dim,
            HAS_BIAS=has_bias,
        )
```

**要点分析**：
- 此示例当前仍保留了 `num_stages` 和 `num_warps`（GPU 遗留），迁移到 NPU 时应移除
- `key=["N", "K"]` 选择输出维度和归约维度作为调优触发条件
- Grid 使用 `lambda META:` 动态计算，确保分核数随 Tiling 参数变化
- `GROUP_SIZE_M` 固定为 8，不参与搜索

### 示例 4：Fused Matmul 反向权重 Kernel（含 SPLIT_K）

来源：`fused_matmul_npu_v3.py`

```python
def bwd_w_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "SPLIT_K": SK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
        )
        for BM in [86, 128]
        for BN in [128]
        for BK in [128, 256]
        for SK in [8, 16]
        for s in [3, 4]
    ]
    return configs

@triton.autotune(
    configs=bwd_w_autotune_config(),
    key=["M", "N"],
)
@triton.jit
def fused_matmul_bwd_w_kernel(
        dy_ptr, x_ptr, dw_ptr, LOCK_W,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
):
    ...
```

**要点分析**：
- `SPLIT_K` 参数用于 K 维度并行归约，减少 atomic 冲突
- `key=["M", "N"]` 选择输入和输出维度作为调优触发条件
- 注意 `BLOCK_SIZE_M` 使用了非 2 的幂次值 86，这在 NPU 上可能需要对齐处理，建议优先使用 2 的幂次

---

## 调试与验证

### 打印 Autotune 信息

设置环境变量查看 autotune 过程和最优配置：

```bash
export TRITON_PRINT_AUTOTUNING=1
```

输出示例：
```
Ascend autotuning parse split axes: {'x': 'BLOCK_M'}, split axis pid dims: {'x': 0}
Generated configs number: 12
Triton autotuning for function matmul_kernel finished after 2.35s;
best config selected: Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=1, num_stages=1);
```

### 验证 Autotune 生效

1. **功能一致**：输出结果与优化前一致
2. **Autotune 触发**：运行时日志显示正在尝试不同配置
3. **性能提升**：不同规模下选择了合适的 Tiling
4. **缓存命中**：相同 key 值不重复调优

### 常见问题

**Q: autotune 会增加首次运行时间吗？**

A: 会，但通常可接受。首次运行时尝试所有配置找最优，之后直接使用缓存结果。控制配置数量可减少首次开销。

**Q: 如何确定 key 参数？**

A: 选择代表数据规模的变量。当这个变量变化时，最优 Tiling 可能不同。常见选择：`n_elements`、`seq_len`、`M`、`N`、`K`。

**Q: Tiling 值选到边界怎么办？**

A: 如果最优配置总是在候选列表的边界（如总是 2048），说明可能需要更大的值，扩展候选列表。

**Q: autotune 调优时间太长怎么办？**

A: (1) 减少候选配置数量；(2) 设置 `TRITON_AUTOTUNE_PARALLEL_COMPILE=1`（默认开启）启用并行编译；(3) 使用 `prune_configs_by` 参数提前剪枝不可行配置。

**Q: 如何强制重新调优？**

A: 需要重启 Python 进程或清除 Triton 缓存目录，调优结果按 key 值缓存于进程内。

---

## 相关文档链接

- [03-autotune-guide.md](../docs_triton_ascend/05-Performance-Optimization/03-autotune-guide.md) -- Autotune 与 AutoTilingTuner 详解
- [03-tiling-and-grid.md](../docs_for_triton_agent/03-tiling-and-grid.md) -- NPU Tiling 与 Grid 配置
- [00-hardware-quick-ref.md](../docs_for_triton_agent/00-hardware-quick-ref.md) -- 910_95 硬件规格速查
- `flash_attention_npu_v8.py` -- Flash Attention NPU 实现及 autotune 配置
- `fused_matmul_npu_v3.py` -- Fused Matmul NPU 实现及 autotune 配置
- [autotuner.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/autotuner.py) -- AutoTilingTuner 源码实现
- [tile_generator.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/tile_generator.py) -- 分块配置生成器源码
