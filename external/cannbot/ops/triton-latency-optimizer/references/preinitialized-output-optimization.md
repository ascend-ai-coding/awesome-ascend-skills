# 输出预初始化优化（Output Preinitialization）

## 思想

若输出张量有大量位置需写入同一默认值（常为 0），不要在 kernel 内逐位置判断填充。
**在 host 侧把输出预初始化为该默认值，kernel 只遍历并写入有效位置。**

- 迭代空间从 `O(output_size)` 降到 `O(valid_size)`
- 消除 `if valid` / `tl.where` 分支与冗余写入
- 向量化更友好，避免 host 侧重复 padding 拷贝

## 触发条件

出现以下任一模式即考虑本优化：

1. kernel 内用 `if valid` 或 `tl.where(valid, val, default)` 区分默认位置与计算位置
2. kernel 扫描**输出侧全图**逐位置判断是否落在有效输入范围内
3. 稀疏写入：只写部分位置，其余需保持已知常数
4. 累加场景（`atomic_add`）但输出未先清零

## 适用前提

- 默认值**可静态预初始化**：0、bias、`-inf`、`torch.full` 等常量。若依赖运行时数据则不适用。
- 目标位置**可从源位置直接推导**，无需从输出坐标反查源坐标。

## 代码对照

**优化前：输出侧全图扫描 + 边界判断**

```python
out = torch.empty(...)
total = N * H_out * W_out
for idx in range(pid, total, tl.num_programs(0)):
    n, h, w = decode(idx)
    valid = (h < H_in) & (w < W_in)        # 每位置都要判断
    val = tl.load(x_ptr + src_off) if valid else 0.0
    tl.store(out_ptr + off, val)            # 无效位置也写入默认值
```

**优化后：host 预置零 + 源侧迭代**

```python
# Host 侧
out = torch.zeros((N, H_out, W_out, C), device=x.device, dtype=x.dtype)

# Kernel：只遍历有效源位置，直接推目标坐标
for idx in range(pid, N * H_in * W_in, tl.num_programs(0)):
    n, h, w = decode(idx)
    out_h = h * SH + pad_h                  # 源坐标 → 目标坐标
    out_w = w * SW + pad_w
    val = tl.load(x_ptr + src_off)
    tl.store(out_ptr + dst_off, val)        # 无判断、无默认值写入
```

## 关键点 / 常见错误

| 错误 | 正确做法 |
|------|---------|
| 用 `torch.empty` 分配输出 | `torch.zeros` / `torch.full` 预初始化 |
| 已预置零，仍 `tl.where(valid, val, 0)` 填充 | 只写有效位置，默认位置交给初始化 |
| 仍遍历 `N*H_out*W_out` 输出侧 | 遍历 `N*H_in*W_in` 源侧 |
| `atomic_add` 前未清零 | 输出先 `torch.zeros`，否则累加历史脏值 |

## 性能收益（ConvTranspose2d 实测）

| 实现 | 几何平均加速比 |
|------|---------------|
| kernel 内条件填充 + 输出侧全图扫描 | 0.1962x |
| 预置零 + 源侧迭代 | 0.3247x |

收益来自：迭代空间缩小、消除分支与冗余写入、向量化提升。

## 实例：Pack + Padding 融合（Conv / Pool / Padding 类算子）

当算子需要把输入先 padding 到更大空间，再按另一种 layout pack 时，常见实现是：

```python
x_pad = torch.zeros((N, C, D_pad, H_pad, W_pad), ...)
x_pad[..., P:P+D, P:P+H, P:P+W].copy_(x)      # 一次完整 memory pass
x_packed = pack_kernel(x_pad, ...)              # 再读一次 x_pad
```

这导致输入数据被**读两遍**（copy 一遍 + pack 一遍），并且 `torch.zeros` 也是一次完整写入。

### 针对 pack+padding 的改造

把上面通用模式应用到 pack + padding 场景：

```python
# Host：不再分配 x_pad，直接置零目标 packed workspace
x_packed = torch.zeros((N, G, C1_pg, D_pad, H_pad, W_pad, C0), ...)

# Kernel：grid = N * G * C1_pg * D * H（只覆盖原始 x 的有效区域）
# 从 x(n, g, d_in, h_in) 读取 W×C0 tile
x_tile = tl.load(x_block_ptr)   # shape=(W, C0)

# 写回 x_packed(n, g, c1, d_in+P, h_in+P, P:P+W)
# 其余 padding 位置保持 zero-init 不变
tl.store(x_packed_ptr + interior_offset, x_tile)
```

与通用模式相比，这里的特异性在于：

- 目标 layout 是 pack 后的多维格式（如 `[N,G,C1,D,H,W,C0]`），不是简单的 `[N,H,W,C]`。
- 需要同时处理 D/H 方向的 padding（`+P`）和 W 方向的 padding（列偏移 `P:P+W`）。
- 源侧迭代维度是 `(n,g,c1,d_in,h_in)`，而不仅仅是 `(n,h,w)`。

这样省掉了 `x_pad = torch.zeros(...) → copy_(x) → pack_kernel(x_pad)` 中的额外 memory pass。

### 触发条件

- 输入需要先 `torch.zeros + copy_` padding 到中间 buffer，再被另一个 kernel 读取/转置。
- pack kernel 的每个输出位置**只对应一个连续源区域**（如 `[P:P+W]`），且源坐标可直接从目标坐标推导。
- 默认值就是 0（或 `torch.full` 可预置的常量）。

### 收益与风险

- **收益**：省掉 `x_pad` 的 zero + copy，输入数据从“copy 一次 + pack 读一次”降为“pack 直接读一次”；同时迭代空间缩小到有效源区域，分支消除。在中小 shape / 带宽 bound 场景收益显著。
- **风险**：
  - 目标坐标必须由源坐标直接推导，不能反过来从输出坐标反查源。
  - `tl.make_block_ptr` 的 `block_shape` 若取非 16 的 channel 数，可能被编译器 scalarize，需保持 C0 对齐或保留 boundary_check。
  - 当 pack 需要把多个不连续源位置合并到同一目标位置（如 gather / im2col）时，本模式不适用。
  - 若源有效区域不是规则矩形（如 dilation / 非对称 padding），需保证 grid 能覆盖所有有效源位置且目标映射正确。

### 已验证数据

`8_ConvStandard3d`（ascend910b1，60 cases）从 0.3725x 提升至 **0.7789x** vs torch。

## 相关优化点

- 优化点 2（Tiling）：非连续规约轴需调整分块
- 优化点 16（连续拷贝聚合）：纯拷贝型算子可进一步聚合
