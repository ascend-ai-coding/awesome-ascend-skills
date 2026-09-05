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

## 相关优化点

- 优化点 2（Tiling）：非连续规约轴需调整分块
- 优化点 16（连续拷贝聚合）：纯拷贝型算子可进一步聚合
