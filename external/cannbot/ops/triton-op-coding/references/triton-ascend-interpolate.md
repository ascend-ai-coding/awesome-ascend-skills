# Triton-Ascend Interpolate 算子开发参考文档

> 适用于 `interpolate` 算子及相关运算。

---

## 一、算子特性与难点

| 特性 | 说明 |
|---|---|
| **多模式支持** | nearest / bilinear / bicubic / area 四种插值模式 |
| **坐标映射复杂** | `align_corners=True/False` 两套坐标公式 |
| **边界处理敏感** | 边缘像素 clamp 逻辑直接影响精度 |
| **模式间差异大** | nearest 简单，但 bicubic 涉及 4×4 邻域和权重函数 |

---

## 二、核心问题与解决方案

### 问题 1：坐标映射公式精度

- **现象**：bilinear / bicubic 模式大面积数值不匹配，相对误差达 10%~200%。
- **根因**：浮点除法顺序和整数除法截断导致坐标计算偏差。

**PyTorch 正确公式：**

- `align_corners=False`：`src = scale * (dst + 0.5) - 0.5`，其中 `scale = src_size / dst_size`（浮点除法）
- `align_corners=True`：`src = dst * (src_size - 1) / (dst_size - 1)`

**✅ 解决方案：在 host 端预计算 scale，避免 kernel 内重复除法。**

```python
# Host 端预计算
inv_scale_h = float(torch.tensor(H_in / H_out, dtype=torch.float32))
inv_scale_w = float(torch.tensor(W_in / W_out, dtype=torch.float32))

# Kernel 内只做乘法和加减
src_y = (h_out.to(tl.float32) + 0.5) * inv_scale_h - 0.5
```

---

### 问题 2：边界 Clamp 逻辑

- **现象**：边缘像素值与 PyTorch 不一致。
- **根因**：PyTorch 的 bilinear 插值对越界坐标有特殊处理——当 `y0 < 0` 时，`y0` 和 `y1` 都可能被调整为 0，且对应权重需重新计算。

**✅ 解决方案：严格参考 PyTorch `upsample_bilinear2d` 的边界处理。**

```python
y0 = tl.cast(src_y, tl.int32)
if src_y < 0.0:
    y0 = y0 - 1  # floor 对负数需要调整
y1 = y0 + 1

# clamp 到有效范围
if y0 < 0:
    y0 = 0
if y1 >= H_in:
    y1 = H_in - 1
if x0 < 0:
    x0 = 0
if x1 >= W_in:
    x1 = W_in - 1
```

---

### 问题 3：Bicubic 权重函数系数

- **现象**：bicubic 模式全部失败，误差极大。
- **根因**：使用了错误的插值核函数。PyTorch 使用 **Keys' bi-cubic（A=-0.75）**，而非 Catmull-Rom。

**✅ 解决方案：使用正确的 Keys' 权重公式。**

```python
A = -0.75

# |t| <= 1
wy = ((A + 2.0) * t_y - (A + 3.0)) * t_y * t_y + 1.0

# 1 < |t| < 2
wy = ((A * t_y - 5.0 * A) * t_y + 8.0 * A) * t_y - 4.0 * A
```

---

### 问题 4：Bicubic 权重归一化

- **现象**：边界处像素值系统性偏差。
- **根因**：边界 clamp 后，4×4 邻域中部分像素被重复计算，权重和不再为 1。

**✅ 解决方案：累加实际使用的权重和，最后做归一化。**

```python
val = 0.0
w_sum = 0.0
for jj in range(4):
    for ii in range(4):
        pixel = tl.load(...)
        weight = wy * wx
        val += weight * pixel
        w_sum += weight

if w_sum != 0.0:
    val = val / w_sum
```

---

### 问题 5：最近邻取整方式

- **现象**：nearest 模式部分失败。
- **根因**：PyTorch 的 nearest 使用 `floor(src + 0.5)` 等价于 round-to-nearest，但需确认 `align_corners` 的坐标映射。

**✅ 解决方案：根据是否使用 size 参数选择不同的计算路径。**

```python
if use_size == 1:
    src_yi = h_out * H_in // H_out  # 整数向下取整
else:
    src_yi = tl.cast(h_out.to(tl.float32) / scale_h, tl.int32)
```

---

### 问题 6：Triton JIT 语法限制

| 限制 | 错误信息 | 解决方案 |
|---|---|---|
| **不支持 `break`** | `unsupported AST node type: Break` | 用 `range(block_start, block_end)` 替代 |
| **不支持 `float(jj)`** | `float() argument must be a string or a real number` | 直接用 `jj`（循环变量在 Triton 中已是标量） |
| **`tl.where` 对 int 类型问题** | 比较结果不正确 | 用 `if-else` 标量控制流替代 |

---

### 问题 7：Ascend NPU 向量加载精度

- **现象**：向量化后精度大面积失败。
- **根因**：Ascend NPU 对分散向量加载（scattered vector load，即索引跨越不同行/通道）支持不佳，导致数据读取错误。

**✅ 解决方案：采用标量循环 + 内部 tiling 策略。**

```python
# 每个 program 处理 BLOCK_SIZE 个像素，但每个像素用标量计算
for block_start in range(pid * BLOCK_SIZE, num_outputs, tl.num_programs(0) * BLOCK_SIZE):
    block_end = min(block_start + BLOCK_SIZE, num_outputs)
    for idx in range(block_start, block_end):
        # 标量处理每个像素
        ...
```

---

## 三、验证策略

| 阶段 | 方法 | 目的 |
|---|---|---|
| **单像素对比** | 手写 Python 逐像素计算 | 快速定位坐标 / 公式问题 |
| **单 shape 验证** | 小 tensor（4×4 → 8×8） | 确认边界处理正确 |
| **全量验证** | 73 个 shape 组合 | 覆盖所有模式和参数组合 |

---

## 四、性能优化经验

| 优化点 | 效果 | 注意事项 |
|---|---|---|
| **入参静态化（`tl.constexpr`）** | 编译期优化 | `mode`、`align_corners`、`scale` 等 |
| **BLOCK_SIZE tiling** | 减少 kernel 启动开销 | 标量循环内每个 program 处理多个像素 |
| **Host 端预计算** | 减少 kernel 内计算 | `scale`、`inv_scale` 在 host 计算后传入 |

---

## 五、关键代码结构

```python
@triton.jit
def interpolate_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in, H_out, W_out,
    mode: tl.constexpr,          # 静态化
    align_corners: tl.constexpr,
    use_size: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    inv_scale_h: tl.constexpr,   # host 预计算
    inv_scale_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_outputs = N * C * H_out * W_out
    # 标量循环处理 BLOCK_SIZE 像素
    for block_start in range(pid * BLOCK_SIZE, num_outputs, tl.num_programs(0) * BLOCK_SIZE):
        block_end = block_start + BLOCK_SIZE
        if block_end > num_outputs:
            block_end = num_outputs
        for idx in range(block_start, block_end):
            # 解码 (n, c, h, w)
            # 根据 mode 分支处理
            # 严格遵循 PyTorch 坐标映射和边界处理
            ...
```

---

## 六、调试 Checklist

- [ ] 坐标映射公式与 PyTorch 源码一致
- [ ] 边界 clamp 逻辑处理负坐标（floor 对负数需调整）
- [ ] bicubic 权重使用 Keys' bi-cubic（`A = -0.75`）
- [ ] bicubic 边界处做权重归一化
- [ ] 浮点除法在 host 端预计算
- [ ] 避免 Triton JIT 不支持的语法（`break`、`continue`、`float()` 等）
- [ ] Ascend NPU 避免分散向量加载，使用标量循环
- [ ] 多 shape 全量验证通过后再做性能优化
