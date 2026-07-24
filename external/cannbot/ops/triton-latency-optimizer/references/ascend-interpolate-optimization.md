# Ascend Interpolate 算子系统性优化

## 概述

本文档覆盖 Ascend NPU 上 Triton Interpolate 算子（`torch.nn.functional.interpolate`，4D NCHW）
的系统性性能优化方法。作为 `latency-optimizer` 的补充，应在完成通用优化点之后按 Phase 顺序执行。

Interpolate 是一个**计算特征高度异构**的算子：`nearest` / `bilinear` / `bicubic` / `area` 四种 mode
在 `align_corners=True/False`、上/下采样、整数倍/分数倍采样比例下的最优数据访问模式完全不同。
**单一通用 kernel 无法同时优化所有路径**，建议采用多策略分派架构。

目标性能：geomean ≥ 1.0x vs torch（归档阈值），用户目标 0.6x。

## 适用算子

- `interpolate`（4D NCHW，支持 nearest / bilinear / bicubic / area）
- `upsample_nearest2d`
- `upsample_bilinear2d`
- `upsample_bicubic2d`
- `adaptive_avg_pool2d`（对应 area 模式）

---

## ⚠️ 核心优化点 Checklist（迭代前必查，违反即性能不达标）

> **背景**：这些优化点都是从真实踩坑中提炼的"绝对重点"。历史上出现过"代码 73/73 精度通过、
> 但 geomean 只有 0.27x"的案例，根因就是漏掉了其中某一条。**每一条都对应 10x 量级的性能差距**，
> 不是"锦上添花"而是"达标前提"。Phase 4 每轮迭代前必须逐条核对当前代码是否已应用，
> 没应用的必须先补上再谈其他优化。

### 🔴 P0 级（漏一条直接 < 0.3x，必查必改）

- [ ] **C1. bilinear 0.5x 下采样必须用 contiguous load + reshape sum，禁止 strided load**
  - 触发：`mode='bilinear' and not align_corners and H_in==2*H_out and W_in==2*W_out`
  - 正确做法：每行一次连续 load `2*BLOCK_W` 元素 → `reshape((BLOCK_W,2))` → `tl.sum(axis=1)` → `*0.25`
  - ❌ 错误做法：用 `x0 = w_idx*2; x1 = x0+1` 做 4 次离散 strided `tl.load`
  - 实测影响：0.005x → 5x（**1000 倍**），单条决定 geomean 能否过 0.6x
  - 详见 Phase 3.3

- [ ] **C2. bilinear 0.5x 下采样必须特化分派，禁止走通用 bilinear 路径**
  - 同一触发条件，但 dispatch 时漏判 `half_down` 会让它走 `bilinear_general_vec_kernel`（4 次 gather + 坐标计算）
  - `_route` 中必须有 `if (not align_corners) and H_in==2*H_out and W_in==2*W_out: bilinear_half_pool_kernel[...]` 的显式分支
  - 详见 Phase 11 决策树

- [ ] **C3. 2x / 0.5x 整数倍 nearest 必须特化，禁止走通用 gather 路径**
  - 2x 上采样：contiguous load + `reshape([BLOCK_IN,1]) * ones([1,2])` broadcast 复制像素
  - 0.5x 下采样：contiguous strided load（步长 2，但仍是连续 vector load 的子集，非 gather）
  - ❌ 错误做法：用 `tl.gather(in_row, src_x)` 取像素
  - 详见 Phase 3.1 / 3.2

- [ ] **C4. 行预载必须用 `MAX_IN_W = triton.next_power_of_2(W_in)` constexpr，禁止裸 `tl.arange(0, W_in)`**
  - `W_in` 非 constexpr 会触发 dynamic-shape load，退化为标量循环
  - 所有需要"整行 load 到 UB"的 kernel（nearest/bilinear/bicubic/ac=True）都必须 pad 到 2 的幂
  - 详见 Phase 2.1

### 🟡 P1 级（漏一条导致特定 case 拖累，需逐条核对）

- [ ] **C5. 上采样（H_out > H_in）必须用 2D 垂直分块 BLOCK_H=2 / MAX_KH=3，禁止逐行独立 load**
  - 上采样时多个输出行共享输入行，逐行 load 重复访存
  - 单 program 处理 2 行输出、一次性 load 3 行输入复用
  - 详见 Phase 4

- [ ] **C6. ac=True 必须 host 预计算坐标和权重，kernel 内禁止重算 `scale * h`**
  - PyTorch C++ 的 ac=True 用 float32 计算，kernel 内重算有精度差异导致 verify 失败
  - bicubic ac=True 还要 host 预计算完整 4×4 索引和权重
  - 详见 Phase 1.1 / 1.2

- [ ] **C7. bicubic ac=True 必须 16 项标量逐项累加，禁止向量化重排累加顺序**
  - 向量化累加改变顺序，与 PyTorch 标量顺序不一致，verify 失败
  - 详见 Phase 6.2

- [ ] **C8. ac=False 负坐标 floor 必须用 `tl.where(src_y < y0_f, y0-1, y0)` 修正**
  - `src_y = scale*(i+0.5)-0.5` 可能为负，`tl.cast(src_y, int32)` 向零截断而非 floor
  - 详见 Phase 7 边界处理

- [ ] **C9. 三次权重必须用 `tl.where` 嵌套分段，禁止 `if` 分支**
  - `if at < 1.0:` 在 vector 数据上不生效
  - 详见 Phase 7

- [ ] **C10. grid_size 必须 `min(total_rows, num_cores)`，禁止超核数**
  - 上采样 total_rows 极大，多余 program 空跑
  - ❌ 也不要盲目乘倍数（如 `num_cores * 8`），实测会严重退化
  - 详见 Phase 9.1

### 🟢 P2 级（优化点，达标后锦上添花）

- [ ] **C11. 低精度 ac=True 的 scale 必须用 NPU vdiv（`_npu_scale`），禁止 CPU float32 除法**
  - CANN vdiv 与 CPU float32 舍入不同，低精度 ac=True verify 不过
  - 详见 Phase 5

- [ ] **C12. 所有 kernel 调用加 `multibuffer=True`**
  - 内存密集型 kernel 的默认编译选项
  - 详见 Phase 10

- [ ] **C13. nearest 通用路径跨行行复用（缓存 `last_n, last_c, last_src_yi`）**
  - 相邻输出行映射到同一输入行时，仅当变化时重新 load
  - 详见 Phase 2.3

### 迭代检查流程（Phase 4 每轮必做）

1. 读当前 `generated_code.py` / `optimized_code.py`
2. 逐条对照 C1–C13，标注每条 ✅已应用 / ❌未应用 / ⚪不适用
3. ❌未应用的优先补上（P0 > P1 > P2），补完再 benchmark
4. 禁止跳过 checklist 直接做"自创优化"——历史上所有自创优化都退化了

---

## 总体架构：多策略分派

**核心原则**：在 `ModelNew.forward()` 中实现 `_route()` 方法，按
`(mode, align_corners, H_out vs H_in, W_out vs W_in, dtype)` 分派到 9 个特化 kernel。

不同场景的最优数据访问模式完全不同：

| 场景 | 最优数据访问模式 |
|------|----------------|
| 2x 整数倍上采样 | contiguous load + reshape broadcast 复制像素 |
| 0.5x 整数倍下采样 | strided contiguous load（步长 2） |
| bilinear 0.5x 下采样 (ac=False) | 退化为 2x2 avg pool，reshape + sum |
| 上采样（通用） | 2D 垂直分块，复用输入行 |
| 通用 nearest/bilinear/bicubic | 行预载到 UB + tl.gather 按列取数 |
| ac=True bicubic | host 预计算 4×4 索引和权重 + 标量累积 |

提醒：用单一 kernel 统一处理所有 mode 和采样比例会丢失所有特化机会，建议按场景分派。

---

## Phase 1：Host 侧坐标与权重预计算

### 触发条件

`align_corners=True` 路径，或 bicubic 需要在 kernel 内计算 4×4 邻域下标和 16 个 Keys' 权重。

### 问题

- PyTorch C++ 的 ac=True 坐标用 `float32` 计算（`scale = (H_in-1)/(H_out-1)`），Triton kernel
  内重新计算会有精度差异导致 verify 失败。
- bicubic 的 16 项累加若向量化会改变累加顺序，与 PyTorch 标量顺序不一致。
- 同一行/列的坐标、权重对所有 channel/batch 重复计算。
- int32 floor、cast、分支判断容易退化为标量循环。

### 优化策略

在 `ModelNew.forward()` 中用 `numpy.float32` 预计算坐标、下标、权重，并缓存到 `_coord_cache`：

#### 1.1 bilinear/bicubic ac=True 坐标预计算

```python
def _precompute_ac_true_coords(self, H_in, W_in, H_out, W_out):
    key = (H_in, W_in, H_out, W_out)
    if key not in self._coord_cache:
        import numpy as np
        # 注意用 np.float32 匹配 PyTorch C++ 的 float32 计算路径
        scale_y = np.float32(H_in - 1) / np.float32(max(H_out - 1, 1))
        y0_l, y1_l, yl_l = [], [], []
        for i in range(H_out):
            s = np.float32(i) * scale_y
            y0 = int(np.float32(np.floor(s)))
            y1 = y0 + 1 if y0 < float(s) else y0
            y0_l.append(max(0, min(y0, H_in - 1)))
            y1_l.append(max(0, min(y1, H_in - 1)))
            yl_l.append(float(s - np.float32(y0)))
        # ... 同理计算 x 方向 ...
        self._coord_cache[key] = (y0, yl, x0, x1, xl)
    return self._coord_cache[key]
```

#### 1.2 bicubic ac=True 完整 4×4 索引和权重预计算

```python
def _precompute_bicubic_ac_true_all(self, H_in, W_in, H_out, W_out):
    key = ('bicubic_all', H_in, W_in, H_out, W_out)
    if key not in self._coord_cache:
        import numpy as np
        scale_y = np.float32(H_in - 1) / np.float32(max(H_out - 1, 1))
        scale_x = np.float32(W_in - 1) / np.float32(max(W_out - 1, 1))
        A = -0.75

        def _cubic(t):
            t = abs(float(t))
            if t <= 1.0:
                return float(((A + 2.0) * t - (A + 3.0)) * t * t + 1.0)
            else:
                return float(((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A)

        def _compute_dim(L_in, L_out, scale):
            idx_m1, idx_0, idx_p1, idx_p2 = [], [], [], []
            w_m1, w_0, w_p1, w_p2 = [], [], [], []
            for i in range(L_out):
                s = np.float32(i) * scale
                y0 = int(np.float32(np.floor(s)))
                if s < 0:
                    y0 -= 1
                wy = float(s - np.float32(y0))
                idx_m1.append(max(0, min(y0 - 1, L_in - 1)))
                idx_0.append(max(0, min(y0, L_in - 1)))
                idx_p1.append(max(0, min(y0 + 1, L_in - 1)))
                idx_p2.append(max(0, min(y0 + 2, L_in - 1)))
                w_m1.append(_cubic(-1.0 - wy))
                w_0.append(_cubic(0.0 - wy))
                w_p1.append(_cubic(1.0 - wy))
                w_p2.append(_cubic(2.0 - wy))
            return (torch.tensor(idx_m1, dtype=torch.int32),
                    torch.tensor(idx_0, dtype=torch.int32),
                    torch.tensor(idx_p1, dtype=torch.int32),
                    torch.tensor(idx_p2, dtype=torch.int32),
                    torch.tensor(w_m1, dtype=torch.float32),
                    torch.tensor(w_0, dtype=torch.float32),
                    torch.tensor(w_p1, dtype=torch.float32),
                    torch.tensor(w_p2, dtype=torch.float32))

        y_all = _compute_dim(H_in, H_out, scale_y)
        x_all = _compute_dim(W_in, W_out, scale_x)
        self._coord_cache[key] = (y_all, x_all)
    return self._coord_cache[key]
```

### 精度要点

- 提醒：用 `np.float32`，避免 Python `float`（64 位）导致精度偏差。
- 下标 clamp 在 host 完成，kernel 内不再判断边界。
- fp16/bf16 `align_corners=True` 的 scale 建议通过 NPU `vdiv` 计算（见 Phase 5）。
- 缓存到 `_coord_cache` 字典，key 为 `(H_in, W_in, H_out, W_out)`，避免重复预计算。

---

## Phase 2：UB 行预加载 + `tl.gather` 离散访存

### 触发条件

bilinear/bicubic/nearest 通用路径需要按运行时列下标从同一行取出多个像素，存在全局内存离散访问。

### 优化策略

#### 2.1 MAX_IN_W pow2 constexpr padding

建议用 `MAX_IN_W = triton.next_power_of_2(W_in)`（受 cap 限制，默认 4096）把输入宽度 pad 到 2 的幂，作为 constexpr 传入。

**Why:** Ascend 上 fixed-shape vector load 需要编译期常量长度；若 W_in 非 constexpr 会触发 dynamic-shape load，退化为标量循环。

```python
MAX_IN_W = triton.next_power_of_2(W_in)  # 受 TRITON_INTERP_MAX_IN_W cap (默认 4096)
# kernel 内:
r_offs = tl.arange(0, MAX_IN_W)
r_mask = r_offs < W_in
row = tl.load(input_ptr + base + r_offs, mask=r_mask, other=0.0)
```

#### 2.2 行预载 + tl.gather

整行 IW 一次性 load 到 UB（pad 到 MAX_IN_W），后续用 `tl.gather(row, x_idx, 0)` 按列下标向量取数：

```python
@triton.jit
def bilinear_general_vec_kernel(input_ptr, output_ptr,
                                N, C, H_in, W_in, H_out, W_out,
                                align_corners: tl.constexpr,
                                scale_h, scale_w,
                                BLOCK_W: tl.constexpr,
                                MAX_IN_W: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        # in-kernel 坐标计算 (ac=False 时 scale = H_in/H_out)
        src_y = scale_h * (h_f + 0.5) - 0.5  # ac=False
        y0 = tl.cast(src_y, tl.int32)
        # ... 边界处理 ...
        ylambda = src_y - tl.cast(y0, tl.float32)

        input_base = (n * C + c) * H_in * W_in
        # 一次性 load 两行到 UB
        row0 = tl.load(input_ptr + input_base + y0 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            # in-kernel 计算 x0, x1, xlambda
            # ...

            # 4 次 gather 取出双线性插值的 4 个像素
            i00 = tl.gather(row0, x0, 0)
            i01 = tl.gather(row0, x1, 0)
            i10 = tl.gather(row1, x0, 0)
            i11 = tl.gather(row1, x1, 0)

            top = x0lambda * i00 + xlambda * i01
            bot = x0lambda * i10 + xlambda * i11
            val = y0lambda * top + ylambda * bot

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### 2.3 跨行行复用（仅 nearest 通用路径）

nearest 通用路径中，相邻输出行可能映射到同一输入行（`src_yi` 相同）。缓存 `(last_n, last_c, last_src_yi)`，仅当变化时重新 load：

```python
last_n, last_c, last_src_yi = -1, -1, -1
in_row = tl.zeros((MAX_IN_W,), dtype=tl.float32)

for row in range(row_start, row_end):
    # ... 计算 n, c, src_yi ...
    if (n != last_n) | (c != last_c) | (src_yi != last_src_yi):
        in_row = tl.load(input_ptr + input_row_off + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        last_n, last_c, last_src_yi = n, c, src_yi
    # ... 用 tl.gather(in_row, src_xi, 0) 取像素 ...
```

### 适用条件

- bilinear / bicubic / nearest 通用路径。
- `W_in` 不超过 UB 预算（`MAX_IN_W * sizeof(dtype)` 在几十 KB 内，cap 默认 4096）。

---

## Phase 3：2x / 0.5x 整数倍采样特化

### 触发条件

整数倍上/下采样时，输出像素到输入像素的映射是规则的，gather 是随机访问，远慢于 contiguous。

### 3.1 2x 上采样 reshape broadcast 复制像素（nearest）

**触发**：`H_out == 2*H_in 且 W_out == 2*W_in`。

```python
@triton.jit
def nearest_2x_upsample_kernel(input_ptr, output_ptr,
                               N, C, H_in, W_in, H_out, W_out,
                               BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    BLOCK_IN: tl.constexpr = BLOCK_W // 2

    for row in range(pid, total_rows, num_programs):
        # ... 计算 n, c, h, src_h = h // 2 ...

        for w_start in range(0, W_out, BLOCK_W):
            in_w_start = w_start // 2
            in_offs = tl.arange(0, BLOCK_IN)
            in_idx = in_w_start + in_offs
            in_mask = in_idx < W_in

            vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            # [a, b, c] -> [[a],[b],[c]] * [[1,1]] -> [[a,a],[b,b],[c,c]] -> [a,a,b,b,c,c]
            vals = tl.reshape(vals, [BLOCK_IN, 1])
            ones = tl.full([1, 2], 1.0, tl.float32)
            vals_dup = vals * ones
            out_vals = tl.reshape(vals_dup, [BLOCK_W])

            tl.store(output_ptr + output_row_off + w_idx, out_vals, mask=mask)
```

**推广**：3x/4x 上采样把 `ones` 改为 `[1, 3]`/`[1, 4]`。

### 3.2 0.5x 下采样 strided contiguous load（nearest）

**触发**：`H_in == 2*H_out 且 W_in == 2*W_out`。

```python
@triton.jit
def nearest_half_downsample_kernel(input_ptr, output_ptr,
                                   N, C, H_in, W_in, H_out, W_out,
                                   BLOCK_W: tl.constexpr):
    # ... 同上 ...
    src_h = 2 * h  # 输出 h 对应输入 2h
    for w_start in range(0, W_out, BLOCK_W):
        in_w_start = 2 * w_start
        in_offs = tl.arange(0, BLOCK_W)
        in_idx = in_w_start + 2 * in_offs    # 步长 2
        in_mask = in_idx < W_in
        vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                       mask=in_mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + output_row_off + w_idx, vals, mask=mask)
```

### 3.3 bilinear 0.5x 下采样退化为 2x2 avg pool（ac=False）

> ## ⚠️⚠️⚠️ 绝对重点优化点（C1/C2，漏此条 geomean 必然 < 0.3x）⚠️⚠️⚠️
>
> **这是整个 interpolate 优化中单点收益最大的优化**。历史实测：
> - **极端 case**：漏掉它时 bilinear 0.5x 下采样单 case 仅 0.005x，补上后跳到 5x（1000 倍差距）
> - **geomean**：iter_3 用 strided load → geomean 0.27x；opt_iter_1 改 contiguous → 0.70x（单点改动决定达标）
>
> 直接决定 geomean 能否过 0.6x。
>
> **强制要求**：
> 1. **必须特化分派**（C2）：`_route` 中显式判断 `half_down` 并分派到 `bilinear_half_pool_kernel`，禁止走通用 bilinear 路径。
> 2. **必须 contiguous load + reshape sum**（C1）：每行一次连续 load `2*BLOCK_W` 元素，`reshape((BLOCK_W,2))` + `tl.sum(axis=1)`。
> 3. **❌ 禁止 strided load**：`x0=w_idx*2; x1=x0+1` 然后 4 次离散 `tl.load` 是错误做法，Ascend 上 strided load 无法 coalesce，比 contiguous 慢 100 倍。
>
> 迭代 checklist C1/C2 检查的就是这一节。

**触发**：`mode='bilinear' and not align_corners and H_in==2*H_out and W_in==2*W_out`。

**Why:** ac=False + 0.5x 下采样时，bilinear 的 4 个采样点正好是 2x2 输入块，权重全是 0.25，等价于均值池化。用 `reshape([BLOCK_W,2]) + tl.sum` 实现远快于通用 bilinear 的 gather。

```python
@triton.jit
def bilinear_half_pool_kernel(input_ptr, output_ptr,
                              N, C, H_in, W_in, H_out, W_out,
                              BLOCK_W: tl.constexpr):
    QUARTER = 0.25
    # ... 同上 ...
    y0 = 2 * h
    y1 = min(y0 + 1, H_in - 1)

    for w_start in range(0, W_out, BLOCK_W):
        in_start = 2 * w_start
        in_offs = tl.arange(0, 2 * BLOCK_W)
        in_idx = in_start + in_offs
        in_mask = in_idx < W_in

        row0 = tl.load(input_ptr + input_base + y0 * W_in + in_idx,
                       mask=in_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + in_idx,
                       mask=in_mask, other=0.0).to(tl.float32)

        # [2*BLOCK_W] -> [BLOCK_W, 2] -> sum axis=1
        row0_pairs = tl.reshape(row0, [BLOCK_W, 2])
        row1_pairs = tl.reshape(row1, [BLOCK_W, 2])
        row0_sum = tl.sum(row0_pairs, axis=1)
        row1_sum = tl.sum(row1_pairs, axis=1)

        val = (row0_sum + row1_sum) * QUARTER
        tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

**推广**：0.25x 下采样可推广为 4x4 avg pool。

---

## Phase 4：2D 垂直分块复用输入行（上采样）

### 触发条件

`H_out > H_in`（上采样）时，多个输出行共享同一批输入行，逐行独立处理会重复 load 输入。

### 优化策略

让单个 program 处理一个 2D tile：`BLOCK_H=2`、`MAX_KH=3`，一次性 load 3 行输入供 2 行输出复用。

```python
@triton.jit
def bilinear_2d_tiled_vec_kernel(input_ptr, output_ptr,
                                 N, C, H_in, W_in, H_out, W_out,
                                 align_corners: tl.constexpr,
                                 scale_h, scale_w,
                                 BLOCK_W: tl.constexpr,
                                 MAX_IN_W: tl.constexpr,
                                 BLOCK_H: tl.constexpr,
                                 MAX_KH: tl.constexpr):
    pid = tl.program_id(0)
    num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
    total_tiles = N * C * num_h_tiles
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    kh_offs = tl.arange(0, MAX_KH)

    for tile in range(pid, total_tiles, num_programs):
        # ... 计算 n, c, h_start ...

        # 计算 y_min (h_start 对应的 floor(src_y))
        # ...

        # 一次性 load [y_min, y_min + MAX_KH) 行输入到 flat UB buffer
        y_idx = y_min + kh_offs
        y_idx = tl.where(y_idx < 0, 0, y_idx)
        y_idx = tl.where(y_idx >= H_in, H_in - 1, y_idx)
        rr = y_idx[:, None]
        cc = r_offs[None, :]
        block2d = tl.load(input_ptr + input_base + rr * W_in + cc,
                          mask=(cc < W_in), other=0.0).to(tl.float32)
        flat_buf = tl.reshape(block2d, [MAX_KH * MAX_IN_W])

        for dh in range(BLOCK_H):    # 2 行输出复用同一批输入
            h = h_start + dh
            # ... 计算 y0, y1, ylambda ...
            ry0 = y0 - y_min
            ry1 = y1 - y_min

            for w_start in range(0, W_out, BLOCK_W):
                # ... 计算 x0, x1, xlambda ...

                # 从 flat_buf gather 4 个像素
                fx0 = ry0 * MAX_IN_W + x0
                fx1 = ry0 * MAX_IN_W + x1
                fx2 = ry1 * MAX_IN_W + x0
                fx3 = ry1 * MAX_IN_W + x1

                i00 = tl.gather(flat_buf, fx0, 0)
                i01 = tl.gather(flat_buf, fx1, 0)
                i10 = tl.gather(flat_buf, fx2, 0)
                i11 = tl.gather(flat_buf, fx3, 0)

                # ... 双线性插值 ...
                tl.store(output_ptr + output_base + dh * W_out + w_idx, val, mask=mask)
```

### TILE 选择

| 场景 | BLOCK_H | MAX_KH | BLOCK_W |
|------|---------|--------|---------|
| 上采样（bilinear/bicubic） | 2 | 3 | 128~512 |
| 上采样（4x 比例） | 4 | 5 | 128~256 |
| 下采样 | 1 | 1~2 | 256~512 |

约束：UB 占用 `< 192KB`，即 `MAX_KH * MAX_IN_W * sizeof(dtype) < 192KB`。

---

## Phase 4.5：bilinear `align_corners=True` 路径说明

### 触发条件

`mode='bilinear' and align_corners=True`，尤其是 `dtype in (fp16, bf16)`。

### v1 实际处理

v1 中 ac=True bilinear **不单独建 kernel**，直接复用 `bilinear_2d_tiled_vec_kernel`（上采样）或 `bilinear_general_vec_kernel`（下采样），仅通过 `align_corners: tl.constexpr` 参数区分坐标计算分支（`if align_corners == 1: src_y = scale * h` else `src_y = scale * (h + 0.5) - 0.5`）。

### 精度要点

提醒：ac=True 的 `scale = (H_in-1)/(H_out-1)` 必须用 `_npu_scale`（见 Phase 5）匹配 CANN vdiv 舍入，否则低精度 verify 不过。坐标计算在 kernel 内完成（`align_corners` 是 constexpr，分支在编译期消除）。

---

## Phase 5：CANN 舍入对齐（`_npu_scale`）

### 触发条件

`align_corners=True` 且 `dtype in (fp16, bf16)` 路径，出现坐标缩放导致的精度边缘失败。

### 问题

CPU float32 除法舍入与 NPU `vdiv` 可能不同，导致 kernel 坐标表与 PyTorch/CANN 不完全一致，
低精度路径 verify 不过。

### 优化策略

通过 NPU `vdiv` 计算 scale，再回 CPU 生成坐标表。用 `_SCALE_CACHE` 字典缓存避免重复同步：

```python
_SCALE_CACHE = {}

def _cpu_scale(num, den):
    """CPU float32, 无 host-device sync, 用于非低精度路径."""
    if den == 0:
        return 0.0
    return float(np.float32(num) / np.float32(den))

def _npu_scale(num, den):
    """NPU vdiv, 匹配 CANN 舍入. 带缓存避免重复同步."""
    if den == 0:
        return 0.0
    key = (num, den)
    if key not in _SCALE_CACHE:
        a = torch.tensor([num], dtype=torch.float32, device='npu')
        b = torch.tensor([den], dtype=torch.float32, device='npu')
        _SCALE_CACHE[key] = (a / b).cpu().item()
    return _SCALE_CACHE[key]

def _get_scales(pairs, use_npu=False):
    """批量获取多个 scale, use_npu=True 时用单次 batched NPU vdiv."""
    if not use_npu:
        return [_cpu_scale(num, den) for num, den in pairs]
    # ... 批量 NPU vdiv, 仅对未缓存的 (num, den) ...

def _need_npu_scale(dtype, align_corners):
    """仅低精度 + ac=True 需要 NPU vdiv."""
    return align_corners and dtype in (torch.float16, torch.bfloat16)
```

### 适用

- **仅** bilinear/bicubic `align_corners=True` 且 `dtype in (fp16, bf16)` 时启用。
- 其他场景用 `_cpu_scale`（无同步开销）。

---

## Phase 6：Bicubic `align_corners=True` 精度特化

### 触发条件

bicubic `align_corners=True` 出现随机 1 像素 fp32 边缘失败。

### 问题

- kernel 内向量计算 Keys' 权重时，浮点舍入与 PyTorch C++ 参考存在差异。
- 16 项向量化累加改变顺序，与 PyTorch 标量顺序不一致。

### 优化策略

#### 6.1 host 预计算 4×4 索引和权重（见 Phase 1.2）

#### 6.2 kernel 内标量逐项累加（16 项展开）

```python
@triton.jit
def bicubic_ac_true_vec_v2_kernel(input_ptr, output_ptr,
                                  y_idx_m1_ptr, y_idx_0_ptr, y_idx_p1_ptr, y_idx_p2_ptr,
                                  y_w_m1_ptr, y_w_0_ptr, y_w_p1_ptr, y_w_p2_ptr,
                                  x_idx_m1_ptr, x_idx_0_ptr, x_idx_p1_ptr, x_idx_p2_ptr,
                                  x_w_m1_ptr, x_w_0_ptr, x_w_p1_ptr, x_w_p2_ptr,
                                  N, C, H_in, W_in, H_out, W_out,
                                  BLOCK_W: tl.constexpr, MAX_IN_W: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    ZERO = tl.full((), 0.0, tl.float32)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        # ... 计算 n, c, h ...
        # 加载 4 行输入到 UB (row0..row3)
        # 加载 4 个 y 索引和 4 个 y 权重 (标量)

        for w_start in range(0, W_out, BLOCK_W):
            # 加载 4 个 x 索引和 4 个 x 权重 (向量)
            # 16 次 gather 取出 16 个像素 p00..p33

            # 标量逐项累加 (16 项全部展开, 顺序与 PyTorch C++ 一致, 对应 C7)
            val = ZERO
            w_sum_total = ZERO
            val = val + wy0 * wx0 * p00; w_sum_total = w_sum_total + wy0 * wx0
            val = val + wy0 * wx1 * p01; w_sum_total = w_sum_total + wy0 * wx1
            val = val + wy0 * wx2 * p02; w_sum_total = w_sum_total + wy0 * wx2
            val = val + wy0 * wx3 * p03; w_sum_total = w_sum_total + wy0 * wx3
            val = val + wy1 * wx0 * p10; w_sum_total = w_sum_total + wy1 * wx0
            val = val + wy1 * wx1 * p11; w_sum_total = w_sum_total + wy1 * wx1
            val = val + wy1 * wx2 * p12; w_sum_total = w_sum_total + wy1 * wx2
            val = val + wy1 * wx3 * p13; w_sum_total = w_sum_total + wy1 * wx3
            val = val + wy2 * wx0 * p20; w_sum_total = w_sum_total + wy2 * wx0
            val = val + wy2 * wx1 * p21; w_sum_total = w_sum_total + wy2 * wx1
            val = val + wy2 * wx2 * p22; w_sum_total = w_sum_total + wy2 * wx2
            val = val + wy2 * wx3 * p23; w_sum_total = w_sum_total + wy2 * wx3
            val = val + wy3 * wx0 * p30; w_sum_total = w_sum_total + wy3 * wx0
            val = val + wy3 * wx1 * p31; w_sum_total = w_sum_total + wy3 * wx1
            val = val + wy3 * wx2 * p32; w_sum_total = w_sum_total + wy3 * wx2
            val = val + wy3 * wx3 * p33; w_sum_total = w_sum_total + wy3 * wx3

            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

### 要点

- 提醒：权重和顺序需与 PyTorch C++ 实现一致。
- 累加顺序固定，建议不要向量化重排。
- 若仍出现精度问题，可回退到 `bicubic_ac_true_scalar_kernel`（纯标量逐像素）作为兜底。

---

## Phase 7：三次权重向量化分段（ac=False bicubic）

### 触发条件

bicubic ac=False 通用路径，kernel 内需要计算 Keys' 三次权重。

### 优化策略

提醒：用 `tl.where` 嵌套做向量分段，避免 `if` 分支（Triton kernel 内 `if` 对 vector 数据无效）：

```python
A = -0.75
def _cubic_weight(t):
    at = tl.abs(t)
    at2 = at * at
    at3 = at2 * at
    # |t| <= 1: (A+2)*|t|^3 - (A+3)*|t|^2 + 1
    w1 = (A + 2.0) * at3 - (A + 3.0) * at2 + 1.0
    # 1 < |t| < 2: A*|t|^3 - 5A*|t|^2 + 8A*|t| - 4A
    w2 = A * at3 - 5.0 * A * at2 + 8.0 * A * at - 4.0 * A
    return tl.where(at < 1.0, w1, tl.where(at < 2.0, w2, 0.0))
```

### 边界处理（ac=False）

ac=False 时 `src_y = scale * (i + 0.5) - 0.5`，可能为负。floor 需要用 `tl.where(src_y < y0_f, y0 - 1, y0)` 修正：

```python
src_y = scale_h * (h_f + 0.5) - 0.5
y0 = tl.cast(src_y, tl.int32)
y0_f = tl.cast(y0, tl.float32)
y0 = tl.where(src_y < y0_f, y0 - 1, y0)  # 负数 floor 修正
```

### 完整 kernel 骨架（`bicubic_vec_kernel`）

提醒：ac=False 通用 bicubic 在 kernel 内计算坐标和三次权重，4 行预载到 UB，16 次 gather 取像素。
权重用 `tl.where` 嵌套分段，累加顺序与 PyTorch C++ 对齐（先按行聚合 4 个 x 像素，再按列聚合 4 行）。

```python
@triton.jit
def bicubic_vec_kernel(input_ptr, output_ptr,
                       N, C, H_in, W_in, H_out, W_out,
                       align_corners: tl.constexpr,
                       scale_h, scale_w,
                       BLOCK_W: tl.constexpr,
                       MAX_IN_W: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    A = -0.75

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        # in-kernel 计算 src_y, in_y (floor with 负数修正), t_y
        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        src_y = sh * (h_f + 0.5) - 0.5  # ac=False
        in_y = tl.cast(src_y, tl.int32)
        in_y_f = tl.cast(in_y, tl.float32)
        # 负数 floor 修正: src_y < in_y_f 时 in_y - 1 (对应 C8)
        in_y = tl.where(src_y < in_y_f, in_y - 1, in_y)
        t_y = src_y - tl.cast(in_y, tl.float32)

        # 4 个 y 邻域下标 (clamped)
        yy0 = in_y - 1; yy1 = in_y; yy2 = in_y + 1; yy3 = in_y + 2
        # ... 每个都用 tl.where clamp 到 [0, H_in-1] ...

        # 4 个 y 权重 (tl.where 分段, t = -1-wy, -wy, 1-wy, 2-wy)
        # wy0 = _cubic_weight(-1.0 - t_y) ... 等

        w_sum_y = wy0 + wy1 + wy2 + wy3

        input_base = (n * C + c) * H_in * W_in
        # 4 行预载到 UB
        row0 = tl.load(input_ptr + input_base + yy0 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + yy1 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row2 = tl.load(input_ptr + input_base + yy2 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row3 = tl.load(input_ptr + input_base + yy3 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            # in-kernel 计算 src_x, in_x, t_x, 4 个 x 邻域下标, 4 个 x 权重
            # ... 同 y 方向 ...

            # 16 次 gather
            p00 = tl.gather(row0, xx0, 0); p01 = tl.gather(row0, xx1, 0)
            p02 = tl.gather(row0, xx2, 0); p03 = tl.gather(row0, xx3, 0)
            # ... row1, row2, row3 同理 ...

            # 按行聚合: r_i = sum_j wx_j * p_ij
            r0 = wx0 * p00 + wx1 * p01 + wx2 * p02 + wx3 * p03
            r1 = wx0 * p10 + wx1 * p11 + wx2 * p12 + wx3 * p13
            r2 = wx0 * p20 + wx1 * p21 + wx2 * p22 + wx3 * p23
            r3 = wx0 * p30 + wx1 * p31 + wx2 * p32 + wx3 * p33

            # 按列聚合
            val = wy0 * r0 + wy1 * r1 + wy2 * r2 + wy3 * r3

            w_sum_total = w_sum_x * w_sum_y
            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

---

## Phase 7.5：bicubic `align_corners=True` 精度兜底 kernel（v1 未启用）

### 说明

v1 实际只使用 `bicubic_ac_true_vec_v2_kernel`（Phase 6.2）处理所有 bicubic ac=True cases。
归档代码中保留了 3 个兜底 kernel 作为历史探索，**v1 未分派**，仅在 v2 出现 1 像素边缘失败时
才需要回退。提醒：下次生成时**不需要实现这些 kernel**，v2 已足够。

| 兜底 Kernel（仅供参考） | 精度 | 性能 | 适用场景 |
|--------|------|------|---------|
| `bicubic_ac_true_vec_kernel` | 中高 | 中 | host 只预算坐标，in-kernel 算权重 |
| `bicubic_ac_true_hybrid_kernel` | 高 | 弱 | UB 行 load + 标量累加 |
| `bicubic_ac_true_scalar_kernel` | 最高（与 PyTorch 完全一致） | 最弱 | 纯标量逐像素，最后兜底 |

若 v2 真的出现精度问题（历史上未发生），可参考归档代码 `interpolate_v1_20260623.py` 实现兜底 kernel。

---

## Phase 8：Area 模式行预载 + 动态窗口 gather

### 触发条件

`mode='area'`，输出像素对应输入的动态大小窗口 `[istartH:iendH, istartW:iendW]`。

### 优化策略

行预载 + 按动态窗口 gather 累加：

```python
@triton.jit
def area_vec_kernel(input_ptr, output_ptr,
                    N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W: tl.constexpr,
                    MAX_KW: tl.constexpr,
                    MAX_IN_W: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        # ... 计算 n, c, h ...
        istartH = (h * H_in) // H_out
        iendH = ((h + 1) * H_in + H_out - 1) // H_out
        kH = iendH - istartH

        for w_start in range(0, W_out, BLOCK_W):
            w_idx = w_start + tl.arange(0, BLOCK_W)
            mask = w_idx < W_out

            istartW = (w_idx * W_in) // W_out
            iendW = ((w_idx + 1) * W_in + W_out - 1) // W_out

            sum_val = tl.full([BLOCK_W], 0.0, tl.float32)
            for ih in range(istartH, iendH):
                # 整行 load 到 UB
                in_row = tl.load(input_ptr + input_base + ih * W_in + r_offs,
                                 mask=r_mask, other=0.0).to(tl.float32)
                # 按动态窗口 gather
                for iw_offset in range(MAX_KW):
                    iw = istartW + iw_offset
                    valid = (iw < iendW) & mask
                    pixel = tl.gather(in_row, iw, 0)
                    sum_val = tl.where(valid, sum_val + pixel, sum_val)

            kW = iendW - istartW
            area = tl.cast(kH, tl.float32) * tl.cast(kW, tl.float32)
            val = tl.where(area > 0.0, sum_val / area, sum_val)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

`MAX_KW = min((W_in + W_out - 1) // W_out, 64)`，覆盖最大可能的窗口宽度。

---

## Phase 9：Grid 与 Block Size 选择

### 9.0 获取核数

提醒：`num_cores` 决定 grid_cap，需从 triton runtime driver 读取 vector core 数。
读取失败时回退到 48（AICore 默认值）。

```python
def _get_num_cores():
    try:
        device = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(device)
        num_cores = props.get("num_vectorcore", -1)
        if num_cores <= 0:
            num_cores = props.get("num_aicore", 48)
        return max(num_cores, 1)
    except Exception:
        return 48
```

### 9.1 Grid 动态限制为核数

建议 `grid = (min(total_rows, grid_cap),)`，`grid_cap = num_cores`（可被 `TRITON_INTERP_GRID_CAP` 环境变量覆盖）。

**Why:** 上采样时 `total_rows = N*C*H_out` 可能极大，超 grid 上限；且多余 program 会空跑。

```python
grid_cap = int(os.environ.get('TRITON_INTERP_GRID_CAP', str(self.num_cores)))
grid = (min(total_rows, grid_cap),)
```

### 9.2 BLOCK_W 选择

`BLOCK_W` 选择 pow2，cap 512：

```python
def _select_block_w(self, W_out):
    cap = int(os.environ.get('TRITON_INTERP_BLOCK_W_CAP', '512'))
    if W_out <= cap:
        return triton.next_power_of_2(W_out)
    return cap
```

### 9.3 MAX_IN_W 选择

`MAX_IN_W` 选择 pow2，cap 4096：

```python
def _select_max_in_w(self, W_in):
    cap = int(os.environ.get('TRITON_INTERP_MAX_IN_W', '4096'))
    if W_in <= cap:
        return triton.next_power_of_2(W_in)
    return cap
```

### 9.4 1D Grid + 交织循环（program 间负载均衡）

每个 program 用 `for row in range(pid, total_rows, num_programs)` 织循环处理多行，避免尾部 program 空跑：

```python
for row in range(pid, total_rows, num_programs):
    # ... 处理第 row 行 ...
```

或分段式（适合行复用）：

```python
rows_per_prog = (total_rows + num_programs - 1) // num_programs
row_start = pid * rows_per_prog
row_end = min(row_start + rows_per_prog, total_rows)
for row in range(row_start, row_end):
    # ...
```

---

## Phase 10：编译选项 `multibuffer`

### 触发条件

插值 kernel 为内存密集型，大量 global load/store。

### 优化策略

建议 kernel 调用开启 `multibuffer=True`：

```python
nearest_vec_kernel[grid](..., multibuffer=True)
bilinear_general_vec_kernel[grid](..., multibuffer=True)
# ... 所有 9 个分派 kernel ...
```

对 tile 较小、计算密集度低的 kernel 可尝试 `unit_flag=True/False` 对比，但 `multibuffer=True` 是默认必备。

---

## Phase 11：Host 侧多策略分派决策树（完整）

```python
def _route(self, x, size, scale_factor, mode, align_corners, ...):
    N, C, H_in, W_in = x.shape
    if size is not None:
        H_out, W_out = int(size[0]), int(size[1])
    elif scale_factor is not None:
        H_out = int(H_in * scale_factor)
        W_out = int(W_in * scale_factor)

    output = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total_rows = N * C * H_out
    grid_cap = int(os.environ.get('TRITON_INTERP_GRID_CAP', str(self.num_cores)))
    grid = (min(total_rows, grid_cap),)
    BLOCK_W = self._select_block_w(W_out)
    MAX_IN_W = self._select_max_in_w(W_in)
    need_npu_scale = _need_npu_scale(x.dtype, align_corners)

    if mode == 'nearest':
        use_size = 1 if size is not None else 0
        if H_out == 2 * H_in and W_out == 2 * W_in:
            # 2x 上采样特化
            nearest_2x_upsample_kernel[grid](..., multibuffer=True)
        elif H_in == 2 * H_out and W_in == 2 * W_out:
            # 0.5x 下采样特化
            nearest_half_downsample_kernel[grid](..., multibuffer=True)
        else:
            # 通用 gather 路径
            scale_h, scale_w = _get_scales([(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
            nearest_vec_kernel[grid](..., multibuffer=True)

    elif mode == 'bilinear' and align_corners and H_out > 1 and W_out > 1:
        ac = 1
        scale_h, scale_w = _get_scales([(H_in-1, H_out-1), (W_in-1, W_out-1)], use_npu=need_npu_scale)
        if H_out > H_in:
            # 2D 垂直分块 (上采样)
            bilinear_2d_tiled_vec_kernel[grid_2d](..., BLOCK_H=2, MAX_KH=3, multibuffer=True)
        else:
            bilinear_general_vec_kernel[grid](..., multibuffer=True)

    elif mode == 'bilinear':
        ac = 1 if align_corners else 0
        # ... scale 计算 ...
        if not align_corners and H_in == 2 * H_out and W_in == 2 * W_out:
            # 0.5x 下采样退化为 avg pool
            bilinear_half_pool_kernel[grid](..., multibuffer=True)
        elif H_out > H_in:
            # 2D 垂直分块 (上采样)
            bilinear_2d_tiled_vec_kernel[grid_2d](..., BLOCK_H=2, MAX_KH=3, multibuffer=True)
        else:
            bilinear_general_vec_kernel[grid](..., multibuffer=True)

    elif mode == 'bicubic':
        ac = 1 if align_corners else 0
        if align_corners:
            # host 预计算 4x4 索引和权重 + 标量累积
            y_all, x_all = self._precompute_bicubic_ac_true_all(H_in, W_in, H_out, W_out)
            bicubic_ac_true_vec_v2_kernel[grid](..., multibuffer=True)
        else:
            scale_h, scale_w = _get_scales([(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
            bicubic_vec_kernel[grid](..., multibuffer=True)

    elif mode == 'area':
        max_kw = min((W_in + W_out - 1) // W_out, 64)
        area_vec_kernel[grid](..., MAX_KW=max_kw, multibuffer=True)

    return output
```

### 设计原则

- 接口不变，仅内部路由。
- 无匹配时回退到通用 kernel（行预载 + gather）。
- 路由开销 `< 0.1ms`。

---

## 9 个分派 Kernel 清单（v1 实际使用）

| Kernel | 触发条件 | 数据访问模式 |
|--------|---------|------------|
| `nearest_vec_kernel` | 通用 nearest | UB 缓存行 + tl.gather + 跨行行复用 |
| `nearest_2x_upsample_kernel` | H_out==2*H_in 且 W_out==2*W_in | contiguous load + reshape broadcast |
| `nearest_half_downsample_kernel` | H_in==2*H_out 且 W_in==2*W_out | strided contiguous load (步长 2) |
| `bilinear_general_vec_kernel` | 通用 bilinear（含 ac=True 下采样） | in-kernel 坐标计算 + 行预载 + gather |
| `bilinear_half_pool_kernel` | ac=False + 0.5x 下采样 | 退化为 2x2 avg pool, reshape + sum |
| `bilinear_2d_tiled_vec_kernel` | 上采样 (H_out>H_in，含 ac=True) | 2D 垂直分块 BLOCK_H=2 复用 MAX_KH=3 行 |
| `bicubic_vec_kernel` | ac=False 通用 bicubic | in-kernel 三次权重计算 (tl.where 分段) |
| `bicubic_ac_true_vec_v2_kernel` | ac=True bicubic | host 预计算 4×4 索引和权重 + 标量累积 |
| `area_vec_kernel` | 通用 area | 行预载 + 动态窗口 gather 累加 |

提醒：归档代码中还定义了 3 个 bicubic 兜底 kernel（见 Phase 7.5），**v1 未分派**，
仅作精度兜底备用，下次生成不需要实现。

---

## 常见陷阱与避免方法

### 陷阱 1: 用单 kernel 统一处理多 mode
- **问题**: 所有 mode 走同一套逻辑，丢失特化机会
- **解决**: 多策略分派（Phase 11），按 (mode, ac, 采样比例) 选择特化 kernel

### 陷阱 2: W_in 非 2 的幂导致 dynamic-shape load
- **问题**: `tl.arange(0, W_in)` 当 W_in 非 constexpr 退化为标量循环
- **解决**: `MAX_IN_W = triton.next_power_of_2(W_in)` (Phase 2.1)

### 陷阱 3: 整数倍上/下采样用 gather
- **问题**: 2x 上采样/0.5x 下采样用 gather 实现是随机访问，远慢于 contiguous
- **解决**: 2x 用 reshape broadcast (Phase 3.1)，0.5x 用 strided load (Phase 3.2)

### 陷阱 4: bilinear 0.5x 下采样走通用路径 【⚠️ 高频踩坑，对应 C1/C2】
- **问题**: ac=False + 0.5x 下采样时 bilinear 退化为 2x2 avg pool，但仍走通用 gather 路径浪费
- **解决**: 退化为 reshape + sum 的 avg pool (Phase 3.3)
- **⚠️ 进阶陷阱**: 即便特化了 `bilinear_half_pool_kernel`，如果 kernel 内部用 `x0=w_idx*2; x1=x0+1` 做 4 次 strided `tl.load`，性能依然极差。**必须用 contiguous load `2*BLOCK_W` + reshape + sum**，这是 strided vs contiguous 的 100 倍差距，不是 gather vs contiguous 的差距。
- **实测**: 极端单 case 0.005x → 5x；geomean iter_3 strided → 0.27x，opt_iter_1 contiguous → 0.70x。详见 Phase 3.3 警告块。

### 陷阱 5: 上采样逐行独立 load 输入
- **问题**: 上采样时多个输出行共享输入行，逐行 load 重复访存
- **解决**: 2D 垂直分块 BLOCK_H=2、MAX_KH=3 (Phase 4)

### 陷阱 6: ac=True kernel 内重新计算坐标
- **问题**: PyTorch C++ 的 ac=True 坐标用 float32 计算，kernel 内重算有精度差异，verify 失败
- **解决**: host 用 numpy.float32 预算坐标和权重 (Phase 1)

### 陷阱 7: bicubic 16 项向量化累加
- **问题**: 向量化累加改变顺序，与 PyTorch 标量顺序不一致，verify 失败
- **解决**: 标量逐项累加 `val = val + w*pixel` (Phase 6.2)

### 陷阱 8: 三次权重用 if 分段
- **问题**: `if at < 1.0: ...` 在 vector 数据上不生效
- **解决**: `tl.where` 嵌套 (Phase 7)

### 陷阱 9: grid_size 超核数
- **问题**: 上采样 total_rows 极大，直接 `grid=(total_rows,)` 超上限或空跑
- **解决**: `grid = (min(total_rows, num_cores),)` (Phase 9.1)

### 陷阱 10: 低精度 ac=True 用 CPU 计算 scale
- **问题**: CANN vdiv 与 CPU float32 除法有微小差异，低精度 ac=True verify 不过
- **解决**: `_npu_scale` (NPU vdiv) + `_SCALE_CACHE` 缓存 (Phase 5)，仅低精度 ac=True 需要

### 陷阱 11: nearest 通用路径逐行重新 load
- **问题**: 相邻输出行映射到同一输入行时，逐行重新 load 浪费
- **解决**: 缓存 `(last_n, last_c, last_src_yi)`，仅当变化时重新 load (Phase 2.3)

### 陷阱 12: ac=False 负坐标 floor 错误
- **问题**: ac=False 时 `src_y = scale * (i + 0.5) - 0.5` 可能为负，`tl.cast(src_y, tl.int32)` 向零截断而非 floor
- **解决**: `y0 = tl.where(src_y < y0_f, y0 - 1, y0)` 修正 (Phase 7)

---

## 验证规则

每个 Phase 独立执行：

1. 修改后检查 `references/checklist.md`。
2. 执行 `verify.py`，要求 `passed_cases == total_cases`。
3. 执行 `benchmark.py`，性能不劣化则保留。

### ⚠️ Phase 4 迭代额外强制步骤（防止漏掉重点优化点）

> 历史教训：Phase 4 迭代时容易"自创优化"而忽略本文档提炼的绝对重点优化点，导致性能不达标。
> 以下步骤强制执行，不可跳过。

**每轮 Phase 4 迭代前**（benchmark 之前）：

1. **逐条核对 C1–C13 checklist**（见文档开头的"⚠️ 核心优化点 Checklist"章节）
2. 对当前 `generated_code.py` / `optimized_code.py` 做如下 grep / 代码审查：
   - `grep -n "strided\|x0.*\*.*2\|w_idx.*\*.*2" optimized_code.py` — 若在 `bilinear_half_pool_kernel` 内命中 strided load 模式，**必须先改成 contiguous + reshape sum**（C1）
   - `grep -n "bilinear_half_pool\|half_down" optimized_code.py` — 若 bilinear dispatch 没有 `half_down` 特化分支，**必须补上**（C2）
   - `grep -n "tl.arange(0, W_in)\|tl.arange(0, H_in)" optimized_code.py` — 若有非 constexpr 的 `tl.arange(0, W_in)`，**必须改 `tl.arange(0, MAX_IN_W)`**（C4）
   - `grep -n "num_cores.*\*\|grid.*\*.*[0-9]" optimized_code.py` — 若 grid_cap 被乘以倍数，**必须改回 `min(total_rows, num_cores)`**（C10）
3. 任何 ❌ 未应用的 P0/P1 项，**必须先补上再 benchmark**，禁止跳过直接尝试自创优化
4. 自创优化（不在 C1–C13 范围内的改动）必须 benchmark 验证 ≥ 1.0x 才保留，否则回退

**判定规则**：
- benchmark 后 geomean < 0.6x 且有 P0 项未应用 → **禁止写 report**，必须先补 P0 项重测
- benchmark 后 geomean 退化（< 上一轮）→ 立即回退到上一轮代码，检查是否违反了某条 C 约束

## 性能预期

按 mode 聚合的典型加速比区间（归档参考）：

| Mode | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| area (downsample) | 5 | 18x ~ 33x | 极快，部分超 profiler 分辨率 |
| bilinear (downsample) | ~15 | 1.5x ~ 6x | 表现优异 |
| bilinear (upsample, ac=F) | ~10 | 0.3x ~ 1.8x | 中等表现 |
| bilinear (upsample, ac=T) | ~8 | 0.2x ~ 0.6x | 较弱，gather 开销 |
| bicubic (ac=F) | 4 | 0.15x ~ 3.4x | 高方差 |
| bicubic (ac=T) | 3 | 0.07x ~ 1.9x | 高方差 |
| nearest | 8 | 0.3x ~ 6.4x | 中等表现 |
| 全量 | 73 | ~1.16x | 达标 1.0x 归档阈值 |

**关键观察**:
1. 下采样路径是性能优势区（1.5x~33x），输出少、输入行复用充分。
2. 上采样路径较弱（0.2x~1.8x），输出多、每行独立计算，bicubic ac=True 标量累积是精度妥协。
3. 多策略分派是核心：不同 mode 和采样比例的最优数据访问模式完全不同。
4. 未来优化方向：上采样路径的向量化精度匹配（消除标量累积）、bicubic ac=True 的 gather 开销优化。


## 参考资料

- `latency-optimizer/references/checklist.md`
- `latency-optimizer/references/vector_core_partition.md`
- `latency-optimizer/references/scalar_to_vector.md`
- `latency-optimizer/references/discrete_memory_access.md`
- `latency-optimizer/references/loop-invariant-hoisting.md`
