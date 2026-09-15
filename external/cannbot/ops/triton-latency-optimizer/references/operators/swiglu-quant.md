# 激活函数 + 量化融合（Activation + Quantization Fusion）优化经验

**以 SwiGLUQuant 为典型案例，提炼可复用于同类算子的设计约束与优化技巧。**

**算子类别**: 激活函数 + 静态/动态量化融合（典型如 `swiglu_quant`、 gated activation + quant 等）
**典型特征**: 输入经激活函数变换后量化为 `int8` / `int4`（`torch.quint4x2`）；支持 smooth scale、offset、group index、per-row dynamic scale
**性能基准**: SwiGLUQuant 几何平均 **5.6929x** vs 纯 PyTorch 参考，52/52 cases 通过；相对原生算子约 **0.70x**

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 必须保持与 PyTorch 参考的 bit-exactness（除非明确允许近似）

- **必须**使用与参考实现完全一致的数学公式。
  - 示例（SwiGLU 的 sigmoid）：`t = exp(-abs(x)); num = where(x >= 0, 1.0, t); sig = num / (1.0 + t)`。
  - 动态 scale：`scale = float(int_scale) / max(max_abs, 1e-10)`，对应 `inv_max = 1.0 / max(...); scale = inv_max * int_scale_f`。
- **禁止**使用与参考语义不一致的近似实现（如 `tl.sigmoid`、tanh 近似、向量化变体等），除非经过全量精度验证。
- **Why**: 量化输出对 1-ULP 差异敏感；近似激活会改变 int4/int8 边界附近的舍入结果，导致大量字节差异。
- **How to apply**: 在 kernel 内用 `.to(tl.float32)` 展开精确公式；动态模式用 inline rcp-mul 模拟参考的浮点语义。

### L1.2 int4 布局必须对齐目标框架约定

- **必须**按 `packed = ((q_odd & 0x0F) << 4) | (q_even & 0x0F)` 打包，每字节存一对相邻元素（以 `torch.quint4x2` 为例）。
- **禁止**使用 `int16` view 或 bitcast 快速打包，除非已在目标架构上验证无 UB 溢出且性能净提升。
- **Why**: 原生算子与 `torch.quint4x2` 采用此布局；其他布局即使数学正确也会字节不一致。
- **How to apply**: 用 `tl.reshape(q, (BLOCK // 2, 2))` + `tl.split` 拆分偶奇，再 `& 0x0F`、`<< 4`、`|`，最后 `.to(tl.int8)` 存储。

### L1.3 禁止在生成的 Triton kernel 代码中调用 `torch_npu` 接口

- **必须**通过 Triton runtime 查询核数：`triton.runtime.driver.active.utils.get_device_properties(device_idx).get('num_vectorcore', 40)`。
- **禁止**使用 `torch_npu.npu.npu_config.get_device_limit` 等 torch_npu API 作为 kernel 调度参数来源。
- **Why**: 该任务是生成纯 Triton kernel，torch_npu 接口会污染生成代码并违反约束。
- **How to apply**: 在 `ModelNew.__init__` 中缓存 `triton.runtime.driver.active`，并通过它获取当前 stream / 核数。

### L1.4 动态量化与静态量化的输出 scale 语义不同

- **静态模式（quant_mode=0）**: `out_scales` 应返回与输入前缀形状同形的零张量（用户不关心，但形状必须一致）。
- **动态模式（quant_mode=1）**: `out_scales` 为 per-row 的 `float(int_scale) / max_abs`，形状为前缀形状。
- **Why**: 验证脚本会检查返回张量形状；混淆会导致形状错误。

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧调度模板

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._launch_cache = {}
        self._driver = triton.runtime.driver.active
        self._stream_cache = {}

    def _stream(self, device_idx):
        s = self._stream_cache.get(device_idx)
        if s is None:
            s = self._driver.get_current_stream(device_idx)
            self._stream_cache[device_idx] = s
        return s

    def forward(self, x, *, smooth_scales=None, offsets=None, group_index=None,
                activate_left=False, quant_mode=0, group_list_type=0, dst_type=None):
        x_shape = x.shape
        prefix_shape = x_shape[:-1]
        prefix_dims = x.numel() // x_shape[-1]
        half_dim = x_shape[-1] // 2

        is_int4 = dst_type is not None and (str(dst_type) == 'int4' or dst_type == torch.quint4x2)
        int_scale, clip_min, clip_max = (7, -8, 7) if is_int4 else (127, -128, 127)

        # 占位空张量，保证 kernel 签名稳定
        if smooth_scales is None:
            smooth_scales = torch.empty(1, dtype=torch.float32, device=x.device)
        if offsets is None:
            offsets = torch.empty(1, dtype=torch.float32, device=x.device)
        if group_index is None:
            group_index = torch.zeros(1, dtype=torch.int32, device=x.device)

        # 输出缓冲区
        if is_int4:
            temp_out = torch.empty((prefix_dims, half_dim // 2), dtype=torch.int8, device=x.device)
        else:
            temp_out = torch.empty((prefix_dims, half_dim), dtype=torch.int8, device=x.device)

        out_scales = (torch.empty(prefix_dims, dtype=torch.float32, device=x.device)
                      if quant_mode == 1 else torch.zeros(prefix_shape, dtype=torch.float32, device=x.device))

        x_ptr = x.view(prefix_dims, 2 * half_dim)
        # 根据 half_dim / is_int4 / quant_mode / activate_left 选择 kernel 与 block size，并缓存 compiled kernel
        ...
        return out_quant.view(...), out_scales
```

### L2.2 Kernel 路径选择树

| 模式 | 推荐路径 | 关键参数 |
|------|---------|---------|
| static, 单 tile 可覆盖 | `*_static_single_tile_kernel`（单 tile + 手动 DB） | `BLOCK_SIZE_COL = next_pow2(dim) ≤ 4096` |
| static, 多 tile | `*_static_tiled_kernel` | `BLOCK_SIZE_COL` 按维度阈值选 1024/2048/4096 |
| dynamic | `*_dynamic_fused_db_kernel` / `*_dynamic_fused_kernel` | `BLOCK_ROWS=4`，`BLOCK_SIZE_COL` 按维度选 |
| int4 pack | 独立 `pack_int8_to_int4_kernel`（静态）或 kernel 内 inline pack | 静态独立 pack 更易控制 UB |

### L2.3 Block Size 选择经验

- **int4**: `BLOCK_SIZE_COL = min(next_pow2(dim), 4096)`。
- **int8**: `dim > 2048 → 4096`; `> 1024 → 2048`; 否则 `1024`。
- **注意**: BLOCK 过大（如 8192/16384）在 int4 pack 时容易触发 UB 溢出；4096 是常见安全上限。

---

## Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

### L3.1 降低 Python 侧 launch overhead

- **技巧**: 在 `ModelNew` 中缓存 `triton.runtime.driver.active`、`get_current_stream(device_idx)` 和 `CompiledKernel` 对象。
- **效果**: SwiGLUQuant 本会话中 vs native 从 **0.5766x → 0.6952x**（+20.5%）。
- **How to apply**:
  ```python
  self._driver = triton.runtime.driver.active
  self._stream_cache = {}
  self._launch_cache = {}
  ```
  每次 forward 用 `self._stream(x.device.index)` 替代 `triton.runtime.driver.active.get_current_stream(...)`。

### L3.2 手动双缓冲替代或补充 `multibuffer=True`

- **技巧**: 对 static 单 tile 路径，在 kernel 内手动 prefetch 下一行数据（无条件 clamped prefetch），当前行计算与下一行 load 重叠。
- **对比**: 自动 `multibuffer=True` 在部分 case 下会触发 UB 溢出或 autotune 问题；手动 prefetch 更可控。
- **效果**: 在 SwiGLUQuant static int4 等 case 上，MTE2 与 VECTOR 重叠度良好，无显著结构性气泡。
- **注意**: prefetch 索引必须用 `tl.minimum(row_idx + num_progs, prefix_dims - 1)` 等无条件写法；条件 prefetch 在偶数 shape 下可能 miscompile。

### L3.3 动态 scale 的 inline rcp-mul

- **技巧**: 不要写成 `scale = int_scale_f / max_abs`，而是：
  ```python
  inv_max = 1.0 / tl.maximum(max_abs, 1e-10)
  row_scale = inv_max * int_scale_f
  ```
- **Why**: 与参考的 `float(int_scale) / y_max` 在 NPU rcp-mul 语义下更一致，scale 相对误差更小且分布稳定。

### L3.4 量化 clip 的指令链

- **推荐链**: `nearbyint` → `to(tl.int32)` → `tl.maximum` → `tl.minimum` → `to(tl.int8)`。
- **避免**: `tl.clamp` 或 `tl.clip` 在某些 BLOCK 大小下会触发 UB 溢出。
- **Why**: 这条链在 Ascend 上被验证可稳定编译且不溢出。

### L3.5 Host 侧按分支方向特化

- **技巧**: 若算子存在互斥分支（如 `activate_left` 决定只算一支），不要在 kernel 内用 `tl.where` 同时算两支；在 Host 侧根据分支参数选择只计算一支的 kernel。
- **Why**: 另一支完全浪费 vector 周期；SwiGLUQuant 中同时算两支比 host 特化慢约 2.1x。

### L3.6 Group index / smooth scale 的条件加载

- **技巧**: 对 `smooth_scales` / `offsets` 使用 scalar load（1D）或 vector block load（2D），并通过 `in_bounds` 条件保护：
  ```python
  if has_group_index:
      gid = row_idx // group_size
      gid = tl.minimum(gid, num_groups - 1)
      in_bounds = row_idx < num_groups * group_size
  ```
- **Why**: 避免越界行错误使用 group 参数；条件写 outside-boundary 时保持 identity（乘 1.0 或加 0.0）。

### L3.7 性能验证的 noise 控制

- **技巧**: 共享卡上同一次运行内的 A/B 对比会受 ±3% geomean 噪声影响；应做 min-of-3 rounds 或同进程内交替 interleave 测量。

---

## Layer 4: 典型坑表（Agent 应避免）

| 坑 | 现象 | 修复 |
|---|------|------|
| 激活函数近似 | int4 大量字节错 | 换与参考一致的精确公式 |
| int16 view 快速 pack | BLOCK ≥ 8192 时 UB 溢出 | 用 `reshape + split + & 0x0F + << 4 + \|` |
| 条件 prefetch | 偶数 shape miscompile | 无条件 clamped prefetch |
| `tl.clamp` 替代 max/min | 动态 int4 BLOCK 4096 UB 溢出 | 用 `nearbyint → int32 → max → min → int8` |
| kernel 内同时算互斥分支 | 速度慢 2x+ | Host 侧按分支参数特化 |
| 未缓存 driver/stream | 小 shape 被 enqueue 开销吃掉 | 缓存 `driver.active` 和 `get_current_stream` |
| 动态 scale 直接用除法 | scale 1-ULP 差异导致量化 ±1 | 改用 `inv_max * int_scale_f` |

---

## 与 latency-optimizer 优化点的对应关系

| 本类算子高频优化点 | 对应 latency-optimizer 序号 | 说明 |
|-----------------|---------------------------|------|
| 入参静态化（`activate_left`, `is_int4`, `quant_mode` 等） | 1 | 作为 `tl.constexpr` 传入，触发编译器特化 |
| Tiling / block size 选择 | 2 | int4/int8 采用不同 block size 阈值 |
| 分核优化 / grid 设置 | 3 | `grid = min(prefix_dims, num_cores)` 或按 tile 数 |
| Scalar 转 Vector（动态 scale） | 5 | tile 级 vector scale 计算避免标量除法 |
| Pass 消除 / 动态两 pass 合并 | 7 | 动态模式需两 pass；尝试合并时受 UB 限制 |
| 循环不变量外提（group scale/offset） | 10 | gid、scale_scalar 在外层循环计算 |
| Autotune 参数（BLOCK, NSTAGES, FAST_I8） | 13 | 可通过环境变量或 autotune 网格微调 |
| 混合策略（static/dynamic/int4/int8 多路径） | 14 | Host 侧根据参数选择 kernel |
| 冗余边界运算消除 | 17 | 检查 `tl.where(in_bounds, ...)` 是否冗余 |

---

## 何时停止优化

- 当 `latency-optimizer` 已无命中点时。
- 当进一步收益需要**改变激活/量化的数学公式**以匹配原生近似时，必须与用户确认精度容忍度（当前实现保持与 PyTorch 参考 bit-exact）。
- 当尝试的优化反复导致 UB 溢出或验证失败时，保留已验证的最稳版本。

---

## 如何复用到其他激活+量化融合算子

1. **替换激活函数**：保持 L1.1 的 bit-exact 原则，把 sigmoid 换成目标激活（如 GELU、SILU 等）的参考公式。
2. **调整量化位宽**：根据目标 dtype（int8/int4）调整 `int_scale`、`clip_min`、`clip_max` 和 pack 方式。
3. **保留 Host 调度骨架**：缓存 driver/stream、多路径 kernel 选择、动态/静态分支处理均可复用。
4. **按分支参数特化**：若算子有互斥计算路径，沿用 L3.5 的 Host 侧特化思路。
