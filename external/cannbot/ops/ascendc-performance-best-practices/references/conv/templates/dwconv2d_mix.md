# Depthwise Conv2D 混合核模板（`__mix__(1,2)`）

## 1. 模板用途

面向 direct-launch 的 **纯 Depthwise Conv2D**（`groups == C`、channel multiplier=1、NCHW）。  
算法对应简化组卷积：单个 opt-group 的 weight **全载 BL1**，AIV 做 NCHW→NZ，AIC 做 Cube。

## 2. 输入输出布局

| 张量 | Layout | Shape |
|------|--------|-------|
| `x` | NCHW | `[N, C, H, W]` |
| `weight` | — | `[C, Kh, Kw]` |
| `bias` | 1D | `[C]`（plugin 当前强制有） |
| `y` | NCHW | `[N, C, Ho, Wo]` |

## 3. 实现现状与使用条件

### 3.1 当前实现模板

| 模板 | 核类型 | 类名 | 编译期实例 | 运行期驱动 |
|------|--------|------|-----------|-----------|
| Depthwise mix | `__mix__(1,2)` | `DepthwiseConv2dKernel<T, L1PP>` | 6（{FP16/FP32/BF16} × {L1_PP=0,3}） | `DwTiling` 字段 |

编译期按 dtype × L1 乒乓特化（6 个实例）；其余 tiling（核切分、L1/L0 tile、enlarge 等）由运行期 `DwTiling` 字段驱动。

### 3.2 已实现能力

| 能力 | 说明 |
|------|------|
| dtype | FP16 / FP32 / BF16 |
| weight 全载 | 按 opt-group 一次性载 BL1（AIV NCHW→NZ→BL1，可双缓冲，两 AIV 按 `g%2` 交替） |
| AL1 | 尽量全载/切分 |
| L1 乒乓 | 编译期 `{0, 3}`（AL1/BL1 同开同关） |
| AIV/AIC 流水线 | AIV NZ 变换 → AIC Cube → CrossCore 握手 |
| enlarge 打包 | 多组拼接减少 per-group 固定开销 |
| bias | 必带（kernel 有 `hasBias` 字段，plugin 当前强制） |
| HF32 | FP32 可选（`hf32Enable` 字段） |
| 核切分 | `batchDim × groupDim × hoDim`（batch 可折入核内复用 weight） |

### 3.3 使用条件

| 条件 | 要求 |
|------|------|
| groups | `== C`，且 cin/cout per group = 1（channel multiplier=1） |
| Layout | contiguous NCHW |
| dtype | FP16 / FP32 / BF16 |
| L1 ping-pong | `{0, 3}`（AL1/BL1 同开同关） |
| 核切分 | `nDim == woDim == 1`（无真 HW 宽 Wo 核切） |
| Weight L1 再切 | 不允许（简化模板前提） |
| L1/L0/UB | 见 `ValidateTiling` + `ValidateMemory`：L1≤512KiB，L0A/B 半缓冲≤32KiB，L0C≤128KiB，UB≤256KiB |

### 3.4 未实现 / 不支持

| 能力 | 状态 | 扩展落点 |
|------|------|---------|
| 可选 bias | kernel 已有 `hasBias`；plugin 未放开 | [usage_guide.md §6.3](usage_guide.md) P0 |
| 真 HW `woDim>1` | 不支持 | [usage_guide.md §6.3](usage_guide.md) P1 |
| AL1/BL1 独立乒乓 | 仅 `{0,3}` | [usage_guide.md §6.3](usage_guide.md) P2 |
| Weight BL1 再切 | 不允许（需新模板） | [usage_guide.md §6.3](usage_guide.md) P2 |
| channel multiplier>1 | 不支持（需新模板） | [usage_guide.md §6.3](usage_guide.md) P3 |
| NHWC | 不支持 | [usage_guide.md §6.3](usage_guide.md) P3 |

## 4. 类名与入口

```cpp
namespace dwconv2d;

template <typename DTYPE, uint8_t L1_PINGPONG>
class DepthwiseConv2dKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const DwTiling& tiling);
    __aicore__ inline void Process();
};
```

共 6 个编译期实例：`{half,float,bfloat16_t} × {0,3}`。

```cpp
template <typename T, uint8_t L1PP>
__global__ __aicore__ __mix__(1, 2) void dwconv2d_kernel(..., DwTiling tiling);

extern "C" void launch_dwconv2d(..., const DwTiling& tiling, void* stream);
```

## 5. TilingData：`DwTiling`

定义在 `dwconv2d_tiling.template` → `dwconv2d_tiling.h`。要点：

| 字段组 | 含义 |
|--------|------|
| `hin/win/hout/wout/...` | 几何 |
| `enlarge/groupOpt/...` | opt-group 打包 |
| `batchDim/groupDim/hoDim/...` | 核切分 |
| `hoL1/hoL0/kL0/nL0` | L1/L0（`ho*` 是 M 空间 tile） |
| `l1PingPong` | 0 或 3 |
| `maxMPerCore/...` | `FinalizeDerived` 填的派生量 |

Host：`DwConv2dGetTiling(DwTilingIn, DwTiling&)`。

## 6. Host tiling 要点

```
BuildGeom → 核切分(M_MODE vs BASIC_BLOCK) → FillL0L1/DecideAl1
  → 可选行对齐 hoL1 精修 → ValidateAll → enlarge 回退 / BuildMinimal
```

## 7. 流水线

1. **AIV**：按 opt-group weight NCHW→UB→NZ→BL1（可双缓冲，两 AIV 按 `g%2`）  
2. **AIC**：AL1 载 fmap → L0 mad（`kL0` 切）→ fixpipe  
3. **CrossCore** 握手

策略索引见上级 [conv_common_perf_experience.md](../conv_common_perf_experience.md)：  
`loop_axis_simplification` / `weight_nz_transform` / `aiv_aic_pipeline` / `scalar_minimization`。

## 8. 不适用 / 替代

| 场景 | 去向 |
|------|------|
| 标准 Conv2D（groups=1，大 Cin） | [conv2d_cube.md](conv2d_cube.md) |
| 真 HW `woDim>1` / 独立乒乓 / BL1 再切 / multiplier>1 / NHWC | 见 [§3.4](#34-未实现--不支持) + [usage_guide.md §6.3](usage_guide.md) |

## 9. 文件清单

| 模板 | 落地 |
|------|------|
| `dwconv2d_tiling.template` | `dwconv2d_tiling.h`（DwTiling + ValidateTiling + tiling API） |
| `dwconv2d_tiling_impl.template` | `dwconv2d_tiling.cpp` |
| `dav310/dwconv2d_kernel.template` | `dwconv2d_kernel.h` |
| `dav310/dwconv2d_kernel_entry.template` | `dwconv2d_kernel.cpp` |
| `dav310/dwconv2d_launch.template` | `dwconv2d_launch.h` |
