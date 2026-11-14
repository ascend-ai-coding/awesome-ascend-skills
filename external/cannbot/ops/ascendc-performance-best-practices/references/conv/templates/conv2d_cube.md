# Conv2D Cube 模板（标准卷积）

## 1. 模板用途

面向 direct-launch 的 **标准 Conv2D**（groups=1、NCHW、必带 bias）Cube 内核。  
一份 `__cube__` kernel，编译期仅按 **dtype**（fp16 / fp32 / bf16）特化；L1 全载/切分、乒乓、splitM、splitN、split-K（`kAccum`）、HF32 等全部由运行期 `Conv2DCase` 字段驱动。

## 2. 输入输出布局

| 张量 | Layout | Shape |
|------|--------|-------|
| `x` | NCHW | `[N, Ci, Hi, Wi]` |
| `filter` | OIHW | `[Co, Ci, Kh, Kw]` |
| `bias` | 1D | `[Co]`（当前路径强制有 bias） |
| `y` | NCHW | `[N, Co, Ho, Wo]` |

属性：`strides[2]`、`pads[4]`、`dilations[2]`。

## 3. 实现现状与使用条件

### 3.1 当前实现模板

| 模板 | 核类型 | 类名 | 编译期实例 | 运行期驱动 |
|------|--------|------|-----------|-----------|
| Conv2D Cube | `__cube__` | `Conv2DKernel<T>` | 3（FP16 / FP32 / BF16 按 dtype 特化） | `Conv2DCase` 字段 |

编译期仅按 dtype 特化；L1 全载/切分、乒乓、splitM、splitN、split-K（`kAccum`）、HF32 全部由运行期 `Conv2DCase` 字段驱动。

### 3.2 已实现能力

| 能力 | 说明 |
|------|------|
| dtype 特化 | FP16 / FP32 / BF16 |
| 多核切分 | `batchDim × mDim × nDim`（batch 优先绑核，可吃满 36 核） |
| AL1/BL1 | 全载与切分；L1 乒乓（`l1PingPong` bit0=AL1，bit1=BL1） |
| L0 双缓冲 | 约定 `l0PingPong == 3` |
| splitM | `mAL1 > mL0`；前提：AL1 全载 Cin + AL1 单缓冲 |
| splitN | `nBL1 > nL0`；同 splitM 前提 + `nBL1 % nL0 == 0` |
| split-K | FP32 `kAccum`（L0C 分段 flush + FP32 atomic） |
| HF32 | FP32 Cube operand 舍入到 HF32（`hf32` 字段） |
| bias | 必带（bias→C2，`cmatrixSource=true`） |
| hoistWeight | BL1 全载且单缓冲时权重外提 |

### 3.3 使用条件

| 条件 | 要求 |
|------|------|
| groups | =1（非 depthwise / 非 group conv） |
| Layout | contiguous NCHW |
| dtype | FP16 / FP32 / BF16 |
| 多核 | `batchDim × mDim × nDim ≤ aicoreNum`；无多核切 K |
| L1/L0 | 见 `ValidateCase`：L1≤512KiB，L0A/B 半缓冲≤32KiB，L0C≤256KiB |
| splitM | `mAL1 > mL0` 时要求 `CinTileA == AlignedCi` 且 AL1 不乒乓 |
| splitN | `nBL1 > nL0` 时要求 `nBL1 % nL0 == 0` + AL1 全载 Cin + AL1 单缓冲 |
| split-K | 仅 FP32；`kAccum % kL0 == 0` |

### 3.4 未实现 / 不支持

| 能力 | 状态 | 扩展落点 |
|------|------|---------|
| 可选 bias | 未做 | [usage_guide.md §6.2](usage_guide.md) P0 |
| groups≠1 | 不支持（需新模板） | [usage_guide.md §6.2](usage_guide.md) P2 |
| NHWC layout | 不支持 | [usage_guide.md §6.2](usage_guide.md) P2 |
| 融合 epilogue | 不支持 | [usage_guide.md §6.2](usage_guide.md) P2 |

## 4. 类名与入口

```cpp
namespace conv2d_cases;

template <typename T>
class Conv2DKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, Conv2DCase tiling);
    __aicore__ inline void Process();
};
```

Kernel 入口（`conv2d_kernel_entry.template` → `conv2d_kernel.cpp`）：

```cpp
template <typename T>
__global__ __aicore__ __cube__ void conv2d_kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                  Conv2DCase tiling)
{
    Conv2DKernel<T> op;
    op.Init(x, weight, bias, y, tiling);
    op.Process();
}

extern "C" void launch_conv2d(..., const Conv2DCase& tiling, void* stream);
```

## 5. TilingData：`Conv2DCase`

定义在 `conv2d_tiling.template` → `conv2d_tiling.h`。关键字段：

| 字段 | 含义 |
|------|------|
| `n..dilationW` | 问题几何与属性 |
| `batchDim/mDim/nDim/coreM` | 多核切分 |
| `mAL1/kAL1/kBL1/nBL1` | L1 tile |
| `mL0/kL0/nL0` | L0 tile |
| `kAccum` | 单核内 split-K；0=整段 K 写一次 GM |
| `l1PingPong` | bit0=AL1，bit1=BL1 |
| `l0PingPong` | 约定为 3（L0 双缓冲） |
| `hf32` | FP32 可选 HF32 |

Host API（`conv2d_tiling.template` + `conv2d_tiling_impl.template`）：

```cpp
Conv2DTilingIn in{...};  // shape/attrs + aicoreNum
Conv2DTilingOut out{};
bool ok = Conv2DDoTiling(in, out);
Conv2DCase tiling = ToCaseView(in, out);
```

## 6. Host tiling 要点

- BasicBlock 搜索：核切分 → L0 → L1（含 splitM / kAccum / HF32 约束）
- `ValidateCase` 失败则 plugin `TORCH_CHECK` 报 outside supported constraints

## 7. Kernel 数据流（摘要）

```
GM(x) ──Dn2Nz──► AL1 ─Load3d─► L0A ──┐
GM(W) ──Nd2Nz/Dn2Nz──► BL1 ─Load2d─► L0B ──┼─ Mmad (+ bias BT) ──► L0C ──Fixpipe──► GM(y)
GM(bias) ──► BiasL1/BT ─────────────┘
```

- **splitM**：`mAL1 > mL0`，要求 AL1 全载 Cin + AL1 单缓冲  
- **splitN**：`nBL1 > nL0`，同一套 AL1 全载 + 单缓冲前提；BL1 窗内按 `nL0` 切子块，
  `LoadBl0` 用 `MStartPosition = nOffset/16` 取窗内 fractal、BT 每子块重载一次。
  收益是 fmap 的 GM 重读次数从 `coreN/nL0` 降到 `coreN/nBL1`（splitM 与 splitN 共用
  `ComputeWindow`，可同时开）
- **split-K**：`kAccum != 0`，L0C 分段 flush + FP32 atomic（仅 FP32）  
- **hoistWeight**：BL1 全载且单缓冲时权重外提

## 8. 关键优化（与 `conv_common_perf_experience.md` 对照）

标准 Conv2D 单次 Mmad 的 `kL0` 通常较大，对 Scalar 不如 depthwise 敏感；重点在：

- 合理 `batchDim` 吃满核（避免只切 M/N 卡在 32 核）
- L1 全载 vs 切分 + 乒乓与读放大权衡
- FP32 长 K 用 `kAccum` 控误差

## 9. 不适用 / 替代

| 场景 | 去向 |
|------|------|
| `groups == C` 纯 depthwise | [dwconv2d_mix.md](dwconv2d_mix.md) |
| 可选 bias / NHWC / groups≠1 / 融合 | 见 [§3.4](#34-未实现--不支持) + [usage_guide.md §6.2](usage_guide.md) |

## 10. 文件清单

| 模板 | 落地 |
|------|------|
| `conv2d_tiling.template` | `conv2d_tiling.h`（Conv2DCase + ValidateCase + tiling API） |
| `conv2d_tiling_impl.template` | `conv2d_tiling.cpp` |
| `dav310/conv2d_kernel.template` | `conv2d_kernel.h` |
| `dav310/conv2d_kernel_entry.template` | `conv2d_kernel.cpp` |
| `dav310/conv2d_launch.template` | `conv2d_launch.h` |
