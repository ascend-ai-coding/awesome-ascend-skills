# Scatter 的精度处理与四个工程陷阱

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 总览见 [scatter_design.md](scatter_design.md)。本文是"性能优化过程中会把你打回原形"的那些事。
> 前两节是精度（不处理就过不了判据），后两节是**只在特定环境暴露、本地全绿也照样挂**的陷阱。

## 1. 半精度的累积型 reduce 必须在 fp32 域做

### 结论

| reduce | fp16 / bf16 的走法 |
|---|---|
| `add` / `multiply`（累积型） | **必须走 fp32 累加器**：`cast in → fp32 域累加 → cast out`，整条链只舍入一次 |
| `amin` / `amax` | 直接用原生 half/bf16 原子。**不产生舍入**，绕 workspace 反而白付两趟 cast |
| `none` | 只搬比特，不涉及 |

### 为什么硬件原子加救不了

`asc_atomic_add(__gm__ half*)` **是存在的**——但声明在 `simt_api/asc_fp16.h`，**不在** `device_atomic_functions.h`，
很容易以为没有。真正的问题是：**它每次操作都把结果舍回 fp16**。

实测把软件版换成硬件原生原子后，MARE **逐位重现** `0.020576`，与软件版完全一致 —— 说明误差来源不是软件实现，
而是"每步截回 fp16"这件事本身。golden 是 fp64 累加，torch CPU 的 half 累加也内部提升到 float。**绕不过去。**

### 代价与优化

fp32 workspace 要付两趟 cast（`data → f32`、`f32 → y`）。**优化点不是消掉它，而是别用 SIMT 做它**：

> 把 cast 从 SIMT 标量循环改成 **SIMD 的 MTE 搬运 + `AscendC::Cast`**：
> 某用例 **1565us → 89us**；67M 元素的大用例上两趟 cast 的占比从约 **47% 降到约 21%**（640us / 3000us）。

剩下的 21% 绕不过去——硬件没有精度够用的 fp16 原子加。

```cpp
template <typename DST, typename SRC>
__global__ __aicore__ __vector__ void scatter_cast_kernel(GM_ADDR dst, GM_ADDR src, int64_t count)
{
    /* 每核连续分片，TILE=16384 元素，DataCopyPad 进 → Cast → DataCopyPad 出 */
}
```

## 2. bf16 的舍入模式必须是 RINT

```cpp
if constexpr (AscendC::IsSameType<DST, bfloat16_t>::value) {
    // f32 -> bf16 必须 round-to-nearest-even，与 torch 对齐。
    // CAST_ROUND 是 round-half-away-from-zero，会在 tie 上差 1 ULP。
    AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_RINT, n);
} else {
    AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_NONE, n);
}
```

`CAST_ROUND` 看起来是"正常的四舍五入"，但它是 **round-half-away-from-zero**；
torch 用的是 **round-to-nearest-even**。差别只在 tie 上的 1 ULP —— 足以让 bf16 的 add 在通用路径和纤维路径**双双挂精度**。

> 一般规律：**任何 f32 → 窄浮点的落地都先确认舍入模式与参照实现一致**，这类 bug 表现为"只差一点点"，
> 极容易被误判成"精度阈值该放宽"。阈值不是用来放宽的。

## 3. 陷阱 1：输出必须是连续张量

**症状**：报"精度不达标"，但同时给出 `MERE = 0.000000, MARE = 0.000000` —— 数值其实完全正确。
日志深处才是真因：

```
AclNN_Inner_Error(EZ9999): Op StridedSlice does not has any binary.
RuntimeError: copy_d2d_baseformat_opapi ... device error type 0, error code is 561103
```

**机理**：评测框架对输出做 `output.cpu()`，非连续张量要先 `OptimizeContiguous` → 调 `StridedSlice`，
而部分 CANN 包（实测 950PR runner 的 **9.1.0-beta.1**）**没有 `StridedSlice` 的算子二进制**。
开发机上的 9.2.0 有，所以**本地怎么测都测不出来**。

**踩过的具体形态**：为省掉打包键路径的 unpack pass，把 int32 workspace 的低 16 位用
`as_strided(sizes, stride×2)` 直接当输出返回。**本地四项回归全绿、分数还更高（73.54 / 20-20），平台 7/20、41.31。**

**规避**：输出一律返回 `torch::empty_like(data)` 这类连续张量，unpack 那一趟老实付掉。

## 4. 陷阱 2：快路判据必须排除通用路径

`usePackArb / useWinPass / useWarpLast / useAtomicEP` **每一条都要带 `extentsMatch`**。

漏掉 `extentsMatch` 时，"2 字节 + none + starved" 会让 `usePackArb` 恒真：`y` 被重绑到 workspace 的
`stride×2` 低半字视图，而通用分支在派发函数里排在最前、按连续下标写 —— **输出以 2 倍元素步长整体错位**，
fp16/bf16 的 none 全挂。

公开用例里往往一条通用路径都不触发，所以**标准集满分也照样挂隐藏集**。

## 5. 回归网怎么搭

四项，前三项纯正确性、约两分钟跑完。**改完代码提交前必须全绿。**

| 测试 | 组数 | 拦的是什么 | 需要 NPU |
|---|---|---|---|
| CPU 语义模拟器 | 50 | 逐行复现 kernel 的索引算术与遍历顺序，与 torch golden 严格对拍。拦语义错 | 否 |
| 通用路径回归 | 147 | 非 dim 维范围不等的所有组合（陷阱 2） | 是 |
| **输出连续性** | 200 | 断言 `is_contiguous()` **并真的调 `.cpu()`**（陷阱 1） | 是 |
| 完整评测 | 20 | 精度 + 性能总分 | 是 |

> 第三项的关键在**真的调 `.cpu()`**。只断言 `stride` 不够 —— `as_strided` 那次的 stride 检查是能过的，
> 只有真正触发 D2H 拷贝才会踩到缺失的 `StridedSlice` 二进制。
> **回归要复现的是"平台会做的那个动作"，不是"我认为等价的那个检查"。**

## 6. 陷阱 3：`__vector__` 不能漏

含 SIMT VF 的 `__global__` 核函数**必须显式标 `__vector__`**（`__attribute__((aiv))`）：

```cpp
template <typename T, typename IDX>
__global__ __aicore__ __vector__ void scatter_warp_last_kernel(...)
{
    asc_vf_call<ScatterWarpLastVF<T, IDX>>(dim3(SCATTER_THREADS), ...);
}
```

漏掉时 `ld.lld` 报 `auto derivate failed`，报错信息与真因隔得很远。纯 SIMD 核不需要。

## 7. 陷阱 4：性能读数会被同机负载污染

多进程共卡的 runner 会把两个评测进程压在同一张卡的两个 device 上，部分用例时延被拉高**一个数量级**。

**判别方法（比看总分可靠得多）**：**找同一条代码路径、同一量级规模的用例互相比对**。

| 用例（同为打包键路径） | 污染跑 | 干净跑 |
|---|---|---|
| 1.05M 元素 | 794.4us | **44.0us** |
| 约 1M 元素 ×3 条 | 约 40us | 约 40us |

一比就露馅。另外两条辅助判据：

- **代码缺陷会按 dtype / reduce / shape 结构分布，不会按用例序号整齐断开。**
  "前 10 条正常、后 10 条整齐慢 10~15 倍、五条不同代码路径同时塌" 一定是机器不是代码。
- 结果里的 `device` 字段是 `npu:6,7` 而不是 `npu:0` 就要警惕。

遇到异常低分**先看 runner，重跑一次再下结论**，别急着改代码。
