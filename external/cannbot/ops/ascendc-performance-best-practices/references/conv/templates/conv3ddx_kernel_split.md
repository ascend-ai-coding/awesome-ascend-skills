# Conv3DDX kernel split 模板（stride≥2 反卷积 kernel 拆分）

## 1. 模板用途

面向 direct-launch 的 **Conv3D 输入梯度（Conv3DDX / conv3d_backprop_input）** 中
空间 stride > 1 的场景：把"前放大（插零）"路径中"与 0 相乘"的无效计算按**输出相位**
拆解为若干 **stride=1 稠密子卷积**，恢复 MTE2 有效带宽并消除无效 MMAD。

覆盖两种拆分模式（选型规则见 §2.5）：
- **SplitHW**（本模板已实现）：strideH=strideW=2，H/W 双向拆分，收益最大；
- **SplitH**（§10 专章，ops-nn 已实现、模板未抽取）：仅 H 方向拆分，覆盖 SplitHW
  准入之外的 stride>1 场景（大 kernel、strideH>2、strideW=1 等），复杂度受控。

一份 `__mix__(1,2)` kernel，编译期仅按 **dtype**（fp16 / bf16）特化；子核描述、
B1 全载、workspace 中转、输出 interleave 全部由运行期 `Conv3ddxKsCase` 字段驱动。

> 本模板来自 ops-nn `conv3d_backprop_input_v2` arch35（DAV_3510 / Ascend950）
> kernel split 模板（tiling 注册优先级 98）的知识抽取。**本目录自包含**：模板所需的
> 全部基座方法（Cube 映射、CalFmapH 几何、基本块/stepK 搜索、核间切分、L0 乒乓）
> 已内联在本文与 [conv3ddx_kernel_split_tuning_guide.md](../conv3ddx_kernel_split_tuning_guide.md)
> §11 中，不依赖其他模板文档。

## 2. kernel 拆分理论（为什么 kernel split 是稀疏计算优化）

### 2.1 传统"前放大"路径及其代价

传统 DX 实现将 dY 按 stride 插零膨胀，再与空间翻转后的卷积核做 stride=1 卷积：

```
dX = Conv_stride=1( Pad_bp[ Dilate_(Sh,Sw)(dY) ], W̃ )
```

以 stride=2 为例（真实 dY → 膨胀平面 D）：

```
原始 dY            前放大后的 D
[a  b]             [a  0  b]
[c  d]      =>     [0  0  0]
                   [c  0  d]
```

传统数据流中，"0"既占据 L1/L0 的存储与搬运带宽，也进入 img2col 后的 A 矩阵消耗
Cube 算力。内部区域真实元素密度 ≈ `1/(Sh×Sw)`，因此 stride 越大浪费越高：

| 浪费 | 机制 | stride=2、kernel=3×3 量级 |
|------|------|--------------------------|
| MTE2 带宽 | GM→L1 的 Dn2Nz 以 `dstNzNStride=strideW` 写带空洞的 L1 数据（空洞=零） | 有效带宽 ≤ 1/4 |
| 无效 MMAD | K 轴 = Cout×HkWk 全长参与，相位不匹配的乘加全在乘零 | ≈ 4/9 |
| Load3D pad 洪泛 | 反向 pad 被 stride 放大（`bpPad ≈ Wi − (Wo−1)·stride + …`），撞 padList=255 上限触发降级 | 频繁 |

### 2.2 精确的相位同余条件（优化成立的根因）

把正向卷积写成 cross-correlation 形式，dX 的一个空间元素：

```
dX[i,j] = Σ_{r∈[0,Kh)} Σ_{c∈[0,Kw)} W[r,c] · dY[ (i+Ph−r)/Sh, (j+Pw−c)/Sw ]
```

只有当 dY 下标为整数且在有效范围内时该项才存在，因此 **r、c 必须满足同余条件**：

```
r ≡ i + Ph (mod Sh)        c ≡ j + Pw (mod Sw)
```

即：**对任意固定的 dX 输出相位 `(i+Ph) mod Sh, (j+Pw) mod Sw`，只有 kernel 中一个
固定 stride 相位组的元素能乘到真实 dY**；前放大路径的其余 kernel 元素只会命中 0。
拆分方案直接不发起这些无效乘法。

> 上式用正向 kernel 坐标说明数学关系；设备实现在 B 矩阵装载时做空间逆序（filter
> 翻转），子核内部元素顺序可能相反，但每个元素所属的相位组不变，结果等价。

### 2.3 子 kernel 的数学定义

对相位 `a ∈ [0,Sh)`、`b ∈ [0,Sw)`，定义子 kernel（只保留落在原核范围内的元素）：

```
W^(a,b)[u,v] = W[a + u·Sh, b + v·Sw]
```

该定义把完整 kernel 划分为最多 `Sh×Sw` 个互不重叠的子 kernel。当前位置 `(i,j)`
选择 `a = (i+Ph) mod Sh`、`b = (j+Pw) mod Sw`。

**子 kernel 尺寸公式**（长度 K、stride S 的维度，相位 a）：

```
K^(a) = floor((K−1−a)/S) + 1    （a < K 时；a ≥ K 时该组为空，不得进入 MMAD）
baseIter = K mod S
K ≥ S 时所有相位组非空：baseIter 个组长为 ceil(K/S)，其余组长为 floor(K/S)；
baseIter = 0 时所有组等长。哪个物理相位属于较长组取决于 kernel 是否已逆序，
长度分布本身不变。
```

**4×4 kernel、stride=2 的完整例子**（4 个 2×2 子核，按正向 kernel 坐标）：

```
W^(0,0) = [w00 w02]   W^(0,1) = [w01 w03]
           [w20 w22]             [w21 w23]
W^(1,0) = [w10 w12]   W^(1,1) = [w11 w13]
           [w30 w32]             [w31 w33]
```

相位与 dX 位置的对应（Ph=Pw=0 为例）：`dX 的 (i mod 2, j mod 2)` = (0,0)→W^(0,0)、
(0,1)→W^(0,1)、(1,0)→W^(1,0)、(1,1)→W^(1,1)。完整计算从"带棋盘格 0 的 dY × 4×4
kernel"变为 4 个只含真实数据的计算 + 结果交叉合并。

**正确性不变量**（拆分实现必须同时满足）：

1. 所有子 kernel 元素的并集恰好等于原 kernel，且任意两组不重叠
   （Σ 各组元素数 = Kh×Kw，每个元素只出现一次）；
2. 对任意有效 dX 坐标，基于 pad 计算出的相位只选择一个正确子 kernel；
3. 子 kernel 对应的 dY 窗口、边界裁剪和输出位置与前放大公式完全一致
   （反向 pad 由正向 pad、kernel 逆序和 output padding 推导，需按相位分别计算，
   不能把 Ph/Pw 当固定偏移）；
4. 输出交叉重排把每个紧凑结果写回唯一的全局 dX 坐标，不丢失也不重复。

### 2.4 拆分后的收益

拆分后每个相位面的计算是 **stride=1 的普通内积**（相位面内相邻 dx 点对应的 dedy
步长恰为 1）：

| 收益 | 机制 | 量级 |
|------|------|------|
| MTE2 带宽恢复 | A1 装载无洞：`dnNum=1, nValue=curHoSize×realWo` 一段连续 NZ | ×stride² |
| 无效乘加消除 | K = Cout×subHkWk（≈HkWk/stride²）全部有效 | ×stride²（乘法量） |
| Load3D pad 收缩 | 子核 pad 按相位独立重算，远离 255 上限 | 消除切 HkWk 降级 |
| A1/B1 全载机会 | K 短了，L1 可驻留整个 Cout 分形（`b1FullLoad`） | 消除 B1 重搬 |

**代价**：每个相位面算出的 dx 是**交错的**（像素网格中每 stride×stride 取 1），
输出侧必须把 stride² 个相位块 interleave 回原始排布——由 AIV 核完成（见 §7）。

> **一句话**：stride>1 的反卷积在稠密 MMAD 视角下是 1/stride² 的结构化稀疏；
> kernel split 把结构化稀疏转化为 stride² 个稠密小子问题 + 一次输出重排。

### 2.5 SplitHW 与 SplitH 两种模式及选择规则

| 模式 | 选择条件 | 子核数 | dY 空洞处理 | 输出方式 | 主要取舍 |
|------|---------|-------|------------|---------|---------|
| 不拆分 | stride=1、kernel 不满足拆分条件，或 tiling 未使能 | 0 | 传统前放大 | 常规写回 | 控制最简单，stride>1 时零值开销高 |
| **SplitHW**（本模板） | Sh=Sw=2，且 Kh、Kw ∈ {2,3,4} | Sh×Sw=4 | H/W 均无前放大 | AIC/AIV 协同交叉重排 | 有效计算最充分；需相位切换、workspace、同步与重排 |
| **SplitH**（§10 专章） | strideH ≥ 2 且 Kh ≥ strideH 的其余场景（详见 §10.3 准入） | Sh | 只消除 H 方向前放大，W 保持完整行 | 按有效行直接写出（无 AIV 重排） | 复杂度受控（相位维度进核间切分），收益低于 SplitHW |

SplitH 的子 kernel 为 `W^(a)[u,v] = W[a+u·Sh, v]`（W 方向不拆）。不全维切分的原因：
组数放大到 Sh×Sw 后，相位切换、pad 推导、地址计算、输出重排和标量控制开销会抵消
省下的 Cube 计算。"可拆分"首先要求 stride>1 且对应 kernel 维度不小于 stride。

```
stride=1 或不满足拆分条件 ──> 常规 DX 路径（本模板不覆盖）
Sh=Sw=2 且 Kh、Kw∈{2,3,4} ──> SplitHW（本模板）
其他已启用的 stride>1 场景 ──> SplitH（§4.4 扩展落点）
```

## 3. 输入输出布局

| 张量 | Layout | Shape |
|------|--------|-------|
| `dedy`（out_backprop） | NCDHW | `[N, Cout, Do, Ho, Wo]` |
| `filter` | NCDHW（OIHW-D 扩展为 `[Cout, Cin, Kd, Kh, Kw]`） | `[Cout, Cin, Kd, Kh, Kw]` |
| `dx` | NCDHW | `[N, Cin, Di, Hi, Wi]` |

属性：`strides[3]`（sH=sW=2）、`dilations[3]`（=1）、`pads[6]`（上下/左右对称）。

## 4. 实现现状与使用条件

### 4.1 当前实现模板

| 模板 | 核类型 | 类名 | 编译期实例 | 运行期驱动 |
|------|--------|------|-----------|-----------|
| Conv3DDX kernel split | `__mix__(1,2)` | `Conv3ddxKsKernel<T>` | 2（FP16 / BF16） | `Conv3ddxKsCase` 字段 |

### 4.2 已实现能力

| 能力 | 说明 |
|------|------|
| dtype | FP16 / BF16（fp32 模板侧兼容，未验证） |
| 拆分模式 | **SplitHW**（strideH=strideW=2，kernel ∈ 1×1~4×4）与 **SplitH**（strideH≥2、Kh∈[strideH,16]）双模式，tiling.splitMode 驱动同一份 kernel 骨架；SplitH 设计见 §10 |
| 子核轮转 | 4 个子核在基本块内步进（W 先 H 后），`splitIndex = splitHIndex*strideW + splitWIndex` |
| B1 全载 | filter 单次进 L1，子核切换用 Load2d `MStartPosition` 跳相位，不重搬（`b1FullLoad`） |
| 输出 interleave | L0C→UB 三段对齐 fixpipe + AIV Interleave（交叉）+ DataCopyPad 隔行直写 NCDHW |
| workspace 中转 | UB 放不下 4 块相位缓冲时 `useWorkspace=1`，L0C→GM(ws)→UB→interleave→GM |
| 多核切分 | `batch × dout × mCnt(splitHiWi) × nCnt(Cin)`，M 按相位面行（splitWi）对齐 |
| D 轴 | Dk 外层归约完整保留（D 维不参与 H/W 相位拆分，由常规 D/K 循环调度） |

### 4.3 使用条件（准入表）

| 条件 | 要求 | 理由 |
|------|------|------|
| stride | `strideH == strideW == 2` | 当前模板（SplitHW）只验证 s=2；strideH≥2 的其他场景走 SplitH（§10） |
| kernel | `kh, kw ∈ {1,2,3,4}` | 子核十字切与 pad 公式按 1~4 建模 |
| dilation | H/W 方向 =1 | 拆分后 dilation 进 Load3D 的路径未建模 |
| groups | =1 | group 循环轴未进模板 |
| layout | dedy / dx = NCDHW | UB 重排后 NCDHW→GM 要求 32B 对齐，NDHWC 无法实现 |
| filter | NCDHW（可扩 NDHWC，见 §10） | |
| pad | 上下对称、左右对称 | 子核 pad 推导前提 |
| shape 下限 | `Hi×Wi ≥ 256 或 Cout1G > 1`；`Wi ≥ 12`；`CinG ≥ 16` | 太小时拆分搬运开销不回本 |
| 负向经验 | `Hi > Wi` 或 `CinG > CoutG` 时慎用 | MTE2 bound 场景实测可能劣化 |

### 4.4 未实现 / 不支持

| 能力 | 状态 | 扩展落点 |
|------|------|---------|
| bias | 不支持 | MMAD `cmatrixSource` + BT 路径，见 [usage_guide.md §6](usage_guide.md) |
| strideH≠strideW、stride≥3 | 不支持 | 相位列表扩为 `strideH×strideW` 个，尺寸公式去 s=2 特化（§2.3 通用公式已就绪） |
| SplitH 模板代码 | **已抽取**（tiling 契约/impl/kernel 均含 splitMode==H 分支，对照索引见 tuning guide §15） | bias/vecTrans/NDHWC 等增强见 §4.4 其余行 |
| split-K（kSegment） | 不支持 | K 过长时 Cout 分段 + workspace fp32 累加 |
| NDHWC dx 输出 | 不支持 | Fixpipe ROW_MAJOR + 去重排（相位面在 NHWC 下天然连续） |

## 5. 类名与入口

```cpp
namespace conv3ddx_ks;

template <typename T>
class Conv3ddxKsKernel {
public:
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR dx, GM_ADDR workspace,
                                const Conv3ddxKsCase& tiling);
    __aicore__ inline void Process();
};
```

Kernel 入口（`dav310/conv3ddx_ks_kernel_entry.template` → `conv3ddx_ks_kernel.cpp`）：

```cpp
template <typename T>
__global__ __aicore__ __mix__(1, 2) void conv3ddx_ks_kernel(GM_ADDR filter, GM_ADDR dedy, GM_ADDR dx,
                                                            GM_ADDR workspace, Conv3ddxKsCase tiling)
{
    Conv3ddxKsKernel<T> op;
    op.Init(filter, dedy, dx, workspace, tiling);
    op.Process();
}

extern "C" void launch_conv3ddx_ks(GM_ADDR filter, GM_ADDR dedy, GM_ADDR dx, GM_ADDR workspace,
                                   const Conv3ddxKsCase& tiling, void* stream);
```

## 6. TilingData：`Conv3ddxKsCase`

定义在 `conv3ddx_ks_tiling.template` → `conv3ddx_ks_tiling.h`。关键字段：

| 字段组 | 字段 | 含义 |
|--------|------|------|
| 问题几何 | `n..groups` | 原样来自算子入参 |
| 子核描述 | `splitHk[4], splitWk[4], splitWi, splitHi` | 4 个子核尺寸与相位面尺寸（§2.3 公式派生） |
| 多核切分 | `usedCoreNum, calRound, tailCnt, singleCoreM, singleCoreN` | blockDim 与每核 M/N 空间 |
| L1 tile | `baseM, baseN, baseK, stepKa, stepKb` | 相位面 M / Cin N / Cout×subHkWk K，L1 驻留步数 |
| 缓冲开关 | `a1Db, b1Db, c0Db, b1FullLoad` | L1/L0 乒乓与 B1 全载 |
| 重排 | `useWorkspace, workspaceBytes` | UB 不足时 L0C→workspace 中转 |

Host API：

```cpp
Conv3ddxKsTilingIn in{...};      // shape/attrs + aicoreNum
Conv3ddxKsTilingOut out{};
bool ok = Conv3ddxKsDoTiling(in, out);
Conv3ddxKsCase tiling = ToCaseView(in, out);
```

## 7. Kernel 数据流（摘要）

```
GM(dedy) ──DataCopy Dn2Nz（连续，无洞）──► AL1 ─Load3D(子核 Kh'×Kw', subPad)──► L0A ─┐
GM(filter) ─Dn2Nz─► BL1（全载，一次）─Load2d(mStartPosition=子核相位)─► L0B ─┼─ Mmad ─► L0C
                                                                                   │
   AIC：L0C ──Fixpipe 三段对齐（head/mid/tail）──► UB[rearrangeWIndex 分格] / GM(ws) ─┘
   AIV：凑齐 strideW 个相位块 → Interleave（交叉）→ DataCopyPad 隔行直写 GM(dx NCDHW)
```

- **A1 连续化**：拆分后 `dstNzNStride=1, dnNum=1`，前放大的洞被 kernel 拆分吸收；
- **B1 全载 + mStartPosition 选行**：子核切换不重搬 B1，Load2d 起点直接跳相位：
  `mStartPosition = ((splitHk−1)·sH·Kw + (splitWk−1)·sW + splitHIndex·Kw + splitWIndex)·blockBaseN`；
- **三段对齐 fixpipe**：M 按 splitWi 切块无法保证整行，UB 指令又要求 16 元素对齐，
  输出行切 head/mid/tail 三段独立 AlignUp 后搬出；
- **4 块 workspace Ping/Pong 流水**（workspace 路径，见 tuning guide §8）：
  一次交叉重排需要两轮配对 Cube 结果，DDR 中 4 块同尺寸 workspace 两批轮转
  （WS0/WS1 批次 0 → WS2/WS3 批次 1 → 复用 WS0/WS1…），
  "两输入块均写完才启动 AIV、AIV 重排结束才允许 AIC 覆盖"由跨核事件双向保证；
- **AIC↔AIV 握手**：AIC 每算完 strideW 个相位块 `CrossCoreSetFlag` 通知 AIV，
  AIV interleave 完成后反向通知；AIC 写 UB 与 AIV 读 UB 双向事件乒乓。

**输出交叉重排的数学形式**（W 方向，stride=2）：

```
srcEven = [x0, x2, x4, x6, ...]     dst[2t]   = src0[t]
srcOdd  = [x1, x3, x5, x7, ...]  →  dst[2t+1] = src1[t]
dst     = [x0, x1, x2, x3, ...]
```

先 W 方向交叉、再 H 方向交叉（H 由 GM 写地址的 `rearrangeHIndex×Wi` 行偏移还原），
即可把 4 个相位块还原为连续的 dx tile。该过程只改变内存布局，不改变数值。

## 8. Host tiling 要点

1. **准入**（`Conv3ddxKsAdmit`）：按 §4.3 表逐条拦截；
2. **子核派生**：`splitHiWi = ceil(Hi/sH)×ceil(Wi/sW)`、
   `splitHkWkC0 = ceil(Kh/sH)×ceil(Kw/sW)×K0`、`kValue = splitHkWkC0 × Cout1G`
   ——即"子核替代原核、stride 归一为 1、M 换相位面"之后的内积 tiling；
3. **基本块**：kernel>1 时 `512/128`（M 大 N 小掩盖 B 侧转换），kernel=1 时 `256/256`
   （计算访存比最高的方块）；`baseN` 钳位到 `nValue`；**baseK 从 L0 容量上限向下找
   第一个同时整除最大与最小子核分形的值**（保证任一子核完整进 L0B）；
4. **stepK 搜索**：L1 对半分（`EqualL1`）与阶梯降级（`Ladder`）二选一，容量模型
   `aL1 = CalFmapH(baseM, splitWi)×Wo×coutNum×B字节`、`bL1 = baseN×stepKb×baseK×B字节`
   （CalFmapH 几何与容量兜底链见 tuning guide §11）；
5. **B1 全载判定**：`cout1B1 ≥ CoutG` 且 `baseN ≥ CinG` 时 `b1FullLoad=1`，
   stepKb = ceil(kValue/baseK)；
6. **workspace 判定**：重排需 4 块 `hi×alignWi×baseN` 缓冲，`>UB` 或 `Wi%16≠0` 时
   `useWorkspace=1`，大小 = 每核缓冲 × usedCoreNum；
7. **核间切分**：`totalCnt = N×Do×ceil(splitHiWi/singleCoreM)×ceil(CinG/singleCoreN)`，
   `singleCoreM` 对齐 splitWi 行；分不满核时按行均分做负载均衡（tuning guide §11.5）。

## 9. 关键优化（与理论对照）

| 优化 | 判定/条件 | 收益模型 |
|------|----------|---------|
| kernel 拆分本身 | §4.3 全部满足 | MTE2 ×s²、乘法 ×~s²（kernel3 时 9/5） |
| B1 全载 | L1 容量允许 + dk==1 | 消除每个基本块的 filter 重搬 |
| kSCoutFullLoad（进阶） | A/B 均全载且 CoutG≤256 | L0C 常驻 + 步数下降（ops-nn 有独立模板，本模板未抽） |
| 三段对齐 fixpipe | Wi 不对齐 16 | 避免 M 切块跨行时整行多搬 |
| 1×1 kernel 特判 | kh=kw=1 | 第二相位恒零：AIC 跳过、AIV 直搬（不 interleave） |
| 尾相位 M 收缩 | Hi/Wi 不整除 s | `subKernelM = Hi/sH×splitWi`，尾相位少搬少算 |

**失效场景**：`Hi > Wi` 或 `CinG > CoutG`（MTE2 bound 时 interleave 开销反超）、
小 shape（`Hi×Wi < 256` 且 `Cout1G == 1`）、`Wi < 12`。

## 10. SplitH（H 单拆）专章

SplitH 是 kernel split 在 SplitHW 准入之外的第二形态：**只按 H 相位拆分**，
W 方向保持完整行处理。ops-nn 中与 SplitHW 同属 kernel split tiling 模板
（`TryKernelSplitHW` 失败后 `TryKernelSplitH` 兜底，kernelSplitMode=2）。
**模板代码已抽取**：tiling 契约（`splitMode`/`splitHkBase`/`kSTail*` 字段）、
host tiling（`Conv3ddxKsAdmitMode` 三态准入 + H 流程）、kernel（相位维解码/
行取样 B 装载/三段直写，AIV 早退）均含 H 分支
（源码对照索引见 tuning guide §15）。

### 10.1 与 SplitHW 的结构差异总览

| 维度 | SplitHW | SplitH |
|------|---------|--------|
| 相位数 | Sh×Sw（stride=2 时 4 个） | **Sh 个**（W 不拆） |
| 相位归属 | 核内轮转（IterateForKernelSplit 步进 splitIndex） | **核间切分维度**（kSCoreIdx，totalCnt × strideH） |
| 新 M 轴 | splitHiWi = ceil(Hi/sH)×ceil(Wi/sW)（W 也压缩） | splitHiWi = ceil(Hi/sH)×**Wi**（W 保持原宽） |
| A1 装载 | 整段连续（dnNum=1） | **行连续**：H 向无洞（strideH 等效 1），W 向仍按 `dstNzNStride=strideW` 留洞或逐行 |
| K 轴 | Cout×subHk×subWk×K0 | Cout×**subHk×Kw**×K0（Wk 不拆） |
| 输出 | L0C→UB/ws + AIV Interleave 重排 | **L0C 三段对齐直写 GM**（带 skip 行距），无 AIV 参与重排 |
| 尾相位处理 | RecalcBaseUseMForKernelSplit（核内） | kSTailIdx/kSTailM 尾相位收缩（核间，见 §10.4） |

### 10.2 KernelSplitPara 差异（SetParamForKernelSplit(isOnlyH=true)）

| 参数 | SplitH 取值 | SplitHW 取值（对照） |
|------|------------|---------------------|
| `curWi` | Wi（不 +1 对齐） | Wi%strideW≠0 时 +1（保证 W 重排正确） |
| `curStrideW` | **strideW**（W 向保持原 stride） | 1 |
| `strideHW` | strideH | strideH×strideW |
| `maxSplitKw` | **Kw**（不拆） | ceil(Kw/sW) |
| `splitHiWi` | ceil(Hi/sH)×**Wi** | ceil(Hi/sH)×ceil(Wi/sW) |
| `subSplitHkWkC0` | (Kh/sH)×**Kw**×K0 | (Kh/sH)×(Kw/sW)×K0 |
| `maxSplitKh` | ceil(Kh/sH) | ceil(Kh/sH) |

### 10.3 准入链（CheckKernelSplitHEnable）

SplitHW 尝试失败后进入（前置公共准入——dtype/groups/dilation/pad 对称/offsetX 与
SplitHW 一致）。SplitH 专属条件：

| 条件 | 要求 | 理由 |
|------|------|------|
| shape 下限 | `CinG ≥ 16 且 CoutG ≥ 16 且 Wi ≥ 32` | 经验阈值：cin/cout/wi 较小时转置及输出重排性能差、拆分不回本 |
| kernel | `strideH ≤ Kh ≤ 16`（ops-nn 注释标注支持 kernelHW[5,16]、strideH∈[2,Kh−1]） | 相位非空前提 + 子核规模上限 |
| stride | `strideH ≥ 2`（strideW 不限，含 1） | 消 H 方向插零的收益来源 |
| pad | **pad=0**（IsBaseShapeFitKernelSplitH 路径按 pad=0 建模） | H 单拆的输出直写地址按无 pad 推导 |
| filter | NDHWC（ops-nn 当前实现约束） | kernelSplitStrideB 与 B 装载视图按 NDHWC 建模 |
| L1 最小容量 | `minA1 + minB1 ≤ L1`：`minA1 = A字节×curHo×Wo×strideW×K0`、`minB1 = B字节×splitHkWkC0×16`（cout1 均取 1） | cout1 最小时放不下则不拆 |

### 10.4 核间切分与尾相位（与 SplitHW 最大的调度差异）

**相位维度进核间**：`totalCnt = batch × group × Do × mCnt × nCnt × strideH`，
`kSCoreIdx = basicBlockIdx / batchDepthMNCnt` 解出该核负责的 H 相位。
`SetSingleCoreInfo` 中 `kSCnt = strideH`（SplitHW 为 1）。

**相位恢复与起始偏移**：

```
kernelStartIdx = padUp % strideH（≠0 时取 strideH − padUp%strideH）
kernelIdx = (kSCoreIdx + kernelStartIdx) % strideH      // 该核实际相位
```

**子核尺寸二选一**（H 相位组不等长，见主文档 §2.3 尺寸公式）：

```
splitBaseHk = ceil(Kh/strideH)      // baseIter 个较长相位组
splitTailHk = Kh/strideH            // 其余相位组
baseHkIter  = (splitTailHk==splitBaseHk) ? strideH : Kh − splitTailHk×strideH   // 较长组的个数
curSplitHk  = (kSCoreIdx < baseHkIter) ? splitBaseHk : splitTailHk              // 按相位序号选择
```

**尾相位 M 收缩**（`kSTailIdx = Hi % strideH ≠ 0` 时，下半相位面的行数少 1）：

```
kSTailM_   = (Hi/strideH) × Wi          // 尾相位的 M 总长
kSTailMCnt_ = ceil(kSTailM_/singleCoreM)
kernelIdx ≥ kSTailIdx 的核：mCnt 换用 kSTailMCnt_，越界基本块 continue 跳过，
                            mCoreUse 尾块换用 kSTailM_ − (kSTailMCnt_−1)×singleCoreM
```

### 10.5 数据通路差异（kernel 侧）

**A 装载（H 方向连续化）**：`CalcLoadToA1Dn2NzParams` 中 `strideH 强制 1`
（kernel 拆分已吸收 H 向插零），W 向保持原逻辑——strideW>1 时仍按
`dstNzNStride=strideW` 留洞或按行搬运；`CalFmapHForKernelSplit` 的行宽用
`curWi=Wi`（非 splitWi），`curHoIdx = mStartIdx/Wi − curBackPropPadUp`。

**子核 backprop pad**：`SetSingleShapeParams(curSplitHk, curBackpropPadUp)` 注入，
`CalcSubKernelBackPadUp(Kh, sH, bpPad, idx)` 公式：

```
subKernelSize = (Kh + sH − idx − 1) / sH
missPointNum  = (Kh − idx − bpPad − 1 > 0) ? (Kh−idx−bpPad−2)/sH + 1 : 0
subPadUp      = (subKernelSize − 1) − missPointNum
```

**B 装载**：`InitKernelSplitStrideB` 设 `kernelSplitStrideB_`（NDHWC 下 = Wk×Cin），
`CalcBlockKSOffset` 按 `kSCoreIdx × kernelSplitStrideB_` 平移 filter 偏移；
单子核场景 `splitIndex_=0`、`splitWkList_[0]=Kw`，L1→L0B 走常规逆序路径。

**输出直写（无 AIV）**：`LoadL0c2GmDnForKernelSplitH`（NCDHW）——L0C NZ 布局
`(1, cin1, splitHi, Wi, cin0)` 三段对齐后 **fixpipe 直写 GM**，行距带 skip：

```
skipDstOffset = Wi × (strideH − 1)                    // 相位面行距 → 原始 dx 行距（NCDHW；NDHWC 再 ×Cin）
dstOffset     = cin×DiHiWi + di×HiWi + mSize + (mSize/Wi)×skipDstOffset
mid 段 dstDnMatrixStride = skipDstOffset + Wi         // 每写一行相位面，跳 (strideH−1) 行原始 dx
```

即 H 方向的"交错还原"不需要 Interleave，由 **fixpipe 目的地址的隔行 stride**
完成（每行写完跳 strideH−1 行）。因 W 未拆，x 方向天然连续，无重排必要——
这是 SplitH 免 AIV 的根本原因。

### 10.6 模板实现状态（已抽取）与后续增强落点

**已实现**（本模板三件套均已含 splitMode==H 分支）：
- tiling 契约：`splitMode`、`splitHkBase/splitHkTail/baseHkIter`（尺寸公式 baseIter 机制）、
  `kernelStartIdx`（相位恢复）、`kSTailIdx/kSTailM/kSTailMCnt`（尾相位）、`subPadUpH[16]`；
  baseK 采用 **整除 K 段** 契约（`cout1×subHkWkC0 % baseK == 0`，与 HW 的倍数契约方向相反）
- host tiling：`Conv3ddxKsAdmitMode` 三态准入（HW 优先、H 兜底）、H 流程
  （KernelSplitPara §10.2、totalCnt×strideH、尾相位字段、workspace 恒 0）
- kernel：相位维解码 + `InjectSubKernelForCore`（子核尺寸/pad 按原始序号注入）、
  A1 行连续（W 留洞）、B1 GM 行循环取样重组、`StorePhaseBlockDirectGm` 三段直写
  （skipDstOffset 隔行）、AIV 早退

**后续增强落点**（对照 ops-nn，未抽取部分）：

| 增强 | ops-nn 参考 | 落点 |
|------|------------|------|
| NDHWC filter/dx | `LoadToB1Nd2NzForKernelSplit` / `LoadL0c2GmNdForKernelSplitH`（ROW_MAJOR） | LoadToB1Block/StorePhaseBlockDirectGm 加布局分支 |
| bias | `ComputeForBias` + BT 路径 | ComputePhase 首 K `cmatrixSource` |
| vecTrans（前置 NZ 化 filter） | `enableVecTransFlag` 判定 + AIV 前置转置 | 独立 AIV 阶段 |
| B1 全载（H） | `InitFullLoadFlag` H 分支 | LoadToB1Block H 换整驻留 + mStartPosition |

## 11. 不适用 / 替代

| 场景 | 去向 |
|------|------|
| stride=1 反卷积 | 无稀疏可消，用传统前放大路径（Dn2Nz 留洞 + Load3D + 逆序 Load2d 三段式通路，方法已内联在 tuning guide §11） |
| SplitHW 与 SplitH 均不满足准入（stride=1、groups>1、dilation>1、bias 等） | 传统路径（或先扩展落点 §4.4） |
| Cout 过大无法全载 | 传统路径 + SplitK |

## 12. 文件清单

| 模板 | 落地 |
|------|------|
| `conv3ddx_ks_tiling.template` | `op_kernel/conv3ddx_ks_tiling.h`（Conv3ddxKsCase + ValidateCase + tiling API） |
| `conv3ddx_ks_tiling_impl.template` | `op_host/conv3ddx_ks_tiling.cpp`（g++） |
| `dav310/conv3ddx_ks_kernel.template` | `op_kernel/conv3ddx_ks_kernel.h` |
| `dav310/conv3ddx_ks_kernel_entry.template` | `op_kernel/conv3ddx_ks_kernel.cpp`（bisheng） |
| `dav310/conv3ddx_ks_launch.template` | `op_kernel/conv3ddx_ks_launch.h` |
