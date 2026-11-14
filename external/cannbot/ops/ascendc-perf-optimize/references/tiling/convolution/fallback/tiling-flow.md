# Convolution 族 — Tiling 生成流程

> Conv2D / DepthwiseConv 通用的 Tiling 推导流程，覆盖多核切分 → 单核 L1/L0/UB 三级 Buffer 划分 → TilingKey 的完整链路。
>
> 算法实现见 [demo/](demo/) 目录，`main.cpp` 注释标注了每步对应的原始代码路径。

---

## 1. 流程总览

```
给定：Conv2D 算子 + Shape (N, Ci, Hi, Wi, Co, Kh, Kw) + dtype + attributes

Step 0 — 算子分类与 Group 决策：
  ├─ groups=1 → 普通卷积
  ├─ groups>1, opt-group weight 能装入 UB → OPT_GROUP（合并多组为 enlarge 组）
  └─ groups>1, opt-group 不满足 UB 约束 → ORI_GROUP（逐组处理）

Step 1 — 切分模式决策（GetTilingSplitMode）：
  ├─ outputOrder = M → M-split（沿 batch × M × Co × groups 切分）
  └─ outputOrder = HW → HW-split（沿 batch × Ho × Wo × Co × groups 切分）

Step 2 — 多核算力分配（NumBlocksDecision）：
  ├─ M-split: 回溯搜索 {batch, m, n, group} 四维因子组合
  │   └─ 目标：最小化总代价 = loadFmap + loadWeight × BW_coeff + loadOutput + cubeCompute
  └─ HW-split: 回溯搜索 {batch, ho, wo, n, group} 五维因子组合
      └─ 目标：最大化核利用率（总块数尽可能接近 aicoreNum）

Step 3 — 单核 Shape 计算（Conv2dApiTilingSetShape）：
  ├─ M-split: singleCoreM = CeilDiv(Align(Ho×Wo, m0), mDim)
  │           singleCoreCo = Align(CeilDiv(Align(Co, n0), nDim), n0)
  └─ HW-split: singleCoreHo = CeilDiv(Ho, hoDim), singleCoreWo = CeilDiv(Wo, woDim)
  └─ singleCoreBatch = CeilDiv(Batch, batchDim)

Step 4 — API Tiling 引擎配置（Conv2dOpTilingSetShape）：
  ├─ 设置 org/single shape、dtype、format、padding/stride/dilation
  └─ 配置 group 参数（enlarge, singleGroups, singleGroupOpt）

Step 5 — 单核 Tiling 计算（GetConv2dApiTiling → Conv2dTiling::GetTiling）：
  ├─ M-split → Mmode（M 优先 tiling）
  └─ HW-split → HWmode（HW 优先 tiling）
  内部执行：
  ├─ L1 Tiling：确定 kAL1/kBL1/mAL1/nBL1/hoAL1/woAL1 切分 + 双缓冲策略
  ├─ L0 Tiling：L1 → L0 子切分 + L0C 双缓冲检查
  └─ UB Tiling：Weight UB transpose 参数 / DMA fmap copy 参数

Step 6 — RunInfo 输出（GetConv2dOpsTiling）：
  └─ 组装多核 numBlocks + 单核 tiling data → kernel 下发参数

Step 7 — TilingKey 计算（SetTilingKey）：
  ├─ fmpTiling / weightTiling：L1 全载程度
  ├─ l1PingPong / l0PingPong：各级 ping-pong buffer 使能
  ├─ outputOrder / iterOrder / groupType：调度模式标记
  └─ weightUbTrans / fmapCopyMode / innerBatch：数据搬运模式
```

---

## 2. 多核算力分配详解

### 2.1 因子生成

对每个可切分维度，调用 `CalcCommFactor(num, aicoreNum)` 生成该维度的所有合法因子（能整除 num 且 ≤ aicoreNum）。

对于 batch/groups 维度，若 batch ≥ 2×aicoreNum，允许使用 aicoreNum 的全部因子；否则只使用 batch 自身的因子。

### 2.2 M-split 代价模型

```
TotalCost = (loadFMCost + loadWtCost × BW_coeff + loadOutCost) / 128 + cubeCost
```

- `loadFMCost = CeilDiv(batch,batchDim) × CeilDiv(groups,groupDim) × m1 × ci1 × k0`
- `loadWtCost = CeilDiv(groups,groupDim) × ci1 × kh × kw × k0 × CeilDiv(batch,batchDim)`（非 opt-group 时再乘 N 维度）
- `loadOutCost = CeilDiv(batch,batchDim) × CeilDiv(groups,groupDim) × CeilDiv(co1×n0,nDim) × m1`
- `cubeCost = CeilDiv(batch,batchDim) × CeilDiv(groups,groupDim) × CeilDiv(co1,nDim) × ci1 × kh × kw × m1`
- `BW_coeff = 4`（NCHW 格式非 group 卷积），否则 `= 1`

回溯搜索四维因子组合 (batchDim, mDim, nDim, groupDim)，选择总代价最小的组合。

### 2.3 HW-split

HW-split 采用简化策略：在 batch × ho × wo × n × group 五维因子空间中回溯搜索，最大化 `total = batchDim × hoDim × woDim × nDim × groupDim`（接近 aicoreNum），同分值时选 waste 最小的。

---

## 3. 单核 Tiling 算法

### 3.1 Mmode（M-split）

- L1 tiling：沿 mAL1、nBL1、kAL1/kBL1 三维切分
- 策略层次：FullLoad（fmap 或 weight 全载 L1）→ K-only FullLoad → No FullLoad
- 支持 innerBatch 机制（1×1 卷积时单次 load 处理多 batch）
- L0 tiling：mL0 × kL0 × nL0

### 3.2 HWmode（HW-split）

- L1 tiling：沿 hoAL1、woAL1、kAL1/kBL1、nBL1 四维切分
- 额外支持 DMA kernel split（khL1/kwL1）用于大 kernel 场景
- L0 tiling：hoL0 × woL0 × kL0 × nL0

---

## 4. TilingKey 编码

TilingKey 将 tiling 结果编码为固定格式的查找键，用于匹配预编译的 kernel 变体：

| 字段 | 含义 | 取值 |
|------|------|------|
| `fmpTiling` | Fmap L1 全载程度 | 0=FULLLOAD_AL1, 1=ONLY_M_FULLLOAD, 2=OTHER |
| `weightTiling` | Weight L1 全载程度 | 0=FULLLOAD_BL1, 1=ONLY_N_FULLLOAD, 2=OTHER |
| `l1PingPong` | L1 双缓冲使能 | 0=ALL_CLOSE, 1=A开B关, 2=A关B开, 3=ALL_OPEN |
| `l0PingPong` | L0 双缓冲使能 | 同上 |
| `outputOrder` | 输出顺序 | 0=HW-mode, 1=M-mode |
| `iterOrder` | L1 迭代顺序 | 0=M_FIRST, 1=N_FIRST |
| `groupType` | 分组类型 | 0=NORMAL, 1=ORI_GROUP, 2=OPT_GROUP |
| `weightUbTrans` | Weight UB transpose | 0=关闭, 1=启用 |
| `fmapCopyMode` | Fmap 搬运模式 | 0=LOAD3D, 1=DMA |
| `innerBatch` | Inner batch 模式 | 0=SINGLE, 1=1x1_MULTI, 2=MULTI |
