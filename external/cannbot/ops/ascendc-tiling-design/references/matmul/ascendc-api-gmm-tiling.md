# Ascend C GMM 高阶 API Tiling 策略

> **适用路径**：Ascend C **Matmul 高阶 API** 上的 GroupedMatmul（`MatmulImpl` + `GMMBaseParams` / `GMMArray`）。
> **适用平台**：Atlas A2 / A3（Ascend910B、Ascend910_93，NpuArch `DAV_2201`）。
> **不适用**：Ascend 950（`DAV_3510`）——950 侧 GMM 实现路线不同，勿套用本文档。
> **扩展策略**：当新架构出现时，按新架构 NpuArch 新增 reference 文件或更新现有平台适配说明。
>
> 适用算子：GroupedMatmul（分组矩阵乘），每组 M/K/N 可不同。
> MatMul 通用 Tiling（L2/Block/BaseBlock 三级、batch 处理、Split-K）见 [ascendc-api-matmul-tiling.md](ascendc-api-matmul-tiling.md)。本文档仅覆盖 GMM 差异化设计。

---

## 算子公式与分组方式

```
y_i[M_i, N] = x_i[M_i, K_i] × w_i[K_i, N]      i = 1, ..., g
```

> - `x_i` / `w_i` / `y_i`: 第 i 组的左矩阵 / 右矩阵 / 输出
> - `M_i` / `K_i` / `N`: 第 i 组的 M / K / N 维度（N 通常各组相同）

| groupType | 含义 | K 轴约束 | M/N 约束 | 典型场景 |
|-----------|------|----------|----------|---------|
| **SPLIT_M (0)** | 沿 M 轴分组 | 各组 K 相同 | M 不同；N 可同可不同 | 正向训练 / 量化推理 |
| **SPLIT_K (1)** | 沿 K 轴分组 | K 不同 | M/N 各组相同 | 反向求 weight 梯度 |
| **NO_SPLIT (-1)** | 不固定轴分组 | 任意 | 任意 | 多 tensor 通配 |

### Tensor 组合

| 缩写 | x 形态 | weight 形态 | y 形态 | M/K/N 来源 |
|------|--------|------------|--------|-----------|
| s-s-s | 单 tensor `[M,K]` | 单 tensor `[G,K,N]` | 单 tensor `[M,N]` | M 由 groupList 提供 |
| s-m-s | 单 tensor `[M,K]` | 多 tensor list `[(K,N),(K,N),...]` | 单 tensor `[M,N]` | M 由 groupList，N 各不同 |
| m-m-s | 多 tensor list | 多 tensor list | 单 tensor `[M,N]` | M 由 groupList，K/N 各不同 |
| m-m-m | 多 tensor list | 多 tensor list | 多 tensor list | M/K/N 完全独立 |

---

## Tiling 数据结构

> **字段来源**：`GMMBaseParams`、`GMMArray[128]`、`TCubeTiling` 字段定义来自 CANN 9.0.0 / asc-devkit 9.0.0（GroupedMatmul 算子 Tiling 头文件），具体版本以本地 `$ASCEND_HOME_PATH/include/` 为准；版本差异请用 `ascendc-docs-search` skill 查询。

### GMMBaseParams

| 字段 | 类型 | 说明 |
|------|------|------|
| groupNum | uint32 | 分组数量 |
| coreNum | uint32 | 实际使用的核数 |
| groupType | int32 | 分组方式（0=SPLIT_M, 1=SPLIT_K, -1=NO_SPLIT） |
| activeType | uint32 | 激活函数类型（0=无） |
| hasBias | uint32 | 是否有 bias |
| quantParam | uint32 | perToken=1, perGroup>1, else=0 |
| singleN | uint32 | 单核 N 方向大小（0=连续写，非0=动态分块值） |
| singleWeight / singleX / singleY | uint32 | Tensor 合并标志（1=单 tensor） |
| groupListType | uint32 | groupList 类型（0=cumsum, 2=sparse_m） |
| m / k / n | uint64 | 总 M / K / N |
| ubBaseK / ubBaseN | uint32 | UB 计算分块的 K/N 步长 |
| ubCalSize | uint32 | UB 每分块计算量（元素数） |
| ubRestBytes | uint32 | UB 剩余字节 |
| vBaseM | uint64 | Vector 侧 M 方向分块步长 |
| workspaceSize | uint64 | workspace 总大小 |
| withOffset | uint32 | 反量化是否有 offset 参数 |
| isOutputDisableL2Cache | uint32 | SplitK 大输出时跳过 L2 |
| isPreTiling | uint64 | 静态 Tiling 标志 |
| quantGroupNum | uint64 | 按 group 量化的组数 |

### GMMArray

| 字段 | 类型 | 说明 |
|------|------|------|
| mList[128] | int32 | 各组 M 值；单 tensor 时 `mList[0] = -1` |
| kList[128] | int32 | 各组 K 值；单 tensor 时仅 `kList[0]` 有效 |
| nList[128] | int32 | 各组 N 值；单 tensor 时仅 `nList[0]` 有效 |

### TCubeTiling

复用通用 MatMul 的 `TCubeTiling` 结构（由 `MatmulApiTiling::GetTiling` 生成），GMM 在此基础上覆写以下参数：

| 覆写参数 | 说明 |
|---------|------|
| baseM / baseN / baseK | GMM 按场景重新计算（不直接用 GetTiling 结果） |
| stepKa / stepKb | 根据 L1 双缓冲容量重新计算 |
| depthA1 / depthB1 | depthX = stepX × 2（双缓冲） |
| stepM / stepN | 固定为 1 |
| dbL0C | 固定为 1（禁用双缓冲） |

---

## 多核切分

### 基本块划分

各组 MatMul shape 不同，不能为每组单独配置 MatmulImpl 实例。GMM 以 **baseM × baseN** 为统一基本块，按 maxM / maxN 划分 M/N 方向块数：

```
mDim = ceil(maxM / baseM)        // （考虑分组后等效 maxM）
nDim = ceil(maxN / baseN)
单组 totalBlocks_perGroup = mDim × nDim
总 totalBlocks = mDim × nDim × groupNum
```

### 对角线分核

当基本块数和核数存在整除关系时，相邻核同步访问左矩阵相同地址会导致 bank conflict。对策：将对角线方向的基本块分配给同一轮次的核。

设 mDim × nDim = coreNum，则遍历顺序改为蛇形：

```
blockSeq = diagonal_order(mDim, nDim)
// 每个核处理不同轮: roundIdx×coreNum + blockSeq[coreIdx % totalBlocks]
```

对角线按阈值分组以充分利用 L2 Cache：

1. `min(Tm, Tn) >= numCore` —— 对角线分组不小于物理核数
2. `dtype_size × (Tm × baseM × K + K × Tn × baseN + Tm × baseM × Tn × baseN) <= L2_size` —— 一组数据 fit L2

### 动态核数优化

当 `mDim × nDim < aicNum` 时，启动所有核会导致部分核空转。GMM 在 s-s-s+SPLIT_M+A8W8/A16W16/A4W4 场景下启用动态核数：

```cpp
usedCoreNum = min(aicNum, mDim × nDim)
```

### 动态 singleN 优化

增大 singleN 可减少 N 方向块数，改善负载均衡（仅 NZ 格式 + 单 tensor 场景）：

```cpp
for (uint32_t factor = 1; factor <= aicNum; ++factor) {
    bestSingleN = ceil(maxN / factor);
    if (bestSingleN % baseN != 0 && bestSingleN != maxN) continue;
    curTaskNum = mDim × ceil(maxN / bestSingleN) × groupNum;
    if (curTaskNum / AlignUp(curTaskNum, aicNum) >= 0.95)  // 有效任务比
        return bestSingleN;
}
```

---

## BaseBlock 参数

### baseN

按场景预设，不随 L0 容量推算：

| 场景 | baseN |
|------|-------|
| 默认 | 256 |
| A8W8 量化单组且 tuningConfig∈(128,256] | 512（配合 baseM=256, baseK=128） |
| A16W8 MSD 且 N≥2048 | 512 |
| A4W4 + tuningConfig≤64 | 128 |

### baseK

由 L0B 容量递推：

```
baseK = Floor( l0BSize / 2 )/ (baseN × dtype_size)   // 2 = 双缓冲
baseK = AlignDown(baseK, 16)
```

### baseM

由 L0A 和 L0C 取最小值，再 16 对齐（上限 MAX_BASEM）：

```
maxBaseM  = l0CSize / (baseN × FP32_SIZE)         // L0C 容量约束
candBaseM = l0ASize / 2 / (baseK × dtype_size)    // L0A 双缓冲约束
baseM = AlignDown(min(candBaseM, maxBaseM), 16)
if (baseM > MAX_BASEM) baseM = MAX_BASEM;
```

---

## L1 深度计算

`stepKa/stepKb` 决定 K 方向 L1 滚动窗口大小（双缓冲）：

```cpp
l1ASize = baseM > baseN ? 256KB : (availableL1 - 256KB);
l1BSize = availableL1 - l1ASize;
stepKa = (l1ASize / 2) / (baseM × baseK × dtypeSize);  // 2 = 双缓冲
stepKb = (l1BSize / 2) / (baseN × baseK × dtypeSize);
// 对齐：stepKa / stepKb 互取整
depthA1 = stepKa × 2;   depthB1 = stepKb × 2;
```

---

## UB Buffer 分块规划

GMM 按场景选用不同的 UB 分块策略，由 `ubDivideBlkNum / ubIoBlkNum / ubBlockAlign` 控制。

### 量化场景 UB 划分

| 子场景 | ubDivideBlkNum | 描述 |
|--------|---------------|------|
| A8W8 perToken / 带激活 | 动态值 | buffer 复用多 |
| A8W8 perTensor 无激活 + fp16 输出 | 静态值 | 简单直接 |
| A8W8 perTensor 无激活 + bf16 输出 | 静态值（不同） | dtype 差异 |
| A4W4 | A4W4_BLOCK_NUM | 固定分块 |

### UB 分块大小计算

```
ubCalSize = ubSize / ubDivideBlkNum（元素数）
ubCalSize = AlignDown(ubCalSize, ubBlockAlign)
ubBaseK  = perTokenOrPerGroupSize  // per-group 场景用 groupSize
ubBaseN  = min(BEST_UB_BASEN, AlignUp(ubCalSize / ubBaseK, MIN_UB_BASEN))
```

### workspace 分配

量化场景需要 workspace 存放 MatMul 中间结果（int32）：

```
workspaceSize = 4 × baseM × baseN × usedCoreNum × sizeof(int32_t)
// 4 路流水: CV 并行时最多 4 轮结果同时在 workspace 中
```

反量化场景（A16W8）的 workspace 用于预重排 x / 存放 per-group scale 等：

```
workspaceSize = sum_over_groups(K × N × weight_dtype_size)
// 性能优化模式时 ×2（双片 workspace）
```

---

## AIV/AIC 调度比例

GMM 使用 `GET_TPL_TILING_KEY` 宏编码 8 维模板参数，其中 `AIV_AIC_RATIO` 决定 AIV/AIC 核数比：

| 比例 | 值 | 触发条件 |
|------|-----|---------|
| Cube Only | AIC 独占，AIV 返回 | 非量化 fp16/bf16/fp32 |
| 1:1 | AIC+AIV 对等并行 | A8W8 + (perTensor 无激活 或 int32 输出) |
| 2:1 | AIV 核数 = AIC×2 | perToken + (K≤1024 或 K≥2048) + token≥128；或 GELU 激活 |

**选择逻辑**：
- K≤1024: Vector bound → 双 Vector 缓解
- 1024\<K≤2048: 平衡区 → 双 Vector 争抢 MTE2 带宽可能劣化，仍用 1:1
- K≥2048: Vector 又成瓶颈 → 双 Vector 提升带宽利用率

---

## Host 端 Tiling 流程

```
1. Init 阶段
   ├─ GMMGetAttrs: 读取 transA/B, groupType, splitItem, dtype, wFormat(NZ/ND)
   ├─ 判定场景: A8W8 / A16W8 / A4W4 / A16W16 / A8W4 / A16W4
   ├─ CheckTensorListLength: 校验 tensor list 不超过 MAX_TENSOR_CONT(128)
   └─ PrepareTilingData: 按 groupType+tensor组合提取 M/K/N list
        ↓
2. CalMMTiling
   ├─ baseN 按场景固定
   ├─ baseK = l0BSize/2 / (baseN×dtypeSize) → AlignDown(16)
   └─ baseM = min(l0ASize/2 / (baseK×dtypeSize), l0CSize / (baseN×4))
        → AlignDown(16), clip MAX_BASEM
        ↓
3. usedCoreNum = CalUsedCoreNum(aicNum)
   └─ s-s-s+SPLIT_M 时 min(aicNum, mDim×nDim)
        ↓
4. GMMSetMMTiling
   ├─ 调用 MatmulApiTiling::GetTiling(TCubeTiling)
   ├─ CalcStepKaKb: 按 L1 剩余空间算滚动深度
   └─ 覆写 baseM/N/K, stepKa/Kb, depthA1/B1, dbL0C=1
        ↓
5. FullLoadK (可选): ND+fp16/bf16+大M+特定N 时全载K轴
        ↓
6. DivideUbAndSetWorkspace
   ├─ 量化: UB分块 + perToken/perTensor workspace
   └─ 伪量化: antiquant workspace + perGroupNum
        ↓
7. DynamicTilingSingleN: NZ格式 动态调整 singleN
        ↓
8. GMMSetTplTilingKey
   ├─ 编码 8 维模板key（dtype + trans + groupType + tilingType +
   │                                    A4W4tpl + A16W8tpl + AIV比例 + 定轴搬移）
   └─ SetTilingKey
        ↓
9. 序列化 GMMTilingData → rawTilingData buffer
   SetBlockDim(usedCoreNum)
```

---

## 关键优化项

| 优化 | 条件 | 策略 |
|------|------|------|
| **FullLoadK** | ND、fp16/bf16、大M、N 匹配特定值、K 适中 | L1 全载 K 轴，增大 depthA1/B1 减少 GM 访问 |
| **FullLoadA** | A8W8/A4W4 PerChannel、singleN 动态增大后 | 用剩余 L1 空间整载左矩阵 |
| **定轴搬移** | A8W8、K=2048/7168、N=7168/4096、groupNum=4 | 专用数据通路，workspace 替代 UB buffer |
| **静态 Tiling API** | A8W8+s-s-s+无 bias+无激活+预匹配 | 编译期固化 baseM/N/K=128/256/128，零运行时开销 |
| **输出 skip L2** | SplitK+s-s-s 且 `M×N×groupNum×dtype > L2` | `isOutputDisableL2Cache=1` |

---

## 设计检查清单

- [ ] groupType 与 tensor 组合匹配（SPLIT_M/SPLIT_K/NO_SPLIT + s/m 组合）
- [ ] GMMArray mList/kList/nList 正确填充（单 tensor 与多 tensor 语义不同）
- [ ] baseM/baseN/baseK 由 L0 容量递推，对齐到 16，非零
- [ ] NZ weight: 校验内轴为 32B 倍数；`nzFactor` 正确
- [ ] usedCoreNum = min(aicNum, mDim × nDim)，blockDim 正确
- [ ] GMMSetTplTilingKey 的 8 维编码与 kernel 模板参数一致
- [ ] workspaceSize 足够覆盖 `CV阶段数 × baseM × baseN × usedCoreNum × sizeof(int32)`
- [ ] FullLoadK 仅在条件满足时启用，否则保持 stepKa=1
- [ ] 对角线分核仅当 `nDim >= coreNum` 时生效
