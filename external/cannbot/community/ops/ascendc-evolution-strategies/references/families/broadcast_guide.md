# Broadcast 族策略选型指南

> 算子族：`broadcast` / `elementwise`（带广播）
> 适用算子：`Add`, `Mul`, `Sub`, `Div`, `Maximum`, `Minimum` 等输入 shape 不完全相同的算子
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
输入 shape 不完全相同？
├─ 否 → 走 elementwise_guide.md
└─ 是 → Broadcast 路径
    ├─ 合轴后只剩 1 维 → OneDim（可能有标量输入）
    └─ 合轴后 > 1 维 → 选择广播方式
        ├─ DAV_2201 → UB Broadcast（静态接口 / 搬运指令 fallback）
        └─ DAV_3510 → 广播发生在 GM→UB 还是 UB 内部？
            ├─ GM→UB 阶段 → NDDMA Broadcast
            └─ UB 内部广播 → UB Broadcast 动态接口（rank 1~9）
```

## 族特定策略

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P113** 广播对齐 Tiling | L1 | ✅ | 多维广播 | 合轴 + UB 切分 + 多核切分 + 广播方式决策 |
| **P12** 广播与掩码操作 | L0 | ✅ | 带 Mask / 广播维度 | 用 BinaryRepeat / Broadcast API 替代手动循环 |
| **P7** 32B 对齐 | L0 | — | 尾块 / 非对齐 | DataCopyPad、维度对齐 |
| **P1** 双缓冲 | L0 | ✅ | tile 循环 | CopyIn / Compute / CopyOut 流水化 |

---

## Ascend C Broadcast Tiling 系统化设计

> 来源：插件本地 [`tiling-design/broadcast/patterns.md`](../tiling-design/broadcast/patterns.md)
>
> 本章节沉淀 Broadcast 族在 Ascend C 上的 Tiling 设计方法论。

### 1. 合轴（DimensionCollapse）

所有 Broadcast 分支的公共前置步骤：

1. **补维**：输入 shape 维度不足输出时，左侧补 1。
2. **标记广播轴**：为每个轴计算 flag 位图，`dim=1` 的输入对应 bit 置 1。
3. **合并相邻同 flag 轴**：相邻两轴所有输入 flag 相同则合并（维度相乘）。
4. **计算 stride**：从右到左累乘；广播轴 stride 置 0。

**示例**：`Add(x=[4,3,8], y=[1,3,8])`

```
flag: 轴0=01(y 广播), 轴1=00, 轴2=00
合轴：轴1+轴2 flag=00 → 合并
x: [4, 24]   strides: [24, 1]
y: [1, 24]   strides: [0,  1]
out:[4, 24]  strides: [24, 1]
```

### 2. 场景判定流程

```
给定: N 个输入 shape + 1 个输出 shape（所有输入连续）
Step 0: 补维 + 合轴 → dims[N+1][≤8] + strides[N+1][≤8]
Step 1: 分支判定
  ├─ 合轴后只剩 1 维 → OneDim → [onedim.md]
  └─ 合轴后 > 1 维 → 选择广播方式
      ├─ DAV_2201 → UB Broadcast（优先静态接口）
      └─ DAV_3510 → 广播方式决策链（见下表）
```

### 3. 广播方式决策链（DAV_3510）

| 优先级 | 条件 | 选择 |
|---|---|---|
| 1 | 用户强制指定 NDDMA 或 UB BRC | 遵从 |
| 2 | NLast 场景，尾轴 >= dcache/2 | UB BRC |
| 3 | dtype 为 INT8/FP16/BF16 且尾轴 32B 对齐 | UB BRC |
| 4 | 其他 | NDDMA |

> **NLast**：尾轴不需要广播（stride≠0），但非尾轴需要广播（stride=0）。尾轴数据量大时 NDDMA 反复读会刷 dcache，不如 UB 内 Broadcast API。

### 4. UB 切分

从最内轴向外累乘，找到第一个放不下的轴作为 `ubSplitAxis`：

```python
curProduct = 1
ubSplitAxis = 0
allFit = true
for i = shapeLen-1 downto 0:
    curProduct *= dims[i]
    if curProduct > maxElemNum:
        ubSplitAxis = i
        curProduct /= dims[i]
        allFit = false
        break

if allFit:
    curProduct /= dims[0]

if shapeLen == 1:
    ubFormer = maxElemNum
else:
    ubFormer = maxElemNum / curProduct

ubOuter = ceil(dims[ubSplitAxis] / ubFormer)
ubTail  = dims[ubSplitAxis] - (ubOuter-1) * ubFormer
```

其中：

```python
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)
```

### 5. 多核切分

```python
fusedProduct = ubOuter × (ubSplitAxis 之前所有轴乘积)
blockFormer  = ceil(fusedProduct / coreNum)
blockNum     = ceil(fusedProduct / blockFormer)
blockTail    = fusedProduct - (blockNum - 1) * blockFormer
```

### 6. NDDMA 超过 5 轴的处理

```python
axesAfterSplit = shapeLen - ubSplitAxis
≤ 5 → WITHOUT_LOOP：一次 NDDMA 调用完成（DataCopy<T,5,config>）
> 5 → WITH_LOOP：最内 5 轴交给 NDDMA，外层轴 Kernel for-loop 遍历
```

### 7. 对齐规则

- **OneDim**：128B（`CACHE_LINE`）对齐。
- **多维**：256B（`REPEAT`）对齐。

### 8. 设计检查清单

- [ ] 是否完成补维 + 合轴？
- [ ] 是否按 flag 位图合并相邻同类型轴？
- [ ] 广播轴 stride 是否置 0？
- [ ] 是否根据 DAV 版本和条件选择 UB BRC / NDDMA？
- [ ] `ubSplitAxis` 是否从最内轴向外累乘得到？
- [ ] `maxElemNum` 是否按 bufferNum 与 dtype 位宽计算？
- [ ] NDDMA 维度超过 5 时是否拆分为 WITH_LOOP？
- [ ] 对齐是否满足 OneDim 128B / 多维 256B？
