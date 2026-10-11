# Broadcast 族 — Tiling 流程

> Broadcast 族二元/多元逐元素算子（Add, Mul, Sub）的 Tiling 推导流程。

---

## 1. 算法分支决策树

```
给定: N个输入shape + 1个输出shape + dtype + ubSize + coreNum

Step 0 — 合轴 (DimensionCollapse):
  补维(左补1) → 标记广播轴flag → 合并相邻同flag轴 → 计算strides

Step 1 — 分支判定:
  ├─ 合轴后仅1维 → OneDim (纯Elementwise,可能有标量输入)
  │
  └─ 合轴后>1维 → 选择广播方式:
      ├─ DAV_2201 → UB Broadcast 静态接口 (rank=1/2, 32B对齐约束)
      │
      └─ DAV_3510 → 广播发生在哪个阶段?
          ├─ GM→UB搬入阶段 → 按优先级决策链选择:
          │   ① 用户强制 → 遵从
          │   ② NLast场景 且 尾轴≥dcache/2 → UB BRC动态接口
          │   ③ dtype为INT8/FP16/BF16 且 尾轴32B对齐 → UB BRC动态接口
          │   ④ 其他 → NDDMA
          │
          └─ UB内部广播 (中间结果需广播) → UB BRC动态接口 (rank 1~9)
```

---

## 2. OneDim 分支

**条件**: 合轴后只剩1维。

**流程**:
1. UB切分: `ubFormer = (ubSize / bufferNum) 对齐到 128B`，标量输入用TensorScalar接口
2. 多核切分: `blockFormer = ceil(ubOuter / coreNum)`
3. 核利用率不足时缩小ubFormer重算

**数据流**:
```
GM → DataCopyPad → UB[ubFormer] → Compute(Adds/Muls标量优先) → DataCopyPad → GM
```

---

## 3. UB Broadcast 分支

**条件**: 合轴后多维，UB内调用Broadcast API展开。

**流程**:
1. 从最内轴向外累乘，找到放不下的轴作为 `ubSplitAxis`
2. UB切分: `ubFormer = maxElemNum / curProduct`
3. 多核切分: 展平ubSplitAxis及其外层轴，均分给多核
4. 搬运指令优化: DataCopyPad dummy填充 → Copy → GatherMask (省tmpBuffer)

**广播方式子决策** (axis=-1):
- Broadcast静态接口约束(srcShape对齐)满足 → Broadcast API
- 否则 M>2 → DataCopyPad填充+Copy+GatherMask
- M≤2 → 逐行Duplicate

---

## 4. NDDMA Broadcast 分支 (DAV_3510专属)

**条件**: GM→UB搬入阶段，NDDMA硬件stride=0自动复制。

**流程**:
1. UB切分、多核切分与UB Broadcast相同
2. 判定schMode: 剩余轴≤5 → WithoutLoop; >5 → WithLoop
3. WithoutLoop: 一次DataCopy<T,5>完成
4. WithLoop: 最内5轴给NDDMA，外层Kernel for-loop遍历
5. FuseAxis优化: 合并广播模式相同的相邻轴

---

## 5. 通用UB切分公式

```
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)

从最内轴向外累乘outputDims:
  curProduct *= dims[i]
  if curProduct > maxElemNum:
    ubSplitAxis = i; curProduct /= dims[i]; break

ubFormer = maxElemNum / curProduct
ubOuter = ceil(dims[ubSplitAxis] / ubFormer)
ubTail = dims[ubSplitAxis] - (ubOuter-1) * ubFormer
```

## 6. 通用多核切分公式

```
fusedProduct = ubOuter × (ubSplitAxis之前所有轴乘积)
blockFormer = ceil(fusedProduct / coreNum)
blockNum = ceil(fusedProduct / blockFormer)
blockTail = fusedProduct - (blockNum-1) * blockFormer
```

## 7. 对标量输入的处理

| 场景 | 方案 |
|------|------|
| OneDim + 有对应TensorScalar接口 | Adds/Muls/Subs 直接对标量 |
| OneDim + 无TensorScalar接口 | Duplicate展开 + TensorTensor接口 |
| 多维 | strid=0 → NDDMA硬件复制或UB Broadcast |
