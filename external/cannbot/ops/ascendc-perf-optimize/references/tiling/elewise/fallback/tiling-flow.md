# Elementwise 族 — Tiling 流程

> Elementwise 族（Sin, Cos, Abs, Exp, Add, Mul 等逐元素算子）Tiling 推导流程。
> 所有输入输出 Shape 完全相同，无跨元素依赖。

---

## 1. 场景判定

```
给定: N个输入shape + M个输出shape

Step 1 — Shape判定:
  所有输入输出shape完全相同? → EleWise, 展平为dim0处理
  否则 → Broadcast分支

Step 2 — dtype×运算判定 (决定Compute路径):
  运算为Add/Sub + dtype∈{FP16,BF16} + 输入量级未知?
    → 升精度分支: Cast→FP32计算→Cast, 需额外FP32中间Buffer
    → 否则: 原dtype直接计算
```

---

## 2. 多核切分

```
coreNum = min(
  (dim0 * minDtypeBits + MIN_TILING_BITS - 1) / MIN_TILING_BITS,
  availableCoreNum
)

blockFormer = AlignUp(CeilDiv(dim0, coreNum), 512)
blockNum = CeilDiv(dim0, blockFormer)
blockTail = dim0 - (blockNum - 1) * blockFormer
```

确保每核至少4KB数据，元素数对齐到512。

---

## 3. UB切分

```
原dtype直算分支:
  bufferDivisor = bufferNum * elemBytes

升精度分支:
  bufferDivisor = bufferNum * elemBytes + K * sizeof(float)

maxElemNum = (ubSize - extraSize) * 8 / bufferDivisor
ubFormer = AlignDown(maxElemNum, 256*8/minDtypeBits)

ubLoopFormer = CeilDiv(blockFormer, ubFormer)
ubTailFormer = blockFormer - (ubLoopFormer-1) * ubFormer
```

---

## 4. Kernel 执行模型

```
for blockIdx in [0, blockNum):
  isLastBlock = (blockIdx == blockNum - 1)
  loopNum = isLastBlock ? ubLoopTail : ubLoopFormer
  tailNum = isLastBlock ? ubTailTail : ubTailFormer

  for i in [0, loopNum - 1):
    ProcessTile(offset, ubFormer)
    offset += ubFormer
  ProcessTile(offset, tailNum)

ProcessTile: CopyIn → Compute → CopyOut
```

**区分首/尾block**：尾block数据量可能小于首block，循环次数不同。

---

## 5. 升精度分支 UB 预算

| 项 | 原dtype直算 | 升精度分支 |
|---|------------|-----------|
| 输入/输出Buffer | half/bf16 | half/bf16 (不变) |
| FP32中间Buffer | — | K份 ubFormer*sizeof(float) |
| UB总占用 | bufferNum*ubFormer*elemBytes | 左边 + K*ubFormer*4 |
