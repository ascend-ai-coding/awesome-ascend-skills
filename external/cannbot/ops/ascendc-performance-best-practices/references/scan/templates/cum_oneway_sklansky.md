# One-way Sklansky 并行前缀扫描

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

代码：[dav310/cum_oneway_sklansky.template](dav310/cum_oneway_sklansky.template)
提取自：`ops-math/math/cumsum/op_kernel/arch35/cumsum_base/cumsum_oneway_sklansky.h`

## 解决的问题

单行扫描轴 R 很长时，流式串行扫描的 O(R) 步串行延迟成为瓶颈。Sklansky（分治扇出）把前缀扫描做成 **`ceil(log2(R))` 层**，每层对整行向量载荷做一次结合律 `combine`，把串行深度从 O(R) 降到 O(log R)。

## 适用条件

- Cumsum / Cumprod / Cummax / Cummin（**无索引变体**）：`combine` 满足结合律。
- R 全量入 UB（否则叠加 UbChunk 层，见 [cum_core_partition_scan](cum_core_partition_scan.md) 与 guide 的三层骨架）。
- **带索引（argmin/argmax）不用本模板** —— Sklansky 层内组合会丢失取得极值的具体步，改用流式扫描。

## 执行流程（8 行正向示例）

```
层0: 锚点 0,2,4,6  →  combine 到 1,3,5,7
层1: 锚点 1,5      →  combine 到 2,3 与 6,7
层2: 锚点 3        →  combine 到 4,5,6,7
```

第 k 层：半组宽 `2^k`、组宽 `2^(k+1)`；正向锚点 = 组内第 `2^k-1` 行，把锚点行 `combine` 到其后 `2^k` 行。尾组（不足整组）单独处理。

## 关键技术

| 技术 | 要点 |
|------|------|
| **整行向量 combine** | 扫描维 R 是标量层号 k；每个前缀节点载荷是整行向量（`alignN×mFold` 元素）。combine 是整行 `Add`/`Min`/… |
| **VF 逐 VL 块** | `RegTensor` + `MaskReg`，逐 `VL=64` 块处理行，尾部 `UpdateMask` |
| **窄 N 折叠（twoway）** | 窄 N 时把长 R 折叠进空闲向量道：段内 Sklansky → `Gather` 紧凑重排 → 段间 Sklansky。见 cumsum `cumsum_twoway_sklansky.h` |

## combine 替换点

`CombineRow` 中的 `Add(b, a, b, mask)`：
- Cumsum → `Add`；Cumprod → `Mul`；Cummax/Cummin（无索引）→ `Max`/`Min`。

## 关键代码索引

- `BaseProcess`：`for k in [0, ceil(log2(rFactor)))` 逐层。
- `ComputeLevel`：算半组宽/组宽/组数，定位锚点行与目标行。
- `CombineRow`：整行向量 combine（← 替换点）。
