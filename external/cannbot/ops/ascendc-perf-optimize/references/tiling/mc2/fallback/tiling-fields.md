# MC² 通算融合族 — 通算切分参数字段语义

> 均分切分算法涉及的通算切分参数字段定义、取值规则和推导出处。
>
> 基础 matmul tiling 字段见 [matmul/fallback/tiling-fields.md](../../matmul/fallback/tiling-fields.md)，本表仅覆盖 MC² 特有的通算切分参数。

---

## 1. 通算切分参数（均分算法输出）

| 字段 | 类型 | 含义 | 取值规则 | 推导出处 |
|------|------|------|---------|---------|
| `tileCnt` | uint32 | 总块数（通算切分的 tile 数量） | 均分：`M / longMSize`；短块：`longBlockCnt + shortBlockCnt` | §3.2 / §3.3 |
| `longBlockCnt` | uint32 | 长块数量 | 均分：`= tileCnt`；短块：`floor(M / longMSize)` | §3.2 / §3.3 |
| `longMSize` | uint32 | 长块的 M 维大小 | 满核下限：`minMBlockCnt × baseM`；均分取满足约束的最小值 | §3.2 |
| `shortBlockCnt` | uint32 | 短块数量 | 均分：`0`；短块降级：`1` | §3.2 / §3.3 |
| `shortMSize` | uint32 | 短块的 M 维大小 | 均分：`0`；短块降级：`M - longBlockCnt × longMSize` | §3.3 |
| `shortBlockPos` | uint8 | 短块位置（0=前, 1=后） | 均分：`0`（无短块）；短块降级默认 `1`（drain 端） | §3.3 |

---

## 2. 派生计算量

| 字段 | 公式 | 含义 |
|------|------|------|
| `nBlockCnt` | `ceil(N / baseN)` | N 方向 block 数（固定） |
| `mBlockCnt` | `ceil(longMSize / baseM)` | 单 tile 的 M 方向 block 数 |
| `totalBlocks` | `mBlockCnt × nBlockCnt` | 单 tile 的总 block 数 |
| `utilization` | `min(totalBlocks, N_core) / N_core` | 单 tile 的核利用率 |
| `minMBlockCnt` | `max(1, ceil(N_core / nBlockCnt))` | 满核所需最小 M 方向 block 数 |

---

## 3. matmul tiling 引用字段

均分切分算法本身不生成 matmul tiling data，而是引用 host tiling 引擎的输出：

| 引用 | 来源 | 说明 |
|------|------|------|
| `long_qbmm_tiling` | matmul/fallback/ (SWAT/FullLoad/StreamK) | 长块的完整 matmul tiling，以 `longMSize` 为 M 规模 |
| `short_qbmm_tiling` | matmul/fallback/ (SWAT) | 短块的 matmul tiling（仅 shortBlockCnt > 0 时），以 `shortMSize` 为 M 规模 |
| `local_qbmm_tiling` | comm-compute/local_matmul.md | 本 rank 全量 M 的独立 matmul tiling |

> **均分算法下 short_qbmm_tiling 不生效**（shortBlockCnt=0），仅需生成长块和 local 的 matmul tiling。

---

## 4. 降级标记字段

| 字段 | 类型 | 含义 | 取值 |
|------|------|------|------|
| `degradeLevel` | uint8 | 降级等级（0=主路径, 1=放宽对齐, 2=短块, 3=放宽利用率） | 主路径: 0；降级 A: 1；降级 B: 2；降级 C: 3 |
| `alignGranularity` | uint32 | longMSize 的实际对齐粒度 | 主路径: `baseM`；降级 A: `16` |

> 降级标记用于下游分析：若 degradeLevel ≥ 2，建议采集隔离测试数据后切换至长短块搜索算法。
