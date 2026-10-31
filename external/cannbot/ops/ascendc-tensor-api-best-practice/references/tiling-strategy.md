# Tiling 决策树与多核切分

本参考在 DESIGN/PLAN 与性能优化节点按 implementation_route=tensor_api_custom 条件加载。承接 [api-dataflow-layout.md](api-dataflow-layout.md) 的「逐级内存预算——任一级超限先改 tile」：本文给出**怎么改**的决策树。

## 1. Tiling 职责与三原则

Tiling = 把输入数据切分后分块计算。**Host 侧算切分参数（标量计算，AI Core 不擅长），Device 侧 Kernel 按参数搬数/计算**。固定 shape 的简单单核算子可省 Host Tiling；动态 shape 或多核并行必须在 Host 算好传给 Kernel。

三原则：
1. **DataBlock 为最小切分单位**（32B 对齐，搬运与矢量计算的硬约束）。
2. **充分利用 UB**：块越大、搬数次数越少，减少 GM 往返。
3. **均衡利用多核**：计算量尽量均分到各核，减少尾核空闲。

## 2. 核心概念

| 概念 | 含义 |
|---|---|
| 主块 | 单核一次处理的最大块，长度 `mainTileLength`（按 UB 可用空间定，32B 对齐） |
| 尾块 | 核内最后一个 < 主块的数据块，长度 `lastTileLength` |
| 整核 | 核间无法完全均分时，多处理数据的核，数量 `formerNum` |
| 尾核 | 少处理数据的核 |

## 3. 四场景决策树（按剩余数据落点）

Host 侧先算三个量，再按落点选场景：

```
alignNum = 32 / sizeof(T)                          # 一个 DataBlock 的元素数
totalLengthAligned = AlignUp(totalLength, alignNum) # 对齐后总量
mainTileNum  = totalLengthAligned / (numBlocks × mainTileLength)   # 每核主块数
mainTileRemainder = totalLengthAligned % (numBlocks × mainTileLength) # 主块后剩余
tailBlockNumEachCore = mainTileRemainder / (numBlocks × alignNum)   # 每核基础尾块 DataBlock 数
remainingTailLength  = mainTileRemainder % (numBlocks × alignNum)   # 仍无法均分的剩余
```

| 落点 | 场景 | 各核形态 | 需要的 Tiling 字段 |
|---|---|---|---|
| `mainTileRemainder == 0` | 主块均分 | 只含主块 | blockLength, tileNum |
| `tailBlockNumEachCore>0 且 remainingTailLength==0` | 尾块均分 | 主块 + 等长尾块 | + lastTileLength |
| `tailBlockNumEachCore==0 且 remainingTailLength>0` | 尾核切分 | 整核(主块+1 DataBlock 尾块) / 尾核(只主块) | formerNum, former{Length,TileNum,LastTileLength}, tail{Length,TileNum} |
| `都 > 0` | 尾核尾块切分 | 整核+尾核都有尾块，整核多 1 DataBlock | 上一行 + tailLastTileLength |

**结构体字段随场景递进**：场景 0 只需 `blockLength/tileNum`；场景 1 加 `lastTileLength`；场景 2 用 `former*`/`tail*` 两组；场景 3 再加 `tailLastTileLength`。

**整核数**：`formerNum = remainingTailLength / alignNum`；整核比尾核多一个 DataBlock。

**尾块合并主块**：场景 3 里 `formerLastTileLength = baseLastTileLength + alignNum`，若恰好等于 `mainTileLength`，则并入主块（`formerTileNum += 1`，尾块长度归 0），避免多一次搬数。

## 4. 核间负载均衡

当 `L2 Cache 切分后` 每核计算量不被核数整除时，固定「前 N 核多算一块」会造成尾核始终空闲（计算拖尾）。调整**拖尾核的位置**（让多算的核分散在时间轴上），可让最后一轮所有核都接近满载，实现全局负载最优。大 E / 大 batch 场景尤其要检查这一步。

## 5. Kernel 侧消费 Tiling

Kernel 用 Tensor API 侧核 ID `block_idx` + Tiling 参数算出本核 GM 偏移，按 `tileNum` 循环搬入/算/搬出，末尾按 `lastTileLength` 处理尾块。核数取 `num_blocks`。尾块必须显式处理、padding 不参与 Golden（与 [api-dataflow-layout.md](api-dataflow-layout.md) 的 mmad 门禁一致）。

## 6. 决策检查清单

1. dtype → `alignNum`（fp16/bf16=16，fp32=8，int8/fp8=32）。
2. 主块长度由 UB 逐级预算反推——UB 容量（256KB 基线）扣掉输出切片与元数据后，按 dtype 字节宽度反推可容纳的最大元素数（32B 对齐），不是拍脑袋；逐级预算表细节见 api-dataflow-layout.md §内存预算。
3. 按 §3 落点选场景，尾块/尾核字段缺一个都会算错数据。
4. 尾块合并主块的边界（`formerLastTileLength == mainTileLength`）必须处理。
5. 多核负载：检查是否拖尾，必要时调整尾核分布。

## 事实源

- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_impl/vector_programming/multi_core_tiling/`（overview + 四场景）
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/tiling_strategy/`（核间负载均衡）
- 样例：`$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/02_features/02_tiling_selector/vector_tiling_strategy/`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。字段类型与对齐事实以当前头文件/手册为准。
