# Catlass 后处理（Epilogue）组件清单

**若下表无所需运算**：先走 [custom-epilogue.md](./custom-epilogue.md)（设计：确认 catlass 是否已有 Tile → 无则先设计自定义 Tile），再在 codegen 中落盘 `op_kernel/custom_epilogue/`。

Epilogue 组件分为 **Tile 级**（单 tile 内操作）和 **Block 级**（协调多个 tile）两层，按功能分为以下类别。

## Tile 级组件

| 类别 | 可选组件 | 说明 |
|------|---------|------|
| 激活函数 | `TileElemWiseGelu`, `TileElemWiseSilu` | GELU、SILU 激活；**无**现成 `TileElemWiseTanh` 时在自定义 Tile 中用 **Exp/Muls/Adds/Div** 实现 \(\tanh\)（与 Silu 同双缓冲范式），或评估 `AscendC::Tanh` 与固定 `COMPUTE_LENGTH`、小 \(M\times N\) 的运行期 UB 风险 |
| 逐元素运算 | `TileElemWiseAdd`, `TileElemWiseMul`, `TileElemWiseMuls` | 加法、乘法、标量乘 |
| 广播运算 | `TileBroadcastAdd`, `TileBroadcastMul`, `TileBroadcastInplaceByColumn`, `TileBroadcastInplaceByRow` | 广播加/乘、按列/行原地广播 |
| 数据搬运 | `TileCopy`（含 `CopyGMToUB`、`CopyUBToGM` 变体） | GM ↔ UB 搬运 |
| 类型转换 | `TileCast` | 精度转换 |
| 数据重排 | `TileSwizzle` | 内存布局重排 |

## Block 级组件

| 类别 | 可选组件 | 说明 |
|------|---------|------|
| 通用后处理 | `BlockEpilogue` | 组合 Tile 组件的标准流水 |
| 单源逐元素 | `BlockEpilogueElemwiseOneSource` | 带一个额外输入（如加偏置） |
| 无源逐元素 | `BlockEpilogueElemwiseNoSource` | 无额外输入的后处理 |
| Softmax | `BlockEpilogueOnlineSoftmaxNoMask` | 在线 Softmax |
| 量化反量化 | `BlockEpiloguePerTokenDequant`, `BlockEpilogueW4A4PerTokenPerChannelDequant` | 反量化 |
| GEMM/GEMV | `BlockEpilogueGEMM`, `BlockEpilogueGEMV` | GEMM/GEMV 专用 |
| Flash Attention | `BlockEpilogueFARescaleO`, `BlockEpilogueFASoftmax` | FA 后处理 |
| MLA | `BlockEpilogueMLARescaleO`, `BlockEpilogueMLASoftmax` 等 | MLA 后处理 |

## 组装模式

`BlockEpilogue` 将多个 Tile 组件组合为后处理流水线，典型模式：

| 模式 | 流水线 | 典型用例 |
|------|--------|---------|
| 纯矩阵乘 | 无 Epilogue（`BlockEpilogue = void`） | 基础 Matmul |
| 激活函数 | Copy入 → GELU/SILU → Copy出 | Matmul + GELU |
| 加偏置 | Copy入 → Add(额外源) → Copy出 | Matmul + Bias |
| 加偏置+激活 | Copy入 → Add → GELU → Copy出 | Matmul + Bias + GELU |
| 广播+运算 | Copy入 → BroadcastAdd → Mul → Copy出 | 融合后处理 |

## 选型原则

1. **按需组合**：只选需要的环节，不要全部引入
2. **Tile 组件按序排列**：Copy入 → 计算 → Copy出
3. **Block 组件接收 Tile 列表**：`BlockEpilogue<DispatchPolicy, InType, OutType, Tile...>`
4. **无后处理时设 `BlockEpilogue = void`**
