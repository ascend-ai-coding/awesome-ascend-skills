# 自定义 Epilogue（设计阶段）

**侧重点**：在定稿组件选型前，判断 **catlass 仓库是否已有** 所需 Tile/流水线；若无，先在设计文档中**定义**自定义 Tile 的数学与接口契约，再交给 **catlass-operator-code-gen** 落盘头文件与 kernel 引用。

---

## 1. 何时走「自定义 Tile」

1. 查 [epilogue-components.md](./epilogue-components.md) 与 `catlass/include/catlass/epilogue/tile/`、`examples/`（如 `03_matmul_add`）。
2. **已有** 与需求一致的 Tile（如 `TileElemWiseAdd`）→ **不要** 自定义；设计文档写清选用哪个**现成**符号即可。
3. **没有** 现成 Tile（新逐元素公式、新融合方式）→ 进入本节：**先设计，后 codegen**。

---

## 2. 设计文档须补充的内容（无代码）

在「核心组件选型 → BlockEpilogue」中**单列一小节**（或表格），至少包含：

| 项 | 说明 |
|----|------|
| **自定义 Tile 名称** | 如 `TileMyFusion`，与 catlass 内置不重名 |
| **数学/行为** | 对 `ubIn0`（matmul 结果）、`ubIn1`（额外源，若 policy 需要）的运算定义 |
| **DispatchPolicy** | `EpilogueAtlasA2ElemWiseNoSource` / `OneSource` / `TwoSource`（与额外 GM 输入个数一致） |
| **与 BlockEpilogue 的组装** | `TileCopy` + 自定义 Tile 的顺序；`CType`/`XType`/`DType` 与 I/O dtype 一致 |
| **computeLength** | 与 L1/L0、UB 约束一致的设计值（见下游 codegen 文档） |
| **参考** | 最接近的 example（如 `03_matmul_add`）仅作**结构类比**，非照抄 |

设计阶段**不写**可编译 C++，只写清上表，便于 codegen 生成 `op_kernel/custom_epilogue/*.hpp` 并在 `*_impl.h` 中 `using`。

---

## 3. 与下游 codegen 的衔接

- 落盘约定见 **catlass-operator-code-gen**：[custom-epilogue.md](../catlass-operator-code-gen/references/custom-epilogue.md)。
- 设计文档中若出现自定义 Epilogue，**实现方案纲要**须点明：kernel 侧增加 `custom_epilogue/` 头文件（目录下自动扫描，无需额外编译选项配置）。

---

## 4. 检查清单（设计）

- [ ] 已检索 catlass `epilogue/tile` 与 epilogue-components，确认「无现成组件」或说明为何不沿用
- [ ] 设计文档已写清自定义 Tile 名称、数学、DispatchPolicy、与 BlockEpilogue 组装关系
- [ ] 未在设计文档中粘贴大段可编译 kernel 代码
