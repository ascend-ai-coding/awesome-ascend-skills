# AICore / AscendC 编程约束（Catlass 算子必遵）

本文档将 **AscendC/AICore 编程约束** 合入 Catlass 侧文档。Catlass 算子**设计**与**代码生成**均须遵守以下内容；设计阶段选型时**只选用 Catlass 已有组件**，不得设计需在 Kernel 内自实现的计算逻辑。

---

## 1. 严正约束：AICore 与 Catlass 唯一实现

### 1.1 AICore 上标量计算极慢

- **禁止以 SIMT（单线程标量循环）视角编写算子代码**。
- 禁止在 Kernel 内手写逐元素标量循环（如 `for (i) for (j) out[i,j] = ...`）。
- AICore 架构与 GPU 不同，标量循环极慢，必须使用向量/块级接口。

### 1.2 仅使用 Catlass 提供的实现

- Kernel 中**只能**使用 Catlass 提供的 Block/Tile/Kernel 组合。
- **不得**自行实现：矩阵乘、逐元素加、逐元素乘、拷贝等计算逻辑。
- 可用组件示例：`BlockMmad`、`BasicMatmul`、`MatmulEpilogue`、`BlockEpilogue`、`TileElemWiseAdd`、`TileCopy` 等（以 catlass 仓内头文件为准）。

### 1.3 Catlass 中已有的 mmad/epilogue 实现

- 矩阵乘 + 偏置 / 加 D 等需求，应使用：
  - **Kernel**：`Gemm::Kernel::MatmulEpilogue`
  - **BlockEpilogue**：如 `EpilogueAtlasA2ElemWiseOneSource` + `TileElemWiseAdd` + `TileCopy`
- **参考实现**：
  - `catlass/examples/03_matmul_add/matmul_add.cpp`
  - `catlass/include/catlass/gemm/kernel/matmul_epilogue.hpp`
- 不得用自写标量或自写向量循环替代上述 Catlass 组件。

### 1.4 无现成组件时的处理

- 若需求在 Catlass 中**无**现成组件：
  - 应在**设计文档**中改为选用 Catlass 已有组件，或
  - 明确说明需**扩展 Catlass 库**后再实现，不得设计「在 Kernel 内自实现计算」的方案。

---

## 2. 设计阶段选型须知

- 输出设计文档时，**核心组件结构**必须全部来自 Catlass 已有 Kernel/Block/Tile/Epilogue；不得写出「自写标量/向量循环实现某某运算」的实现方案。
- Epilogue 选型请结合 [epilogue-components.md](epilogue-components.md) 与 catlass 仓内 `matmul_epilogue.hpp`、`03_matmul_add` 等确定可用组合。

---

## 3. 反模式（NEVER）

- 在 AICore Kernel 内**手写标量/逐元素循环**或**自实现矩阵乘、逐元素加**。
- 以 SIMT 方式在 Kernel 内写 `for (i) for (j) out[i,j] = ...` 或等价标量逻辑。
- 设计文档中给出「自写 C+D、自写 GELU」等依赖非 Catlass 组件的 Kernel 实现方案。

---

## 4. 参考资料（Catlass 仓内）

| 路径 | 说明 |
|------|------|
| catlass/examples/03_matmul_add/matmul_add.cpp | 矩阵乘 + 逐元素加（D = A*B + X）的完整参考 |
| catlass/include/catlass/gemm/kernel/matmul_epilogue.hpp | MatmulEpilogue Kernel 定义 |
| catlass/include/catlass/epilogue/block/block_epilogue_elemwise_one_source.hpp | BlockEpilogue（单源逐元素） |
| catlass/include/catlass/epilogue/tile/tile_elemwise_add.hpp | TileElemWiseAdd |

以上约束以 **Catlass 侧文档** 为准，供 catlass-operator-design 与 catlass-operator-code-gen 共同遵守；不依赖对 AscendC 相关 skill 的修改权限。
