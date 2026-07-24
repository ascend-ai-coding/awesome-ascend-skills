# Blaze Custom Epilogue 层模块手册

> **定位**：blaze_custom 路径下 Epilogue 层各模块的使用手册。Epilogue 层运行在 AIV 核心上，接收 AIC 通过 CopyL0C2UB 写入 UB 的 matmul 结果，执行 vector 后处理并写回 GM。

---

## §1 模块清单

| 模块 | 头文件 | 类型 | 说明 |
|------|--------|------|------|
| `EpilogueFusionRegBase` | `epilogue/epilogue_fusion_regbase.h` | 通用 RegBase 框架 | 使用 `__VEC_SCOPE__` + `Reg::` API，中间值不写回 UB |
| `MulEpilogue` | `epilogue/epilogue_fusion_membase.h` | 参考 MemBase 实现 | Output = matmul(A,B) × D，使用 `AscendC::Mul` |
| `CvSync` 常量 | `epilogue/cv_sync_constants.h` | 同步常量集 | MODE=4, AIC_TO_AIV_FLAG=8, AIV_TO_AIC_FLAG=5, COUNT_ID_MAX=15, COUNT_FLAG=3 |

**设计文档入口**：

- 通用设计方法：`development/epilogue-dev-guide.md`
- MemBase 设计：`development/epilogue-membase-design.md`
- RegBase 设计：`development/epilogue-regbase-design.md`

**RegBase 核心特性**：

- 计算核心使用 `__VEC_SCOPE__` 块，内部声明 `Reg::RegTensor<T>` 和 `Reg::MaskReg`
- 中间值全程驻留向量寄存器，不落 UB，减少 UB 中间 buffer 数量
- 通过 `Reg::LoadAlign` 加载、`Reg::StoreAlign` 写回
- 支持 `Reg::Add/Sub/Mul/Div`、`Reg::Adds/Muls/Divs`、`Reg::Axpy`、`Reg::Exp/Log/Abs/Sqrt`、`Reg::Compare/Select` 等 API

**MemBase 参考实现（MulEpilogue）**：

- 使用标准 AscendC API（`AscendC::Mul`）
- 三路 UB buffer：`cLocal_`（matmul 结果）+ `dLocal_`（第二路输入）+ `cLocalTmp_`（计算结果）
- StageNum = 2（UB 分区数）

---

## §2 RegBase vs MemBase 选择

| 判断维度 | 选 RegBase | 选 MemBase |
|---------|-----------|-----------|
| 公式复杂度 | 复杂公式（GELU / SwiGLU / LayerNorm），3+ 中间值 | 简单 vector 操作（Mul / Add / Div） |
| 可用 API | 需要 `Reg::` 系列 API（Exp/Log/Axpy 等） | 已有现成 `AscendC::` API（如 `AscendC::Mul`） |
| UB 占用 | 较少（2-3 blocks） | 较多（3-4 blocks） |
| AIV 效率 | 更高（寄存器级操作，无 UB 读写开销） | 一般（需 UB 中间 buffer 读写） |
| 开发难度 | 较高（需理解 RegTensor/MaskReg/VF 循环） | 较低（类标准 AscendC 开发） |

**决策建议**：优先 RegBase。仅当融合操作为单一 vector op 且已有对应 `AscendC::` API 时选 MemBase。

设计细节请查阅：

- `development/epilogue-dev-guide.md`
- `development/epilogue-membase-design.md`
- `development/epilogue-regbase-design.md`

---

## §3 接口契约

普通 C+V 与 MX C+V 的 Epilogue 类必须提供以下四个接口，与 `MatmulKernelFused` / `MxMatmulKernelFused` 兼容：

| 接口 | 签名 | 说明 |
|------|------|------|
| `Params` | `struct Params { ... }` | 参数结构体，含 GM_ADDR 字段 |
| `Init` | `void Init(Params, baseM, baseN, ProblemShape)` | 初始化 UB 布局、绑定 GM 地址 |
| `operator()` | `void operator()(BlockShape, gmOffset, flagId)` | 逐 tile 执行后处理 |
| `GetTensor` | `auto GetTensor()` | 返回 `cLocal_`（UB Tensor，供 Kernel 层 CopyL0C2UB 使用） |

Grouped C+V 使用 context-based Epilogue，`operator()` 签名不同：

```cpp
__aicore__ inline void operator()(BlockShape blockShape, TileContext tileContext, int64_t flagId)
```

`TileContext` 由 `GroupMatmulKernel` 传入，包含 `groupIdx`、`prefixM`、`mOffset`、`nOffset`、`writeM`、`totalM`、`totalN`、`totalK` 等字段。普通 linear offset Epilogue 不直接兼容 Grouped C+V。

### 兼容性矩阵

| Epilogue 类型 | 普通 C+V | MX C+V | Grouped C+V |
|---------------|----------|--------|-------------|
| `MulEpilogue` / 普通 linear offset Epilogue | 可用 | 可用，但必须按 `GetTaskRation()/GetSubBlockIdx()` 消费 SplitM 后的 UB | 不直接可用 |
| MX SplitM-aware Epilogue | 不需要 | 可用 | 不直接可用 |
| context-based grouped Epilogue | 不需要 | 不需要 | 必需 |

**Params 结构体约定**：

- RegBase：`GM_ADDR extraInputAddr`（额外输入）+ `GM_ADDR outputGmAddr`（输出）
- MemBase（MulEpilogue）：`GM_ADDR multiplierGmAddr`（乘数 D）+ `GM_ADDR outputGmAddr`（输出）

**BlockShape / ProblemShape**：

- `BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>` — `{curM, curN, 1, 1}`
- `ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>` — `{m, n, k, 1}`

**辅助方法**：

- `InitParams(Params)` — `__host_aicore__` 静态方法，用于 host 端参数透传

---

## §4 UB 占用情况

| 路径 | UB 分区 | 说明 |
|------|---------|------|
| **RegBase** | 2-3 blocks | `cLocal_`（matmul 结果，固定）+ `extraBuf_`（额外输入，1 行）+ `bf16Out_`（输出，stageRows 行） |
| **MemBase** | 3-4 blocks | `cLocal_`（matmul 结果，固定）+ `dLocal_`（第二路输入，stageSize）+ `cLocalTmp_`（计算结果，stageSize） |

**RegBase UB 布局计算**：

```
matmulAreaBytes = ceilDiv(baseM, taskRation) × nAlign × sizeof(L0CDataType)
extraBufBytes   = nAlign × sizeof(ComputeType)
remainBytes     = TOTAL_UB_SIZE - matmulAreaBytes - extraBufBytes
stageRows_      = remainBytes / (nAlign × sizeof(OutputType))
```

**MemBase UB 布局计算**：

```
matmulArea = ceilDiv(baseM, taskRation) × nAlign    // 元素数
lastUBBytes = TOTAL_UB_SIZE - matmulArea × sizeof(DataType)
stageSize_ = min(lastUBBytes / StageNum / sizeof(DataType), matmulArea)
```

RegBase 通过 `ReinterpretCast` 划分 UB 区域，MemBase 通过 `LocalTensor` 偏移划分。RegBase 的 stage 循环按行切分（`rowsThisStage`），MemBase 按元素数切分（`curStageSize`）。

---

## §5 命名空间说明

| 模块 | 命名空间 | 说明 |
|------|---------|------|
| `EpilogueFusionRegBase` | 全局作用域 | 不在任何 namespace 内 |
| `MulEpilogue` | 全局作用域 | 不在任何 namespace 内 |
| `CvSync` | `CvSync::` namespace | 同步常量集合 |

Epilogue 类定义在全局作用域，不归属 `Kernel::` 或 `Block::` namespace。Kernel 层通过模板参数直接引用 Epilogue 类名（如 `MatmulKernelFused<..., EpilogueFusionRegBase>`）。
