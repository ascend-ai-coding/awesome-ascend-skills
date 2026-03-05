---
name: ascendc
description: AscendC transformer/GMM/MoE 算子与 Matmul/Cube Kernel 的统一开发规范。用于在 ops-transformer 下新增或修改 op_host、tiling/infershape、op_kernel（含 MatmulImpl/Cube 调用）、以及对应的 CANN aclnn 示例和单测。
keywords:
  - ascend
  - ascendc
  - kernel
  - npu
  - 开发环境
  - 算子
  - 昇腾
---

# AscendC Transformer 算子开发

指导 Agent 按现有模式开发/修改 AscendC 的 FFN、GMM、MoE 类算子及对应 CANN `aclnn_*` 示例。**具体约定与代码模板见同目录下 `references/`；执行任务前请先通读本 SKILL.md，再按「References 索引」打开对应文档。**

---

## 如何读取本 Skill（能力与使用方式）

- **本 skill 能做什么**：在 ops-transformer 工程中新增/修改 AscendC 算子（op_host、tiling、infershape、op_kernel）、Matmul/Cube 调用、以及 aclnn 示例与单测；所有约定与模板集中在 `references/` 中，本文件提供索引与工作流。
- **推荐读取顺序**：① 本 SKILL.md 全文（When to Use、References 索引、Overall Workflow、各 Step）；② 根据当前任务从「References 索引」表打开对应 `references/0X-*.md`；③ **以 CANN 官方文档与 references 中的「官方文档参考」为准**，工程内同类型算子可作实现参考。
- **references 路径**：与 SKILL.md 同级的 `references/` 目录，文档名为 `01-type-format.md`～`08-genop.md`，表中已按主题与何时查阅列出。

---

## 使用本 Skill 的方式（Agent 必读）

1. **先读本文件**：确认 When to Use、Overall Workflow、References 索引与各 Step 摘要。
2. **按需读 references**：根据当前任务打开下表对应文档（如做 tiling 则读 `06-tiling.md`，做 Matmul 则读 `02-kernel-guide.md`），按文档内约定实现或修改代码。
3. **以官方文档为参考示例**：规范与示例以 CANN 官方文档（算子开发、Ascend C API、Tiling、单算子调用等）及 references 内列出的官方链接为准；工程内同类型算子可作实现参考，不凭空发明模式。

---

## When to Use

- **算子层面**：在 `ops-transformer` 下新增或修改 FFN / GMM / MoE / 路由类 AscendC 算子（含前向、反向、路由融合等）。
- **Kernel 层面**：在 AscendC `op_kernel` 中实现或调整 Matmul/Cube 调用（如 `MatmulImpl`、分块 GEMM、AIC/AIV 协作、确定性 GMM、`grouped_matmul_finalize_routing` 风格）。
- **Tiling / Infershape**：补充或修改 `*_tiling*.h/.cpp`、`*_infershape.cpp`，或理解 shape→tiling→kernel 的完整映射。
- **示例与单测**：编写或调整 CANN `aclnn_*` 示例与 Python/CPP 单测，接口、dtype、格式与 op_host/op_kernel 精确对齐。
- **对齐与重构**：重构、修 bug 或新增功能时，严格沿用现有 FFN/GMM/MoE 模式。

---

## References 索引（按执行顺序与主题）

| 文档 | 说明 | 何时查阅 |
|------|------|----------|
| [references/01-type-format.md](references/01-type-format.md) | op_host 类型/格式：DataType·Format·UnknownShapeFormat 个数约定、JSON 映射 | 定义或修改 Input/Output 时 |
| [references/02-kernel-guide.md](references/02-kernel-guide.md) | Kernel：GlobalTensor/TQue、CopyIn/Compute/CopyOut；Matmul/Cube 模板；GMM 转置说明 | 写/改 op_kernel、Matmul 调用时 |
| [references/03-op-host-examples.md](references/03-op-host-examples.md) | FFN/GMM/MoE 的 Input/Output/Attr 定义示例代码 | 写 *_def.cpp 时 |
| [references/04-op-kernel-skeletons.md](references/04-op-kernel-skeletons.md) | FFN/GMM/MoE 的 op_kernel 命名空间与主类骨架 | 写/改 op_kernel 主类时 |
| [references/05-json-types-flow.md](references/05-json-types-flow.md) | JSON + graph/types.h 驱动 op_host/infershape/tiling 对齐流程 | 接口与 JSON/types 对齐时 |
| [references/06-tiling.md](references/06-tiling.md) | Tiling 实现：标准 C++ 与宏定义两种方式及取值/写回差异 | 实现或修改 tiling 时 |
| [references/07-aclnn-template.md](references/07-aclnn-template.md) | aclnn 示例通用模板与生成步骤 | 写 test_aclnn_* 时 |
| [references/08-genop.md](references/08-genop.md) | genop 命令、生成结构、生成后定制与常见问题 | 从零生成新算子目录时 |

**官方文档为参考示例**：本 skill 以 CANN 官方文档为规范与示例来源，不依赖本地工程路径。

- **Ascend C API**：[Ascend C API 列表](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)（LocalTensor、GlobalTensor、TQue、Matmul 等）
- **Tiling**：[Host侧Tiling实现 - 基本流程](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html)、[使用标准C++语法定义Tiling结构体](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00024.html)
- **样例代码**：[Gitee ascend/samples - operator/ascendc](https://gitee.com/ascend/samples/tree/master/operator/ascendc)

各 references 文档内均设有「官方文档参考」小节，列出对应官方链接。

---

## Overall Workflow

1. **以官方文档与 references 为准**：规范与示例见 CANN 官方文档及本 skill 的 references；工程内可在 `ops-transformer/ffn/`、`gmm/`、`moe/` 下对照同类型算子的 `*_def.cpp`、`*_tiling*.h/.cpp`、`op_kernel/*.h`、`examples/test_aclnn_*.cpp` 作实现参考。
2. **op_host 定义图算子接口**：Input/Output/Attr、AICore 配置、OP_ADD。见 [01-type-format](references/01-type-format.md)、[03-op-host-examples](references/03-op-host-examples.md)。
3. **op_kernel 实现 AscendC 内核**：Init、Process、队列与 UB 管理、Matmul 时见 [02-kernel-guide](references/02-kernel-guide.md)、[04-op-kernel-skeletons](references/04-op-kernel-skeletons.md)。
4. **完成 tiling、infershape 与注册**：见 [06-tiling](references/06-tiling.md)、[05-json-types-flow](references/05-json-types-flow.md)。
5. **编写或更新 CANN 示例与单测**：见 [07-aclnn-template](references/07-aclnn-template.md)。
6. **若有 Python 前端**：按现有 test 模板补用例并做数值/形状校验。

**新算子从零开始时**：先执行 `bash build.sh --genop=op_class/op_name`，再按 [08-genop](references/08-genop.md) 与上述步骤定制。

---

## Step 1：复用现有模式

- **参考示例以官方文档为准**：类型/格式见 [01-type-format](references/01-type-format.md)（含 CANN/graph/types.h）；Kernel 与 Matmul 见 [02-kernel-guide](references/02-kernel-guide.md)（含 Ascend C API、Gitee samples）。工程内可对照同类型算子（FFN/GMM/MoE 的 op_host、tiling、op_kernel、examples）做实现参考。
- **行为准则**：以官方规范与 references 为准，先完整复制同类型算子骨架再做最小必要修改；保持命名、宏（如 ASCEND_IS_AIC）、队列与 UB 管理、AICore 与芯片配置一致。

---

## Step 2：op_host 定义

- **模式**：继承 `OpDef`，在 `namespace ops` 内定义；Input 用 `Input("name")` + `.ParamType(REQUIRED/OPTIONAL)` + `.DataType({...})`、`.Format({...})`、`.UnknownShapeFormat({...})`（个数一致）；Output 同理；属性用 `.Attr("name").AttrType(...).Int/Float/ListInt(...)`；AICore 用 `OpAICoreConfig` 并 `AddConfig("ascend910b", config)`；最后 `OP_ADD(YourOpClassName)`。
- **示例**：[03-op-host-examples](references/03-op-host-examples.md)。新算子从参考算子完整复制类与构造函数，只改类名、输入输出名与个数、DataType/Format、属性与默认值；需 aclnn 时沿用 `"aclnnSupport.value", "support_aclnn"`。

---

## Step 3：op_kernel 实现

- **共性**：命名空间与算子一致；包含 `kernel_operator.h`，矩阵类用 `lib/matmul_intf.h`；类型别名与 `MatmulImpl` 按参考算子；用模板区分 dtype/量化/激活等。
- **骨架**：[04-op-kernel-skeletons](references/04-op-kernel-skeletons.md)。确认是否基于 `MatmulImpl` 及 tiling 字段；只增删 GM 输入、调整 ComputeDequantAndActivate 等业务逻辑；保持队列/UB、PipeBarrier、DataCopyPad、SetAtomicAdd 等模式不变。

### Matmul / Cube 子流程

- **何时**：新增或修改基于 Matmul 的内核（如 GMM、MoE finalize routing）。
- **步骤概要**：（1）按 [02-kernel-guide](references/02-kernel-guide.md) 与官方 Matmul 高阶 API 定义 `MatmulType`/`MatmulImpl`，按 A:GM+ND、B:GM+NZ、C:GM+ND 复用或微调；（2）Init 中把 Host 传入的 GM 绑定为 `GlobalTensor`，保存 tiling 的 baseM/baseN/baseK、stepKa/stepKb、coreNum/parallNum 等；（3）Process 中按 tiling 划分 M×N×K block，算 A/B/C 的 GM 与 workspace 偏移；（4）每 block 按 [02-kernel-guide](references/02-kernel-guide.md) 调用 `SetOrgShape`/`SetSingleShape`/`SetTensorA`/`SetTensorB`/`Iterate`+`GetTensorC`，仅 AIC 执行；需 AIV 协作或确定性时参考官方多核/同步文档与 Gitee 样例，不自行发明同步方案。详见 [02-kernel-guide](references/02-kernel-guide.md)。

---

## Step 4：Tiling / Infershape 与 JSON 对齐

- 在 `op_host/` 下找 `*_tiling*.h/.cpp`、`*_infershape.cpp`，按官方 Tiling 文档与同类型算子分析 tiling 参数与 shape→tiling 映射。
- **JSON + types.h**：[05-json-types-flow](references/05-json-types-flow.md)（JSON 接口 → op_host 映射 → infershape 对齐 → tiling 校验 → op_kernel 命名一致）。
- **Tiling 两种方式**：[06-tiling](references/06-tiling.md)。标准 C++：结构体在 **op_kernel** 目录，Host 用 `GetTilingData<YourTilingData>()` 后直接对成员赋值，无需 SaveToBuffer/SetDataSize；Kernel 用 `REGISTER_TILING_DEFAULT(YourTilingData)`。宏定义：op_host 用 BEGIN/TILING_DATA_FIELD_DEF/REGISTER_TILING_DATA_CLASS；Host 需先创建 tiling 实例，set 完后 **必须** `SaveToBuffer` + `SetDataSize` 写回 context。**参考示例**：见 [06-tiling](references/06-tiling.md) 内「官方文档参考」— 标准 C++ 见官方「使用标准C++语法定义Tiling结构体」与 Gitee MatmulCustomMultiCore；宏定义见官方「基本流程」Add 算子示例。

---

## Step 5：CANN aclnn 示例

- **流程**：Init(aclInit/SetDevice/CreateStream) → 为每个输入/输出 CreateAclTensor → aclnnXxxGetWorkspaceSize → 按需 aclrtMalloc workspace → aclnnXxx(...) → aclrtSynchronizeStream → 拷回并打印 → 销毁张量/释放内存/ResetDevice/aclFinalize。
- **模板**：[07-aclnn-template](references/07-aclnn-template.md)。新示例时复制模板，替换 include、dtype、张量构造与 aclnnXxx 调用，保持 CHECK_RET 与成对释放。

---

## Step 6：测试与验证

- 若有 Python 单测：以 CANN 单测规范为准，构造典型与边界 shape，用参考实现或简单算法算期望值，断言 shape/dtype 与数值误差在可接受范围；工程内若有同类型 `test_npu_*` 可作模板参考。

---

## 约束与示例

- **约束**：不凭空发明模式，先搜索并对齐相邻算子；改文件前先通读；涉及芯片/动态 shape/确定性时与现有算子一致；示例与测试宜小且可手算验证。
- **示例**：用户要求“新增类似 grouped_matmul_finalize_routing 的 GMM 路由算子”时，Agent 应以 **官方文档与 references 为参考示例**：按 [02-kernel-guide](references/02-kernel-guide.md)、[05-json-types-flow](references/05-json-types-flow.md)、[06-tiling](references/06-tiling.md) 实现 op_host、tiling、op_kernel 与 Graph→kernel 映射；按 [07-aclnn-template](references/07-aclnn-template.md) 写 aclnn 示例并补单测；工程内同类型算子（如 gmm 目录下同名或相近算子）可作实现参考。

---

## genop 与通用示例生成

- **genop**：在 ops-transformer 下执行 `bash build.sh --genop=op_class/op_name` 生成新算子目录与占位文件。详见 [08-genop](references/08-genop.md)。
- **aclnn 示例生成**：从 op_host/op_kernel 提取输入输出与属性，按 [07-aclnn-template](references/07-aclnn-template.md) 填充分支；缺失处用 FILL IN 注释并提示用户补全。
