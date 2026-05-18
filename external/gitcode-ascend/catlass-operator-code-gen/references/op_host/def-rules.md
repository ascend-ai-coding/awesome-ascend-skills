# OpDef（`*_def.cpp`）

具体注册范式以 catlass 工程示例与 CANN 规范为准；本节仅列 **Catlass 场景**下与设计文档对齐的要点。

## 逻辑来源

- **逻辑来源**：按设计文档的 I/O 信息表注册 Input/Output/DataType/Format。
- **芯片配置**：`this->AICore().AddConfig("ascend910b")`。

## 检查清单（def）

- [ ] `*_def.cpp` 中只包含算子原型定义（Input/Output/DataType/Format）和芯片配置（`AddConfig`）
- [ ] `*_def.cpp` 中**没有** `SetTiling` 或 `SetInferShape` 调用
- [ ] OpDef 的 Input/Output/DataType 与设计文档一致
- [ ] **DataType 与 Format 列表长度一致**：每个 `Input`/`Output` 上 `.DataType({...})`、`.Format({...})`、`.UnknownShapeFormat({...})` 的**元素个数必须相同**（例如仅支持一种 dtype 时写 `.Format({ge::FORMAT_ND})` 单项，勿写两个 `FORMAT_ND` 只配一个 dtype），否则 opbuild 报 `Element num of DataType and Format is not aligned` / `The dtype size of input[0] ... is 0`。

与 tiling、infershape 分文件并列及端到端顺序见 [SKILL.md](../SKILL.md)「Step 4」；目录树见 [code-structure.md](./code-structure.md)。tiling / infershape 细则见 [tiling-rules.md](./tiling-rules.md)、[infershape-rules.md](./infershape-rules.md)。
