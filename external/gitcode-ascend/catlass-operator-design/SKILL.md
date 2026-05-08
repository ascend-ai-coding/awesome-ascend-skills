---
name: external-gitcode-ascend-catlass-operator-design
description: 将用户基于CATLASS开发算子的需求转变为具体的设计文档
original-name: catlass-operator-design
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# CATLASS 算子设计

## 核心工作流

```
需求分析 → 阅读 catlass 仓库 → 组件选型 → 输出设计文档
```

---

## 前置条件

| 检查项 | 说明 |
|--------|------|
| catlass 仓库 | 须在 **OPS_PROJECT_ROOT** 下可访问 `catlass/`（含 `include/`、`examples/`） |
| 必读参考 | 设计前**必须**阅读本 skill 的 `references/` 文件 |

### 必读参考

| 文件 | 用途 |
|------|------|
| [design-document.md](references/design-document.md) | 设计文档模板 |
| [matmul-templates.md](references/matmul-templates.md) | 模板清单与选型指南 |
| [epilogue-components.md](references/epilogue-components.md) | Epilogue 组件分类与组合模式 |
| [custom-epilogue.md](references/custom-epilogue.md) | 自定义 Epilogue 契约 |

---

## 算子命名（强制）

- **`op_name`（snake_case）必须包含子串 `catlass`**，例如：`catlass_matmul_add`、`catlass_fused_gemm_bias`
- **禁止**单独使用易与框架或其它厂商算子冲突的短名（如 `matmul_add`、`gemm_add`）
- **OpDef / 类名（CamelCase）** 与目录名一致映射，如 `CatlassMatmulAdd`、`ops/catlass_matmul_add/`

---

## 选型与适配原则

- 从 catlass **example** 和 **include** 中确定要用的模板，**不是**把 example 整份粘贴到算子工程
- 大部分算子工程基于 msopgen 小工程，example 中的 Host 调用代码需要改写为工程中的 op_host/op_kernel 分离结构
- 设计阶段只产出**选型决策**，代码由 code-gen 生成

---

## NEVER / ALWAYS

**NEVER**：使用不含 `catlass` 的算子名定稿设计；把 example 整份照抄；设计文档中写大量代码块；在信息不足时臆测需求

**ALWAYS**：算子名含 `catlass`，目录名与类名一致映射；条件不明则追问；设计前存在 catlass 仓库并阅读其文档与代码；设计文档用**选型表格**描述组件

---

## 需求输入

用户可能**直接说出需求**，也可能**不说**。执行本 skill 时：**若认为信息不足，则继续追问**。

需确认的信息：`op_name`（须含 catlass）、功能要点、I/O 与 dtype、布局、目标 SoC、参考 example（如有）、约束条件。

---

## 需求分析

- 确定 **`op_name`：须含 `catlass`**
- 提取：算子功能、数学公式、I/O、dtype、布局、目标 SoC、约束

---

## 阅读 catlass 并选型

- **阅读** catlass 仓库：`examples/`、`include/` 下的文档与代码
- 在 `examples/` 中按功能找**相似示例**，在设计文档中写明：选中的 example 路径、选型理由
- 若无完全匹配：基于 [matmul-templates.md](references/matmul-templates.md) 与仓库中现有样例，说明**变通方案**
- **Epilogue**：在 [epilogue-components.md](references/epilogue-components.md) 与 `catlass/include/catlass/epilogue/tile/` 中检索；**无**现成 Tile 时按 [custom-epilogue.md](references/custom-epilogue.md) 先写设计契约

---

## 组件选型

选型结果用**概念表格**记录在设计文档中，不写代码。

设计文档须**显式写出**选型表格：ArchTag、BlockMmad（L1/L0 TileShape、数据类型）、BlockEpilogue（各环节 Tile/Block 组件）、BlockScheduler、Kernel 类型。

---

## 输出设计文档

### 文档章节要求

1. **概述**：功能、场景、数学公式
2. **输入输出信息表**：变量名、数据类型、Shape、布局、描述
3. **核心组件选型**：用选型表格描述各层组件
4. **参考 Example 与模板选型**：选中的 example、选型理由
5. **TilingKey 分支设计**：各 key 对应的条件和 Kernel 分支内容
6. **Workspace**：大小计算逻辑来源
7. **接口与使用**：aclnn 接口、Host Tiling 流程、Kernel 入口概念
8. **扩展性**：可替换的组件
9. **实现方案纲要**：Host 侧与 Kernel 侧要点

### 输出格式

设计文档须为独立 Markdown 文件，命名建议 `design_<op_name>.md`，存放于 **USER_OP_PROJECT** 或用户指定路径。

---

## 质量验证

### 设计文档检查清单

- [ ] 算子名含 `catlass`，snake_case 目录名与 CamelCase 类名一致
- [ ] 含完整的输入输出信息表
- [ ] 核心组件选型已明确：ArchTag、BlockMmad、BlockEpilogue、BlockScheduler、Kernel
- [ ] 参考 example 已写明，选型理由清晰
- [ ] TilingKey 分支设计已列出
- [ ] 交付路径 USER_OP_PROJECT 已明确

---

## 参考资料

| 文件 | 用途 |
|------|------|
| [design-document.md](references/design-document.md) | 设计文档模板与章节规范 |
| [matmul-templates.md](references/matmul-templates.md) | 模板清单与选型指南 |
| [epilogue-components.md](references/epilogue-components.md) | Epilogue 组件分类与选型 |
| [custom-epilogue.md](references/custom-epilogue.md) | 自定义 Tile Epilogue 的设计契约 |
| catlass/examples/ | 开发形式与多种模板参考 |
| catlass/docs/3_API/gemm_api.md | Gemm API 分层模型、组件对照表 |
