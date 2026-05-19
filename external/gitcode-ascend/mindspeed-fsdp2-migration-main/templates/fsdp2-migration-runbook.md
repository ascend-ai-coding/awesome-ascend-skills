# MindSpeed-MM FSDP2 迁移运行手册（Runbook）

## 1. 目的

为开发者与 AI Agent 提供一套统一操作流程，使用 5 个 FSDP2 Skill 完成完整迁移。

## 2. 适用范围

- 源仓：任意开源模型仓库
- 目标仓：MindSpeed-MM FSDP2 后端
- 验收标准：
  - 功能门禁通过
  - 可靠性门禁通过
  - 至少一次分布式端到端训练跑通

## 3. Skill 执行顺序

1. `mindspeed-fsdp2-migration-main`
2. `mindspeed-fsdp2-model-migration`
3. `mindspeed-fsdp2-data-migration`
4. `mindspeed-fsdp2-config-migration`
5. `mindspeed-fsdp2-verification`
6. `mindspeed-fsdp2-migration-main` 汇总收口

## 4. 执行步骤

### 步骤 A：发现与映射（Main）

- 识别源仓入口：训练入口、模型入口、数据入口。
- 识别目标仓落点：模型插件、数据插件、配置文件路径。
- 产出源到目标的兼容映射矩阵。
- 校验真实运行资产：`model_path`、`dataset_path`、可选 `image_root`。

### 步骤 B：模型迁移（Model Skill）

- 生成模型插件实现。
- 完成注册与加载签名兼容。
- 验证 forward 输出满足 `.loss` 契约。
- 重点检查 `from_pretrained` 签名与 `ModelHub` 调用链兼容。

### 步骤 C：数据迁移（Data Skill）

- 优先复用源仓预处理核心逻辑。
- 实现数据注册、`__getitem__` 与 `collate_fn` 契约。
- 验证输入字段与模型匹配。
- 验证路径类型兼容（`str` 与 `list[str]`）。

### 步骤 D：配置迁移（Config Skill）

- 将源仓参数映射到 YAML。
- 对齐 `model_id`、`dataset_type`、`training.plugin`。
- 强制执行 strict 与 extra 字段分层。
- 确保运行资产已映射到配置字段而非占位值。

### 步骤 E：验证（Verification Skill）

- 执行功能门禁。
- 执行可靠性门禁。
- 执行一次分布式端到端训练命令。
- 生成证据产物。
- 任一门禁失败时，输出失败用例并回流责任技能修复。

### 步骤 F：汇总收口（Main）

- 合并迁移结果。
- 输出“有证据支撑”的结论。
- 列出未解决风险与下一步动作。
- 禁止“无证据通过”或“部分通过上线”。

## 5. 标准输入模板

```yaml
source_repo_path: /abs/path
target_repo_path: /abs/path
model_identity:
  name: your_model
  modality: vlm|llm|audio|video
source_entrypoints:
  train: /abs/path/train.py
  model: /abs/path/modeling.py
  dataset: /abs/path/dataset.py
runtime_assets:
  model_path: /abs/path/model
  dataset_path: /abs/path/train.jsonl
  image_root: /abs/path/images
constraints:
  editable_paths:
    - examples/fsdp2
    - mindspeed_mm/fsdp/models
    - mindspeed_mm/fsdp/data
  no_core_modification: true
  reuse_first: true
acceptance:
  functional: true
  reliability: true
  distributed_e2e_run_once: true
```

## 6. 标准输出包

```yaml
changeset_manifest: [...]
compatibility_matrix: {...}
verification_evidence: {...}
risk_register: [...]
next_actions: [...]
```

## 7. 失败回流规则

- 模型失败 -> `mindspeed-fsdp2-model-migration`
- 数据失败 -> `mindspeed-fsdp2-data-migration`
- 配置失败 -> `mindspeed-fsdp2-config-migration`
- 验证门禁失败 -> 按根因归属回流到对应 Skill

## 8. 证据要求

- 每条验证必须记录：命令、退出码、时间戳、日志摘要。
- 没有证据，不得宣称完成。
- 报告结论与证据文件必须一一对应。

## 9. 完成判定

- 功能门禁通过。
- 可靠性门禁通过。
- 一次分布式端到端训练跑通。
- 最终汇总报告结论均有证据支撑。
