# 端到端迁移时序（单次运行）

## 步骤 0：发现阶段

- 识别 `source_repo_path`、`target_repo_path`、`source_entrypoints`、`model_identity`。

## 步骤 1：映射阶段

- 构建源仓与 MindSpeed-MM 模块映射：
  - 模型插件路径
  - 数据插件路径
  - YAML 配置字段

## 步骤 2：模型实施

- 调用 `mindspeed-fsdp2-model-migration`，输入来自步骤 1。
- 产出模型插件与验收清单。

## 步骤 3：数据实施

- 调用 `mindspeed-fsdp2-data-migration`，输入来自步骤 1。
- 产出数据插件与验收清单。

## 步骤 4：配置实施

- 调用 `mindspeed-fsdp2-config-migration` 生成或修复 YAML。
- 校验注册关联与 strict/extra 分层。

## 步骤 5：验证

- 调用 `mindspeed-fsdp2-verification`：
  - 功能门禁
  - 可靠性门禁
  - 一次分布式端到端运行（预期成功）

## 步骤 6：汇总收口

- 使用 `templates/migration-report-template.md` 产出最终迁移报告。
- 附上 evidence JSON 与风险清单。
