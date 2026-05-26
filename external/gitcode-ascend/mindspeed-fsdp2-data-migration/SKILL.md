---
name: external-gitcode-ascend-mindspeed-fsdp2-data-migration
description: 用于将数据预处理与数据加载契约迁移到 MindSpeed-MM FSDP2。适用于实现数据集注册、预处理复用、collate 行为与输入字段兼容时。
original-name: mindspeed-fsdp2-data-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed FSDP2 数据迁移

## 目标

将源数据集逻辑迁移到 MindSpeed-MM FSDP2 数据插件结构中，保持预处理语义并确保 trainer/模型输入兼容。

## 适用场景

- 迁移任务需要在 `mindspeed_mm/fsdp/data/` 下创建数据插件。
- 需要复用源预处理函数或进行最小化适配。
- 需要实现 `dataset_type` 注册、`__getitem__` 与 `collate_fn`。

## 不适用场景

- 任务是模型加载或模型注册。
- 任务仅做配置映射且不涉及数据行为变更。
- 任务是最终验证门禁执行。

## 输入

- 源数据集入口文件
- 目标 `dataset_type`
- 预期模型输入字段
- 数据与路径格式假设
- `runtime_assets.dataset_path`

## 输出

- 数据插件实现
- 带通过/失败状态的数据迁移检查清单
- 对“非直接拷贝改动”的适配理由

## 必检项

1. 注册契约：
   - `@data_register.register("<dataset_type>")`
2. 数据契约：
   - `__getitem__` 返回模型前向所需字段
   - `collate_fn` 能安全处理变长 batch
3. 复用契约：
   - 兼容时应优先复用源预处理函数
4. 稳健性契约：
   - 安全处理路径类型变体（`str`、`list[str]`）
   - 对无效 schema 假设抛出清晰错误

## 防偏差执行协议

1. 实施前必须完成：
   - 校验 `runtime_assets.dataset_path` 存在且可读
   - 抽样读取前 N 条样本，确认 schema 字段存在
2. 实施后必须产出：
   - `dataset_schema_report.md`
   - `fields_contract_report.md`
3. 失败即停止：
   - `__getitem__` 字段不全或 collate 崩溃时，禁止进入配置/验证阶段

## 禁止行为

- 禁止把路径类型假设为仅 `str`
- 禁止静默吞掉 schema 错误
- 禁止删除模态关键字段（`pixel_values`、`image_flags` 等）而不声明
- 禁止以“样例数据通过”替代“真实路径抽样验证”

## 实施规则

- 先拷贝源仓核心预处理逻辑。
- 适配范围仅限导入、注册、字段映射和契约对齐。
- 不得静默丢弃模态关键字段。
- 明确记录每个函数不能直接复用的原因。

## 退出标准

- 数据集可通过 `build_mm_dataset` 构建。
- 数据字段与 collate 输出符合模型预期。
- 数据侧清单结论均有证据支撑。

## 渐进式资源

- `checks/data-acceptance.md`
- `checks/data-contract-gates.md`
- `errors/data-error-dictionary.md`
- `errors/triage-severity.md`
- `examples/minimal-io.md`

仅在验收或数据错误分诊需要详细判据时读取这些文件。
