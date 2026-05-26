---
name: external-gitcode-ascend-mindspeed-fsdp2-config-migration
description: 用于将源训练设置映射到 MindSpeed-MM FSDP2 YAML 契约。适用于创建或修复 model_id/dataset_type/plugin
  对齐、strict/extra 分层与分片配置时。
original-name: mindspeed-fsdp2-config-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed FSDP2 配置迁移

## 目标

将源训练/配置语义转换为合法的 MindSpeed-MM FSDP2 YAML，并保证注册关联正确、字段分层安全。

## 适用场景

- 迁移任务需要新建或修复 `examples/fsdp2/*_config.yaml`。
- `model_id`、`dataset_type`、`training.plugin` 映射缺失或不一致。
- 数据参数分层错误导致 dataclass strict-field 报错。

## 不适用场景

- 任务是模型代码适配。
- 任务是数据预处理实现。
- 任务是最终测试执行与报告产出。

## 输入

- 源训练参数或脚本
- 目标模型/数据注册 id
- 所需并行/FSDP2 模块匹配模式
- `runtime_assets`（至少含 `model_path`、`dataset_path`）

## 输出

- YAML 配置草案/更新
- 映射表（源参数 -> 目标 YAML 字段）
- 带通过/失败状态的配置迁移检查清单

## 必检项

1. 注册关联：
   - `model.model_id` 与模型注册一致
   - `data.dataset_param.dataset_type` 与数据注册一致
   - `training.plugin` 能导入模型与数据插件包
2. 字段分层：
   - strict dataclass 字段保留在 `basic_parameters`
   - 自定义扩展字段放入 `*_extra`
3. 并行配置健全性：
   - `parallel.fsdp_plan.apply_modules` 指向有效模块
   - dtype/recompute 选项语法合法
4. 配置必填字段完整性：
   - `model.model_id`、`model.model_name_or_path`
   - `data.dataset_param.dataset_type`
   - `data.dataset_param.basic_parameters.dataset`
   - `data.dataset_param.preprocess_parameters.model_name_or_path`
   - `data.dataloader_param` 下 `dataloader_mode`、`sampler_type`、`shuffle`、`drop_last`、`pin_memory`、`collate_param`
   - `data.dataloader_param.collate_param.model_name`
   - `training.lr`、`training.micro_batch_size`、`training.train_iters`

## 防偏差执行协议

1. 实施前必须完成：
   - `model_id`、`dataset_type`、`training.plugin` 三元目标值已冻结
   - 真实运行资产（model/dataset）已映射到配置字段
2. 实施后必须产出：
   - `config_mapping_table.md`
   - `linkage_check_report.md`
3. 失败即停止：
   - 任一注册关联失败时，禁止进入验证阶段

## 禁止行为

- 禁止把扩展字段写入 strict dataclass 区块
- 禁止 `training.plugin` 只写模型插件不写数据插件
- 禁止 `apply_modules` 使用与模型结构无关的硬编码模式
- 禁止直接复用他模型的 `apply_modules` 列表而不做匹配验证
- 禁止遗漏必填字段并依赖隐式默认值
- 禁止在未跑解析检查时宣称配置可用

## 实施规则

- 先保持源配置语义，再适配配置形态。
- 不得将自定义字段放入 strict schema 区块。
- 配置改动必须显式且可追溯。
- 每个非平凡映射决策都要给出理由。

## 退出标准

- YAML 可解析且注册关联一致。
- 配置分层不引入 strict-field schema 错误。
- 配置侧清单结论均有证据支撑。

## 渐进式资源

- `checks/config-acceptance.md`
- `checks/config-contract-gates.md`
- `errors/config-error-dictionary.md`
- `errors/triage-severity.md`
- `examples/minimal-io.md`

仅在字段分层检查或配置错误分诊需要详细判据时读取这些文件。
