---
name: external-gitcode-ascend-mindspeed-fsdp2-model-migration
description: 用于模型侧迁移到 MindSpeed-MM FSDP2 注册与加载契约。适用于实现模型插件、加载签名兼容、token/embedding
  更新与前向兼容时。
original-name: mindspeed-fsdp2-model-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed FSDP2 模型迁移

## 目标

在满足框架契约的前提下，将模型层迁移到 MindSpeed-MM FSDP2，并保持源模型语义。

## 适用场景

- 迁移任务需要在 `mindspeed_mm/fsdp/models/` 下创建模型插件。
- 需要新增或修复 `model_id` 注册及 ModelHub 兼容性。
- 需要适配模型加载签名（`from_pretrained`、`_from_config`）。

## 不适用场景

- 任务是数据预处理或 dataloader 行为调整。
- 任务仅涉及 YAML 映射。
- 任务仅为验证阶段问题分诊。

## 输入

- 源模型入口文件
- 目标 `model_id`
- 必需的特殊 token 行为
- 目标配置中的 FSDP2 约束
- `runtime_assets.model_path`

## 输出

- 模型插件实现
- 带通过/失败状态的模型迁移检查清单
- 每一处“非复用改动”的理由说明

## 必检项

1. 注册契约：
   - `@model_register.register("<model_id>")`
2. 构建契约：
   - `from_pretrained` 与 `ModelArguments` 调用链兼容
   - `_from_config` 路径有效
3. 语义契约：
   - 源模型需要特殊 token 时，必须保留其处理流程
   - 仅在必要时进行 embedding resize/init
4. 训练契约：
   - 前向路径可接收 trainer 输入并返回 `.loss`

## 防偏差执行协议

1. 实施前必须完成：
   - 校验 `runtime_assets.model_path` 存在且可读
   - 对照最相似案例抽取签名模式
2. 实施后必须产出：
   - `model_contract_checklist.md`
   - `signature_compat_report.md`
3. 失败即停止：
   - `from_pretrained` 或 `_from_config` 任一不兼容，禁止继续后续阶段

## 禁止行为

- 禁止把 `from_pretrained` 写成仅单参数签名
- 禁止仅靠“代码阅读”判定兼容，必须给调用证据
- 禁止在未确认 token-id 对齐时修改 token 常量
- 禁止删除源模型关键后处理逻辑而不给出等价替代

## 实施规则

- 优先复用源逻辑；仅在框架契约不兼容时做适配。
- 修改范围仅限模型插件区域。
- 不引入对源仓包路径的外部硬依赖。
- 记录每项适配的精确原因。

## 退出标准

- 模型可通过 ModelHub 按目标 `model_id` 构建。
- 加载签名冲突已解决。
- 模型侧清单结论均有证据支撑。

## 渐进式资源

- `checks/model-acceptance.md`
- `checks/model-contract-gates.md`
- `errors/model-error-dictionary.md`
- `errors/triage-severity.md`
- `examples/minimal-io.md`

仅在验收或错误分诊需要详细判据时读取这些文件。
