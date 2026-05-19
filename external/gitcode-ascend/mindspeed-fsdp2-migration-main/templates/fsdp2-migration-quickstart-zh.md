# MindSpeed-MM FSDP2 迁移快速上手（Quickstart）

## 1. 目标

用最少步骤完成一次“任意开源模型 -> MindSpeed-MM -> FSDP2”迁移闭环，并满足：

- 功能门禁通过
- 可靠性门禁通过
- 一次分布式端到端训练跑通

## 2. 前置准备

- 已确认可编辑目录：
  - `examples/fsdp2/`
  - `mindspeed_mm/fsdp/models/`
  - `mindspeed_mm/fsdp/data/`
- 源仓入口已定位：
  - 训练入口
  - 模型入口
  - 数据入口
- 真实运行资产已准备：
  - `model_path` 可读
  - `dataset_path` 可读
  - `image_root`（如需要）可读

## 3. 六步执行

### 第零步：K0 知识准备（强制）

- 阅读：
  - `docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2.md`
  - `docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2_DEVELOPER.md`
- 扫描：
  - `mindspeed_mm/fsdp/models/`
  - `mindspeed_mm/fsdp/data/datasets/`
  - `examples/fsdp2/`
- 交付：
  - `architecture_chain.md`
  - `contract_matrix.yaml`
  - `similar_case_selection.md`
  - `preflight_risk_register.yaml`
- 未完成 K0 禁止进入后续步骤

### 第一步：总控建图（Main）

- 启动 `mindspeed-fsdp2-migration-main`
- 输出：
  - 源-目标映射矩阵
  - 子任务分派（Model/Data/Config）

### 第二步：模型迁移（Model）

- 启动 `mindspeed-fsdp2-model-migration`
- 完成：
  - `model_id` 注册
  - 加载签名兼容
  - forward `.loss` 契约

### 第三步：数据迁移（Data）

- 启动 `mindspeed-fsdp2-data-migration`
- 完成：
  - `dataset_type` 注册
  - `__getitem__` 字段契约
  - `collate_fn` 契约

### 第四步：配置迁移（Config）

- 启动 `mindspeed-fsdp2-config-migration`
- 完成：
  - `model_id/dataset_type/plugin` 三元一致
  - strict 与 extra 字段分层
  - FSDP2 配置有效

### 第五步：验证收口（Verification + Main）

- 启动 `mindspeed-fsdp2-verification`
- 完成：
  - 功能门禁
  - 可靠性门禁
  - 一次分布式端到端训练
- 若失败：必须输出失败用例并路由到责任技能
- 返回 Main 汇总报告

## 4. 最小输入模板

```yaml
source_repo_path: /abs/path/source
target_repo_path: /abs/path/MindSpeed-MM
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
```

## 5. 通过标准

- 至少一次分布式端到端训练成功
- 报告中的每条结论有证据可追溯
- 失败项有明确回流 Skill 与修复建议

