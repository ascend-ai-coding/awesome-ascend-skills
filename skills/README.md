# Skills

本目录是 Awesome Ascend Skills 中**所有本地 skills 的统一入口**。

规则：

- 本地 skills 的唯一正式路径是 `skills/<domain>/...`
- root 不再承载本地 `SKILL.md`
- `external/` 单独保留为外部同步目录

## 功能域导航

- [`base/`](base/)：基础环境、设备、容器、PyTorch NPU 基础能力
- [`inference/`](inference/)：推理、模型转换、量化、评测
- [`training/`](training/)：训练链路、通信、MindSpeed-LLM
- [`profiling/`](profiling/)：Profiling 采集与性能分析
- [`ops/`](ops/)：算子开发、迁移与调优
- [`agent-tools/`](agent-tools/)：工程案例、issue 分析、社区反馈分析与开源合入流程
- [`ai-for-science/`](ai-for-science/)：AI for Science 专项域

## 维护约定

- 新增本地 skill 时，先判断功能域，再放入对应 `skills/<domain>/...`
- 若需要 bundle 暴露或 README 导航入口，同时更新 `.claude-plugin/marketplace.json` 与 root `README.md`
- 如果目录结构发生迁移，README、validator、CI、marketplace 与交叉链接必须在同一轮更新中完成
