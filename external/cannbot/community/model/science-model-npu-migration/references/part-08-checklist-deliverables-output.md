# 分册 8：Checklist、mig_docs 交付、对话输出格式

> 对应**执行步 6**（归档收口）；无主流程独立 §。完整迁移在 part-05 评测达标后，按 [workflow.md](workflow.md) **闭环检查**收口。

## Checklist（可直接复制粘贴给进度）

任务进度：

- [ ] **可行性预判（part-02）**：已完成迁移链路、IO 契约、风险分级、Phase 1～3 计划；结论与风险表已写入 `Mig_report` **§2.2**
- [ ] **门禁（part-03）**：迁移开始时已确保 **`mig_docs/`** 目录存在；已按 4.0.1 生成/读取 **`mig_docs/working/environment.md`**
- [ ] **门禁（part-03）**：已阅读 README/requirements 等并完成 4.0.3 适配判定（AUTO / MANUAL_STOP / UNKNOWN）；**未因 part-02 通过而跳过**
- [ ] 若为 **MANUAL_STOP**：已输出手动需求清单并已停止自动化 NPU 训练/推理步骤（待用户补齐后再刷新 `environment.md`）
- [ ] 明确目标硬件与 CANN/驱动/工具链版本
- [ ] 收集工程依赖与基线推理入口（requirements/setup/脚本）
- [ ] 记录基线：优先检索项目**完整训练日志**并摘录；否则精度指标 + 延迟/吞吐（默认 **GPU**，可复现输入或 §2.6 接口）
- [ ] 已在 `Mig_Readme.md` 填写数据集与测试用途对照（短测/冒烟/训练/推理等）及下载方式与过程；若无完整训练日志，已按推荐顺序在 **§2.6** 准备 **GPU** 训练与推理调用接口（NPU 数据可先就绪）；`Mig_report.md` / `Compare.md` 与之一致
- [ ] 确认框架/工程栈与输入/输出 shape（固定/动态策略）
- [ ] 填写环境信息快照表（见 [part-03-environment.md](part-03-environment.md) §4.2 表）
- [ ] 环境验证：`npu-smi`、框架 Ascend 插件可导入、运行时依赖
- [ ] 若处于沙箱：`npu-smi info` 与框架插件检查在沙箱内**检测不到或不稳定**时，已在**沙箱外**用同一组命令复检，并记录两侧结果与最终采信结论（见 part-03 §4.0.1）
- [ ] 环境快照：`mig_docs/working/environment.md`：已读 / 本次已更新或生成
- [ ] 本轮变更是否已同步更新 `Mig_report` / `Mig_Readme` / `Compare` / `environment.md`（在 `mig_docs/` 内）
- [ ] 若出现失败/回滚：`Mig_report.md` **§7** 是否已更新（含现象、复现、根因、修复、验证）；**§8** 是否注明运行日志路径
- [ ] 本机环境检测结论：可直接测试 / 仅输出环境目标清单落地方案与待补齐项
- [ ] 已按 [`environment-setup-objectives.md`](environment-setup-objectives.md) 对齐环境准备目标，并在 `environment.md` / 对话中留下对应证据或缺口说明
- [ ] FP16 路径确定（精度目标先行）
- [ ] **步 4 收口（part-04 §5.0，含 §5.6 smoke）**：`Mig_report` §4～§6、`Mig_Readme` §4～§5 已填；smoke 已在 NPU 跑通并勾选 §6
- [ ] 迁移后最小验证：NPU 前向 + 少量样本 + 记录启动命令与日志
- [ ] Golden 样本：基线与 NPU 数值一致性抽查
- [ ] （可选）代码级迁移适配推理 smoke：固定输入、输出 shape/无 NaN（由环境检测和用户意愿决定）
- [ ] （可选，若迁移后仍训练）代码级迁移适配训练 smoke：单 batch 前向+反向一步或极少 step
- [ ] （可选，若迁移后仍训练）**训练短测**（part-05 **§8.1.1**，**代码迁移后**）：**loss** 相对起点**稳定下降约 30%～50%** 即通过并**停止**，**未**在同配置下做多余重复短测
- [ ] （可选，若迁移后仍训练）正式训练阶段：loss/约定指标不明显时已**适当加大 iteration**；若 **>500** step（或 `Mig_Readme` §5.3 约定值）仍无改善，已按 **part-05 §8.2** 转查**代码/模型实现**并记录 `Mig_report` **§7**
- [ ] 运行验证：权重/checkpoint 能稳定加载、无异常日志
- [ ] （可选）精度对比：与基线差异在允许范围内
- [ ] （可选）性能评估：p50/p95（或约定指标）与吞吐统计方法一致；**推荐顺序**：先在 **NPU** 完成短测/约定性能并落数；baseline **优先项目训练日志**，否则在 `Mig_Readme.md` **§2.6** 给出 **GPU** 接口（用户自测并回填）；已在 `Compare.md` §2.0～2.1、§4 注明来源
- [ ] **性能评估与精度对比完成后**，已按 [workflow.md](workflow.md)「文档一致性校验矩阵」执行统一数据同步（含 `environment.md` 与 `Mig_report` §3 版本一致）
- [ ] 形成回归结论文档及迁移报告（含问题材料清单，便于复盘）
- [ ] 训练/推理测试完成后，已执行“归档总结收口”：统一汇总迁移操作、迁移环境、预判与验证结论、训练/推理效果
- [ ] **`Summary.md` 已定稿**（最终交付；含 **当前进程与适配状态（显式快照）**：成功时 baseline 来源〔项目日志 / GPU 实测〕、环境、数据集、指标摘要；失败时算子/依赖等勾选）
- [ ] 已执行文档结构整理：删除重复/无效文档、清理缓存与临时文件、统一目录入口与命名
- [ ] **`mig_docs/` 过程文档 + 最终交付**：`working/` 内四份过程文档是否已随进度填写；**`Summary.md`** 是否在步 6 定稿；**`mig_docs/working/environment.md`** 是否与 part-03 / objectives 一致
- [ ] **任务结束时** 上述交付物（含 **`environment.md`**）是否已与当前状态一致（见下方「规范输出」）

## mig_docs 规范输出（交付物）

- **目录**：迁移**开始**即应存在 **`mig_docs/`**；若为空，从 **`science-model-npu-migration`** skill 仓库按 [deliverables-index.md](deliverables-index.md) 复制 `mig_docs/` 模板（含 **`.gitignore`**、**`Summary.md`** 与 **`working/`** 下四份过程模板）及本 index 文件，并随进度填写。工程级忽略（`.venv/`、`__pycache__/` 等）见 deliverables-index「版本控制与临时产物」。
- **维护**：迁移过程中可阶段性更新；在**性能评估与精度对比完成后**，必须按 [workflow.md](workflow.md)「文档一致性校验矩阵」执行统一数据同步（`Mig_report` / `Mig_Readme` / `Compare` / `Summary` / `environment.md` 关键数据一致）。
- **失败与回滚留痕**：若出现运行失败、算子不支持、精度/性能异常，写入 **`Mig_report.md` §7**，并与 `Compare.md` 等结论一致。
- **归档总结**：训练/推理测试完成后，定稿 **`mig_docs/Summary.md`（最终交付）**，覆盖：迁移操作、迁移环境、**预判与验证**（part-02 + part-03 + 迁移后验证）、训练/推理效果、最终结论与后续建议；**须含显式「当前进程与适配状态」快照**（成功：baseline 来源〔项目训练日志 / GPU 实测〕、测试环境、数据集与指标摘要；失败：算子/依赖缺失等勾选与说明）。**勿**在 `Mig_report.md` 重复维护与 Summary 同内容的归档章节。
- **收口清理**：任务收尾时应执行“结构清理”并记录结果：删除重复/过期文档，清理缓存与临时文件（如 `__pycache__/`、`*.pyc`、`*.tmp`、`*.log`），确认目录中仅保留有效交付物。

| 文件 | 必填内容要点 |
|------|----------------|
| **`mig_docs/working/Mig_report.md`** | 元信息、成功标准、**§2.2 可行性预判**、**数据集与测试用途摘要**、环境快照、**代码级迁移要点**、**启动命令**、**代码/配置/依赖变更清单**、算子与后处理变更、验证勾选、风险与回滚、日志路径 |
| **`mig_docs/working/Mig_Readme.md`** | 环境准备（§2.4 对齐 [`docs/environment-setup-objectives.md`](environment-setup-objectives.md)；§2.5 **`environment.md`**；依赖、CANN set_env、设备检查）、**§2.6 基线 GPU 训练与推理调用接口**（**无完整训练日志时**默认 GPU；NPU 短测/性能落数后补齐）、**数据集与测试用途对照**（§3.1 含**训练短测**说明：loss **30%～50%** 达标即停）、下载方式与过程、数据与预处理（与 IO 契约一致）、**NPU 推理命令与参数表**；若存在 NPU 训练则写训练入口、**§5.3**（正式训练迭代，与短测区分）、与迁移前差异，否则明确「仅推理」 |
| **`mig_docs/working/Compare.md`** | 基线 vs 昇腾环境表、**基线选取：项目训练日志优先 / 否则默认 GPU**（§2.1）、**数据集与测试用途**、**测量口径**、精度对比表、Golden 样本摘要、性能对比表（§4 注明 baseline 来源）、瓶颈与原始日志路径 |
| **`mig_docs/Summary.md`（最终交付）** | 迁移全程摘要、**显式快照**（baseline 来源：项目日志或 GPU 实测）、关键决策、测试结论（训练/推理）、达标性判断、遗留问题与下一步计划 |

**命名约定**：迁移报告文件名为 **`Mig_report.md`**（英文 report，避免拼写为 reprot）。

**对话中的输出规范**（除写文件外，回复正文应包含）：

1. 说明四份核心文档与 **`mig_docs/working/environment.md`** 是否已创建/更新及**相对仓库根路径**（例如 `mig_docs/working/Mig_report.md`）；若本轮有失败/回滚，说明 **`Mig_report` §7** 是否已更新。
2. 用简短摘要覆盖：`Mig_report` 中的主要变更行数或关键文件列表；`Compare` 中的达标结论一行话。
3. 若用户未指定路径，默认建议：项目根目录下的 `mig_docs/`。

## 输出格式（要求 agent 按以下结构输出）

最终回复建议包含：

1) 一段“迁移结论/下一步”（若 **MANUAL_STOP**，结论须为「暂停自动化，待手动项完成」）  
2) 迁移步骤清单（Checklist）  
3) **4.0.3 适配判定**（AUTO / MANUAL_STOP / UNKNOWN）一行结论；若为 MANUAL_STOP，附**手动需求清单**  
4) 环境快照路径与状态（**`mig_docs/working/environment.md`**：已读 / 本次已更新）与本机环境检测结果、执行策略（直接测试 / 仅输出环境目标落地方案与待补齐项）  
5) **`mig_docs/` 交付状态**：本轮创建/更新了哪些文件；四份核心文档与 **`environment.md`** 是否已从 skill 复制/生成并按阶段填写  
6) 可执行的命令（命令模板 + 已知参数填充 + 待确认参数列表；**MANUAL_STOP** 时不应包含误导性的 NPU 全链路执行命令）  
7) 环境信息快照表（[part-03-environment.md](part-03-environment.md) §4.2）与产物命名/路径说明  
8) 精度/性能验证计划（可选执行项需标注触发条件；含 Golden 样本与全量指标；**推荐测量顺序**见 [part-05](part-05-performance-accuracy.md) §7.0 与 `Compare.md` §2.0：先 **NPU** 落数；baseline **优先项目训练日志**，否则经 `Mig_Readme.md` §2.6 补 **GPU** baseline；须在 `Compare.md` 写明来源）  
9) **文档数据同步更新结果**：在性能/精度完成后，已按 [workflow.md](workflow.md)「文档一致性校验矩阵」对 `Mig_report` / `Mig_Readme` / `Compare` / `Summary` / `environment.md` 做统一回填与一致性校验
10) 风险点与回滚策略（详见 [part-06-risk-rollback.md](part-06-risk-rollback.md))
11) 归档总结：定稿 `mig_docs/Summary.md`（最终交付）；须含显式当前进程快照（成功：baseline 来源〔项目日志 / GPU 实测〕、环境、数据集与指标摘要；失败：失败类型勾选与阻塞说明）
12) 收口清理结果：本轮删除了哪些冗余文档/缓存文件；若无可删项需明确“已检查，无冗余”

**仅“检查当前 NPU 适配情况”时的简化输出**：

- 从 [part-03](part-03-environment.md) 直接执行并输出：`environment.md` 状态、沙箱内/外检测、4.0.3 判定（AUTO/MANUAL_STOP/UNKNOWN）、待补齐项。
- **不进入** part-04～05、part-08 完整归档；回复中明确“本次为适配状态检查路径，未执行完整迁移链路”。

---

## 关联索引

- **上一环节**：[part-05-performance-accuracy.md](part-05-performance-accuracy.md)（完整迁移主线）  
- **流程总览**：[workflow.md](workflow.md) 步 6、闭环检查
