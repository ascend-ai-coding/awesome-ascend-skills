# cannbot-knowledge 集成

本参考定义何时以及如何把 `~/cannbot-knowledge` 作为 Tensor API 开发的补充知识源。
知识库保存技术事实；本 Skill 保存执行流程、设计合同和验收门禁，不复制第二份技术正文。

## 1. 适用场景

| 问题 | 首选入口 |
|---|---|
| Ascend C / Tensor API 准确语义、重载、约束 | `ascendc-api-knowledge-query` |
| A5/950 平台容量、微架构、SKU 差异 | `knowledge-query`，domain=common |
| 优化模式、历史 Runbook、实测 Probe | `knowledge-query`，domain=ops，technology=ascendc |
| 固定版本源码级 Operator 卡 | knowledge 的 VV/CV Operator producer 产物 |
| 本 Skill 的流程、DESIGN/PLAN、验收门禁 | 本 Skill references，不查知识库 |

知识库不是事实源替代者。公开 API/ABI/Layout 仍先查 asc-devkit 手册与头文件；知识卡用于
补充平台边界、实测结论、优化经验和排障证据。
精确 API 名查询固定使用 `--match literal`；能力描述或现象查询使用短词组和 `--match text`。

## 2. 查询流程

1. 先确定目标平台。本 Skill 默认 A5/950，查询使用 `--platform 950`；不得把开发机平台
   当作任务平台。
2. 索引缺失、损坏或 stale 时停止，不要扫描全库兜底。知识仓维护者可运行
   `knowledge_index.py build` 重建。
3. 精确 API 名用 `--match literal`；能力描述或现象用短词组和 `--match text`。
4. 脚本结果只是候选摘要；选中后必须读完整卡片，不能只引用 title/description。
5. 默认查 stable；只有明确需要历史线索时才查 draft，并单独标注“平台待核验”。
6. 零命中不证明能力不存在；先补搜准确标识、参数或错误信号，再报告缺口。

常用命令示例（在 skill 根目录执行时，路径按实际安装位置替换）：

    python3 /home/hyolee/cannbot-knowledge/.agents/skills/knowledge-query/scripts/knowledge_query.py \
      --knowledge-root /home/hyolee/cannbot-knowledge search \
      --query "reg_tensor" --domain ops --technology ascendc --type apis \
      --platform 950 --status stable --match literal -k 5 --text

## 3. 证据采用规则

1. 引用知识卡时记录：卡片路径、卡片 status、来源 role、来源 commit/URL、平台和读取时间。
2. `status: stable` 只表示生命周期，不表示技术正确；`verified` 才是真实复核事件。
3. 正文中的 `[MEASURED]` 是设备实测，`[SPEC]` 是厂商主张；两者不能混写。
4. 卡片结论必须与当前目标平台、CANN 版本和任务合同核对；跨平台结论不得直接采用。
5. 与 asc-devkit 手册/头文件冲突时，以当前 asc-devkit 头文件为准，并把冲突写入 DESIGN §11。

## 4. 高价值入口

| 主题 | 卡片 |
|---|---|
| Reg Vector 轻量封装 | `knowledge/ops/ascendc/apis/simd_api/tensor_api/reg_vector_compute/reg_tensor/reg_tensor.md` |
| A5 每核存储与 UB 双口径 | `knowledge/common/platforms/concepts/ascend950_per_core_memory.md` |
| A5 微架构 | `knowledge/common/platforms/concepts/ascend950_microarchitecture.md` |
| A5 Roofline | `knowledge/ops/ascendc/concepts/roofline_model.md` |
| Manual Mmad 尾块/转置陷阱 | `knowledge/ops/ascendc/optimizations/ol_224_manual_mmad_real_k_and_transpose_gating.md` |
| MTE2 并行实测 | `knowledge/ops/ascendc/concepts/mte2_parallel_channels.md` |
| UB bank conflict 实测 | `knowledge/ops/ascendc/concepts/ub_bank_conflict_behavior.md` |

这些入口只作为查询起点，不是完整 API 或优化清单。

## 5. 权限边界

查询保持只读。不要修改知识卡、索引、图谱、SDK 或全局环境；不要为召回放宽平台条件；
不要把检索摘要写成已读证据。若需要沉淀新经验，走知识库贡献流程，而不是回写本 Skill。
