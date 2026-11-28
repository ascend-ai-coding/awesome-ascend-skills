---
name: external-cannbot-infra-knowledge-query
description: 检索 CANN 知识库中的 API、算子、模型、Runtime 和排障经验。当技术问答、设计、实现、调试或优化需要知识证据时使用；先检查并按需安装知识库，再按目标平台检索并阅读全文。
license: CANN-2.0
original-name: knowledge-query
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
---

# 知识检索

## 1. 检查并安装知识库

先定位本 Skill 的实际安装目录，将下文 `<skill-dir>` 替换为该绝对路径。首次进入任务时执行：

```bash
python3 <skill-dir>/scripts/knowledge_setup.py
```

安装器按 `--knowledge-root`、`CANNBOT_KNOWLEDGE_ROOT`、当前目录祖先的
`.cannbot/knowledge.env`、当前所在知识仓、`~/.cannbot/cannbot-knowledge` 顺序定位。
知识根是包含 `knowledge/` 和 `governance/` 的仓库根，不是 `knowledge/` 子目录。

目录不存在时自动从 `https://gitcode.com/cann/cannbot-knowledge.git` 克隆 master；
检查 Python 3.10+、运行依赖和索引，依赖缺失时在知识仓 `.venv/` 中安装，索引缺失或陈旧时构建并验证。
已有有效知识库直接复用，不自动拉取更新或覆盖本地修改。已有不完整目录、网络或安装失败时报错并停止，
不能将失败说成“知识库无结果”。自定义位置用 `--knowledge-root /absolute/path`，后续查询保持同一根。

安装只准备知识数据、运行依赖和检索索引，不安装知识仓中的其他 Skill，不改业务工程的 Agent 提示。
`knowledge_query.py` 每次调用也会执行此检查，避免其他 Skill 直接调用时漏掉准备步骤。
离线环境可复用准备完成的知识仓；仅下载知识仓不代表已经恢复卡片引用的全部源码或 SDK。

## 2. 确定问题与目标平台

- 明确要回答的判断和准确实体。目标平台来自用户任务或工程配置，不默认采用执行机器的平台。
- 平台名必须来自知识仓 Registry；型号映射不清楚时先检索平台知识，仍有歧义再确认。
  首次对同一知识根运行 `discover` 查看实际覆盖；各维度计数不代表它们的组合已有覆盖。
- API、类型或错误码用准确名称和 `--match literal`；能力需求、机制或症状用短词组和 `--match text`。
- 只加已确认的 domain、technology、type 等过滤。默认查 `stable`；项目可配置
  `CANNBOT_KNOWLEDGE_PLATFORM`，显式 `--platform` 优先。平台未知不能用 `agnostic` 代替。

```bash
python3 <skill-dir>/scripts/knowledge_query.py discover --json
python3 <skill-dir>/scripts/knowledge_query.py search \
  --query "DataCopyPad" --domain ops --technology ascendc --type apis \
  --platform a2 --status stable --match literal -k 5
```

示例的 `a2` 仅在任务目标确为 A2 时采用。组合过滤、平台映射和旧调用迁移见
[查询参数与证据](references/query.md)。API 也由本 Skill 直接检索，不转交专用 API Skill。

## 3. 选卡、读全文并形成结论

1. 从 `results` 的 `local_path`、`title`、`description` 选择相关卡片，读取完整 Frontmatter 和正文。
   搜索摘要只是候选，不能当作已经核验的证据。
2. 核对平台、生命周期、版本、前提和失效条件。`stable` 不保证技术结论正确，
   `verified` 记录也不能替代当前任务的适用性判断。
3. 按需要沿卡片链接补读；普通相对链接以卡片目录解析，以 `/` 开头的知识链接以 `knowledge/`
   为根。需要声明或实现时再读固定版本来源、头文件；来源未恢复就报告该部分缺口，不假称已读。
4. 零命中时保持已知平台和技术边界，用准确标识、错误信号或相关概念补搜。
   精确 API 在目标平台 stable 零命中时，可独立补查 `--include-unlocated --status draft`，
   不带平台，结果只作平台待核验线索。错误退出与有效零命中必须区分。
5. 已读证据足够时直接回答，引用实际读过的卡片或固定来源，区分文档事实、推断和实测。
   补搜没有新证据时说明具体缺口；零命中不证明能力不存在。

准备完成后的查询与读卡保持只读。不为提高召回修改知识卡、放宽已确定的平台条件或执行无关工程实验。

## 维护依据

本 Skill 根据 [cann/cannbot-knowledge](https://gitcode.com/cann/cannbot-knowledge) 的
`12693c19df3859c13adb24af59f1267564bfe365` 版本整理。保留通用查询流程，运行时使用完整知识仓中
同版本配套的 Query、索引和 Contract，避免复制独立脚本后规则与数据不一致。
