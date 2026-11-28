---
name: external-cannbot-infra-knowledge-ingest
description: 将文档、固定版本源码说明、开发记录或人工材料整理为可复用知识，新增或增量更新 CANN 知识卡。当需要知识生成、经验沉淀或知识维护时使用；先准备知识库，再查重、提炼证据并校验入库结果。
license: CANN-2.0
original-name: knowledge-ingest
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
---

# 知识生成与入库

## 1. 检查并安装知识库

定位本 Skill 的实际安装目录后，先执行：

```bash
python3 <skill-dir>/scripts/setup_knowledge.py
```

该入口复用随依赖 `knowledge-query` 安装的准备脚本，两者位于同一 Skills 根目录。
通过市场安装 `knowledge-ingest` 会同时安装该依赖；手动复制时保留这两个目录。
脚本按显式 `--knowledge-root`、`CANNBOT_KNOWLEDGE_ROOT`、项目 `.cannbot/knowledge.env`、
当前所在知识仓、`~/.cannbot/cannbot-knowledge` 顺序定位，返回 `root` 和 `python`。
知识库不存在时自动克隆 `https://gitcode.com/cann/cannbot-knowledge.git`，安装缺失运行依赖并准备索引。
已有知识库复用且不自动更新；不完整目录和安装失败会停止，不覆盖已有材料。

将返回的完整仓库根用于后续所有读写。先读取该仓的 `AGENTS.md`、适用的
`governance/specs/` 和 `governance/schemas/profiles.yaml`，遵循当前目录与字段规范。
安装仅准备知识库及查询依赖，不安装或路由到专用知识生产 Skill。

## 2. 查重并确定修改范围

用 `knowledge-query` 按实体、根因或关键结论检索并完整阅读相邻卡片。根据实际内容决定新增、更新、
合并、拆分、废弃或暂缓：同一结论有新证据时优先增量补充，没有新增信息就跳过。
读取待改卡片全文后再编辑，保留原有有效证据、引用和适用范围，不整体替换已有人工整理。

## 3. 通用生成原则

- **证据可追溯**：完整阅读实际采用的材料，记录来源 URL、固定 commit、文档版本或原始记录位置。
  区分来源事实、推导结论和待验证假设；不把工具执行成功写成技术验证完成。
- **提炼可复用知识**：围绕一个明确主题整理问题、适用条件、机制或根因、处理方法、验证结果及限制。
  开发记录保留关键失败与成功证据，不把整份日志或上游文件直接复制成卡片。
- **范围有依据**：平台、版本、标签及可选 Scope 只填写来源支持且有用的内容。
  未知就是未知，不用默认值或 `any` 凑齐字段，不把未提及解释为“不支持”。
- **状态真实**：证据不充分时保持 `draft`；满足当前仓库要求且完成实际复核后才升级状态。
  旧结论失效时保留历史和替代关系，使用当前仓库允许的生命周期值。
- **写入范围明确**：只修改本次任务涉及的卡片与必要导航。新建不覆盖同名文件，更新采用增量融合；
  无法可靠提炼时保留待处理材料并说明缺口，不制造空卡或推断事实。

## 4. 检查与交付

检查卡片内容、Frontmatter、相对链接、受影响的逐层导航和变更日志。提交上库前按知识仓当前质量入口
验证格式与一致性，重建并验证检索索引，再按目标平台和技术执行查询确认可召回。
不因普通入库强制构建图谱、恢复全部源码或运行工程实验。

报告新增、更新、跳过及暂缓的卡片、来源版本、验证结果与待确认项。是否提交、推送或创建 PR
以用户本次要求为准；生成知识本身不代表自动发布。

## 维护依据

通用原则整理自 [cann/cannbot-knowledge](https://gitcode.com/cann/cannbot-knowledge)
`12693c19df3859c13adb24af59f1267564bfe365` 的 knowledge-ingest。这里不携带专用生产器、来源类型路由
或固定卡片模板，具体字段以已准备知识仓的规范为准。
