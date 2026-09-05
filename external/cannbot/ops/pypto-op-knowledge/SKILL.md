---
name: external-cannbot-ops-pypto-op-knowledge
description: PyPTO 算子开发知识库查询技能。串行查询经验表和问题查找表，有结果立即返回，无结果返回"无匹配"。触发关键词：查经验表、查问题表、知识库查询、knowledge
  query。
original-name: pypto-op-knowledge
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO 知识库查询

```
1. 查经验表 → 命中 → 返回 → 结束
            → 未命中 → 进入 2
2. 查问题查找表 → 命中 → 返回 → 结束
               → 未命中 → 返回"无匹配"→ 结束
```

## 步骤 1：查经验表

**文件**：[experience-table.md](references/experience-table.md)（索引）+ `references/experience_classified/` 下 8 个分类文件。

**搜索**：错误码（如 `F00001`、`FC4000`）、错误关键词、异常症状。

1. 在索引表中按错误码 + 二级关键词（`+` 后面的部分）匹配。
2. 命中则打开对应分类文件读取具体条目。
3. 提取**症状 → 根因 → 修复方案**，返回给调用方。

**多条目匹配策略**：同一错误码在索引表中可能出现多次（二级关键词不同）。精确匹配错误码 + 二级关键词的条目优先返回；无精确匹配时返回该错误码的所有候选条目，由调用方根据实际错误日志选择。

未命中：记录搜索关键词，进入步骤 2。

## 步骤 2：查问题查找表

**文件**：[problem-lookup.md](references/problem-lookup.md)（辅助表，路由到官方文档）。

**搜索**：错误码、错误关键词、崩溃症状。在以下四个索引中**并行查找**，任一命中即返回：

| 索引 | 按什么查 | 返回什么 |
|------|---------|---------|
| §〇 关键字索引 | 关键字 | 目标文档 + 章节 |
| §一 错误码路由 | 错误码范围 | 归属组件 + 官方文档 URL |
| §二 运行时症状 | 症状描述 | 官方文档章节（带 URL） |
| §三 修复速查表 | 错误信息 | 根因 + 一句话修复方案 |

未命中：记录搜索关键词，返回"无匹配"。
