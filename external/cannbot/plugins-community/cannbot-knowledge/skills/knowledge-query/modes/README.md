# 检索模式库（modes）

每个模式 = 一条**定制检索路线**（选哪些召回 + 哪条重排 + scope），写成一份 worked-recipe。
agent 读了照着调 `knowledge_query.py` 的工具即可；**也可以不用模式、自由组合**——skill 不锁工作流。

## 现有模式
- [`similar_examples.md`](similar_examples.md) — 相似样例检索：从 `ops/` 找与某算子类似的设计卡（tag/type/paradigm 匹配）。
- [`base_api.md`](base_api.md) — 基础 API 检索：从 `reference/` 找需要用的 asc-devkit API。
- [`optimization_practice.md`](optimization_practice.md) — 优化实践检索：从 `runbooks/` + `examples/` 找优化经验/踩坑。

## 新增一个模式（模板）
在本目录加一份 `<name>.md`，四段：

```
# <模式名> — 一句话用途

## 场景
什么问题用它（触发条件）。

## 路线
- 召回：method + scope + facet（可多路）
- 重排：method
- 默认 status / 别名 / k

## 命令
具体 `python3 scripts/knowledge_query.py --knowledge-root <知识库根> ...` 序列（用归一 scope）。

## 注入
选卡 → get 整卡（@sha 引用）→ neighbors 多跳补周边 → 合成回答。
```

把它登记到上面"现有模式"列表即可。
