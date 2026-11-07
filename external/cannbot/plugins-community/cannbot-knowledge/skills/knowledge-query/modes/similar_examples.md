# similar_examples — 相似样例检索（从 ops/ 找类似算子设计）

## 场景
"我要做 X 算子，有没有结构类似的现成设计可参考？" 按**计算范式 + 类目 + 硬件**找 `ops/` 下相似算子卡。

## 路线
- 召回：`tagtype`，scope=`ops/`，`--kind operator`（或 `--type operator_spec`）+ `--paradigm <范式>`（可加 `--tags <类目>,<硬件>`）。
- 重排：`tagidf`（共享稀有 tag 越多越靠前）；想要"按算子名贴近"也可叠 `bm25f --query <算子名>`。
- status：默认 `verified`（排 stub 占位卡）；**若目的是"找尚未实现的待补算子"，用 `--status all` 或 `--status stub`**。

## 命令
```
# 1) 已知目标算子的范式/类目（看它自己的卡 frontmatter: paradigms / tags）
python3 scripts/knowledge_query.py --knowledge-root <知识库根> recall \
  --method tagtype --scope ops/ --kind operator \
  --paradigm Elementwise --tags activation --status all -k 8

# 2)（可选）按相似度重排：以某张已知卡为种子
python3 scripts/knowledge_query.py --knowledge-root <知识库根> recall --method tagtype --scope ops/ --kind operator --paradigm Elementwise --status all -k 20 \
  | python3 scripts/knowledge_query.py --knowledge-root <知识库根> rerank --method tagidf --seed ops/nn/activation/elu.md -k 8
```

## 注入
选 1–3 张最相似的 → `get` 整卡看其 TilingKey 分发 / UB 内存布局 / 多模板设计 → `neighbors <path> --hops 1` 取其样例/相关算子 → 引用各卡 `resource` @sha。
