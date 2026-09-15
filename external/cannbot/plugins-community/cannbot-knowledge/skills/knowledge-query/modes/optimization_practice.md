# optimization_practice — 优化实践检索（从 runbooks/ + examples/ 找优化经验/踩坑）

## 场景
"这类算子有哪些优化点 / 别人踩过哪些坑？" 从 `runbooks/`（优化点库 + 实战 field-notes）与
`reference/asc-devkit/examples/` 找经验，**按证据质量排序**。

## 路线
- 多路召回（pipeline）：
  - `tagtype`，scope=`runbooks/`，`--tags <问题域,如 multi-core,tiling,precision>`（可 `--severity high`）；
  - `bm25`，scope=`asc-devkit/examples/`，`--query <算子/优化点关键词>`。
- 重排：`quality`（runbooks 的 `quality_score` desc → `severity` → 命中分；examples 无质量分则靠后）。

## 命令
```
python3 scripts/knowledge_query.py --knowledge-root <知识库根> pipeline \
  --recall tagtype,bm25 --rerank quality \
  --query "多核 tiling 广播" --tags multi-core --tags tiling \
  --scope runbooks/ -k 10
# 只看高危踩坑：
python3 scripts/knowledge_query.py --knowledge-root <知识库根> recall \
  --method tagtype --scope runbooks/ --tags precision --severity high -k 10
```
> 注：pipeline 的 `--scope` 作用于全部召回路；若要两路不同 scope，分别 `recall` 再 `rerank` 合并。

## 注入
按 `quality_score`/`severity` 选高置信条目 → `get` 整卡看根因/规避法 → 关联到对应 API（`neighbors`）→ 引用卡 `resource`/来源；**field-notes 无上游 @sha 属正常**（实战记录）。
