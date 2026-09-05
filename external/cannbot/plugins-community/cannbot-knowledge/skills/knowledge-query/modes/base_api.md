# base_api — 基础 API 检索（从 reference/ 找需要的 asc-devkit API）

## 场景
"实现某功能要用哪个 Ascend C API？签名/约束/声明头在哪？" 从 `reference/asc-devkit/` 找 API 卡。

## 路线
- 召回：`bm25`（别名默认开，中英 API 名/术语自动扩展），scope=`asc-devkit/`，`--kind api`。
- 重排：`bm25f`（默认，确定性）。
- 多跳补全：对选中的 API `neighbors` 取**声明头文件 / 配套样例 / 相关接口**。

## 命令
```
python3 scripts/knowledge_query.py --knowledge-root <知识库根> recall \
  --method bm25 --query "矢量 按元素 加法 vector add" \
  --kind api --scope asc-devkit/ -k 8

# 选定后取整卡 + 多跳周边（样例/头文件/相关接口）
python3 scripts/knowledge_query.py --knowledge-root <知识库根> get asc-devkit/api/basic_api/add.md
python3 scripts/knowledge_query.py --knowledge-root <知识库根> neighbors asc-devkit/api/basic_api/add.md --hops 1 -k 8
```

## 注入
`get` 整卡（含 `# 函数原型` 文档+代码链接）→ `neighbors` 把 declares(头文件)/exemplifies(样例)/same_topic(AddRelu 等)一并带出，给一个连通的 API 知识面 → 引用 `resource` @sha。
