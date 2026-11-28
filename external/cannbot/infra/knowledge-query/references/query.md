# 查询参数与证据

执行命令前按 SKILL.md 完成知识库检查；下文 `<skill-dir>` 为本 Skill 的实际安装路径。

```bash
python3 <skill-dir>/scripts/knowledge_query.py --knowledge-root <repo> search \
  --query "<实体或短语>" --platform <目标平台> \
  --domain ops --technology ascendc --type apis --match literal -k 5
```

- `--knowledge-root` 可以放在子命令前后，指向完整仓库。
- `--type` 是路径 Profile（例如 `apis`、`guides`、`runbooks`），以当前知识仓 Registry/Profile 为准。
- `literal` 匹配完整查询串，不是多个词的 AND。文本查询按相关性排序，不保证每个条件都满足。
- 多个 `--platform` 和 `--tag` 是 AND；跨平台比较分开查询。
- `--status` 支持 `stable`、`draft`、`deprecated`、`all`，默认 `stable`。
- `--scope-file` / `--scope-json` 仅用于已确认且 Registry 支持的场景条件，不自动补齐未知字段。
- `--text` 输出便于阅读的候选；默认 JSON 的 `results` 含 `local_path` 等信息。
- `count` 是受 `-k` 限制的返回条数，不是总命中数。已达上限且未找到目标时，细化查询或提高 `-k`。

平台尚需映射时，可用准确产品名检索平台知识：

```bash
python3 <skill-dir>/scripts/knowledge_query.py search \
  --query "Ascend 950" --domain common --technology platforms --match text -k 5
```

读取命中卡及 `<repo>/governance/schemas/registries.yaml` 确认平台 ID。普通技术查询仍要求
明确平台或项目缺省值，不把平台识别查询的例外扩展到所有领域。

旧消费方的 `preflight`、`get`、`neighbors`、`pipeline`、`search --scope runbooks/` 和
`knowledge_query.py build` 已不适用。现在使用 `discover`、`search --type runbooks` 并按
`local_path` 读卡；`discover.root` 是仓库根，业务文档中的 `$KB_ROOT` 若指知识 Bundle，取
`<root>/knowledge`。索引维护通过安装器或知识仓自身的 `knowledge_index.py` 完成。

退出码 `0` 包含有效零命中；`2` 表示参数、知识仓或索引错误，必须检查错误信息。
需要更多参数时使用 `knowledge_query.py search --help`；完整来源、平台和版本限制仍需阅读全文。

卡片的 `sources[].resource` 本地路径以知识仓根解析；头文件路径以对应 CANN 组件根解析。
安装器不批量下载源码、SDK 和文档归档。正文足够回答时无需恢复这些资源；确需补证时按
知识仓登记来源准备所需资源，并明确尚未读取的部分。
