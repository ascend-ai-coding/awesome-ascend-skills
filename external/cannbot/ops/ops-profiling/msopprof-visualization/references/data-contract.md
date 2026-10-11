# 数据合同与真源规则

## 真源

- `collection_manifest.json`：采集块、命令、状态、alias 和 artifact 的唯一真源；
- `feature_catalog.json`：请求特性到采集能力的映射；
- `report_payload.json`：报告页面使用的 canonical payload；
- `report_index.json`：交付回执、请求页面、渲染页面和不可用模块；
- `_internal/run_state.json`：顶层复用门禁。

## 采集块状态

建议使用：

- `ok`：命令和语义校验通过；
- `aliased`：复用另一个语义完整的采集块；
- `partial`：存在部分产物但不能满足完整合同；
- `empty`：返回成功但没有可用语义数据；
- `unavailable`：当前 CLI、架构或输入不支持；
- `failed`：命令失败；
- `skipped`：因依赖、circuit breaker 或用户选择跳过。

alias 必须记录来源块和复用原因。

## 路径

manifest 内 artifact 路径使用相对 manifest 的 POSIX 路径。不得写入仅在采集机器有效的临时绝对路径作为唯一定位信息。

## 可用性

报告中每个请求页面必须具有明确状态：

- 有语义数据：渲染模块；
- 定向请求但缺失：渲染诊断页；
- preset 请求且策略允许省略：写入 `omitted_modules`；
- 不得使用空白图表冒充成功。

## 不可替代的数据

- On-Chip Memory 必须使用 `memory_info.json`；
- Source 必须使用真实源码快照、行映射、指令映射和关系；
- Warp Stall 必须使用 SIMT/PCSampling 证据；
- 指令时间线必须使用受支持的指令级 timeline 数据；
- MemoryDetail 计数器不能替代 allocation address、lifetime 或 bank/group 信息。

## 调试导出

Memory topology 每个 Block 导出：

```text
_debug/memory_edges/block_{ID}_edges.csv
_debug/memory_edges/block_{ID}_bandwidth.csv
_debug/memory_edges/block_{ID}_requests.csv
```

Source 导出：

```text
_debug/source/source_files.csv
_debug/source/source_lines.csv
_debug/source/instructions.csv
_debug/source/source_instruction_relations.csv
```
