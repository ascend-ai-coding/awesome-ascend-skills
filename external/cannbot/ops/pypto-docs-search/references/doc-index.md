# PyPTO 教程 / 工具 / 安装文档索引

按主题查文档。`<path>` 取下列某条目（如 `tutorials/debug/performance`）。**缓存在场**读 `$PYPTO_DEVKIT_DIR/docs/<path>.md`；**无则在线** `https://pypto.gitcode.com/_sources/<path>.md.txt`。
例：`tutorials/debug/performance` → 本地 `$PYPTO_DEVKIT_DIR/docs/tutorials/debug/performance.md` / 在线 `.../_sources/tutorials/debug/performance.md.txt`。
API 文档见 [`api-index.md`](api-index.md)；排障按错误码见 [`error-code-index.md`](error-code-index.md)。各区结构以 `<区>/index` 为准。

## install/ 安装与环境
- `install/prepare_environment` — 环境准备
- `install/build_and_install` — 编译安装

## tutorials/ 教程
- `tutorials/introduction/`：`introduction`、`quick_start`、`program_paradigms`
- `tutorials/development/`：`tensor_creation`、`tensor_operation`、`tiling`、`compile`、`loops`、`conditions`
- `tutorials/debug/`：`debug`、`precision`、`performance`、`matmul_performance_guide`、`debug_case_ffn`、`performance_case_quantindexerprolog`、`performance_case_GDR`
- `tutorials/network_integration/pytorch_integration` — PyTorch 集成
- `tutorials/appendix/`：`faq`、`issue`、`glossary`

## tools/ 配套可视化与分析工具（性能调优相关）
- `tools/introduction/`：`简介`、`安装`、`快速入门`、`数据准备`
- `tools/control_flow/` — 控制流图（查看 / 搜索节点）
- `tools/computation_graph/` — 计算图（查看 / 健康报告 / 搜索 / 对比差异 / 跳转代码行）
- `tools/swimlane_graph/` — 泳道图（跳转计算图 / 搜索 / 测量时间间隔 / 时间范围 / 观测线 / 性能报告 / 着色 / 系统参数）
- `tools/three_column/` — 三栏联动视图
- `tools/others/`、`tools/faq/` — 其他功能、已知问题
