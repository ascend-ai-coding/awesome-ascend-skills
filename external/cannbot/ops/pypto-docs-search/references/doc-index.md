# PyPTO 教程 / 工具 / 安装文档索引

按主题查文档。安装与教程的 `<path>` 取下列条目（如 `guide/programming_guide/tensor/debug/performance`）。**缓存在场**读 `$PYPTO_DEVKIT_DIR/docs/<path>.md`；**无则在线** `https://pypto.gitcode.com/_sources/<path>.md.txt`。
例：`guide/programming_guide/tensor/debug/performance` → 本地 `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/tensor/debug/performance.md` / 在线 `.../_sources/guide/programming_guide/tensor/debug/performance.md.txt`。
API 文档见 [`api-index.md`](api-index.md)；排障按错误码见 [`error-code-index.md`](error-code-index.md)。各区结构以 `<区>/index` 为准。

## install/ 安装与环境

- `install/prepare_environment` — 环境准备
- `install/build_and_install` — 编译安装

## guide/ 教程

- `guide/introduction` — 简介
- `guide/quick_start/tensor/quick_start` — 快速入门
- `guide/programming_guide/tensor/program_paradigms` — 编程范式
- `guide/programming_guide/tensor/development/`：`tensor_creation`、`tensor_operation`、`tiling`、`compile`、`loops`、`conditions`
- `guide/programming_guide/tensor/debug/`：`debug`、`precision`、`performance`、`matmul_performance_guide`、`debug_case_ffn`、`performance_case_quantindexerprolog`、`performance_case_GDR`
- `guide/programming_guide/tensor/pytorch_integration` — PyTorch 集成
- `guide/appendix/`：`faq/index`（常见问题与已知问题分篇入口）、`glossary`

## 配套可视化与分析工具（性能调优相关）

工具文档使用独立的 [官方工具站](https://pypto-tools.gitcode.com/_sources/index.md.txt)，不在主仓缓存中；下列 `<path>` 读取 `https://pypto-tools.gitcode.com/_sources/<path>.md.txt`。

- `introduction/`：`introduction`（简介）、`install`（安装）、`quick_start`（快速入门）、`data_preparation`（数据准备）
- `control_flow/index` — 控制流图（查看 / 搜索节点）
- `computation_graph/index` — 计算图（查看 / 健康报告 / 搜索 / 对比差异 / 跳转代码行）
- `swimlane_graph/index` — 泳道图（跳转计算图 / 搜索 / 测量时间间隔 / 时间范围 / 观测线 / 性能报告 / 着色 / 系统参数）
- `three_column/three_column` — 三栏联动视图
- `others/others`、`appendix/index` — 其他功能、常见问题与已知问题
