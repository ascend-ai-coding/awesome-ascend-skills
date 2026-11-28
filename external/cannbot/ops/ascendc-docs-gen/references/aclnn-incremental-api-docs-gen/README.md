# cann-document-generation

## 安装与运行

- **Python 3.11+**（本仓 `.python-version` 固定 3.11.9）。
- 安装依赖（`requirements.txt` 为唯一依赖源，宽松约束）：

  ```bash
  pip install -r requirements.txt
  ```

- **环境变量**（详见 `WORKFLOW.md`）：
  - `INCREMENTAL_LLM_KEY` — LLM API key，**必填**（缺失时启动期报错退出）
  - `INCREMENTAL_LLM_URL` — LLM base_url，**必填**（缺失时启动期报错退出）
  - `INCREMENTAL_LLM_MODEL` — 模型名，可选，默认 `qwen3.8-max`（单点定义于 `src/settings.py`）
  - 可选：`CANN_PLUGIN_LOG_DIR`（日志目录，默认 `<repo>/data/logs`）、`CANN_OPS_ROOT` 等（见 `WORKFLOW.md`）
- **运行**（主入口为 `src/run_all_cli.py`，双模式：脚本直跑与 `python -m src.run_all_cli` 均可，从仓库根执行）：

  ```bash
  python src/run_all_cli.py --help
  python src/run_all_cli.py \
    --original_op_repo <repo-root> --original_op_path <rel/op/path> \
    --modified_op_repo <repo-root> --modified_op_path <rel/op/path> \
    --output_data_path <dir> [--validate-only] [--md-stem aclnnXxx ...]
  ```

  - 非 TTY（管道/CI）环境下缺少必需参数（4 个 `--*_op_repo`/`--*_op_path` 与 `--output_data_path`）时直接 `exit 2` 报错，不回退交互 TUI；TTY 下省略则进入交互式 Rich/questionary 界面。
  - `--validate-only` 仅做预校验（跳过变更检测与生成）；`--skip-fix` 跳过校验失败的修复步骤。
- **测试**：

  ```bash
  python -m pytest
  ```

  配置位于仓根 `pytest.ini`（`pythonpath`/`norecursedirs`）。


