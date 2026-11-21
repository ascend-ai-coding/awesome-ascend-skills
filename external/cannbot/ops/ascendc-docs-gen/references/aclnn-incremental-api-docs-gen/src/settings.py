# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""单一 env 配置站点：全部环境变量读取与校验的唯一入口。

约定:
- 本模块自身 **import 期零副作用**（不读 env、不抛错），校验全部发生在 load_settings()。
- 必填变量: INCREMENTAL_LLM_KEY / INCREMENTAL_LLM_URL——缺失时 load_settings() 抛 RuntimeError
  （含变量名与 WORKFLOW.md 指引），由各入口在启动期捕获并以非零码退出，
  消除"静默 None 回退 api.openai.com"的暗坑。
- INCREMENTAL_LLM_MODEL 默认 qwen3.8-max（UD5 单点定义，勿在别处复制该字面量）。
- 可选变量仅透传，不在此做强校验。
- PubRefConfig 的非 env 常量（DOC_PARTS / DOC_PART_MAX_WORKERS）不属于本模块，保持原位。
"""

import os
from dataclasses import dataclass
from typing import Optional

# INCREMENTAL_LLM_MODEL 统一默认值（UD5）——全仓唯一定义点
DEFAULT_MODEL = "qwen3.8-max"

# 必填环境变量（缺失即 RuntimeError）。注意勿 import src.llm 来取名字常量——
# llm 包 __init__ 会拉起 client 并反向依赖本模块，构成循环导入。
_REQUIRED_ENV_VARS = ("INCREMENTAL_LLM_KEY", "INCREMENTAL_LLM_URL")

# 报错指引（WORKFLOW.md 由 Task 15 撰写）
_ENV_HINT = "请参考 WORKFLOW.md 配置"


@dataclass(frozen=True)
class Settings:
    """运行期配置快照：由 load_settings() 构造，入口启动期注入下游。"""

    incremental_llm_key: str
    incremental_llm_url: str
    model: str = DEFAULT_MODEL
    # 可选透传变量（None 表示未设置）
    cann_plugin_log_dir: Optional[str] = None
    cann_ops_root: Optional[str] = None
    cann_ops_workspace: Optional[str] = None
    cann_evidence_dir: Optional[str] = None
    test_operator_path: Optional[str] = None


def load_settings() -> Settings:
    """惰性读取并校验环境变量，返回 Settings 快照。

    Raises:
        RuntimeError: 任一必填变量（INCREMENTAL_LLM_KEY / INCREMENTAL_LLM_URL）未设置或为空串。
    """
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        details = "\n".join(
            f"[settings] 环境变量 {name} 未设置——{_ENV_HINT}" for name in missing
        )
        raise RuntimeError(details)
    return Settings(
        incremental_llm_key=os.environ["INCREMENTAL_LLM_KEY"],
        incremental_llm_url=os.environ["INCREMENTAL_LLM_URL"],
        model=os.environ.get("INCREMENTAL_LLM_MODEL") or DEFAULT_MODEL,
        cann_plugin_log_dir=os.environ.get("CANN_PLUGIN_LOG_DIR"),
        cann_ops_root=os.environ.get("CANN_OPS_ROOT"),
        cann_ops_workspace=os.environ.get("CANN_OPS_WORKSPACE"),
        cann_evidence_dir=os.environ.get("CANN_EVIDENCE_DIR"),
        test_operator_path=os.environ.get("TEST_OPERATOR_PATH"),
    )


def apply_settings(settings: Settings) -> None:
    """运行期把 Settings 注入 PubRefConfig（供下游既有 PubRefConfig 读取点兼容）。

    仅在各入口 main()/启动期调用——import 期零赋值零 raise；下游所有
    PubRefConfig 读取点无需改动。PubRefConfig 自身保持普通数据类占位。
    """
    from src.snapshot_construction.pub_ref import PubRefConfig

    PubRefConfig.INCREMENTAL_LLM_KEY = settings.incremental_llm_key
    PubRefConfig.INCREMENTAL_LLM_URL = settings.incremental_llm_url
    PubRefConfig.MODEL = settings.model
