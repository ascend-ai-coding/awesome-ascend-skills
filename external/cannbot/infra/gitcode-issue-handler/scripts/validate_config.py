#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read-only local configuration validation; shared with normal config loading."""

import argparse
import os
from pathlib import Path

from config_validation import DEPRECATED
from handler_config import ConfigError, load_handler_config
from protocol_output import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--config", help="Path relative to repository root, or absolute")
    args = parser.parse_args(argv)
    try:
        os.chdir(Path(args.repository_root).resolve())
        path = Path(args.config or ".cannbot/gitcode-issue-handler/config/classify_config.yaml").resolve()
        config = load_handler_config(path)
        warnings = []
        for field in sorted(DEPRECATED):
            group, key = field.split(".")
            if key in config.get(group, {}):
                warnings.append({"field": field, "message": "已失效；可删除，实际状态由 --status-name 指定"})
        result = {
            "valid": True,
            "next_action": "continue" if config["repo"] else "configure_repository",
            "file": str(path),
            "warnings": warnings,
        }
        code = 0
    except ConfigError as exc:
        result = {"valid": False, "next_action": "fix_configuration", "errors": exc.errors}
        code = 2
    except (OSError, ValueError, RuntimeError) as exc:
        result = {
            "valid": False,
            "next_action": "fix_configuration",
            "errors": [{"message": type(exc).__name__ + ": 检查仓库目录或配置路径"}],
        }
        code = 2
    write_json(result, ensure_ascii=False, indent=2)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
