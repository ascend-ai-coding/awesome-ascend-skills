#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bootstrap the checkout, then run its canonical read-only query CLI."""

import subprocess
import sys

from knowledge_setup import QUERY_DIR, prepare, report_error, split_root


def main(argv=None):
    try:
        args, explicit = split_root(sys.argv[1:] if argv is None else argv)
        if not args or args[0] in ("-h", "--help"):
            print(
                "Usage: knowledge_query.py [--knowledge-root PATH] discover|search [options]"
            )
            print("Missing checkout/dependencies/index are prepared before querying.")
            print("Search: --query TEXT --platform ID [--domain ID] [--technology ID]")
            print(
                "        [--type ID] [--status stable|draft|deprecated|all] [--match text|literal] [-k N]"
            )
            return 0
        if args[0] not in ("discover", "search"):
            raise ValueError(
                "supported commands: discover, search; read SKILL.md for current usage"
            )
        root, python = prepare(explicit)
        result = subprocess.run(
            [
                str(python),
                str(root / QUERY_DIR / "knowledge_query.py"),
                "--knowledge-root",
                str(root),
                *args,
            ],
            check=False,
            timeout=300,
        )
        return result.returncode
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
