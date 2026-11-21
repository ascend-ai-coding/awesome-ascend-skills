#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Use the shared checkout installer shipped by the knowledge-query dependency."""

from pathlib import Path
import subprocess
import sys


def main():
    setup = (
        Path(__file__).resolve().parents[2]
        / "knowledge-query/scripts/knowledge_setup.py"
    )
    if not setup.is_file():
        print(
            "Install knowledge-query beside knowledge-ingest to provide the shared installer.",
            file=sys.stderr,
        )
        return 2
    return subprocess.run(
        [sys.executable, str(setup), *sys.argv[1:]], check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
