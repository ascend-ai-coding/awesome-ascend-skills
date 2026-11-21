# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from common import P, R, S

manifest = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R).decode().strip(),
    "sources": [],
    "working_tree_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=R)).hexdigest(),
}
for group in ["reduce", "scan", "conv", "conversion", "gather", "sort/arch35"]:
    for src in (S / "reference" / group / "templates").rglob("*.template"):
        rel = src.relative_to(S / "reference" / group / "templates")
        name = rel.name[:-9]
        if not Path(name).suffix:
            name += ".h"
        dst = P / "include" / group / rel.parent / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        manifest["sources"].append(
            {
                "source": str(src.relative_to(R)),
                "materialized": str(dst.relative_to(P)),
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            }
        )
for src in (S / "references/common/arch35/regbase/code").glob("*.h"):
    dst = P / "include/regbase" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    manifest["sources"].append(
        {
            "source": str(src.relative_to(R)),
            "materialized": str(dst.relative_to(P)),
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        }
    )
(P / "src").mkdir(exist_ok=True)
(P / "build").mkdir(exist_ok=True)
(P / "logs").mkdir(exist_ok=True)
(P / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Materialized", len(manifest["sources"]), "sources from", manifest["commit"])
