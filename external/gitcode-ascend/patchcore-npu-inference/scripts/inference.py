#!/usr/bin/env python
"""PatchCore 昇腾 NPU 端到端推理脚本（根目录入口）

自动转发到 src/inference.py 的实际实现。
这样评分系统从根目录运行 ``python inference.py ...`` 也能正常工作。
"""

import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 将 src/ 和项目根目录加入 Python 路径
for p in [str(_HERE / "src"), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 导入并运行真正的实现
try:
    from src.inference import (  # noqa: E402
        ALL_CATEGORIES,
        BaihuNN,
        auto_tune_params,
        build_device,
        check_environment,
        detect_cpu_count,
        detect_disk_gib,
        detect_npu,
        detect_npu_mem_gib,
        detect_ram_gib,
        get_dataloader,
        main,
        parse_args,
        run_single_category,
    )
except ImportError as e:
    print(f"[EVAL_ERROR] 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
