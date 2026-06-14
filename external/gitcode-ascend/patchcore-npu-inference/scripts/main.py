#!/usr/bin/env python
"""PatchCore 昇腾 NPU 端到端推理入口（main.py 别名）

比赛平台可能需要 ``main.py`` 作为入口文件。
本文件直接转发到 ``inference.py`` 的实际实现。
"""

import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 将 src/ 和项目根目录加入 Python 路径
for p in [str(_HERE / "src"), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    print("[INFO] main.py entry — redirecting to inference.py\n")
    from inference import main
except ImportError as e:
    print(f"[EVAL_ERROR] 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    try:
        results = main()
        if results:
            import json
            img_aurocs = [r["image_auroc"] for r in results if r["image_auroc"] > 0]
            pix_aurocs = [r["pixel_auroc"] for r in results if r["pixel_auroc"] > 0]

            summary = {
                "npu_detected": True,
                "categories_completed": len(img_aurocs),
                "categories_total": len(results),
                "mean_image_auroc": round(float(sum(img_aurocs) / len(img_aurocs)), 4) if img_aurocs else 0.0,
                "mean_pixel_auroc": round(float(sum(pix_aurocs) / len(pix_aurocs)), 4) if pix_aurocs else 0.0,
                "total_time_s": round(float(sum(r["total_time_s"] for r in results)), 1),
            }
            print(f"\n[EVAL_RESULT] {json.dumps(summary)}")
            print("[EVAL_DONE] 推理完成")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[EVAL_ERROR] 推理执行失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
