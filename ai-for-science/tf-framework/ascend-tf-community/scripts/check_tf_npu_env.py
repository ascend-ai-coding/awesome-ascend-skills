#!/usr/bin/env python3
"""Validate TensorFlow Community plus npu_device prerequisites."""

import argparse
import os
import sys


REQUIRED_ENV = [
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TensorFlow Community NPU environment")
    parser.add_argument(
        "--open-device",
        action="store_true",
        help="Attempt to open the NPU device and run a tiny TensorFlow op",
    )
    args = parser.parse_args()

    failures = 0

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"[WARN] cann-env: missing vars: {', '.join(missing)}")
    else:
        print("[PASS] cann-env: CANN environment variables look set")

    try:
        import tensorflow as tf

        print(f"[PASS] tensorflow: {tf.__version__}")
    except Exception as exc:
        print(f"[FAIL] tensorflow: {exc}")
        return 1

    try:
        import npu_device

        print("[PASS] npu_device: import succeeded")
    except Exception as exc:
        print(f"[FAIL] npu_device: {exc}")
        return 1

    if args.open_device:
        try:
            import tensorflow as tf
            import npu_device

            npu_device.open().as_default()
            value = tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy().item()
            print(f"[PASS] tf-op: simple TensorFlow op succeeded after npu_device init (sum={value:.1f})")
        except Exception as exc:
            print(f"[FAIL] tf-op: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
