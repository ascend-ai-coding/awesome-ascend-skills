"""Offline parser for MindSpore Profiler data (_ascend_ms)."""
import argparse
import os
import shutil
import sys


def _check_msprof() -> None:
    """Ensure msprof is in PATH; required by CANN/profiler analyse."""
    if shutil.which("msprof") is None:
        print(
            "Error: msprof command not found. Please source the correct CANN environment, e.g.:\n"
            "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline parser for MindSpore Profiler data (_ascend_ms)."
    )
    parser.add_argument(
        "profiler_path",
        type=str,
        help="Path to the profiler result directory (e.g., ./result_data)",
    )
    args = parser.parse_args()

    profiler_path = args.profiler_path
    if not os.path.exists(profiler_path):
        print(f"Error: Profiler path '{profiler_path}' does not exist.")
        sys.exit(1)

    _check_msprof()

    try:
        from mindspore.profiler.profiler import analyse
    except ImportError as e:
        print(f"Error: MindSpore not installed or analyse unavailable: {e}")
        sys.exit(1)

    print(f"Starting offline analysis for: {profiler_path}")
    try:
        analyse(profiler_path=profiler_path)
        print("Analysis completed successfully.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
