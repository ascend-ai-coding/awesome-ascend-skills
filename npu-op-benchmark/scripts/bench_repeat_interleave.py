#!/usr/bin/env python3
"""Example wrapper that benchmarks repeat_interleave via bench_op.py.

This script exists only as a convenience example for one operator. It does not
mean the skill is limited to repeat_interleave benchmarks.
"""

import subprocess, sys
from pathlib import Path
bench = Path(__file__).with_name('bench_op.py')
cmd = [sys.executable, str(bench), '--op', 'repeat_interleave', *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd))
