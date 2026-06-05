import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu


def _load_ascend_kernel():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libascend_kernel.so")
    torch.ops.load_library(so_path)


_load_ascend_kernel()
