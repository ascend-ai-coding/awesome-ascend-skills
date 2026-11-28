# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 收集 csrc 下全部 cpp：ops_common.cpp / ops_def_registration.cpp / 各算子 impl cpp
# 用 glob 而非写死列表，新增算子只要把 cpp 丢进 csrc 目录即可
source_files = glob.glob(os.path.join(BASE_DIR, "xops/csrc", "*.cpp"))

ext = NpuExtension(
    name="xops.xops_lib",
    sources=source_files,
    # 只需要 ACL 头文件路径；aclnn 符号全部运行时 dlopen，无需 libraries=
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)

setup(
    name="xops",
    version='1.0',
    ext_modules=[ext],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
