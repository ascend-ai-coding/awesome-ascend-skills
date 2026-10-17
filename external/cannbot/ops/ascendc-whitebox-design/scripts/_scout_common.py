#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import re

PLATFORM_MAP = {
    'DAV_3510': {'arch_num': '3510', 'chip': 'Ascend950', 'is_950': True},
    'DAV_2201': {'arch_num': '2201', 'chip': 'Ascend910B', 'is_950': False},
    'DAV_2002': {'arch_num': '2002', 'chip': 'Ascend310P', 'is_950': False},
}

ARCH_DIR_MAP = {
    'arch35': 'DAV_3510',
    'arch22': 'DAV_2201',
    'arch32': 'DAV_2201',
    'arch38': 'DAV_5102',
}

ARCH_FEATURE_MAP = {
    'regbase': 'DAV_3510',
    'reg_base': 'DAV_3510',
    '_apt': 'DAV_3510',
}

RE_TILING_KEY_IS = re.compile(r'TILING_KEY_IS\(\s*([^)]+?)\s*\)')


def skip_parens(content, start):
    pos = start
    depth = 1
    while pos < len(content) and depth > 0:
        if content[pos] == '(':
            depth += 1
        elif content[pos] == ')':
            depth -= 1
        pos += 1
    return pos if depth == 0 else None


def find_def_after(content, match_end, max_scan=200):
    rest = content[match_end:match_end + max_scan]
    brace_pos = rest.find('{')
    semi_pos = rest.find(';')
    if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
        return True
    return False
