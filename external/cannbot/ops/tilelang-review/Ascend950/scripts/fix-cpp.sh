#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Fix C++ file format issues using clang-format
#
# Usage:
#   fix-cpp.sh <file.cc> [file.h ...]

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"

cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Error: provide at least one C++ file path." >&2
    exit 2
fi

FILE_ARRAY=()
for file in "$@"; do
    case "$file" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hpp|*.hh|*.icc)
            if [ -f "$file" ]; then
                FILE_ARRAY+=("$file")
            elif [ -f "$REPO_ROOT/$file" ]; then
                FILE_ARRAY+=("$REPO_ROOT/$file")
            fi
            ;;
    esac
done

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo "Error: no existing C++ files were provided." >&2
    exit 2
fi

# Check if clang-format is available
if ! command -v clang-format &>/dev/null; then
    echo "Error: clang-format not found. Please install clang-format."
    exit 1
fi

echo "Fixing ${#FILE_ARRAY[@]} C++ file(s)..."

# Run clang-format -i with style from .clang-format file
clang-format -i --style=file "${FILE_ARRAY[@]}"

echo "C++ files have been fixed."
