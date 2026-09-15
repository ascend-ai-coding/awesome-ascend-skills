# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash

# TileLang-Ascend environment check script
# Used to verify if the development environment is correctly configured

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the TileLang-Ascend repository root (the environment being checked).
# The skill may be installed OUTSIDE the repo (e.g. symlinked under
# .opencode/skills/ or .claude/skills/), so we must NOT assume a fixed
# directory depth. Resolution order:
#   1) $TILELANG_DIR  (exported by the plugin SessionStart hook)
#   2) walk up from the current working directory looking for install_ascend.sh
#   3) a ./tilelang-ascend subdirectory under the current / ancestor dirs
find_project_root() {
    if [ -n "${TILELANG_DIR:-}" ] && [ -f "$TILELANG_DIR/install_ascend.sh" ]; then
        echo "$TILELANG_DIR"; return 0
    fi
    local d="$PWD"
    while :; do
        [ -f "$d/install_ascend.sh" ] && { echo "$d"; return 0; }
        [ -f "$d/tilelang-ascend/install_ascend.sh" ] && { echo "$d/tilelang-ascend"; return 0; }
        [ "$d" = "/" ] && break
        d="$(dirname "$d")"
    done
    return 1
}

if ! PROJECT_ROOT="$(find_project_root)"; then
    echo "✗ 未找到 TileLang-Ascend 仓库（目录内缺少 install_ascend.sh）。"
    echo "  请在 tilelang-ascend 仓库目录内运行，或先 clone 该仓库，或设置 TILELANG_DIR 环境变量。"
    exit 1
fi

# Enter project root directory
cd "$PROJECT_ROOT"

echo "========================================"
echo "TileLang-Ascend 环境检查"
echo "========================================"
echo ""

# Check result statistics
TOTAL_CHECKS=4
PASSED_CHECKS=0
FAILED_ITEMS=()

# Check functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED_ITEMS+=("$2")
}

check_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

# [1/4] Check repository integrity
echo "[1/4] 检查代码仓库完整性..."
echo ""

if [ -e ".git" ]; then
    check_pass "Git 仓库存在"
else
    check_fail "Git 仓库不存在" "git_repo"
fi

# Check critical submodules
SUBMODULES=("tvm" "cutlass" "composable_kernel" "pto-isa" "shmem" "catlass")
MISSING_SUBMODULES=()
INCOMPLETE_SUBMODULES=()

echo "检查子模块状态..."
for submodule in "${SUBMODULES[@]}"; do
    submodule_path="3rdparty/$submodule"
    
    status=$(git submodule status "$submodule_path" 2>/dev/null || echo "")
    status_prefix="${status:0:1}"
    
    if [[ -z "$status" ]]; then
        check_fail "子模块 $submodule 不存在" "submodule_incomplete"
        MISSING_SUBMODULES+=("$submodule")
    elif [[ "$status_prefix" == "-" ]]; then
        check_fail "子模块 $submodule 未初始化" "submodule_incomplete"
        MISSING_SUBMODULES+=("$submodule")
    elif [[ "$status_prefix" == "+" ]]; then
        check_fail "子模块 $submodule 不完整" "submodule_incomplete"
        INCOMPLETE_SUBMODULES+=("$submodule")
    elif [[ "$status_prefix" == " " ]]; then
        if [ -d "$submodule_path" ] && [ "$(ls -A $submodule_path 2>/dev/null)" ]; then
            commit_hash=$(echo "$status" | awk '{print $1}')
            check_pass "子模块 $submodule 完整 ($commit_hash)"
        else
            check_fail "子模块 $submodule 目录为空" "submodule_incomplete"
            INCOMPLETE_SUBMODULES+=("$submodule")
        fi
    else
        commit_hash=$(echo "$status" | awk '{print $1}')
        check_pass "子模块 $submodule 完整 ($commit_hash)"
    fi
done

if [ ${#MISSING_SUBMODULES[@]} -eq 0 ] && [ ${#INCOMPLETE_SUBMODULES[@]} -eq 0 ]; then
    check_pass "所有子模块完整"
fi

echo ""

# [2/4] Check compilation status
echo "[2/4] 检查编译安装状态..."
echo ""

if [ -d "build" ]; then
    check_pass "build 目录存在"
    
    if ls build/*.so 1> /dev/null 2>&1 || ls build/lib*.so 1> /dev/null 2>&1; then
        check_pass "编译产物存在"
    else
        check_warn "build 目录存在但未找到 .so 编译产物"
        check_fail "编译产物不存在" "build_products"
    fi
else
    check_fail "build 目录不存在" "build_dir"
fi

echo ""

# [3/4] Check environment variables and auto-fix
echo "[3/4] 检查环境变量..."
echo ""

ENV_FIXED=0

check_and_fix_env() {
    local need_fix=0
    
    if [ -z "$TL_ROOT" ]; then
        echo -e "${YELLOW}! TL_ROOT 未设置${NC}"
        need_fix=1
    fi
    
    if [ -z "$ACL_OP_INIT_MODE" ]; then
        echo -e "${YELLOW}! ACL_OP_INIT_MODE 未设置${NC}"
        need_fix=1
    fi
    
    if [ $need_fix -eq 1 ]; then
        echo ""
        echo -e "${BLUE}>>> 自动执行 source set_env.sh 修复环境变量...${NC}"
        source "$PROJECT_ROOT/set_env.sh"
        ENV_FIXED=1
        echo -e "${GREEN}✓ 环境变量已设置${NC}"
        echo ""
    fi
}

# First check and fix environment variables
check_and_fix_env

# Re-check environment variable status
if [ -n "$TL_ROOT" ]; then
    check_pass "TL_ROOT 已设置: $TL_ROOT"
else
    check_fail "TL_ROOT 未设置" "env_tl_root"
fi

if [ -n "$PYTHONPATH" ]; then
    check_pass "PYTHONPATH 已设置"
else
    check_warn "PYTHONPATH 未设置（可能不影响使用）"
fi

if [ -n "$ACL_OP_INIT_MODE" ]; then
    check_pass "ACL_OP_INIT_MODE 已设置: $ACL_OP_INIT_MODE"
else
    check_warn "ACL_OP_INIT_MODE 未设置（可能影响某些功能）"
fi

echo ""

# [4/4] Run verification test
echo "[4/4] 运行验证测试..."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    check_fail "Python 未安装" "python_not_found"
else
    PYTHON_VERSION=$(python --version 2>&1)
    check_pass "Python 可用: $PYTHON_VERSION"
    
    # Check if tilelang can be imported
    if python -c "import tilelang" 2> /dev/null; then
        check_pass "tilelang 模块可导入"
        
        # Run simple test
        if [ -f "$SCRIPT_DIR/quick_verify.py" ]; then
            echo ""
            echo "运行验证测试..."
            if python "$SCRIPT_DIR/quick_verify.py" 2>&1; then
                check_pass "验证测试通过"
            else
                check_fail "验证测试失败" "verify_test_failed"
            fi
        else
            check_warn "验证测试脚本不存在，跳过运行测试"
        fi
    else
        check_fail "tilelang 模块无法导入" "tilelang_import"
    fi
fi

echo ""

# Output summary
echo "========================================"
if [ ${#FAILED_ITEMS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ 环境检查通过！所有配置正确。${NC}"
    if [ $ENV_FIXED -eq 1 ]; then
        echo -e "${BLUE}  注：环境变量已自动设置，仅当前终端会话有效${NC}"
        echo -e "${BLUE}  后续使用请先执行: source set_env.sh${NC}"
    fi
else
    echo -e "${RED}✗ 环境检查未完全通过${NC}"
    echo ""
    echo "需要修复的问题："
    
    for item in "${FAILED_ITEMS[@]}"; do
        case $item in
            git_repo)
                echo "  - Git 仓库不存在：请确认在正确的项目目录中运行"
                ;;
            submodule_incomplete)
                echo "  - 子模块不完整"
                ;;
            build_dir)
                echo "  - build 目录不存在：运行 bash install_ascend.sh"
                ;;
            build_products)
                echo "  - 编译产物不存在：运行 bash install_ascend.sh 重新编译"
                ;;
            env_tl_root)
                echo "  - TL_ROOT 未设置：运行 source set_env.sh"
                ;;
            python_not_found)
                echo "  - Python 未安装：请安装 Python 3.8+"
                ;;
            tilelang_import)
                echo "  - tilelang 无法导入：运行 source set_env.sh 或检查编译是否成功"
                ;;
            verify_test_failed)
                echo "  - 验证测试失败：检查 NPU 设备是否可用或查看详细错误信息"
                ;;
        esac
    done
fi
echo "========================================"

# Return appropriate exit code
if [ ${#FAILED_ITEMS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi