#!/bin/bash

# ============================================================================
# Ascend-Kernel算子工程检测脚本
# 功能：检测当前目录或子目录中是否存在ascend-kernel工程
# 输出格式：
#   - PROJECT_FOUND:<path>              找到ascend-kernel项目（含csrc/ops/）
#   - PROJECT_FOUND_NO_OPS:<path>       找到ascend-kernel项目但缺少csrc/ops/目录
#   - PROJECT_NOT_FOUND                 未找到ascend-kernel项目
# ============================================================================

# 函数：检查指定路径是否为ascend-kernel项目
# 判断标准：
#   - 必须存在 build.sh 和 CMakeLists.txt
#   - 必须存在 csrc/ 目录
is_ascend_kernel_project() {
    local dir="$1"
    [ -f "${dir}/build.sh" ] && [ -f "${dir}/CMakeLists.txt" ] && [ -d "${dir}/csrc" ]
}

# 函数：检查ascend-kernel项目是否包含ops目录
has_ops_directory() {
    local dir="$1"
    [ -d "${dir}/csrc/ops" ]
}

# 检测策略1：当前目录就是ascend-kernel
if is_ascend_kernel_project "."; then
    if has_ops_directory "."; then
        echo "PROJECT_FOUND:."
    else
        echo "PROJECT_FOUND_NO_OPS:."
    fi
    exit 0
fi

# 检测策略2：当前目录下有ascend-kernel子目录
if [ -d "ascend-kernel" ] && is_ascend_kernel_project "ascend-kernel"; then
    if has_ops_directory "ascend-kernel"; then
        echo "PROJECT_FOUND:ascend-kernel"
    else
        echo "PROJECT_FOUND_NO_OPS:ascend-kernel"
    fi
    exit 0
fi

# 检测策略3：一级子目录中查找ascend-kernel
found_projects=()
found_no_ops=()
for dir in */; do
    dir="${dir%/}"  # 去除末尾斜杠
    if [ -d "$dir" ] && is_ascend_kernel_project "$dir"; then
        if has_ops_directory "$dir"; then
            found_projects+=("$dir")
        else
            found_no_ops+=("$dir")
        fi
    fi
    # 检查子目录下的ascend-kernel
    if [ -d "$dir/ascend-kernel" ] && is_ascend_kernel_project "$dir/ascend-kernel"; then
        if has_ops_directory "$dir/ascend-kernel"; then
            found_projects+=("$dir/ascend-kernel")
        else
            found_no_ops+=("$dir/ascend-kernel")
        fi
    fi
done

# 输出检测结果
if [ ${#found_projects[@]} -eq 1 ]; then
    echo "PROJECT_FOUND:${found_projects[0]}"
    exit 0
elif [ ${#found_projects[@]} -gt 1 ]; then
    echo "MULTIPLE_PROJECTS:${found_projects[*]}"
    exit 0
elif [ ${#found_no_ops[@]} -eq 1 ]; then
    echo "PROJECT_FOUND_NO_OPS:${found_no_ops[0]}"
    exit 0
elif [ ${#found_no_ops[@]} -gt 1 ]; then
    echo "MULTIPLE_PROJECTS_NO_OPS:${found_no_ops[*]}"
    exit 0
else
    echo "PROJECT_NOT_FOUND"
    exit 1
fi
