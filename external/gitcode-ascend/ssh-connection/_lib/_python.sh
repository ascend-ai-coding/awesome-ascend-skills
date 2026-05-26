#!/usr/bin/env bash
# _python.sh — Python 可执行文件检测 + 平台适配
# 用法: source "$SUITE_DIR/_lib/_python.sh"
# 设置 $PYTHON 变量，兼容 python3 / python
# 设置 $IS_WINDOWS 变量，用于平台判断

if command -v python3 &>/dev/null && python3 -c "pass" 2>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "错误: 未找到 Python，请安装 Python 3" >&2
    exit 1
fi

# Windows 平台检测
IS_WINDOWS=false
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
esac

# Windows 下强制 Python 使用 UTF-8 输出，避免 GBK 乱码
if $IS_WINDOWS; then
    export PYTHONIOENCODING=utf-8
fi
