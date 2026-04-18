#!/bin/bash
set -euo pipefail

# 颜色输出函数
red_echo() { echo -e "\033[31m[ERROR] $1\033[0m"; }
green_echo() { echo -e "\033[32m[INFO] $1\033[0m"; }
yellow_echo() { echo -e "\033[33m[WARN] $1\033[0m"; }

# 基础配置
PY_CHECK_SCRIPT="./check_package.py"
DRIVER_PKG="" && FIRMWARE_PKG=""
DRIVER_REGEX="Ascend-hdk-.+-npu-driver_.+_linux-.+\.run"
FIRMWARE_REGEX="Ascend-hdk-.+-npu-firmware_.+\.run"

# 入参检查
if [ $# -ne 2 ]; then
    red_echo "入参错误！正确用法：$0 <NPU包文件夹完整路径> <驱动运行用户>"
    exit 1
fi
NPU_PKG_FOLDER=$1 && RUN_USER=$2 && RUN_GROUP=$2

# 基础环境检查：ROOT权限 & Python3
check_base() {
    [ $EUID -ne 0 ] && red_echo "必须以ROOT权限运行本脚本！" && exit 1
    command -v python3 &> /dev/null || { red_echo "未检测到Python3环境，请先安装后重试"; exit 1; }
    green_echo "基础环境（ROOT/Python3）检查通过"
}

# 提前提取驱动/固件包路径（Python校验前执行）
extract_pkgs() {
    [ ! -d "$NPU_PKG_FOLDER" ] && red_echo "NPU包所在文件夹不存在：$NPU_PKG_FOLDER" && exit 2
    # 提取驱动包并校验数量
    DRIVER_LIST=$(ls -1 "$NPU_PKG_FOLDER" | grep -E "^${DRIVER_REGEX}$")
    [ $(echo "$DRIVER_LIST" | wc -l) -ne 1 ] && red_echo "驱动包数量异常，仅允许1个符合格式的驱动包" && exit 3
    # 提取固件包并校验数量
    FIRMWARE_LIST=$(ls -1 "$NPU_PKG_FOLDER" | grep -E "^${FIRMWARE_REGEX}$")
    [ $(echo "$FIRMWARE_LIST" | wc -l) -ne 1 ] && red_echo "固件包数量异常，仅允许1个符合格式的固件包" && exit 4
    # 拼接绝对路径
    DRIVER_PKG=$(realpath "${NPU_PKG_FOLDER}/${DRIVER_LIST}")
    FIRMWARE_PKG=$(realpath "${NPU_PKG_FOLDER}/${FIRMWARE_LIST}")
    green_echo "包路径提取成功：\n→ 驱动包：$DRIVER_PKG\n→ 固件包：$FIRMWARE_PKG"
}

# 按需添加可执行权限（Python校验前执行，核心）
add_pkg_perm() {
    green_echo "开始检查并添加安装包可执行权限"
    # 驱动包权限处理
    if [ -f "$DRIVER_PKG" ]; then
        [ -x "$DRIVER_PKG" ] && green_echo "驱动包已存在可执行权限，跳过" || { chmod +x "$DRIVER_PKG" && green_echo "驱动包赋权成功" || { red_echo "驱动包赋权失败" && exit 14; }; }
    else
        red_echo "驱动包文件不存在：$DRIVER_PKG" && exit 16
    fi
    # 固件包权限处理
    if [ -f "$FIRMWARE_PKG" ]; then
        [ -x "$FIRMWARE_PKG" ] && green_echo "固件包已存在可执行权限，跳过" || { chmod +x "$FIRMWARE_PKG" && green_echo "固件包赋权成功" || { red_echo "固件包赋权失败" && exit 15; }; }
    else
        red_echo "固件包文件不存在：$FIRMWARE_PKG" && exit 17
    fi
}

# 调用Python脚本做包最终校验（已完成赋权）
python_check() {
    [ ! -f $PY_CHECK_SCRIPT ] && red_echo "Python校验脚本不存在：$PY_CHECK_SCRIPT，请放在同目录" && exit 1
    PY_OUTPUT=$(python3 $PY_CHECK_SCRIPT $NPU_PKG_FOLDER)
    PYC_EXIT_CODE=$?
    case $PYC_EXIT_CODE in
        0) green_echo "Python脚本包校验通过" ;;
        1|2|3|4|5|7) red_echo "Python包校验失败，退出码：$PYC_EXIT_CODE" && exit $PYC_EXIT_CODE ;;
        *) red_echo "Python脚本执行异常，退出码：$PYC_EXIT_CODE" && exit 8 ;;
    esac
}

# 创建驱动运行用户/组（无则新建）
create_user() {
    green_echo "检查并创建运行用户/组：$RUN_USER/$RUN_GROUP"
    if ! id "$RUN_USER" &>/dev/null; then
        getent group $RUN_GROUP &> /dev/null || { groupadd $RUN_GROUP && green_echo "用户组$RUN_GROUP创建成功"; }
        getent passwd $RUN_USER &> /dev/null || { useradd -g $RUN_GROUP -s /sbin/nologin $RUN_USER && green_echo "运行用户$RUN_USER创建成功"; }
    fi
}

# 单个系统依赖检查
check_dep() {
    if [ -x "$(command -v yum)" ]; then
        rpm -q $1 &> /dev/null
    elif [ -x "$(command -v apt)" ]; then
        dpkg -s $1 &> /dev/null
    else
        return 1
    fi
}

# 系统依赖安装：先验后装，仅安装缺失依赖
install_deps() {
    green_echo "开始检查系统基础依赖"
    local missing_deps=()
    # 分系统定义依赖列表
    if [ -x "$(command -v yum)" ]; then
        deps=("gcc" "make" "dkms" "libstdc++-devel" "glibc-devel")
    elif [ -x "$(command -v apt)" ]; then
        deps=("gcc" "make" "dkms" "libstdc++6" "libc6-dev")
    else
        yellow_echo "未检测到yum/apt包管理器，跳过自动依赖安装" && return
    fi
    # 收集缺失依赖
    for dep in "${deps[@]}"; do
        check_dep $dep || { yellow_echo "依赖$dep缺失，将安装"; missing_deps+=($dep); }
    done
    # 安装缺失依赖并二次校验
    if [ ${#missing_deps[@]} -gt 0 ]; then
        [ -x "$(command -v yum)" ] && yum install -y "${missing_deps[@]}" &> /dev/null
        [ -x "$(command -v apt)" ] && apt update -y &> /dev/null && apt install -y "${missing_deps[@]}" &> /dev/null
        for dep in "${missing_deps[@]}"; do
            check_dep $dep || { red_echo "依赖$dep安装失败，请手动安装" && exit 13; }
        done
    fi
    green_echo "系统依赖检查/安装完成"
}

# 安装NPU驱动+固件（严格遵循先驱后固顺序）
install_npu() {
    green_echo "开始安装NPU驱动"
    $DRIVER_PKG --full --install-for-all || { red_echo "NPU驱动安装失败" && exit 10; }
    green_echo "NPU驱动安装完成，开始安装固件"
    $FIRMWARE_PKG --full || { red_echo "NPU固件安装失败" && exit 11; }
    green_echo "NPU驱动+固件安装完成"
}

# 重启确认 + 驱动加载状态验证
verify_npu() {
    read -p "是否立即重启系统（驱动固件生效需重启）？(y/n，默认n): " REBOOT_CHOICE
    REBOOT_CHOICE=${REBOOT_CHOICE:-n}
    [ "$REBOOT_CHOICE" == "y" ] && { green_echo "系统将立即重启..."; reboot; } || yellow_echo "已跳过重启，后续驱动失效请手动执行reboot"
    
    green_echo "开始验证NPU驱动加载状态"
    command -v npu-smi &> /dev/null || { red_echo "未找到npu-smi命令，驱动可能安装失败" && exit 12; }
    echo -e "======================================"
    npu-smi info
    echo -e "======================================"
    green_echo "若上方输出NPU相关信息，说明驱动加载成功；失败请前往昇腾社区求助"
}

# 主执行流程
main() {
    echo -e "========== 昇腾NPU驱动固件安装脚本 ==========\n"
    check_base
    extract_pkgs
    add_pkg_perm
    python_check
    create_user
    install_deps
    install_npu
    verify_npu
    echo -e "\n========== 安装流程全部执行完毕 =========="
}

# 启动主程序
main
