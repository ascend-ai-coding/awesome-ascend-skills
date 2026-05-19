#!/usr/bin/env bash
# =============================================================================
# ATB CSV Tester - 运行脚本
# 用于在 Docker 容器内执行 ATB CSV 测试
# =============================================================================
set -eo pipefail

# -----------------------------------------------------------------------------
# 帮助信息
# -----------------------------------------------------------------------------
show_help() {
    cat << 'EOF'
ATB CSV Tester - 运行 ATB CSV 测试用例

用法:
    bash run-csv-tests.sh [OPTIONS]

必需参数:
    --atb-repo-path PATH      ATB 源码仓库根目录
    --csv-file PATH           CSV 测试文件路径

可选参数:
    --case-range RANGE        测试用例范围，格式: 起始:结束 (如 1:5)
    --cann-path PATH          CANN 安装路径 (默认从容器内自动检测)
    --docker-name NAME        Docker 容器名称 (**必填**)
    --dry-run                 仅显示命令，不实际执行
    -h, --help                显示此帮助信息

示例:
    # 运行全部 linear.csv 测试
    bash run-csv-tests.sh \
        --atb-repo-path /path/to/ascend-transformer-boost \
        --csv-file /path/to/ascend-transformer-boost/tests/apitest/opstest/csv/linear.csv

    # 运行前 5 个测试用例
    bash run-csv-tests.sh \
        --atb-repo-path /path/to/ascend-transformer-boost \
        --csv-file /path/to/ascend-transformer-boost/tests/apitest/opstest/csv/linear.csv \
        --case-range 1:5

    # 在指定 Docker 容器中运行
    bash run-csv-tests.sh \
        --atb-repo-path /path/to/ascend-transformer-boost \
        --csv-file /path/to/ascend-transformer-boost/tests/apitest/opstest/csv/linear.csv \
        --docker-name my-container
EOF
}

# -----------------------------------------------------------------------------
# 日志函数
# -----------------------------------------------------------------------------
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $*"
}

log_ok() {
    echo -e "\033[0;32m[OK]\033[0m $*"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $*"
}

log_err() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
}

# -----------------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------------
ATB_REPO_PATH=""
CSV_FILE_PATH=""
CASE_RANGE=""
CANN_PATH=""
DOCKER_NAME="<YOUR_DOCKER_NAME>"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --atb-repo-path)
            ATB_REPO_PATH="$2"; shift 2 ;;
        --csv-file)
            CSV_FILE_PATH="$2"; shift 2 ;;
        --case-range)
            CASE_RANGE="$2"; shift 2 ;;
        --cann-path)
            CANN_PATH="$2"; shift 2 ;;
        --docker-name)
            DOCKER_NAME="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            show_help; exit 0 ;;
        *)
            log_err "未知参数: $1"; show_help; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# 参数校验
# -----------------------------------------------------------------------------
if [[ -z "$ATB_REPO_PATH" ]]; then
    log_err "错误: --atb-repo-path 为必填参数"
    show_help
    exit 1
fi

if [[ -z "$CSV_FILE_PATH" ]]; then
    log_err "错误: --csv-file 为必填参数"
    show_help
    exit 1
fi

# 规范化路径（去除末尾斜杠）
ATB_REPO_PATH="${ATB_REPO_PATH%/}"
CSV_FILE_PATH="${CSV_FILE_PATH%/}"

# -----------------------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------------------
log_info "=== ATB CSV Tester 启动 ==="

# 检查 Docker 是否运行
if ! docker ps &> /dev/null; then
    log_err "错误: Docker 未运行或当前用户无权限访问 Docker"
    exit 1
fi

# 检查容器是否运行
if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_NAME}$"; then
    log_err "错误: Docker 容器 '${DOCKER_NAME}' 未运行"
    log_info "请先启动容器: docker start ${DOCKER_NAME}"
    exit 1
fi
log_ok "Docker 容器 '${DOCKER_NAME}' 运行正常"

# 检查 ATB 仓库路径（容器外）
if [[ ! -d "$ATB_REPO_PATH" ]]; then
    log_err "错误: ATB 仓库目录不存在: $ATB_REPO_PATH"
    log_err "请通过 --atb-repo-path 参数指定 ATB 仓库路径"
    exit 1
fi

# 检查 ATB_OUTPUT 是否存在（编译产物）
if [[ ! -f "${ATB_REPO_PATH}/output/atb/set_env.sh" ]]; then
    log_err "错误: ATB 编译产物不存在，请先编译测试框架"
    log_err "ATB 仓库: $ATB_REPO_PATH"
    log_err "编译命令: cd ${ATB_REPO_PATH} && bash scripts/build.sh testframework"
    exit 1
fi
log_ok "ATB 仓库目录存在: $ATB_REPO_PATH"

# 检查 CSV 文件（容器外）
if [[ ! -f "$CSV_FILE_PATH" ]]; then
    log_err "错误: CSV 文件不存在: $CSV_FILE_PATH"
    exit 1
fi
log_ok "CSV 测试文件存在: $CSV_FILE_PATH"

# 检查 CSV 文件格式（至少检查分隔符和列数）
FIRST_LINE=$(head -1 "$CSV_FILE_PATH")
COLUMN_COUNT=$(echo "$FIRST_LINE" | tr '|' '\n' | wc -l)
if [[ "$COLUMN_COUNT" -lt 10 ]]; then
    log_err "错误: CSV 文件格式异常，列数不足: $COLUMN_COUNT"
    exit 1
fi
log_ok "CSV 文件格式正确，列数: $COLUMN_COUNT"

# -----------------------------------------------------------------------------
# 获取 ATB_HOME_PATH 并检查 libatb_test_framework.so
# -----------------------------------------------------------------------------
log_info "正在获取 ATB_HOME_PATH 并检查测试框架..."

ATB_HOME_PATH=$(docker exec "$DOCKER_NAME" bash -c \
    "source ${ATB_REPO_PATH}/output/atb/set_env.sh 2>/dev/null && echo \$ATB_HOME_PATH" 2>/dev/null || echo "")

if [[ -z "$ATB_HOME_PATH" ]]; then
    log_warn "未能获取 ATB_HOME_PATH，libatb_test_framework.so 可能不存在"
    log_info "请先编译测试框架: cd ${ATB_REPO_PATH} && bash scripts/build.sh testframework"
    exit 1
fi
log_ok "检测到 ATB_HOME_PATH: $ATB_HOME_PATH"

# 检查 libatb_test_framework.so 是否存在
TEST_FRAMEWORK_LIB="${ATB_HOME_PATH}/lib/libatb_test_framework.so"
if ! docker exec "$DOCKER_NAME" test -f "$TEST_FRAMEWORK_LIB" 2>/dev/null; then
    log_warn "libatb_test_framework.so 不存在于: $TEST_FRAMEWORK_LIB"
    log_info "请先编译测试框架: cd ${ATB_REPO_PATH} && bash scripts/build.sh testframework"
    exit 1
fi
log_ok "libatb_test_framework.so 存在: $TEST_FRAMEWORK_LIB"

# -----------------------------------------------------------------------------
# 获取 CANN 路径（从容器内推断）
# -----------------------------------------------------------------------------
log_info "正在获取 CANN 路径..."

DETECTED_CANN_PATH=$(docker exec "$DOCKER_NAME" bash -c \
    'if [[ -n "$ASCEND_TOOLKIT_HOME" ]]; then
         echo "$ASCEND_TOOLKIT_HOME"
     elif [[ -n "$ASCEND_HOME_PATH" ]]; then
         echo "$ASCEND_HOME_PATH"
     else
         echo ""
     fi' 2>/dev/null || echo "")

if [[ -n "$CANN_PATH" ]]; then
    : # 用户已指定，使用指定的
elif [[ -n "$DETECTED_CANN_PATH" ]]; then
    CANN_PATH="$DETECTED_CANN_PATH"
else
    # 尝试常见路径，优先检查容器内是否存在
    # 优先使用环境变量 $ASCEND_TOOLKIT_HOME 推断路径
    for p in \
        "$ASCEND_TOOLKIT_HOME" \
        /usr/local/Ascend/ascend-toolkit/latest \
        /home/ascend/ascend-toolkit/latest; do
        if docker exec "$DOCKER_NAME" test -f "${p}/set_env.sh" 2>/dev/null; then
            CANN_PATH="$p"
            break
        fi
    done
fi

# 检查 CANN 路径是否存在 set_env.sh
if [[ -n "$CANN_PATH" ]]; then
    if docker exec "$DOCKER_NAME" test -f "${CANN_PATH}/set_env.sh" 2>/dev/null; then
        log_ok "检测到 CANN 路径: $CANN_PATH"
    else
        # 尝试在子目录中查找 set_env.sh（如 cann-9.0.0-beta.2/）
        CANN_SET_ENV=$(docker exec "$DOCKER_NAME" bash -c "find '${CANN_PATH}' -maxdepth 2 -name 'set_env.sh' 2>/dev/null | head -1" 2>/dev/null || echo "")
        if [[ -n "$CANN_SET_ENV" && -f "$CANN_SET_ENV" ]]; then
            CANN_PATH=$(dirname "$CANN_SET_ENV")
            log_ok "检测到 CANN 路径: $CANN_PATH (from subdirectory)"
        else
            log_err "错误: 指定的 CANN 路径不存在 set_env.sh: ${CANN_PATH}"
            log_err "请通过 --cann-path 参数指定正确的 CANN 路径（需包含 set_env.sh）"
            exit 1
        fi
    fi
else
    log_err "错误: 未能从环境变量中检测到 CANN 路径"
    log_err "请通过 --cann-path 参数指定 CANN 安装路径（需包含 set_env.sh）"
    exit 1
fi

# -----------------------------------------------------------------------------
# 构建并显示测试命令
# -----------------------------------------------------------------------------
CSV_TOOL_DIR="${ATB_REPO_PATH}/tests/framework/python/CsvOpsTestTool"

# 构造容器内执行的完整命令（单行形式，避免多命令拼接问题）
if [[ -n "$CANN_PATH" ]]; then
    EXEC_CMD="source ${CANN_PATH}/set_env.sh && source ${ATB_REPO_PATH}/output/atb/set_env.sh && cd ${CSV_TOOL_DIR} && python3 atb_csv_ops_test.py -i ${CSV_FILE_PATH}"
else
    EXEC_CMD="source ${ATB_REPO_PATH}/output/atb/set_env.sh && cd ${CSV_TOOL_DIR} && python3 atb_csv_ops_test.py -i ${CSV_FILE_PATH}"
fi

if [[ -n "$CASE_RANGE" ]]; then
    EXEC_CMD="${EXEC_CMD} -n ${CASE_RANGE}"
fi

echo ""
log_info "=== 执行 CSV 测试 ==="
log_info "ATB 仓库: $ATB_REPO_PATH"
log_info "CSV 文件: $CSV_FILE_PATH"
if [[ -n "$CASE_RANGE" ]]; then
    log_info "测试范围: $CASE_RANGE"
fi
echo ""

if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] 将在容器中执行以下命令:"
    echo ""
    echo "    docker exec ${DOCKER_NAME} bash -c \"${EXEC_CMD}\""
    echo ""
    exit 0
fi

# -----------------------------------------------------------------------------
# 执行测试
# -----------------------------------------------------------------------------
log_info "开始执行测试..."
echo ""

START_TIME=$(date +%s)

docker exec "$DOCKER_NAME" bash -c "$EXEC_CMD"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
log_ok "=== CSV 测试执行完成 ==="
log_info "耗时: ${DURATION} 秒"
