#!/bin/bash
# 环境检查脚本
# 用途：检查 Ascend C 算子运行环境是否正确配置

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================"
echo "Ascend C 环境检查"
echo "================================"
echo ""

ERRORS=0
WARNINGS=0

# 1. 检查 CANN Toolkit 环境
echo -e "${YELLOW}[1/6] 检查 CANN Toolkit 环境...${NC}"
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo -e "${RED}✗ ASCEND_HOME_PATH 未设置${NC}"
    echo "  官方配置方法："
    echo "    # root 用户默认路径"
    echo "    source /usr/local/Ascend/cann/set_env.sh"
    echo "    # 非root用户默认路径"
    echo "    source \$HOME/Ascend/cann/set_env.sh"
    echo "    # 指定路径安装"
    echo "    source \${install_path}/cann/set_env.sh"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ ASCEND_HOME_PATH = $ASCEND_HOME_PATH${NC}"
    
    # 验证 set_env.sh 是否存在
    if [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
        echo -e "${GREEN}  ✓ CANN Toolkit set_env.sh 存在${NC}"
    else
        echo -e "${YELLOW}  ⚠ set_env.sh 不存在，环境可能未正确配置${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo ""

# 2. 检查 CANN Ops 环境（运行态依赖）
echo -e "${YELLOW}[2/6] 检查 CANN Ops 环境（运行态依赖）...${NC}"
if [ -z "$ASCEND_OPP_PATH" ]; then
    echo -e "${YELLOW}⚠ ASCEND_OPP_PATH 未设置${NC}"
    echo "  说明："
    echo "    - 编译算子时：可以跳过（不影响编译）"
    echo "    - 运行算子时：必需（需安装 CANN Ops 包）"
    echo ""
    echo "  解决方法："
    echo "    1. Docker 环境：镜像已包含 CANN Ops，检查是否 source set_env.sh"
    echo "    2. 手动安装："
    echo "       ./Ascend-cann-\${soc_name}-ops_*.run --install --install-path=\${install_path}"
    echo "       source \${install_path}/cann/set_env.sh"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓ ASCEND_OPP_PATH = $ASCEND_OPP_PATH${NC}"
    
    # 验证 vendors 目录
    if [ -d "$ASCEND_OPP_PATH/vendors" ]; then
        vendor_count=$(find "$ASCEND_OPP_PATH/vendors" -maxdepth 1 -type d | tail -n +2 | wc -l)
        echo -e "${GREEN}  ✓ CANN Ops 已安装（$vendor_count 个 vendors）${NC}"
    else
        echo -e "${YELLOW}  ⚠ vendors 目录不存在，CANN Ops 可能未正确安装${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo ""

# 3. 检查自定义算子包（可选）
echo -e "${YELLOW}[3/6] 检查自定义算子包...${NC}"
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo -e "${YELLOW}⚠ 跳过检查（ASCEND_HOME_PATH 未设置）${NC}"
else
    VENDOR_DIRS=$(find "$ASCEND_HOME_PATH/opp/vendors" -maxdepth 1 -type d 2>/dev/null | tail -n +2 || true)
    
    if [ -n "$VENDOR_DIRS" ]; then
        found_custom=0
        for vendor_dir in $VENDOR_DIRS; do
            vendor_name=$(basename "$vendor_dir")
            # 检查多种可能的算子库路径
            op_api_lib=""
            if [ -d "$vendor_dir/op_api/lib" ]; then
                op_api_lib="$vendor_dir/op_api/lib"
            elif [ -d "$vendor_dir/op_impl/ai_core/tbe/op_api/lib" ]; then
                op_api_lib="$vendor_dir/op_impl/ai_core/tbe/op_api/lib"
            fi
            
            if [ -n "$op_api_lib" ] && [ -d "$op_api_lib" ]; then
                so_count=$(find "$op_api_lib" -name "*.so" 2>/dev/null | wc -l)
                echo -e "${GREEN}✓ $vendor_name: $so_count 个算子已安装${NC}"
                
                # 检查 LD_LIBRARY_PATH 是否包含该路径
                if ! echo "$LD_LIBRARY_PATH" | grep -q "vendors/${vendor_name}/"; then
                    echo -e "${YELLOW}  ⚠ LD_LIBRARY_PATH 未配置${NC}"
                    echo "    建议：export LD_LIBRARY_PATH=$op_api_lib:\$LD_LIBRARY_PATH"
                    WARNINGS=$((WARNINGS + 1))
                fi
                found_custom=1
            fi
        done
        
        if [ $found_custom -eq 0 ]; then
            echo -e "${YELLOW}⚠ vendors 目录存在但未找到算子库${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 未安装自定义算子包${NC}"
        echo "  说明：仅运行自定义算子时需要"
    fi
fi

echo ""

# 4. 检查 CANN 工具
echo -e "${YELLOW}[4/6] 检查 CANN 工具...${NC}"
if command -v msprof &> /dev/null; then
    echo -e "${GREEN}✓ msprof 可用${NC}"
else
    echo -e "${YELLOW}⚠ msprof 不可用${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

if command -v cannsim &> /dev/null; then
    echo -e "${GREEN}✓ cannsim 可用${NC}"
else
    echo -e "${YELLOW}⚠ cannsim 不可用（仅 ascend950 需要）${NC}"
fi

echo ""

# 5. 检查日志目录
echo -e "${YELLOW}[5/6] 检查日志目录...${NC}"
LOG_DIR="$HOME/ascend/log/debug/plog"
if [ -d "$LOG_DIR" ]; then
    log_count=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ 日志目录存在: $log_count 个日志文件${NC}"
else
    echo -e "${YELLOW}⚠ 日志目录不存在: $LOG_DIR${NC}"
    echo "  日志将在首次运行后创建"
fi

echo ""

# 6. 检查环境变量配置
echo -e "${YELLOW}[6/6] 检查调试配置...${NC}"
if [ "$ASCEND_SLOG_PRINT_TO_STDOUT" = "1" ]; then
    echo -e "${GREEN}✓ 日志打屏已开启 (ASCEND_SLOG_PRINT_TO_STDOUT=1)${NC}"
else
    echo -e "${YELLOW}⚠ 日志打屏未开启${NC}"
    echo "  建议：export ASCEND_SLOG_PRINT_TO_STDOUT=1"
fi

echo ""
echo "================================"
echo "检查结果"
echo "================================"
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ 发现 $ERRORS 个错误${NC}"
    echo "  请先修复错误再运行算子"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ 发现 $WARNINGS 个警告${NC}"
    echo "  建议修复警告以提高稳定性"
    exit 0
else
    echo -e "${GREEN}✓ 环境检查通过${NC}"
    exit 0
fi
