#!/bin/bash
# msModelSlim Environment Check Script
# Usage: ./check_env.sh

set -e

echo "=========================================="
echo "  msModelSlim Environment Check"
echo "=========================================="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}[✓]${NC} $1"
}

check_fail() {
    echo -e "${RED}[✗]${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# 1. Check Python version
echo "--- Python Version ---"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    check_pass "Python version: $PYTHON_VERSION"
else
    check_fail "Python version too old: $PYTHON_VERSION (need 3.8+)"
fi
echo

# 2. Check CANN environment
echo "--- CANN Environment ---"
if [ -n "$ASCEND_HOME" ]; then
    check_pass "ASCEND_HOME: $ASCEND_HOME"
elif [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
    check_warn "CANN found at /usr/local/Ascend/ascend-toolkit but ASCEND_HOME not set"
    echo "    Run: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
elif [ -d "/usr/local/Ascend/cann" ]; then
    check_warn "CANN found at /usr/local/Ascend/cann but ASCEND_HOME not set"
    echo "    Run: source /usr/local/Ascend/cann/set_env.sh"
else
    check_fail "CANN not found"
fi
echo

# 3. Check PyTorch and torch-npu
echo "--- PyTorch ---"
if python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    check_pass "PyTorch installed"
else
    check_fail "PyTorch not installed"
fi

if python3 -c "import torch_npu; print(f'torch-npu: {torch_npu.__version__}')" 2>/dev/null; then
    check_pass "torch-npu installed"
else
    check_warn "torch-npu not installed (required for NPU)"
fi
echo

# 4. Check msModelSlim
echo "--- msModelSlim ---"
if python3 -c "import msmodelslim; print(f'msModelSlim: {msmodelslim.__version__}')" 2>/dev/null; then
    check_pass "msModelSlim installed"
else
    check_fail "msModelSlim not installed"
    echo "    Install with: git clone https://gitcode.com/Ascend/msmodelslim.git && cd msmodelslim && bash install.sh"
fi

if command -v msmodelslim &> /dev/null; then
    check_pass "msmodelslim CLI available"
else
    check_warn "msmodelslim CLI not in PATH"
fi
echo

# 5. Check NPU devices
echo "--- NPU Devices ---"
if command -v npu-smi &> /dev/null; then
    NPU_COUNT=$(npu-smi info -l 2>/dev/null | grep -c "Total [0-9]* devices" || echo "0")
    if [ "$NPU_COUNT" -gt 0 ]; then
        check_pass "NPU devices found"
        npu-smi info -l 2>/dev/null | head -20
    else
        check_warn "No NPU devices detected"
    fi
else
    check_warn "npu-smi not available"
fi
echo

# 6. Check environment variables
echo "--- Environment Variables ---"
[ -n "$ASCEND_RT_VISIBLE_DEVICES" ] && check_pass "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" || check_warn "ASCEND_RT_VISIBLE_DEVICES not set"
[ -n "$PYTORCH_NPU_ALLOC_CONF" ] && check_pass "PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF" || check_warn "PYTORCH_NPU_ALLOC_CONF not set"
echo

# 7. Summary
echo "=========================================="
echo "  Environment Check Complete"
echo "=========================================="
echo
echo "If all checks passed, you're ready to use msModelSlim!"
echo
echo "Quick Start:"
echo "  msmodelslim quant --model_path /path/to/model --save_path /path/to/output \\"
echo "      --device npu --model_type Qwen2.5-7B-Instruct --quant_type w8a8"
echo
