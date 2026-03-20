#!/bin/bash
# Enhanced environment check with compatibility validation
# Usage: ./check_env_enhanced.sh

echo "=== ATC Environment Compatibility Check ==="
echo "=========================================="
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0
WARNINGS=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    elif [ $1 -eq 1 ]; then
        echo -e "${RED}✗${NC} $2"
        FAILED=$((FAILED + 1))
    else
        echo -e "${YELLOW}⚠${NC} $2"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# 1. Check Python version
echo "=== 1. Python Version Check ==="
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ] && [ "$PYTHON_MINOR" -le 10 ]; then
    print_status 0 "Python version is compatible (3.7-3.10 recommended)"
elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    print_status 1 "Python 3.11+ may cause TBE operator compilation failures"
    echo "  Solution: Create Python 3.10 conda environment:"
    echo "    conda create -n atc_py310 python=3.10 -y"
    echo "    conda activate atc_py310"
else
    print_status 1 "Python version $PYTHON_VERSION may not be compatible"
fi
echo

# 2. Check NumPy version
echo "=== 2. NumPy Version Check ==="
if python3 -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1)
    
    echo "NumPy version: $NUMPY_VERSION"
    
    if [ "$NUMPY_MAJOR" -lt 2 ]; then
        print_status 0 "NumPy version is compatible (< 2.0)"
    else
        print_status 2 "NumPy 2.0+ may cause compatibility issues with some CANN versions"
        echo "  If you encounter errors: pip install \"numpy<2.0\" --force-reinstall"
    fi
else
    print_status 1 "NumPy is not installed"
    echo "  Solution: pip install numpy"
fi
echo

# 3. Check required Python modules
echo "=== 3. Required Python Modules Check ==="
REQUIRED_MODULES=("decorator" "attr" "attrs" "absl" "psutil" "google.protobuf" "sympy")
MISSING_MODULES=()

for module in "${REQUIRED_MODULES[@]}"; do
    MODULE_NAME=$(echo $module | cut -d. -f1)
    if python3 -c "import $module" 2>/dev/null; then
        print_status 0 "Module '$MODULE_NAME' is installed"
    else
        print_status 1 "Module '$MODULE_NAME' is missing"
        MISSING_MODULES+=($MODULE_NAME)
    fi
done

if [ ${#MISSING_MODULES[@]} -gt 0 ]; then
    echo
    echo "Install missing modules:"
    echo "  pip install ${MISSING_MODULES[@]}"
fi
echo

# 4. Check CANN installation
echo "=== 4. CANN Installation Check ==="
if [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
    CANN_PATH="/usr/local/Ascend/ascend-toolkit"
    CANN_VERSION=$(cat $CANN_PATH/latest/version.cfg | grep "Version=" | head -1 | cut -d'=' -f2)
    echo "CANN Path: $CANN_PATH"
    echo "CANN Version: $CANN_VERSION"
    print_status 0 "CANN ascend-toolkit found"
elif [ -f "/usr/local/Ascend/cann/latest/version.cfg" ]; then
    CANN_PATH="/usr/local/Ascend/cann"
    CANN_VERSION=$(cat $CANN_PATH/latest/version.cfg | grep "Version=" | head -1 | cut -d'=' -f2)
    echo "CANN Path: $CANN_PATH"
    echo "CANN Version: $CANN_VERSION"
    print_status 0 "CANN installation found"
elif [ -d "/usr/local/Ascend/cann-8.5.0" ] || [ -d "/usr/local/Ascend/cann-8.3.RC1" ]; then
    # CANN 8.5.0+ uses versioned directory without version.cfg at expected path
    CANN_PATH=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1)
    echo "CANN Path: $CANN_PATH (versioned directory)"
    print_status 0 "CANN installation found (versioned layout)"
elif command -v atc &>/dev/null; then
    # atc is available — CANN env was already sourced
    echo "CANN env already sourced (atc is in PATH)"
    print_status 0 "CANN environment active (detected via atc in PATH)"
else
    print_status 1 "CANN installation not found"
    echo "  Checked:"
    echo "    /usr/local/Ascend/ascend-toolkit/latest/version.cfg"
    echo "    /usr/local/Ascend/cann/latest/version.cfg"
    echo "    /usr/local/Ascend/cann-8.5.0/"
fi
echo

# 5. Check ATC tool
echo "=== 5. ATC Tool Check ==="
if command -v atc > /dev/null 2>&1; then
    ATC_PATH=$(which atc)
    print_status 0 "ATC tool found: $ATC_PATH"
    
    # Try to run atc --help
    if atc --help > /dev/null 2>&1 | head -1; then
        print_status 0 "ATC tool is executable"
    else
        print_status 2 "ATC tool found but may not be properly configured"
    fi
else
    print_status 1 "ATC tool not found in PATH"
    echo "  Solution: Source CANN environment"
    echo "    source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "    OR"
    echo "    source /usr/local/Ascend/cann/set_env.sh"
fi
echo

# 6. Check NPU devices
echo "=== 6. NPU Device Check ==="
if command -v npu-smi > /dev/null 2>&1; then
    NPU_COUNT=$(npu-smi info -l 2>/dev/null | grep "NPU ID" | wc -l)
    if [ $NPU_COUNT -gt 0 ]; then
        print_status 0 "Found $NPU_COUNT NPU device(s)"
        npu-smi info -l 2>/dev/null | grep -E "(NPU ID|Name)" | head -10
    else
        print_status 2 "No NPU devices found (this is OK for development hosts)"
    fi
else
    print_status 2 "npu-smi not found (optional, for NPU management)"
fi
echo

# 7. Check environment variables
echo "=== 7. Environment Variables Check ==="
if [ -n "$ASCEND_HOME" ]; then
    print_status 0 "ASCEND_HOME is set: $ASCEND_HOME"
else
    print_status 2 "ASCEND_HOME not set (will be set by set_env.sh)"
fi

if [ -n "$PYTHONPATH" ]; then
    print_status 0 "PYTHONPATH is set"
    echo "  $PYTHONPATH"
else
    print_status 2 "PYTHONPATH not set (may be needed for Conda environments)"
fi
echo

# Summary
echo "=========================================="
echo "=== Summary ==="
echo "=========================================="

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Environment is ready for ATC conversion.${NC}"
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found. Environment should work but review warnings above.${NC}"
else
    echo -e "${RED}✗ $FAILED error(s) and $WARNINGS warning(s) found.${NC}"
    echo
    echo "Please fix the errors above before running ATC."
    echo
    echo "Quick fix for common issues:"
    echo "  1. Python 3.11+ issue: conda create -n atc_py310 python=3.10 -y"
    echo "  2. NumPy 2.0 issue: pip install \"numpy<2.0\" --force-reinstall"
    echo "  3. Missing modules: pip install decorator attrs absl-py psutil protobuf sympy"
    echo "  4. CANN not found: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

echo
