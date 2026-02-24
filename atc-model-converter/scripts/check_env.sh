#!/bin/bash
# Check CANN environment and ATC tool availability
# Usage: ./check_env.sh

echo "=== ATC Environment Check ==="
echo

# Detect CANN version
detect_cann_version() {
    local version="unknown"
    local path=""
    
    # Check for 8.5.0+ path
    if [ -f "/usr/local/Ascend/cann/latest/version.cfg" ]; then
        path="/usr/local/Ascend/cann"
        version=$(cat "$path/latest/version.cfg" 2>/dev/null | grep "Version=" | cut -d'=' -f2)
    # Check for 8.3.RC1 path
    elif [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
        path="/usr/local/Ascend/ascend-toolkit"
        version=$(cat "$path/latest/version.cfg" 2>/dev/null | grep "Version=" | cut -d'=' -f2)
    fi
    
    echo "$version|$path"
}

echo "=== CANN Version Detection ==="
VERSION_INFO=$(detect_cann_version)
CANN_VERSION=$(echo "$VERSION_INFO" | cut -d'|' -f1)
CANN_PATH=$(echo "$VERSION_INFO" | cut -d'|' -f2)

if [ -n "$CANN_PATH" ]; then
    echo "✓ CANN installation found"
    echo "  Path: $CANN_PATH"
    echo "  Version: ${CANN_VERSION:-Unknown}"
    
    # Version-specific guidance
    if [[ "$CANN_PATH" == *"ascend-toolkit"* ]]; then
        echo "  Type: CANN 8.3.RC1 or earlier"
        echo "  Setup command: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    elif [[ "$CANN_PATH" == *"cann"* ]]; then
        echo "  Type: CANN 8.5.0+"
        echo "  Setup command: source /usr/local/Ascend/cann/set_env.sh"
        
        # Check for ops package
        if [ -d "$CANN_PATH/opp" ]; then
            echo "  ✓ Ops package (opp) found"
        else
            echo "  ✗ Ops package (opp) NOT found - REQUIRED for 8.5.0+"
        fi
    fi
else
    echo "✗ CANN installation not detected"
    echo "  Checked paths:"
    echo "    /usr/local/Ascend/cann/latest/version.cfg"
    echo "    /usr/local/Ascend/ascend-toolkit/latest/version.cfg"
fi

echo
echo "=== ATC Tool Check ==="

# Check ATC command
if command -v atc &> /dev/null; then
    echo "✓ ATC tool found: $(which atc)"
    atc --help 2>&1 | head -3
else
    echo "✗ ATC tool not found in PATH"
    echo
    echo "  Please source the appropriate CANN environment:"
    echo
    if [[ "$CANN_PATH" == *"ascend-toolkit"* ]]; then
        echo "    CANN 8.3.RC1:"
        echo "      source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    elif [[ "$CANN_PATH" == *"cann"* ]]; then
        echo "    CANN 8.5.0+:"
        echo "      source /usr/local/Ascend/cann/set_env.sh"
    else
        echo "    Try one of:"
        echo "      source /usr/local/Ascend/cann/set_env.sh          # For 8.5.0+"
        echo "      source /usr/local/Ascend/ascend-toolkit/set_env.sh # For 8.3.RC1"
    fi
    echo
    echo "  Or use the auto-setup script:"
    echo "    source ./scripts/setup_env.sh"
    exit 1
fi

echo
echo "=== CANN Environment Variables ==="

# Check common environment variables
if [ -n "$ASCEND_HOME" ]; then
    echo "✓ ASCEND_HOME: $ASCEND_HOME"
else
    echo "✗ ASCEND_HOME not set"
fi

if [ -n "$ASCEND_VERSION" ]; then
    echo "✓ ASCEND_VERSION: $ASCEND_VERSION"
fi

if [ -n "$TE_PARALLEL_COMPILER" ]; then
    echo "ℹ TE_PARALLEL_COMPILER: $TE_PARALLEL_COMPILER"
else
    echo "ℹ TE_PARALLEL_COMPILER not set (default: 8)"
fi

echo
echo "=== NPU Device Check ==="
if command -v npu-smi &> /dev/null; then
    echo "✓ npu-smi found"
    npu-smi info -l 2>/dev/null || echo "  No NPU devices found or permission denied"
    
    # Try to get more device info
    npu-smi info -m 2>/dev/null | head -10 || true
else
    echo "✗ npu-smi not found (optional, for NPU device management)"
fi

echo
echo "=== Python Dependencies ==="
python3 -c "import onnxruntime; print('✓ onnxruntime: ' + onnxruntime.__version__)" 2>/dev/null || echo "✗ onnxruntime not installed (optional)"
python3 -c "import onnx; print('✓ onnx: ' + onnx.__version__)" 2>/dev/null || echo "✗ onnx not installed (optional)"

echo
echo "=== Recommendations ==="

if [ -z "$CANN_PATH" ]; then
    echo "  1. Install CANN Toolkit from https://www.hiascend.com/software/cann/community"
fi

if ! command -v atc &> /dev/null; then
    echo "  2. Source CANN environment: source ./scripts/setup_env.sh"
fi

if [[ "$CANN_PATH" == *"cann"* ]] && [ ! -d "$CANN_PATH/opp" ]; then
    echo "  ⚠ For CANN 8.5.0+, install the matching ops package"
fi

if ! python3 -c "import onnxruntime" 2>/dev/null; then
    echo "  • Install onnxruntime (optional): pip install onnxruntime"
fi

if ! command -v npu-smi &> /dev/null; then
    echo "  • This appears to be a development host without NPU devices"
    echo "    Model conversion will work, but inference requires NPU"
fi

echo
echo "=== Quick Start ==="
echo "  1. Source environment: source ./scripts/setup_env.sh"
echo "  2. Check model: python3 scripts/get_onnx_info.py model.onnx"
echo "  3. Convert: atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3"
