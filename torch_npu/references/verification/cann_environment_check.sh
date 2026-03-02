#!/bin/bash
# CANN Environment Detection and Configuration Script
# CANN 环境检测与配置脚本
# 
# This script detects the installed CANN version and provides guidance for environment configuration.
# 本脚本检测已安装的 CANN 版本并提供环境配置指导。

set -e

# Function to detect CANN version
detect_cann_version() {
    local version="unknown"
    local path=""
    
    # Check for 8.5.0+ path (new structure)
    if [ -f "/usr/local/Ascend/cann/latest/version.cfg" ]; then
        path="/usr/local/Ascend/cann"
        version=$(cat "$path/latest/version.cfg" 2>/dev/null | grep "Version=" | cut -d'=' -f2)
    # Check for 8.3.RC1 or earlier path (legacy structure)
    elif [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
        path="/usr/local/Ascend/ascend-toolkit"
        version=$(cat "$path/latest/version.cfg" 1>/dev/null | grep "Version=" | cut -d'=' -f2)
    fi
    
    echo "$version|$path"
}

# Main execution
echo "============================================================"
echo "CANN Version Detection"
echo "============================================================"
echo ""

VERSION_INFO=$(detect_cann_version)
CANN_VERSION=$(echo "$VERSION_INFO" | cut -d'|' -f1)
CANN_PATH=$(echo "$VERSION_INFO" | cut -d'|' -f2)

if [ -n "$CANN_PATH" ]; then
    echo "✓ CANN installation found"
    echo "  Path: $CANN_PATH"
    echo "  Version: ${CANN_VERSION:-Unknown}"
    echo ""
    
    # Version-specific guidance
    if [[ "$CANN_PATH" == *"ascend-toolkit"* ]]; then
        echo "  Type: CANN 8.3.RC1 or earlier (legacy structure)"
        echo "  Setup command: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo ""
        echo "  To configure environment, run:"
        echo "    source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo ""
    elif [[ "$CANN_PATH" == *"cann"* ]]; then
        echo "  Type: CANN 8.5.0+ (new structure)"
        echo "  Setup command: source /usr/local/Ascend/cann/set_env.sh"
        echo ""
        
        # Check for ops package
        if [ -d "$CANN_PATH/opp" ]; then
            echo "  ✓ Ops package (opp) found"
        else
            echo "  ✗ Ops package (opp) NOT found - REQUIRED for 8.5.0+"
            echo "    Please install the ops package before using torch_npu"
        fi
        echo ""
        echo "  To configure environment, run:"
        echo "    source /usr/local/Ascend/cann/set_env.sh"
        echo ""
    fi
    
    echo "============================================================"
    echo "Recommended Actions"
    echo "============================================================"
    echo ""
    echo "1. Source the environment:"
    echo "   $ source $CANN_PATH/set_env.sh"
    echo ""
    echo "2. Verify NPU availability:"
    echo "   $ python3 -c \"import torch; import torch_npu; print(torch.npu.is_available())\""
    echo ""
    echo "3. Check device count:"
    echo "   $ python3 -c \"import torch; import torch_npu; print(torch.npu.device_count())\""
    echo ""
else
    echo "✗ CANN installation not detected"
    echo ""
    echo "  Checked paths:"
    echo "    /usr/local/Ascend/cann/latest/version.cfg (CANN 8.5.0+)"
    echo "    /usr/local/Ascend/ascend-toolkit/latest/version.cfg (CANN 8.3.RC1 or earlier)"
    echo ""
    echo "  Please install CANN from:"
    echo "    https://www.hiascend.com/document"
    echo ""
fi

echo "============================================================"
echo "Additional Information"
echo "============================================================"
echo ""

# Check for NNAE installation (alternative CANN installation)
if [ -d "/usr/local/Ascend/nnae" ]; then
    echo "NNAE installation found at /usr/local/Ascend/nnae"
    echo "  This is an alternative CANN installation method."
    echo "  Check: ls -la /usr/local/Ascend/nnae/"
    echo ""
fi

# Check environment variables
echo "Current environment variables:"
echo "  ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-Not set}"
echo "  ASCEND_OPP_PATH: ${ASCEND_OPP_PATH:-Not set}"
echo ""

# Provide troubleshooting tips
echo "============================================================"
echo "Troubleshooting Tips"
echo "============================================================"
echo ""
echo "If torch_npu operators fail with error code 561000 or 561103:"
echo ""
echo "1. Make sure you have sourced the correct set_env.sh"
echo "2. Check CANN version compatibility with torch_npu version"
echo "3. Verify the opp package is installed (for CANN 8.5.0+)"
echo "4. Try using a Docker container with pre-configured environment"
echo ""
echo "For more details, see:"
echo "  - references/installation/version_compatibility.md"
echo "  - references/verification/verification_guide.md"
echo ""
