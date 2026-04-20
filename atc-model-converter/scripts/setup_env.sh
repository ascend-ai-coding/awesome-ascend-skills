#!/bin/bash
# Auto-detect and setup CANN environment
# Usage: source ./setup_env.sh OR ./setup_env.sh
#
# This script will:
#   1. Detect your CANN installation (8.3.RC1 or 8.5.0+)
#   2. Source the appropriate environment
#   3. Verify atc command is available
#   4. Display SoC version warning

set -e

echo "=============================================="
echo "  ATC Environment Auto-Setup"
echo "=============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect CANN version and path
detect_cann_version() {
    local cann_path=""
    local cann_version=""
    
    # Check for 8.5.0+ path first
    if [ -f "/usr/local/Ascend/cann/latest/version.cfg" ]; then
        cann_path="/usr/local/Ascend/cann"
        cann_version=$(cat "$cann_path/latest/version.cfg" 2>/dev/null | grep "Version=" | cut -d'=' -f2)
    # Check for 8.3.RC1 path
    elif [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
        cann_path="/usr/local/Ascend/ascend-toolkit"
        cann_version=$(cat "$cann_path/latest/version.cfg" 2>/dev/null | grep "Version=" | cut -d'=' -f2)
    # Try to find atc and infer path
    elif command -v atc &>/dev/null; then
        local atc_path=$(which atc)
        if [[ "$atc_path" == *"cann"* ]]; then
            cann_path="/usr/local/Ascend/cann"
        elif [[ "$atc_path" == *"ascend-toolkit"* ]]; then
            cann_path="/usr/local/Ascend/ascend-toolkit"
        fi
        cann_version="detected via atc"
    fi
    
    echo "$cann_path|$cann_version"
}

# Check NPU device and get SoC version
check_npu_device() {
    echo -e "\n${BLUE}[NPU Device Check]${NC}"
    
    if command -v npu-smi &>/dev/null; then
        local npu_info=$(npu-smi info 2>/dev/null | grep -E "Name|Version" | head -5)
        if [ -n "$npu_info" ]; then
            echo "$npu_info"
            
            # Extract SoC name (e.g. "910B3" from lines like "| 0     910B3  | OK |...")
            local soc_name=$(npu-smi info 2>/dev/null | grep -E "^\|\s+[0-9]+\s+[0-9A-Z]" | awk '{print $3}' | head -1)
            if [ -n "$soc_name" ]; then
                echo -e "\n${YELLOW}⚠️  IMPORTANT: SoC Version Matching${NC}"
                echo "   Your device SoC: Ascend$soc_name"
                echo "   ATC command must use: --soc_version=Ascend$soc_name"
                echo -e "   ${RED}Error example: supported socVersion=Ascend910B3, but the model socVersion=Ascend910B${NC}"
                echo
            fi
        else
            echo -e "${YELLOW}Warning: npu-smi info returned empty${NC}"
        fi
    else
        echo -e "${YELLOW}npu-smi not found - running on non-Ascend host?${NC}"
        echo "For conversion on non-Ascend host, ensure you know the target device's SoC version."
    fi
}

# Setup environment based on version
setup_environment() {
    local result=$(detect_cann_version)
    local cann_path=$(echo "$result" | cut -d'|' -f1)
    local cann_version=$(echo "$result" | cut -d'|' -f2)
    
    if [ -z "$cann_path" ]; then
        echo -e "${RED}✗ CANN installation not found!${NC}"
        echo
        echo "Please install CANN Toolkit first:"
        echo "  Download from: https://www.hiascend.com/software/cann"
        echo
        return 1
    fi
    
    echo -e "${GREEN}✓ CANN installation found:${NC}"
    echo "  Path: $cann_path"
    echo "  Version: ${cann_version:-Unknown}"
    echo
    
    # Source the environment
    if [ -f "$cann_path/set_env.sh" ]; then
        echo -e "${BLUE}→ Sourcing environment from: $cann_path/set_env.sh${NC}"
        source "$cann_path/set_env.sh"
    else
        echo -e "${RED}✗ Environment setup script not found at: $cann_path/set_env.sh${NC}"
        return 1
    fi
    
    # Verify atc is available
    echo
    echo -e "${BLUE}[ATC Verification]${NC}"
    if command -v atc &>/dev/null; then
        local atc_version=$(atc --help 2>&1 | head -1 || echo "unknown")
        echo -e "${GREEN}✓ atc command available${NC}"
        echo "  Version info: $atc_version"
    else
        echo -e "${RED}✗ atc command not found after sourcing environment!${NC}"
        echo "  This may indicate an incomplete CANN installation."
        return 1
    fi
    
    # Version-specific additional setup
    if [[ "$cann_path" == *"cann"* ]] && [[ "$cann_version" =~ ^8\.[5-9] ]]; then
        echo
        echo -e "${BLUE}[CANN 8.5.0+ Additional Setup]${NC}"
        
        # Check for ops package requirement
        if [ ! -d "$cann_path/opp" ]; then
            echo -e "${YELLOW}  ⚠ Warning: Ops package (opp) not found!${NC}"
            echo "    For CANN 8.5.0+, you must install the matching ops package."
        else
            echo -e "${GREEN}  ✓ Ops package found${NC}"
        fi
        
        # Set LD_LIBRARY_PATH for non-Ascend hosts
        if ! command -v npu-smi &>/dev/null; then
            local arch=$(uname -m)
            export LD_LIBRARY_PATH="$cann_path/${arch}-linux/devlib:$LD_LIBRARY_PATH"
            echo -e "${GREEN}  ✓ Set LD_LIBRARY_PATH for non-Ascend host development${NC}"
        fi
    fi
    
    return 0
}

# Print usage
print_usage() {
    echo "Usage:"
    echo "  source $0    # To setup environment in current shell (recommended)"
    echo "  $0           # To check environment without modifying current shell"
    echo
    echo "Options:"
    echo "  -h, --help   Show this help message"
    echo
    echo "This script will:"
    echo "  1. Detect your CANN installation (8.3.RC1 or 8.5.0+)"
    echo "  2. Source the appropriate environment"
    echo "  3. Verify atc command is available"
    echo "  4. Display NPU device info and SoC version warning"
    echo
    echo "Example:"
    echo "  source ./setup_env.sh"
    echo "  atc --model=model.onnx --framework=5 --output=model --soc_version=Ascend910B3"
}

# Main
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed, not sourced
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        print_usage
        exit 0
    fi
    
    echo -e "${YELLOW}Note: Running script directly. Environment changes will not persist.${NC}"
    echo "For persistent changes, run: source $0"
    echo
    setup_environment && check_npu_device
else
    # Script is being sourced
    setup_environment && check_npu_device
fi
