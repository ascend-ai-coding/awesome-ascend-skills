#!/bin/bash
# Auto-detect and setup CANN environment
# Usage: source ./setup_env.sh OR ./setup_env.sh

echo "=== ATC Environment Auto-Setup ==="
echo

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
    fi
    
    echo "$cann_path|$cann_version"
}

# Setup environment based on version
setup_environment() {
    local result=$(detect_cann_version)
    local cann_path=$(echo "$result" | cut -d'|' -f1)
    local cann_version=$(echo "$result" | cut -d'|' -f2)
    
    if [ -z "$cann_path" ]; then
        echo "✗ CANN installation not found!"
        echo "  Please install CANN Toolkit first."
        return 1
    fi
    
    echo "✓ CANN installation found:"
    echo "  Path: $cann_path"
    echo "  Version: ${cann_version:-Unknown}"
    echo
    
    # Source the environment
    if [ -f "$cann_path/set_env.sh" ]; then
        echo "→ Sourcing environment from: $cann_path/set_env.sh"
        source "$cann_path/set_env.sh"
        echo "✓ Environment setup complete"
    else
        echo "✗ Environment setup script not found at: $cann_path/set_env.sh"
        return 1
    fi
    
    # Version-specific additional setup
    if [[ "$cann_path" == *"cann"* ]] && [[ "$cann_version" =~ ^8\.[5-9] ]]; then
        echo
        echo "ℹ CANN 8.5.0+ detected. Additional setup:"
        
        # Check for ops package requirement
        if [ ! -d "$cann_path/opp" ]; then
            echo "  ⚠ Warning: Ops package (opp) not found!"
            echo "    For CANN 8.5.0+, you must install the matching ops package."
        else
            echo "  ✓ Ops package found"
        fi
        
        # Set LD_LIBRARY_PATH for non-Ascend hosts
        if ! command -v npu-smi &>/dev/null; then
            local arch=$(uname -m)
            export LD_LIBRARY_PATH="$cann_path/${arch}-linux/devlib:$LD_LIBRARY_PATH"
            echo "  ✓ Set LD_LIBRARY_PATH for non-Ascend host development"
        fi
    fi
    
    return 0
}

# Print usage
print_usage() {
    echo "Usage:"
    echo "  source $0    # To setup environment in current shell"
    echo "  $0           # To check environment without modifying current shell"
    echo
    echo "This script will:"
    echo "  1. Detect your CANN installation (8.3.RC1 or 8.5.0+)"
    echo "  2. Source the appropriate environment"
    echo "  3. Perform version-specific setup"
}

# Main
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed, not sourced
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        print_usage
        exit 0
    fi
    
    echo "Note: Running script directly. Environment changes will not persist."
    echo "For persistent changes, run: source $0"
    echo
    setup_environment
else
    # Script is being sourced
    setup_environment
fi
