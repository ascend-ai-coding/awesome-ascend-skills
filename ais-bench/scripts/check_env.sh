#!/bin/bash
# AISBench Environment Check Script
# Verifies all prerequisites for running AISBench evaluations

set -e

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

echo "========================================="
echo "AISBench Environment Check"
echo "========================================="
echo ""

# Check Python version
echo "--- Python Version ---"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -ge 10 && "$PYTHON_MINOR" -le 12 ]]; then
    check_pass "Python $PYTHON_VERSION (supported: 3.10, 3.11, 3.12)"
else
    check_fail "Python $PYTHON_VERSION (required: 3.10, 3.11, or 3.12)"
fi

# Check ais_bench installation
echo ""
echo "--- AISBench Installation ---"
if command -v ais_bench &> /dev/null; then
    check_pass "ais_bench CLI available"
    ais_bench --version 2>/dev/null || check_warn "Could not determine version"
else
    check_fail "ais_bench CLI not found. Run: pip3 install -e ./ --use-pep517"
fi

# Check Python package
if python3 -c "import ais_bench" 2>/dev/null; then
    check_pass "ais_bench Python package installed"
else
    check_fail "ais_bench Python package not found"
fi

# Check optional dependencies
echo ""
echo "--- Optional Dependencies ---"

# API dependencies
if python3 -c "import aiohttp" 2>/dev/null; then
    check_pass "aiohttp (API support)"
else
    check_warn "aiohttp not found - install requirements/api.txt for service model evaluation"
fi

# HuggingFace dependencies
if python3 -c "import transformers" 2>/dev/null; then
    check_pass "transformers (HF support)"
else
    check_warn "transformers not found - install requirements/hf_vl_dependency.txt for HF models"
fi

# Check NPU environment (if applicable)
echo ""
echo "--- NPU Environment ---"
if command -v npu-smi &> /dev/null; then
    check_pass "npu-smi available"
    npu-smi info 2>/dev/null | head -20 || check_warn "Could not get NPU info"
else
    check_warn "npu-smi not found - NPU not available or CANN not installed"
fi

# Check dataset directory
echo ""
echo "--- Dataset Directory ---"
if [[ -d "ais_bench/datasets" ]]; then
    check_pass "ais_bench/datasets directory exists"
    DATASET_COUNT=$(find ais_bench/datasets -maxdepth 1 -type d | wc -l)
    echo "    Found $((DATASET_COUNT - 1)) dataset folders"
else
    check_warn "ais_bench/datasets directory not found"
fi

# Check environment variables
echo ""
echo "--- Environment Variables ---"
if [[ -n "$ASCEND_HOME" ]]; then
    check_pass "ASCEND_HOME=$ASCEND_HOME"
else
    check_warn "ASCEND_HOME not set"
fi

if [[ -n "$LD_LIBRARY_PATH" ]]; then
    if [[ "$LD_LIBRARY_PATH" == *"ascend"* ]]; then
        check_pass "LD_LIBRARY_PATH contains Ascend paths"
    else
        check_warn "LD_LIBRARY_PATH does not contain Ascend paths"
    fi
fi

echo ""
echo "========================================="
echo "Environment check complete"
echo "========================================="
