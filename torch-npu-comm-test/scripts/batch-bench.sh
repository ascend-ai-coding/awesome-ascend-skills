#!/bin/bash
set -e
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly COMM_BENCH="${SCRIPT_DIR}/comm-bench.py"

usage() {
    cat <<'EOF'
Usage: batch-bench.sh [OPTIONS]

Run comm-bench.py across multiple ops and shapes in sequence.

Options:
    --npus <N>          NPUs per node (default: 8)
    --ops <ops>         Comma-separated operators (default: "all_reduce")
                        e.g., "all_reduce,all_gather,reduce_scatter"
    --shapes <shapes>   Semicolon-separated shapes, each comma-separated
                        e.g., "4096,4096;4096,12288;32768,4096"
    --dtype <dtype>     Data type: fp32|fp16|bf16|int32 (default: fp16)
    --iters <N>         Measured iterations (default: 50)
    --warmup <N>        Warmup iterations (default: 10)
    --output <fmt>      Output format: table|json (default: table)
    --nnodes <N>        Number of nodes for multi-node (default: 1)
    --master-addr <IP>  Master address for multi-node
    --master-port <P>   Master port for multi-node (default: 29500)
    --node-rank <R>     This node's rank for multi-node (default: 0)
    --check             Enable correctness check
    -h, --help          Show this help

Examples:
    # Single-node, multiple ops and shapes
    ./batch-bench.sh --npus 8 \
        --ops "all_reduce,all_gather,reduce_scatter" \
        --shapes "4096,4096;4096,12288;32768,4096" --dtype fp16

    # Multi-node
    ./batch-bench.sh --npus 8 --nnodes 2 \
        --master-addr 175.99.1.2 --node-rank 0 \
        --ops "all_reduce" --shapes "4096,12288" --dtype bf16
EOF
    exit 0
}

NPUS=8
OPS="all_reduce"
SHAPES="1024,1024"
DTYPE="fp16"
ITERS=50
WARMUP=10
OUTPUT="table"
NNODES=1
MASTER_ADDR=""
MASTER_PORT=29500
NODE_RANK=0
CHECK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npus)       NPUS="$2";        shift 2 ;;
        --ops)        OPS="$2";         shift 2 ;;
        --shapes)     SHAPES="$2";      shift 2 ;;
        --dtype)      DTYPE="$2";       shift 2 ;;
        --iters)      ITERS="$2";       shift 2 ;;
        --warmup)     WARMUP="$2";      shift 2 ;;
        --output)     OUTPUT="$2";      shift 2 ;;
        --nnodes)     NNODES="$2";      shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --node-rank)  NODE_RANK="$2";   shift 2 ;;
        --check)      CHECK="--check";  shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

TORCHRUN_ARGS="--nproc_per_node=${NPUS}"
if [[ ${NNODES} -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR}" ]]; then
        echo "ERROR: --master-addr is required for multi-node testing"
        exit 1
    fi
    TORCHRUN_ARGS+=" --nnodes=${NNODES} --node_rank=${NODE_RANK}"
    TORCHRUN_ARGS+=" --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
fi

IFS=',' read -ra OP_ARRAY <<< "${OPS}"
IFS=';' read -ra SHAPE_ARRAY <<< "${SHAPES}"

TOTAL=$((${#OP_ARRAY[@]} * ${#SHAPE_ARRAY[@]}))
CURRENT=0

echo "============================================================"
echo "Batch Communication Benchmark"
echo "============================================================"
echo "Ops:    ${OPS}"
echo "Shapes: ${SHAPES}"
echo "Dtype:  ${DTYPE}"
echo "NPUs:   ${NPUS} x ${NNODES} node(s)"
echo "Iters:  ${ITERS} (warmup: ${WARMUP})"
echo "Total tests: ${TOTAL}"
echo "============================================================"
echo ""

FAILED=0

for op in "${OP_ARRAY[@]}"; do
    for shape in "${SHAPE_ARRAY[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL}] Testing ${op} with shape [${shape}] ..."

        if ! torchrun ${TORCHRUN_ARGS} "${COMM_BENCH}" \
            --op "${op}" --shape "${shape}" --dtype "${DTYPE}" \
            --iters "${ITERS}" --warmup "${WARMUP}" \
            --output "${OUTPUT}" ${CHECK}; then
            echo "  FAILED: ${op} shape=[${shape}]"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "============================================================"
echo "Batch complete: ${CURRENT} tests, ${FAILED} failed"
echo "============================================================"

if [[ ${FAILED} -gt 0 ]]; then
    exit 1
fi
