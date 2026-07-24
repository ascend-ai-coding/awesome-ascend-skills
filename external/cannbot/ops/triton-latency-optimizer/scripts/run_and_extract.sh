#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions
# of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
# You may not use this file except in compliance with the License.
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

# NOTE: The following environment setup lines are intentionally commented out.
# When this script is invoked by the latency-optimizer skill, the execution
# environment (conda, Ascend toolkit, NPU device selection, PATH) is already
# activated by the skill harness. Hard-coding these paths makes the script
# fragile across different deployment environments.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$1"

# NOTE: BISHENGIR_COMPILE is expected to be provided by the skill execution
# environment (e.g., via settings.json env vars or pre-activated shell).
# It should resolve to the absolute path of bishengir-compile.
# If undefined, the script will fail at the compile step with a clear error.
if [ -z "$BISHENGIR_COMPILE" ]; then
    # Fallback: try to locate bishengir-compile from PATH
    BISHENGIR_COMPILE=$(command -v bishengir-compile 2>/dev/null || true)
    if [ -z "$BISHENGIR_COMPILE" ]; then
        echo "ERROR: BISHENGIR_COMPILE is not set and bishengir-compile is not found in PATH."
        echo "Please ensure the Ascend Triton compiler environment is activated before running this skill."
        exit 1
    fi
fi

# Detect IR dump flag supported by the compiler version.
# Priority: env override > --mlir-print-ir-after-all > --bishengir-print-ir-after > --print-after-all
PRINT_IR_FLAG=""
if [ -n "$BISHENGIR_PRINT_IR_FLAG" ]; then
    PRINT_IR_FLAG="$BISHENGIR_PRINT_IR_FLAG"
    echo "INFO: Using user-provided IR dump flag: $PRINT_IR_FLAG"
else
    BISHENGIR_HELP=$($BISHENGIR_COMPILE --help 2>&1 || true)
    if echo "$BISHENGIR_HELP" | grep -q "mlir-print-ir-after-all"; then
        PRINT_IR_FLAG="--mlir-print-ir-after-all"
    elif echo "$BISHENGIR_HELP" | grep -q "bishengir-print-ir-after"; then
        PRINT_IR_FLAG="--bishengir-print-ir-after=hivm-inject-sync"
    elif echo "$BISHENGIR_HELP" | grep -q "print-after-all"; then
        PRINT_IR_FLAG="--print-after-all"
    fi
    if [ -n "$PRINT_IR_FLAG" ]; then
        echo "INFO: Auto-detected IR dump flag: $PRINT_IR_FLAG"
    else
        echo "WARNING: No known IR dump flag found in bishengir-compile. last_pass.mlir may not be generated."
    fi
fi

COMPILE_FAILED=0
COMPILE_FAILURES=""
TOTAL_KERNELS=0
PROCESSED_KERNELS=0
VALIDATED_KERNELS=0

IR_SAVE_ALL="${IR_SAVE_ALL:-true}"
IR_SAVE_TTIR="${IR_SAVE_TTIR:-true}"
IR_SAVE_TTADAPTER="${IR_SAVE_TTADAPTER:-true}"

if [ "$IR_SAVE_ALL" = "false" ]; then
    IR_SAVE_TTIR="false"
    IR_SAVE_TTADAPTER="false"
fi

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "Usage: $0 <python_script.py>"
    echo ""
    echo "  python_script.py  : The Triton kernel script to run"
    echo ""
    echo "Example:"
    echo "  $0 softcap.py"
    exit 1
fi

# NOTE: NPU device selection is managed by the skill harness via
# ASCEND_RT_VISIBLE_DEVICES. Do not hard-code a default device here.
# export ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICE:-2}

echo "===== Compile with TRITON_ALWAYS_COMPILE=1 to trigger autotune and dump IR ====="
export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0
export TRITON_DISABLE_FFTS=1

python "$PYTHON_SCRIPT" 2>&1 | tee /tmp/triton_compile_run.log

echo ""
echo "===== Result: Extracting IR dump directories ====="
DUMP_DIRS=$(grep "Dumping intermediate results to" /tmp/triton_compile_run.log | awk '{print $NF}')
DIR_COUNT=$(echo "$DUMP_DIRS" | wc -l)
echo "Found $DIR_COUNT kernel dump directories"

SEEN_KERNELS=""
UNIQ_DIRS=""

for DUMP_DIR in $DUMP_DIRS; do
    if [ ! -d "$DUMP_DIR" ]; then
        continue
    fi

    TTADAPTER_FILE="$DUMP_DIR/kernel.ttadapter.mlir"
    if [ ! -f "$TTADAPTER_FILE" ]; then
        continue
    fi

    KERNEL_NAME=$(grep -oP '(?<=func\.func @)\w+' "$TTADAPTER_FILE" 2>/dev/null | head -1)
    if [ -z "$KERNEL_NAME" ]; then
        KERNEL_NAME=$(grep -oP '(?<=tt\.func @)\w+' "$TTADAPTER_FILE" 2>/dev/null | head -1)
    fi
    if [ -z "$KERNEL_NAME" ]; then
        KERNEL_NAME=$(grep -oP '(?<=module @)\w+' "$TTADAPTER_FILE" 2>/dev/null | head -1)
    fi
    if [ -z "$KERNEL_NAME" ]; then
        KERNEL_NAME="unknown"
    fi

    if echo "$SEEN_KERNELS" | grep -qw "$KERNEL_NAME"; then
        echo "Skipping duplicate kernel '$KERNEL_NAME' in $DUMP_DIR (autotune config variant)"
        continue
    fi

    SEEN_KERNELS="$SEEN_KERNELS $KERNEL_NAME"
    UNIQ_DIRS="$UNIQ_DIRS $DUMP_DIR|$KERNEL_NAME"
done

UNIQ_COUNT=$(echo "$UNIQ_DIRS" | tr ' ' '\n' | grep -c '|' 2>/dev/null || echo 0)
echo "After deduplication: $UNIQ_COUNT unique kernels"
echo "Unique kernels:$(echo $SEEN_KERNELS | tr ' ' '\n' | grep -v '^$' | sed 's/^/ /')"

idx=0

# NOTE: IR_OUTPUT_DIR must be set by the skill harness before invoking this script.
# If not set, default to the skill's own workspace (scripts/../ir_output/) so that
# generated IR files are persisted alongside the skill and not lost in /tmp.
if [ -z "$IR_OUTPUT_DIR" ]; then
    IR_OUTPUT_DIR="$SCRIPT_DIR/../ir_output"
    echo "INFO: IR_OUTPUT_DIR is not set. Using skill workspace: $IR_OUTPUT_DIR"
fi

mkdir -p "$IR_OUTPUT_DIR"

# NOTE: We do NOT delete any existing files in IR_OUTPUT_DIR. Old IR files from
# previous runs are retained; cp will overwrite only when filenames collide.

for ENTRY in $UNIQ_DIRS; do
    DUMP_DIR=$(echo "$ENTRY" | cut -d'|' -f1)
    KERNEL_NAME=$(echo "$ENTRY" | cut -d'|' -f2)
    idx=$((idx + 1))

    TTADAPTER_FILE="$DUMP_DIR/kernel.ttadapter.mlir"

    echo ""
    echo "===== Extracting Triton dump files for $KERNEL_NAME ====="
    if [ "$IR_SAVE_TTIR" = "true" ] && [ -f "$DUMP_DIR/kernel.ttir.mlir" ]; then
        cp "$DUMP_DIR/kernel.ttir.mlir" "$IR_OUTPUT_DIR/${KERNEL_NAME}_ttir.mlir"
        echo "$KERNEL_NAME: Copied kernel.ttir.mlir ($(wc -l < "$DUMP_DIR/kernel.ttir.mlir") lines)"
    fi
    if [ "$IR_SAVE_TTADAPTER" = "true" ] && [ -f "$DUMP_DIR/kernel.ttadapter.mlir" ]; then
        cp "$DUMP_DIR/kernel.ttadapter.mlir" "$IR_OUTPUT_DIR/${KERNEL_NAME}_ttadapter.mlir"
        echo "$KERNEL_NAME: Copied kernel.ttadapter.mlir ($(wc -l < "$DUMP_DIR/kernel.ttadapter.mlir") lines)"
    fi

    echo ""
    echo "===== Running bishengir-compile with '$PRINT_IR_FLAG' on $KERNEL_NAME ====="
    OUTPUT_FILE="/tmp/kernel_${idx}_full_ir_dump.txt"
    LAST_PASS_EXTRACTED=0
    cd "$DUMP_DIR"
    COMPILE_EXIT_CODE=0

    # NOTE: The default target below is for Ascend910_9382. When the skill is
    # invoked for a different NPU (e.g., Ascend950PR_957c per AGENTS.md),
    # the harness should set BISHENGIR_TARGET before calling this script.
    TARGET="${BISHENGIR_TARGET:-Ascend910_9382}"

    if [ -n "$PRINT_IR_FLAG" ]; then
        "$BISHENGIR_COMPILE" \
            --target="$TARGET" \
            --enable-auto-multi-buffer=False \
            --enable-auto-bind-sub-block=True \
            --enable-hfusion-compile=true \
            --enable-hivm-compile=true \
            --enable-triton-kernel-compile=true \
            $PRINT_IR_FLAG \
            kernel.ttadapter.mlir > "$OUTPUT_FILE" 2>&1 || COMPILE_EXIT_CODE=$?
    else
        echo "WARNING: No IR dump flag available, skipping bishengir-compile IR extraction."
        COMPILE_EXIT_CODE=1
    fi
    cd "$SCRIPT_DIR"

    TOTAL_KERNELS=$((TOTAL_KERNELS + 1))

    if [ $COMPILE_EXIT_CODE -ne 0 ]; then
        COMPILE_FAILED=1
        COMPILE_FAILURES="${COMPILE_FAILURES}  - ${KERNEL_NAME}: exit code ${COMPILE_EXIT_CODE}, see ${OUTPUT_FILE}\n"
        echo "ERROR: bishengir-compile failed for $KERNEL_NAME with exit code $COMPILE_EXIT_CODE"
        echo "  Full output saved to: $OUTPUT_FILE"
        LAST_PASS_NAME_IN_ERROR="(not generated)"
        if [ -f "$OUTPUT_FILE" ]; then
            ERROR_LINES=$(wc -l < "$OUTPUT_FILE")
            echo "  Output file contains $ERROR_LINES lines"
        fi
    else
        PROCESSED_KERNELS=$((PROCESSED_KERNELS + 1))
    fi

    echo ""
    echo "===== Extracting last pass IR for $KERNEL_NAME ====="
    LAST_PASS_EXTRACTED=0

    # Strategy depends on the flag type:
    #   --mlir-print-ir-after-all / --print-after-all  → output contains "IR Dump After" markers
    #   --bishengir-print-ir-after=<pass>              → stdout is already the IR after that pass
    if echo "$PRINT_IR_FLAG" | grep -q "bishengir-print-ir-after"; then
        # Direct IR output: the whole file is the last-pass IR
        if [ -f "$OUTPUT_FILE" ] && [ $(wc -l < "$OUTPUT_FILE") -gt 0 ]; then
            OUTPUT_MLIR="/tmp/${KERNEL_NAME}_last_pass.mlir"
            cp "$OUTPUT_FILE" "$OUTPUT_MLIR"
            LINES=$(wc -l < "$OUTPUT_MLIR")
            echo "$KERNEL_NAME: Direct IR dump saved $LINES lines to $OUTPUT_MLIR"

            WORKSPACE_MLIR="$IR_OUTPUT_DIR/${KERNEL_NAME}_last_pass.mlir"
            cp "$OUTPUT_MLIR" "$WORKSPACE_MLIR"
            echo "$KERNEL_NAME: Copied to workspace as $(basename "$WORKSPACE_MLIR")"
            LAST_PASS_EXTRACTED=1
        fi
    else
        # Marker-based extraction: find the last "IR Dump After" block
        LAST_PASS_LINE=$(grep -n "IR Dump After" "$OUTPUT_FILE" | tail -1 | cut -d: -f1)
        if [ -n "$LAST_PASS_LINE" ]; then
            LAST_PASS_NAME=$(grep "IR Dump After" "$OUTPUT_FILE" | tail -1 | sed 's/.*IR Dump After //' | sed 's/ .*//')
            OUTPUT_MLIR="/tmp/${KERNEL_NAME}_last_pass_${LAST_PASS_NAME}.mlir"
            sed -n "${LAST_PASS_LINE},\$p" "$OUTPUT_FILE" > "$OUTPUT_MLIR"
            LINES=$(wc -l < "$OUTPUT_MLIR")
            echo "$KERNEL_NAME: Last pass is '$LAST_PASS_NAME', saved $LINES lines to $OUTPUT_MLIR"

            WORKSPACE_MLIR="$IR_OUTPUT_DIR/${KERNEL_NAME}_last_pass.mlir"
            cp "$OUTPUT_MLIR" "$WORKSPACE_MLIR"
            echo "$KERNEL_NAME: Copied to workspace as $(basename "$WORKSPACE_MLIR")"
            LAST_PASS_EXTRACTED=1
        else
            if [ $COMPILE_EXIT_CODE -eq 0 ]; then
                echo "WARNING: No 'IR Dump After' found in output for $KERNEL_NAME (bishengir-compile succeeded but no passes logged)"
                LAST_PASS_NAME_IN_ERROR="(no passes found)"
            fi
        fi
    fi
done

echo ""
echo "===== Done ====="
echo ""
echo "===== IR File Validation ====="
VALIDATION_FAILED=0
MISSING_FILES=""
INVALID_FILES=""
VALID_LAST_PASS_COUNT=0
PARTIAL_KERNELS=""

if [ ! -d "$IR_OUTPUT_DIR" ]; then
    echo "ERROR: Output directory $IR_OUTPUT_DIR does not exist"
    exit 1
fi

for ENTRY in $UNIQ_DIRS; do
    DUMP_DIR=$(echo "$ENTRY" | cut -d'|' -f1)
    KERNEL_NAME=$(echo "$ENTRY" | cut -d'|' -f2)

    TTIR_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_ttir.mlir"
    TTADAPTER_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_ttadapter.mlir"
    LAST_PASS_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_last_pass.mlir"

    KERNEL_VALID=1
    KERNEL_MISSING=""

    if [ "$IR_SAVE_TTIR" = "true" ]; then
        if [ ! -f "$TTIR_FILE" ]; then
            KERNEL_MISSING="${KERNEL_MISSING} <kernel_name>_ttir.mlir"
            KERNEL_VALID=0
        elif [ $(wc -l < "$TTIR_FILE") -eq 0 ]; then
            INVALID_FILES="${INVALID_FILES}  - ${KERNEL_NAME}_ttir.mlir: file is empty (0 lines)\n"
            KERNEL_VALID=0
        fi
    fi

    if [ "$IR_SAVE_TTADAPTER" = "true" ]; then
        if [ ! -f "$TTADAPTER_FILE" ]; then
            KERNEL_MISSING="${KERNEL_MISSING} <kernel_name>_ttadapter.mlir"
            KERNEL_VALID=0
        elif [ $(wc -l < "$TTADAPTER_FILE") -eq 0 ]; then
            INVALID_FILES="${INVALID_FILES}  - ${KERNEL_NAME}_ttadapter.mlir: file is empty (0 lines)\n"
            KERNEL_VALID=0
        fi
    fi

    if [ ! -f "$LAST_PASS_FILE" ]; then
        KERNEL_MISSING="${KERNEL_MISSING} <kernel_name>_last_pass.mlir"
        KERNEL_VALID=0
    elif [ $(wc -l < "$LAST_PASS_FILE") -eq 0 ]; then
        INVALID_FILES="${INVALID_FILES}  - ${KERNEL_NAME}_last_pass.mlir: file is empty (0 lines)\n"
        KERNEL_VALID=0
    else
        if grep -q "hivm\.hir\." "$LAST_PASS_FILE" 2>/dev/null || grep -q "llvm\." "$LAST_PASS_FILE" 2>/dev/null; then
            VALID_LAST_PASS_COUNT=$((VALID_LAST_PASS_COUNT + 1))
        else
            INVALID_FILES="${INVALID_FILES}  - ${KERNEL_NAME}_last_pass.mlir: does not contain expected 'hivm.hir' or 'llvm' keywords (pipeline may have stopped early)\n"
            KERNEL_VALID=0
        fi
    fi

    if [ $KERNEL_VALID -eq 1 ]; then
        VALIDATED_KERNELS=$((VALIDATED_KERNELS + 1))
    else
        VALIDATION_FAILED=1
        if [ -n "$KERNEL_MISSING" ]; then
            MISSING_FILES="${MISSING_FILES}  - ${KERNEL_NAME}: missing ${KERNEL_MISSING}\n"
        fi
        PARTIAL_KERNELS="${PARTIAL_KERNELS}  - ${KERNEL_NAME}\n"
    fi
done

echo ""
echo "Expected kernels: $(echo $UNIQ_DIRS | tr ' ' '\n' | grep -c '|' 2>/dev/null || echo 0)"
echo "Validated successfully: $VALIDATED_KERNELS kernels"
echo ""

if [ $COMPILE_FAILED -eq 1 ]; then
    echo "===== bishengir-compile Failures ====="
    echo -e "$COMPILE_FAILURES"
    echo "Recommendation: Check the full output files listed above for detailed error messages."
    echo ""
fi

if [ -n "$MISSING_FILES" ]; then
    echo "===== Missing Files ====="
    echo -e "$MISSING_FILES"
    echo ""
fi

if [ -n "$INVALID_FILES" ]; then
    echo "===== Invalid Files ====="
    echo -e "$INVALID_FILES"
    echo ""
fi

if [ $VALIDATION_FAILED -eq 1 ]; then
    echo "===== Analysis Based on Available Files ====="
    echo ""
    echo "Based on the generated IR files, here are diagnostic suggestions:"
    echo ""

    for ENTRY in $UNIQ_DIRS; do
        # FIX: f1 is DUMP_DIR, f2 is KERNEL_NAME. Previously f1 was used by mistake,
        # causing diagnostic output to show a path instead of the kernel name.
        KERNEL_NAME=$(echo "$ENTRY" | cut -d'|' -f2)
        TTIR_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_ttir.mlir"
        TTADAPTER_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_ttadapter.mlir"
        LAST_PASS_FILE="$IR_OUTPUT_DIR/${KERNEL_NAME}_last_pass.mlir"

        echo "Kernel: $KERNEL_NAME"
        HAS_TTIR=0
        HAS_TTADAPTER=0
        HAS_LAST_PASS=0

        if [ -f "$TTIR_FILE" ] && [ $(wc -l < "$TTIR_FILE") -gt 0 ]; then
            HAS_TTIR=1
        fi
        if [ -f "$TTADAPTER_FILE" ] && [ $(wc -l < "$TTADAPTER_FILE") -gt 0 ]; then
            HAS_TTADAPTER=1
        fi
        if [ -f "$LAST_PASS_FILE" ] && [ $(wc -l < "$LAST_PASS_FILE") -gt 0 ]; then
            if grep -q "hivm\.hir\." "$LAST_PASS_FILE" 2>/dev/null || grep -q "llvm\." "$LAST_PASS_FILE" 2>/dev/null; then
                HAS_LAST_PASS=1
            fi
        fi

        if [ $HAS_TTIR -eq 1 ] && [ $HAS_TTADAPTER -eq 1 ] && [ $HAS_LAST_PASS -eq 0 ]; then
            echo "  - Pipeline stopped before HIVM->LLVM lowering stage"
            echo "    Possible cause: 'hivmc' not found in PATH or internal compiler error"
            echo "    Suggestion: Verify bishengir bin directory is in PATH, check /tmp/kernel_*_full_ir_dump.txt"
        elif [ $HAS_TTIR -eq 1 ] && [ $HAS_TTADAPTER -eq 0 ]; then
            echo "  - Missing ttadapter IR, possible Triton adapter conversion failure"
            echo "    Suggestion: Check Triton dialect to memref adaptation passes"
        elif [ $HAS_TTIR -eq 0 ]; then
            echo "  - Missing ttir IR, possible Triton frontend compilation failure"
            echo "    Suggestion: Check if the Triton kernel syntax is correct"
        elif [ $HAS_LAST_PASS -eq 1 ]; then
            echo "  - Full IR pipeline completed successfully"
        fi
        echo ""
    done
fi

echo "===== Output Summary ====="
echo "IR files location: ${IR_OUTPUT_DIR}/"
echo ""
echo "Files saved per kernel (controlled by IR_SAVE_* variables):"
[ "$IR_SAVE_TTIR" = "true" ] && echo "  - <kernel_name>_ttir.mlir         (Triton frontend IR)"
[ "$IR_SAVE_TTADAPTER" = "true" ] && echo "  - <kernel_name>_ttadapter.mlir   (Triton adapter IR)"
echo "  - <kernel_name>_last_pass.mlir   (BishengIR last pass IR) [always generated]"
echo ""

cd "$IR_OUTPUT_DIR"
FILE_COUNT=$(find . -name "*.mlir" -type f | wc -l)
echo "Total .mlir files generated: $FILE_COUNT"
echo ""

if [ $VALIDATION_FAILED -eq 0 ] && [ $COMPILE_FAILED -eq 0 ]; then
    echo "All kernels validated successfully."
    echo ""
    echo "===== Quick Performance Characteristics ====="
    for LAST_PASS_FILE in *_last_pass.mlir; do
        if [ -f "$LAST_PASS_FILE" ]; then
            KERNEL_NAME=$(basename "$LAST_PASS_FILE" _last_pass.mlir)
            TOTAL_LINES=$(wc -l < "$LAST_PASS_FILE")
            HIVM_INTRINSIC_COUNT=$(grep -c "hivm\.hir\." "$LAST_PASS_FILE" 2>/dev/null || echo 0)
            LLVM_OPS_COUNT=$(grep -c "llvm\." "$LAST_PASS_FILE" 2>/dev/null || echo 0)
            MEMCPY_COUNT=$(grep -c "llvm\.memcpy" "$LAST_PASS_FILE" 2>/dev/null || echo 0)
            LOAD_COUNT=$(grep -c "llvm\.load" "$LAST_PASS_FILE" 2>/dev/null || echo 0)
            STORE_COUNT=$(grep -c "llvm\.store" "$LAST_PASS_FILE" 2>/dev/null || echo 0)

            echo "Kernel: $KERNEL_NAME"
            echo "  Total last_pass lines: $TOTAL_LINES"
            echo "  HIVM intrinsics: $HIVM_INTRINSIC_COUNT ($(awk "BEGIN {printf \"%.1f\", ($HIVM_INTRINSIC_COUNT/$TOTAL_LINES)*100}")% of lines)"
            echo "  LLVM operations: $LLVM_OPS_COUNT"
            echo "  Memory ops: $LOAD_COUNT loads, $STORE_COUNT stores, $MEMCPY_COUNT memcpy"
            echo ""
        fi
    done
    echo "For detailed analysis and optimization recommendations, see:"
    echo "  <skill_dir>/references/IR_triton.md"
else
    echo "Some issues detected. Please review the diagnostics above."
    echo "For troubleshooting, see: <skill_dir>/references/IR_triton.md"
fi