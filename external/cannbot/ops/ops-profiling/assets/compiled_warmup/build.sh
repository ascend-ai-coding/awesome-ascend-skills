#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
npu_arch="${NPU_ARCH:-}"

for arg in "$@"; do
    case "$arg" in
        --soc=*) npu_arch="${arg#*=}" ;;
        *) echo "[ERROR] unsupported argument: $arg" >&2; exit 2 ;;
    esac
done

case "$npu_arch" in
    ascend910b|ascend910_93|ascend950) ;;
    *)
        echo "[ERROR] set NPU_ARCH or pass --soc=ascend910b|ascend910_93|ascend950" >&2
        exit 2
        ;;
esac

torch_dir="$($python_bin -c 'import os, torch; print(os.path.join(torch.utils.cmake_prefix_path, "Torch"))')"
torch_npu_path="$($python_bin -c 'import os, torch_npu; print(os.path.dirname(torch_npu.__file__))')"
build_dir="$script_dir/build/$npu_arch"

cmake -S "$script_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$python_bin" \
    -DTorch_DIR="$torch_dir" \
    -DTORCH_NPU_PATH="$torch_npu_path" \
    -DNPU_ARCH="$npu_arch"
cmake --build "$build_dir" --config Release --parallel "${BUILD_JOBS:-4}"

extension="$script_dir/ops_profiling_warmup/_C.so"
kernel_object="$build_dir/compiled_warmup_kernel.o"
test -f "$extension"
test -f "$kernel_object"

if nm -D --undefined-only "$extension" | grep -Eq ' (ReportAscendProf|StartAscendProf|AscendProfRegister|AscendCGetProfkTypeImpl|GetAscendProfStatus)$'; then
    echo "[ERROR] compiled warmup has unresolved AscendC runtime symbols" >&2
    exit 1
fi
if ! nm "$kernel_object" | grep -Eq ' [Tt] OpsProfilingWarmup$'; then
    echo "[ERROR] compiled warmup kernel name is not exported as OpsProfilingWarmup" >&2
    exit 1
fi

echo "$extension"
