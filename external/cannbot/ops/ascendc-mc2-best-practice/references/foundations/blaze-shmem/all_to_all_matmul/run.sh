#!/bin/bash
# ----------------------------------------------------------------------------
# run.sh：一键编译 + 生成数据 + 跑算子 + 精度比对
#
# 流程：
#   1. cmake 配置 + 编译，产出 build/all_to_all_matmul
#   2. 准备 input/ output/ 目录（output 子目录会被清空，避免 WriteFile 不 truncate 残留）
#   3. python3 scripts/gen_data.py 生成 input_*.bin + cpu_output.bin
#   4. ./build/all_to_all_matmul 跑 precision 模式（单次 kernel）
#   5. python3 scripts/verify_result.py 比对 npu_out.bin vs cpu_output.bin
#
# 用法：
#   bash run.sh                  # 默认 m=2048 k=8192 n=3584 rank_num=4
#   bash run.sh 512 4096 3072 4  # 小 shape 冒烟
#   bash run.sh 512 4096 3072 4 perf  # 第 5 参透传给算子（perf 模式跑性能模板）
# ----------------------------------------------------------------------------
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$PROJECT_ROOT/third_party/shmem/install/shmem/lib:$LD_LIBRARY_PATH"

M="${1:-2048}"
K="${2:-8192}"
N="${3:-3584}"
RANK="${4:-4}"
MODE="${5:-precision}"

echo "===== [1/4] cmake configure + build ====="
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" -DNPU_ARCH=dav-3510 >/dev/null
cmake --build "$PROJECT_ROOT/build" -j

OP_EXE="$PROJECT_ROOT/build/all_to_all_matmul"
if [ ! -x "$OP_EXE" ]; then
    echo "ERROR: $OP_EXE not found or not executable"
    exit 1
fi

# 算子以 exe 所在目录为 baseDir 读写 input/output，做 symlink 对齐
mkdir -p "$PROJECT_ROOT/build/output"
[ ! -e "$PROJECT_ROOT/build/input" ] && ln -sf ../input "$PROJECT_ROOT/build/input"
for r in $(seq 0 $((RANK - 1))); do
    rm -rf "$PROJECT_ROOT/build/output/$r"
    ln -sf "../../output/$r" "$PROJECT_ROOT/build/output/$r"
done

echo "===== [2/4] gen_data.py m=$M k=$K n=$N rank=$RANK ====="
python3 "$PROJECT_ROOT/scripts/gen_data.py" "$M" "$K" "$N" "$RANK"

# 清掉历史 npu_out.bin/shmem_*.bin：算子的 WriteFile 不 truncate，残留旧 size 会污染比对
for r in $(seq 0 $((RANK - 1))); do
    rm -f "$PROJECT_ROOT/output/$r/npu_out.bin" \
          "$PROJECT_ROOT/output/$r/shmem_A.bin" \
          "$PROJECT_ROOT/output/$r/shmem_scale.bin"
done

echo "===== [3/4] run operator (mode=$MODE) ====="
"$OP_EXE" "$M" "$K" "$N" "$RANK" "$MODE"

if [ "$MODE" != "precision" ]; then
    echo "mode=$MODE，跳过精度比对（perf 模式输出仅供 msprof 外部采集）"
    exit 0
fi

echo "===== [4/4] verify_result.py ====="
python3 "$PROJECT_ROOT/scripts/verify_result.py" "$M" "$N" "$RANK" "$PROJECT_ROOT/output"
