export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径

# 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh

#export ASCEND_RT_VISIBLE_DEVICES=2,3,8,9
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=0
export SPDLOG_LEVEL=debug
export ASCEND_MODULE_LOG_LEVEL=ATB=0
export ASDOPS_LOG_TO_FILE=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export LCCL_DETERMINISTIC=1 
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
# 3. 清理旧日志
\rm -rf core.*
\rm -rf log_1/node_*.log

export PROFILING_MODE=dynamic

# 4. 启动分布式服务
#MODEL_PATH="/export/home/shanchenfeng/xllm_build/xllm_qwen_embed/qwen3_omni_moe_thinker"
MODEL_PATH="/export/home/models/Qwen3-Next-80B-A3B-Thinking/"
MASTER_NODE_ADDR="127.0.0.1:9778"                  # Master 节点地址（需全局一致）
START_PORT=18002                                   # 服务起始端口
START_DEVICE=0                           # 起始 NPU 逻辑设备号
LOG_DIR="log_1"                                      # 日志目录
NNODES=4                                           # 节点数（当前脚本启动 2 个进程）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \
    --max_tokens_per_batch=20000 \
    --max_seqs_per_batch=16 \
    --enable_mla=false \
    --block_size=128 \
    --communication_backend="lccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    --enable_graph=false \
    --node_rank=$i  \
    --enable_shm 1 \
    --task="generate" \
    --backend llm  > $LOG_FILE 2>&1 &
done
