#!/bin/bash
# Ascend NPU Environment Setup Script for Wan Models
# Source this script before running inference/training: source env_setup.sh

# ============================================
# Attention Algorithm Selection
# ============================================
# ALGO=0: Default FA operator (fused_attn_score)
# ALGO=1: High-performance FA (ascend_laser_attention) [RECOMMENDED]
# ALGO=3: NPU fused inference attention
export ALGO=1

# ============================================
# Layer Normalization Optimization
# ============================================
# FAST_LAYERNORM=0: Use PyTorch native layer norm
# FAST_LAYERNORM=1: Use MindIE optimized layer norm [RECOMMENDED]
export FAST_LAYERNORM=1

# ============================================
# Communication/Computation Overlap
# ============================================
# OVERLAP=0: Disable overlap
# OVERLAP=1: Enable AllToAll overlap with FA computation [RECOMMENDED]
export OVERLAP=1

# ============================================
# NPU Memory Management
# ============================================
# Enable expandable segments for better memory efficiency
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'

# Alternative: Limit max split size for reduced fragmentation
# export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

# ============================================
# Task Queue Optimization
# ============================================
# TASK_QUEUE_ENABLE=0: Disabled
# TASK_QUEUE_ENABLE=1: Basic task queue
# TASK_QUEUE_ENABLE=2: Advanced task queue with optimization [RECOMMENDED]
export TASK_QUEUE_ENABLE=2

# ============================================
# CPU Affinity (NUMA-aware scheduling)
# ============================================
# Format: <cores_per_rank>:<offsets>
# Example for 8 NPU ranks on 4-socket system:
export CPU_AFFINITY_CONF=0-23:24-47:48-71:72-95

# ============================================
# Tokenizer Parallelism
# ============================================
# Disable for NPU to avoid thread conflicts
export TOKENIZERS_PARALLELISM=false

# ============================================
# Additional Performance Flags
# ============================================
# Enable HCCL fast sync (distributed training)
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

# ============================================
# Verification
# ============================================
echo "✓ Ascend NPU Environment Configured:"
echo "  ALGO=${ALGO} (Attention Algorithm)"
echo "  FAST_LAYERNORM=${FAST_LAYERNORM}"
echo "  OVERLAP=${OVERLAP}"
echo "  PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF}"
echo "  TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE}"
echo "  CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF}"
echo ""
echo "Ready for Wan model inference/training on Ascend NPU!"
