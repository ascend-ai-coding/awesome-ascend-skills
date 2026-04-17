# NPU Docker Launcher Examples

## Example 1: Development Environment (Host Network)

Use this for local development work on an NPU server.

```bash
#!/bin/bash

IMAGE="pytorch/pytorch:2.0.1-cuda11.3-cudnn8"
CONTAINER_NAME="pytorch-dev"
USE_PRIVILEGED="true"
NET_MODE="host"
NPU_CARDS="all"
MOUNT_DIRS="/home/user:/home/user,/data:/data"
WORK_DIR="/home/user"
ENV_VARS="PYTHONUNBUFFERED=1"
ADDITIONAL_MOUNTS="-v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/sbin:/usr/local/sbin"

# Container launch command
docker run -itd \
  --name $CONTAINER_NAME \
  $USE_PRIVILEGED && echo "--privileged" || "" \
  --network host \
  $ADDITIONAL_MOUNTS \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  $(echo $ENV_VARS | xargs -n1 -I{} echo -e {}) \
  $IMAGE
```

**Features:**
- Host networking for easy localhost access
- Privileged mode for all NPU access
- Home and data directories mounted
- Python output unbuffered

## Example 2: Single NPU Card Training

Use this for training on a single NPU card.

```bash
#!/bin/bash

IMAGE="registry.example.com/ascend-pytorch:latest"
CONTAINER_NAME="train-single-npu"
USE_PRIVILEGED="false"
NET_MODE="host"
NPU_CARDS="0"
MOUNT_DIRS="/home/training:/workspace,/data/datasets:/datasets"
WORK_DIR="/workspace"
ENV_VARS="ASCEND_DEVICE_ID=0 ASCEND_GLOBAL_LOG_LEVEL=1"

# Mount specific NPU device
DEVICE_ARGS="--device=/dev/davinci0"

docker run -itd \
  --name $CONTAINER_NAME \
  --network host \
  $DEVICE_ARGS \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  -e $ENV_VARS \
  $IMAGE
```

**Features:**
- Specific NPU card (card 0) mounted
- Non-privileged mode for better security
- ASCEND environment variables configured
- Training workspace and dataset directories

## Example 3: Multi-Card Distributed Training

Use this for training across multiple NPU cards.

```bash
#!/bin/bash

IMAGE="registry.example.com/ascend-mindspore:latest"
CONTAINER_NAME="train-distributed"
USE_PRIVILEGED="true"
NET_MODE="host"
NPU_CARDS="0,1,2,3"
MOUNT_DIRS="/home/training:/workspace,/data/checkpoints:/checkpoints"
WORK_DIR="/workspace"
ENV_VARS="ASCEND_VISIBLE_DEVICES=0,1,2,3 RANK_SIZE=4"

docker run -itd \
  --name $CONTAINER_NAME \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3 \
  -e RANK_SIZE=4 \
  $IMAGE
```

**Features:**
- All four NPU cards accessible
- ASCEND_VISIBLE_DEVICES configured
- RANK_SIZE for distributed training
- Checkpoint directory for model saving

## Example 4: Inference Service with Port Mapping

Use this for running inference as a service.

```bash
#!/bin/bash

IMAGE="registry.example.com/ascend-inference:latest"
CONTAINER_NAME="inference-service"
USE_PRIVILEGED="false"
NET_MODE="custom"
PORT_MAPPING="8080:8000 9000:9000"
NPU_CARDS="0"
MOUNT_DIRS="/models:/models"
WORK_DIR="/models"
ENV_VARS="MODEL_PATH=/models/model.bin"

# Mount specific NPU device
DEVICE_ARGS="--device=/dev/davinci0"

docker run -itd \
  --name $CONTAINER_NAME \
  --network bridge \
  -p $PORT_MAPPING \
  $DEVICE_ARGS \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  -e MODEL_PATH=/models/model.bin \
  $IMAGE
```

**Features:**
- Port mapping for external access
- Bridge network mode
- Single NPU for inference
- Model directory mounted
- API accessible on host port 8080

## Example 5: Auto-Detect NPU Cards

Use this when the number of NPU cards varies across servers.

```bash
#!/bin/bash

IMAGE="pytorch/pytorch:latest"
CONTAINER_NAME="auto-npu"
USE_PRIVILEGED="false"
MOUNT_DIRS="/home:/home"
WORK_DIR="/home"

# Auto-detect available NPU cards
NPU_COUNT=$(ls /dev/davinci* 2>/dev/null | wc -l)

if [ $NPU_COUNT -eq 0 ]; then
    echo "No NPU devices detected. Using privileged mode."
    USE_PRIVILEGED="true"
    DEVICE_ARGS=""
else
    echo "Detected $NPU_COUNT NPU card(s)"
    DEVICE_ARGS=""
    for i in $(seq 0 $((NPU_COUNT - 1))); do
        DEVICE_ARGS="$DEVICE_ARGS --device=/dev/davinci$i"
    done
    export ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((NPU_COUNT - 1)))
fi

docker run -itd \
  --name $CONTAINER_NAME \
  $USE_PRIVILEGED && echo "--privileged" || "" \
  --network host \
  $DEVICE_ARGS \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  $IMAGE
```

**Features:**
- Automatically detects available NPU cards
- Falls back to privileged mode if no devices found
- Generates appropriate device mount arguments
- Sets ASCEND_VISIBLE_DEVICES environment variable

## Example 6: Training with Checkpoint Resuming

Use this for training that needs to resume from checkpoints.

```bash
#!/bin/bash

IMAGE="mindspore/mindspore:latest"
CONTAINER_NAME="train-resume"
USE_PRIVILEGED="true"
NET_MODE="host"
NPU_CARDS="all"
MOUNT_DIRS="/home/training:/workspace,/data/datasets:/datasets,/data/checkpoints:/checkpoints"
WORK_DIR="/workspace"
ENV_VARS="
CHECKPOINT_PATH=/checkpoints/latest.ckpt
RESUME_TRAINING=true
ASCEND_SLOG_PRINT_TO_STDOUT=1
"

docker run -itd \
  --name $CONTAINER_NAME \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  -e CHECKPOINT_PATH=/checkpoints/latest.ckpt \
  -e RESUME_TRAINING=true \
  -e ASCEND_SLOG_PRINT_TO_STDOUT=1 \
  --shm-size=16g \
  $IMAGE
```

**Features:**
- Checkpoint directory mounted
- Training resume configuration
- Logging to stdout enabled
- Large shared memory for distributed training

## Example 7: Debugging Environment

Use this for debugging NPU applications with full device access.

```bash
#!/bin/bash

IMAGE="ascend/debug:latest"
CONTAINER_NAME="debug-npu"
USE_PRIVILEGED="true"
NET_MODE="host"
NPU_CARDS="all"
MOUNT_DIRS="/home/debug:/workspace,/tmp:/tmp"
WORK_DIR="/workspace"
ENV_VARS="
ASCEND_SLOG_PRINT_TO_STDOUT=1
ASCEND_GLOBAL_LOG_LEVEL=0
DEBUG_MODE=1
"

docker run -it \
  --rm \
  --name $CONTAINER_NAME \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit \
  -v $MOUNT_DIRS \
  -w $WORK_DIR \
  -e ASCEND_SLOG_PRINT_TO_STDOUT=1 \
  -e ASCEND_GLOBAL_LOG_LEVEL=0 \
  -e DEBUG_MODE=1 \
  --cap-add=SYS_PTRACE \
  $IMAGE bash
```

**Features:**
- Interactive mode (no -d flag)
- Auto-remove container on exit (--rm)
- Full debug logging enabled
- PTRACE capability for debugging tools
- Ascend toolkit mounted for tool access

## Configuration Patterns Summary

| Pattern | Use Case | Privileged | Network | NPU Cards | Key Features |
|---------|----------|------------|---------|-----------|--------------|
| Dev Env | Local development | Yes | Host | All | Home dir, unbuffered output |
| Single Training | Single card training | No | Host | Specific | Least privilege, specific device |
| Multi Training | Distributed training | Yes | Host | All | Multiple cards, RANK_SIZE |
| Inference | Production service | No | Custom | Specific | Port mapping, bridge network |
| Auto-Detect | Variable NPU count | Auto | Host | Auto | Detects available cards |
| Resume | Training with checkpoint | Yes | Host | All | Checkpoint dir, shm-size |
| Debug | Debugging | Yes | Host | All | Interactive, ptrace, full logs |

## Tips for Customization

1. **Mount additional tools**: Add `-v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit` for toolkit access
2. **Increase shared memory**: Add `--shm-size=16g` for distributed training
3. **Set GPU memory**: Use environment variables like `ASCEND_DEVICE_MEMORY_LIMIT` for memory control
4. **Interactive mode**: Remove `-d` flag and add `--rm` for temporary debugging sessions
5. **Resource limits**: Add `--cpus=8 --memory=32g` for CPU/memory constraints
