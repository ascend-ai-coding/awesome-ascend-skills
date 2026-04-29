#!/bin/bash
# Verl 容器启动脚本
# 用法: bash verl_docker_run.sh <镜像ID> <容器名称> [额外参数]

set -e

IMAGE_ID=$1
CONTAINER_NAME=$2
EXTRA_ARGS="${@:3}"

if [ -z "$IMAGE_ID" ] || [ -z "$CONTAINER_NAME" ]; then
    echo "用法: bash verl_docker_run.sh <镜像ID> <容器名称> [额外参数]"
    echo ""
    echo "示例:"
    echo "  # 从 Harbor 拉取的镜像"
    echo "  bash verl_docker_run.sh hub.openlab-sh.sd.huawei.com/ascend/verl:v1.0 verl_hlm"
    echo ""
    echo "  # 从公网拉取的镜像"
    echo "  bash verl_docker_run.sh quay.io/ascend/verl:latest verl_hlm"
    echo ""
    echo "  # 本地已有镜像"
    echo "  bash verl_docker_run.sh verl:custom verl_custom"
    echo ""
    echo "  # 带额外参数（如限制 NPU 数量）"
    echo "  bash verl_docker_run.sh quay.io/ascend/verl:latest verl_hlm '-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3'"
    exit 1
fi

# 检查镜像是否存在，不存在则拉取
if ! docker image inspect "$IMAGE_ID" &>/dev/null; then
    echo "镜像不存在，正在拉取: $IMAGE_ID"
    docker pull "$IMAGE_ID"
fi

# 检查容器是否已存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "容器已存在: $CONTAINER_NAME"
    echo "选择操作:"
    echo "  1. 启动现有容器"
    echo "  2. 删除并重新创建"
    echo "  3. 取消"
    read -p "请选择 [1/2/3]: " choice
    case $choice in
        1)
            docker start "$CONTAINER_NAME"
            echo "容器已启动，使用以下命令进入:"
            echo "  docker exec -it $CONTAINER_NAME bash"
            exit 0
            ;;
        2)
            docker rm -f "$CONTAINER_NAME"
            echo "已删除旧容器，重新创建..."
            ;;
        3)
            echo "已取消"
            exit 0
            ;;
        *)
            echo "无效选择，已取消"
            exit 1
            ;;
    esac
fi

echo "启动容器: $CONTAINER_NAME"
echo "镜像: $IMAGE_ID"

# 启动容器
docker run -it \
    --user root \
    --ipc=host \
    --network=host \
    --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox \
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /mnt2:/mnt2 \
    --name "$CONTAINER_NAME" \
    --entrypoint=/bin/bash \
    $EXTRA_ARGS \
    "$IMAGE_ID"

echo ""
echo "容器已创建并进入交互模式"
echo "退出后重新进入使用: docker exec -it $CONTAINER_NAME bash"
