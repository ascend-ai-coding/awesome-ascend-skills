# Docker / Ascend（参考）

## 镜像规则

- 用户/运维已指定 `IMAGE_TAG`：直接使用
- 未指定时默认：
  - 910B: `verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1`
  - A3: `verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1`
- 自选 tag：`https://quay.io/repository/ascend/verl?tab=tags&tag=latest`

## 容器命令骨架

```bash
export IMAGE_REPO="quay.io/ascend/verl"
export ASCEND_HW="${ASCEND_HW:-a3}"  # a3 或 910b；也可显式 export IMAGE_TAG=...
case "$ASCEND_HW" in
  910b|910B) export IMAGE_TAG="${IMAGE_TAG:-verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1}" ;;
  a3|A3)     export IMAGE_TAG="${IMAGE_TAG:-verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1}" ;;
  *)         echo "请设置 ASCEND_HW=a3|910b 或显式 export IMAGE_TAG=..."; exit 1 ;;
esac
export IMAGE="$IMAGE_REPO:$IMAGE_TAG"

docker pull "$IMAGE"

export container_name="${container_name:-verl_$(date +%Y%m%d_%H%M%S)}"
export WORKSPACE="$(pwd)"

docker run -itd \
  --network host \
  --shm-size=10g \
  --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /home/:/home \
  -v "$WORKSPACE:/workspace/verl" \
  --name "$container_name" \
  "$IMAGE" bash

docker exec -it "$container_name" bash
```
