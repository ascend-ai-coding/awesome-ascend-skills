# Docker Deployment

## Quick Start

```bash
docker run -it -d \
    --name vllm-server \
    --privileged \
    --network host \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home/weights:/model \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    vllm-ascend:latest \
    /bin/bash
```

## Docker Compose

```yaml
version: '3'
services:
  vllm-server:
    image: vllm-ascend:latest
    container_name: vllm-server
    privileged: true
    network_mode: host
    volumes:
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver
      - /usr/local/sbin:/usr/local/sbin
      - /home/data1:/home/data1
    working_dir: /home/data1
    environment:
      - ASCEND_RT_VISIBLE_DEVICES=0
      - TASK_QUEUE_ENABLE=1
    command: |
      bash -c "
        vllm serve /home/data1/Qwen3-8B \
          --host 0.0.0.0 \
          --port 8000 \
          --trust-remote-code \
          --quantization ascend
      "
```

## Port Mapping (Bridge Network)

```bash
docker run -it -d \
    --name vllm-server \
    --privileged \
    --network bridge \
    -p 8000:8000 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /home/weights:/model \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    vllm-ascend:latest \
    /bin/bash
```

## Multi-Card Container

```bash
docker run -it -d \
    --name vllm-server-tp2 \
    --privileged \
    --network host \
    --shm-size=16g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /home/weights:/model \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -e HCCL_BUFFSIZE=1024 \
    -e HCCL_CONNECT_TIMEOUT=600 \
    vllm-ascend:latest \
    /bin/bash
```

## Remote Container Launch

```bash
# Using sshpass
sshpass -p 'password' ssh user@host "
    docker run -it -d \
        --name vllm-server \
        --privileged \
        --network host \
        -v /home/weights/Qwen3-8B:/model \
        -e ASCEND_RT_VISIBLE_DEVICES=0 \
        vllm-ascend:latest
"

# Execute vLLM in container (background)
sshpass -p 'password' ssh user@host "
    docker exec -d vllm-server bash -c 'vllm serve /model --port 8000 --quantization ascend'
"
```

## Container Management

```bash
# View logs
docker exec <container> tail -f /tmp/vllm.log

# Check process
docker exec <container> ps aux | grep vllm

# Stop vLLM
docker exec <container> pkill -f vllm

# Restart container
docker stop <container> && docker start <container>

# Enter container
docker exec -it <container> bash
```
