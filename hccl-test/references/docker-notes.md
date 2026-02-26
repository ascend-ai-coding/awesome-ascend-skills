# HCCL Test Docker Notes

在 Docker 容器中运行 HCCL Test 的注意事项。

---

## 1. 网络模式要求

### 1.1 必须使用 Host 网络模式

HCCL Test 在 Docker 容器中运行时，**必须使用 host 网络模式**。

```bash
# ✅ 正确：使用 host 网络模式
docker run --network host --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/ascend-toolkit/latest:/usr/local/Ascend/ascend-toolkit/latest \
  ascend-cann:latest

# ❌ 错误：使用 bridge 网络模式（会导致 HCCL 建链失败）
docker run --network bridge ...
```

### 1.2 原因说明

HCCL 使用特定的网络协议进行 NPU 间通信，需要直接访问宿主机的网络栈。Bridge 模式会导致：
- NPU 发现失败
- 建链超时
- 通信异常

---

## 2. 设备映射

### 2.1 必需挂载

```bash
docker run --network host --privileged \
  # NPU 设备
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  \
  # 驱动管理设备
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  \
  # 驱动目录
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  \
  # CANN 工具链
  -v /usr/local/Ascend/ascend-toolkit/latest:/usr/local/Ascend/ascend-toolkit/latest:ro \
  \
  # 日志目录（可选）
  -v ~/ascend/log:/root/ascend/log \
  \
  ascend-cann:latest
```

### 2.2 批量挂载脚本

```bash
#!/bin/bash
# docker-run-hccl.sh

NPU_DEVICES=""
for i in {0..7}; do
    if [[ -e "/dev/davinci$i" ]]; then
        NPU_DEVICES="$NPU_DEVICES --device /dev/davinci$i"
    fi
done

docker run -it --rm \
    --network host \
    --privileged \
    $NPU_DEVICES \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/ascend-toolkit/latest:/usr/local/Ascend/ascend-toolkit/latest:ro \
    -v $(pwd):/workspace \
    ascend-cann:latest \
    /bin/bash
```

---

## 3. 环境变量配置

### 3.1 容器内必需环境变量

```bash
# 在容器内执行
export INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
```

### 3.2 Dockerfile 示例

```dockerfile
FROM ubuntu:22.04

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    openssh-client \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# 安装 MPI（已在镜像中安装 MPICH）
COPY --from=mpich:latest /usr/local/mpich /usr/local/mpich

# 设置环境变量
ENV INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
ENV PATH=/usr/local/mpich/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/mpich/lib:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

CMD ["/bin/bash"]
```

---

## 4. 多机容器测试

### 4.1 注意事项

1. **所有容器必须使用 host 网络模式**
2. **容器间 SSH 免密登录需配置**
3. **CANN 版本需一致**

### 4.2 配置步骤

```bash
# 1. 在每个节点上启动容器（必须使用 host 网络）
docker run -d --network host --privileged \
    --name hccl-test-node \
    -v /root/.ssh:/root/.ssh:ro \
    ascend-cann:latest \
    tail -f /dev/null

# 2. 进入容器
docker exec -it hccl-test-node bash

# 3. 在容器内执行 HCCL 测试（与物理机相同）
./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

---

## 5. 日志查看

### 5.1 容器内日志路径

```bash
# 查看 Ascend 日志
cat ~/ascend/log/debug/plog/plog-*.log | tail -100

# 查看 HCCL 特定日志
cat ~/ascend/log/debug/plog/plog-*_hccl*.log
```

### 5.2 日志挂载到宿主机

```bash
docker run --network host --privileged \
    ... \
    -v ~/ascend/log:/root/ascend/log \
    ascend-cann:latest
```

---

## 6. 故障排查

### 6.1 容器内无法发现 NPU

**现象**: `npu-smi info` 显示无 NPU

**解决**:
1. 检查设备是否正确挂载：`--device /dev/davinci0`
2. 检查驱动目录挂载：`-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro`
3. 检查是否使用 `--privileged` 参数

### 6.2 容器内 HCCL 建链失败

**现象**: retcode: 7 或建链超时

**解决**:
1. 确认使用 `--network host` 模式
2. 确认容器内可以 ping 通其他节点
3. 确认 SSH 免密登录配置正确

### 6.3 权限问题

**现象**: 访问 NPU 设备权限不足

**解决**:
```bash
# 在宿主机上检查设备权限
ls -la /dev/davinci*

# 确保容器使用 --privileged 参数
# 或在宿主机上修改权限（不推荐）
chmod 666 /dev/davinci*
```
