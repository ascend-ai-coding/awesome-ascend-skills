# HCCL Test Common Issues

HCCL性能测试工具常见问题及解决方法。

---

## 1. gethostbyname failed

### 问题现象

执行 `mpirun` 命令时，报错：

```
gethostbyname failed: HW-AI-LC-1-1
```

### 原因

当前节点的主机名无法解析为 IP 地址。

### 解决方法

在 `/etc/hosts` 文件中添加当前节点 IP 地址与对应的主机名信息：

```bash
# 查看主机名
hostname

# 添加到 /etc/hosts
echo "172.16.0.100 $(hostname)" >> /etc/hosts

# EulerOS 需要刷新
nmcli c reload
```

---

## 2. MPI Library Link Error

### 问题现象

执行 `mpirun` 命令时，报错：

```
error while loading shared libraries: libmpi.so.12: cannot open shared object file: No such file or directory
```

### 原因

系统找不到 MPI 的动态链接库。

### 解决方法

在环境变量 `LD_LIBRARY_PATH` 中加入 MPI 的 lib 库路径：

```bash
# MPICH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH

# Open MPI
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
```

或者添加到 `~/.bashrc` 永久生效：

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. "bash: orted: 未找到命令" Error

### 问题现象

集群场景下，执行 `mpirun` 命令时，报错：

```
bash: orted: 未找到命令
--------------------------------------------------------------------------
A daemon (pid 8793) died unexpectedly with status 127 while attempting
to launch so we are aborting.
```

### 原因

集群中存在未退出的 hccl_test 进程。

### 解决方法

利用 MPI 的能力，终止残余的 hccl_test 进程：

```bash
# MPICH 场景
# 准备 Hostfile，确保与测试时使用的相同
mpirun -f hostfile -n 512 pkill -9 -f "all_reduce_test|mpirun"

# Open MPI 场景
mpirun -hostfile hostfile -n 512 pkill -9 -f "all_reduce_test|openmpi"
```

**参数说明**：
- `-f` / `-hostfile`：Hostfile 节点列表文件
- `-n`：需要终止的 NPU 总数（节点数 × 每节点 NPU 数）
- `pkill -9 -f`：强制终止匹配的进程

---

## 4. "retcode: 7" Error

### 问题现象

集群场景下，HCCL Test 工具已启动成功，但打印出表头后报错：

```
the minbytes is 8192, maxbytes is 2147483648, iters is 20, warmup_iters is 5
hccl interface return err ./common/src/hccl_test_common.cc:538, retcode: 7 
This is an error in init_hcclComm.
```

### 原因

集群中与当前节点通信的节点上存在未退出的 hccl_test 进程。

### 解决方法

与问题 3 相同，清理残余进程：

```bash
# MPICH 场景
mpirun -f hostfile -n 512 pkill -9 -f "all_reduce_test|mpirun"

# Open MPI 场景
mpirun -hostfile hostfile -n 512 pkill -9 -f "all_reduce_test|openmpi"
```

清理完成后，再次执行 HCCL Test 工具进行测试。

---
## 5. CANN Version Mismatch (Multi-Node)

### Problem
Multi-node HCCL test fails with connection or initialization errors.

### Cause
CANN versions are inconsistent across nodes.

### Solution

Check and unify CANN versions on all nodes:

```bash
# Check CANN versions on all nodes
for node in 175.99.1.2 175.99.1.3; do
    echo "=== $node ==="
    ssh root@$node "cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | grep runtime_running_version"
done
```

If versions differ, upgrade/downgrade to the same version on all nodes.

---

## 6. NPU Health Status (Alarm/Offline)

### Problem
HCCL test fails on specific NPUs or shows degraded performance.

### Cause
NPU hardware is in Alarm or Offline state.

### Solution

Check NPU health status:

```bash
# Check NPU health
npu-smi info -t health -i 0
```

**NPU Status:**

| Status | Meaning | Action |
|--------|---------|--------|
| OK | Normal | ✅ Can use |
| Alarm | Warning | ⚠️ Investigate |
| Offline | Offline | ❌ Do not use |

If NPU is in Alarm/Offline state, exclude it from hostfile:

```bash
# Example: NPU 0 on 175.99.1.2 is faulty
# hostfile:
# 175.99.1.2:7  (exclude NPU 0)
# 175.99.1.3:8
```

---

## 7. Additional Tips

### Check MPI Environment

```bash
# Check if MPI is installed
which mpirun

# Check MPI version
mpirun --version

# Check library paths
ldconfig -p | grep mpi
```

### Check CANN Environment

```bash
# Check CANN installation path
echo $INSTALL_DIR
ls -la /usr/local/Ascend/ascend-toolkit/latest/

# Check HCCL Test tools
ls -la ${INSTALL_DIR}/tools/hccl_test/bin/
```

### Check Network Connectivity

```bash
# Test inter-node network
ping ${remote_node_ip}

# Test SSH login
ssh ${remote_node_ip} "hostname"
```

### Docker Container Testing

**Important**: When running HCCL Test in Docker, you must use **host network mode**.

```bash
# ✅ Correct: use host network
docker run --network host --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/ascend-toolkit/latest:/usr/local/Ascend/ascend-toolkit/latest \
  ascend-cann:latest

# ❌ Wrong: bridge network mode
docker run --network bridge ...
```

### View Debug Logs

```bash
# View Ascend logs
cat ~/ascend/log/debug/plog/plog-*.log | tail -100

# View HCCL specific logs
cat ~/ascend/log/debug/plog/plog-*_hccl*.log
```

---

## Official References

- [HCCL Performance Test Tool - Common Issues](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/hccltool/HCCLpertest_16_0008.html)
