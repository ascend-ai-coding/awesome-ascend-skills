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

## 5. Additional Tips

### 检查 MPI 环境

```bash
# 检查 MPI 是否安装
which mpirun

# 检查 MPI 版本
mpirun --version

# 检查库路径
ldconfig -p | grep mpi
```

### 检查 CANN 环境

```bash
# 检查 CANN 安装路径
echo $INSTALL_DIR
ls -la /usr/local/Ascend/ascend-toolkit/latest/

# 检查 HCCL Test 工具
ls -la ${INSTALL_DIR}/tools/hccl_test/bin/
```

### 检查网络连通性

```bash
# 测试节点间网络
ping ${remote_node_ip}

# 测试 SSH 登录
ssh ${remote_node_ip} "hostname"
```

---

## Official References

- [HCCL 性能测试工具 - 常见问题及解决方法](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/hccltool/HCCLpertest_16_0008.html)
