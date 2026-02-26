---
name: hccl-test
description: HCCL (Huawei Collective Communication Library) performance testing for Ascend NPU clusters. Use for testing distributed communication bandwidth, verifying HCCL functionality, and benchmarking collective operations like AllReduce, AllGather. Covers MPI installation, multi-node pre-flight checks (SSH/CANN version/NPU health), and production testing workflows.
keywords:
    - hccl
    - 性能测试
    - 集合通信
    - 打流
    - allreduce
    - allgather
    - 多机测试
    - 910B
---

# HCCL Performance Test

HCCL性能测试工具用于测试HCCL（Huawei Collective Communication Library）集合通信的功能正确性以及性能。

## Overview

- **适用场景**：分布式训练场景下的集合通信性能测试
- **源码位置**：`${INSTALL_DIR}/tools/hccl_test`
- **支持版本**：CANN 8.3.RC1, CANN 8.5, CANN 25.RC

### 支持的产品型号

| 产品系列 | 最大 Rank 数 | 备注 |
|----------|-------------|------|
| Atlas 训练系列产品 | 4096 | - |
| Atlas A2 训练系列产品 | 32K | - |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 32K | AlltoAll/AlltoAllV 最大 8K |
| Atlas 300I Duo 推理卡 | - | - |

### 核心算子（推荐测试）

分布式训练场景最常用：

| 算子 | 可执行文件 | 通信模式 | 适用场景 | 推荐度 |
|------|-----------|---------|---------|--------|
| **AllReduce** | `all_reduce_test` | 多对多 | 梯度聚合、参数同步 | ⭐⭐⭐ 必测 |
| **AllGather** | `all_gather_test` | 多对多 | 数据聚合、参数收集 | ⭐⭐⭐ 必测 |
| Broadcast | `broadcast_test` | 一对多 | 配置分发、初始化 | ⭐⭐ 可选 |
| AlltoAll | `alltoall_test` | 多对多 | 数据重排、负载均衡 | ⭐⭐ 可选 |

> **提示**: 完整算子列表见 [references/parameters.md](references/parameters.md)

---

## Quick Reference

```bash
# 1. 前置检查（多机测试必需）
./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3

# 2. 编译工具
cd ${INSTALL_DIR}/tools/hccl_test
make MPI_HOME=/usr/local/mpich ASCEND_DIR=${INSTALL_DIR}

# 3. 快速连通性测试（单机）
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum

# 4. 完整性能测试（多机，推荐）
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

---

## 1. Pre-test Checklist（多机测试必需）

> ⚠️ **重要**: 多机测试前必须完成以下检查，否则可能出现建链超时或测试失败。

### 1.1 SSH 免密登录配置

所有节点间必须配置 SSH 免密登录：

```bash
# 1. 生成 SSH 密钥（如已存在可跳过）
ssh-keygen -t rsa

# 2. 将公钥复制到所有节点（包括本机）
ssh-copy-id -i ~/.ssh/id_rsa.pub root@<node1_ip>
ssh-copy-id -i ~/.ssh/id_rsa.pub root@<node2_ip>

# 3. 验证免密登录
ssh root@<node1_ip> "echo 'SSH OK'"
ssh root@<node2_ip> "echo 'SSH OK'"
```

### 1.2 CANN 版本一致性检查

多机 CANN 版本必须一致，否则会导致测试失败：

```bash
# 检查所有节点的 CANN 版本
for node in 175.99.1.2 175.99.1.3; do
    echo "=== $node ==="
    ssh root@$node "cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | grep runtime_running_version"
done
```

> **注意**: 版本不一致时需统一版本（建议统一为最新 RC 版本）。

### 1.3 NPU 健康状态检查

测试前需确认所有 NPU 状态正常：

```bash
# 检查所有节点的 NPU 健康状态
for node in 175.99.1.2 175.99.1.3; do
    echo "=== $node NPU Health ==="
    ssh root@$node "npu-smi info -t health -i 0"
done
```

**NPU 状态说明**：
| 状态 | 说明 | 操作建议 |
|-----|------|---------|
| OK | 正常 | ✅ 可以使用 |
| Alarm | 告警 | ⚠️ 需排查故障 |
| Offline | 离线 | ❌ 不可使用 |

> 如存在 Alarm 状态，需排除故障卡。例如 NPU 0 故障，使用 7 张卡测试。

### 1.4 一键前置检查脚本

```bash
# 使用提供的检查脚本（推荐）
./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3
```

---

## 2. MPI Installation

HCCL性能测试工具依赖MPI拉起多个进程，默认使用 **MPICH**。

### 2.1 MPICH Installation (Recommended)

**下载地址**：https://www.mpich.org/static/downloads/

| 产品系列 | 推荐版本 |
|----------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | MPICH 4.1.3 |
| Atlas A2 训练系列产品 | MPICH 3.2.1 |
| Atlas 训练系列产品 | MPICH 3.2.1 |
| Atlas 300I Duo 推理卡 | MPICH 3.2.1 |

**安装步骤**：

```bash
# 1. 解压
tar -zxvf mpich-${version}.tar.gz
cd mpich-${version}

# 2. 配置编译选项
# Atlas A3 产品（必须使用 TCP 协议）
./configure --disable-fortran --prefix=/usr/local/mpich --with-device=ch3:nemesis

# 其他产品
./configure --disable-fortran --prefix=/usr/local/mpich

# 3. 编译安装
make -j 32 && make install
```

### 2.2 Open MPI Installation (Alternative)

适用于需要 IPv6 支持的场景。

```bash
tar -zxvf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --disable-fortran --enable-ipv6 --prefix=/usr/local/openmpi
make -j 32 && make install
```

### 2.3 环境配置

```bash
# MPICH 环境
export INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
```

---

## 3. Tool Compilation

```bash
cd ${INSTALL_DIR}/tools/hccl_test

# MPICH
make MPI_HOME=/usr/local/mpich ASCEND_DIR=${INSTALL_DIR}

# Open MPI
make MPI_HOME=/usr/local/openmpi ASCEND_DIR=${INSTALL_DIR}
```

编译成功后，在 `bin` 目录下生成 10 个可执行文件。

---

## 4. Testing Scenarios

### 4.1 快速连通性测试

用于验证 HCCL 基本连通性，数据量较小，执行速度快：

```bash
# 单机快速测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum

# 多机快速测试
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum
```

### 4.2 完整性能测试（推荐）

用于测试大带宽网络性能，数据量到 1GB，更能反映实际训练场景：

```bash
# 单机完整性能测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum

# 多机完整性能测试（推荐）
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

**参数说明**：
- `-b 8K`: 起始数据量 8KB
- `-e 1G`: 结束数据量 1GB（64M 只能测试小数据量）
- `-f 2`: 乘法因子，测试 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1G

### 4.3 核心算子测试

```bash
# 使用 quick-verify.sh 测试核心算子（推荐）
./scripts/quick-verify.sh 8

# 手动测试核心算子
./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
./bin/all_gather_test -p 8 -b 8K -e 1G -f 2 -d fp32
```

### 4.4 Hostfile 配置

**MPICH Format** (`节点IP:卡数`)：

```bash
# 单机测试
175.99.1.3:8

# 双机测试
175.99.1.3:8
175.99.1.4:8
```

> 请将属于同一超节点的 AI Server 信息配置在一起，不支持交叉配置。

---

## 5. Parameters

### 5.1 核心参数速查

| 参数 | 说明 | 示例 |
|------|------|------|
| `-p <npus>` | 单节点参与训练的 NPU 个数 | `-p 8` |
| `-b <size>` | 测试数据量起始值 | `-b 8K` |
| `-e <size>` | 测试数据量结束值 | `-e 1G` |
| `-f <factor>` | 乘法因子 | `-f 2` |
| `-d <type>` | 数据类型: fp32/fp16/int32 | `-d fp32` |
| `-o <op>` | 操作类型: sum/prod/max/min | `-o sum` |
| `-n <iters>` | 迭代次数（默认 20） | `-n 20` |
| `-c <0/1>` | 是否开启结果校验（默认 1） | `-c 1` |

> 详细参数说明见 [references/parameters.md](references/parameters.md)

### 5.2 数据量配置示例

```bash
# 固定数据量测试
-b 100M -e 100M

# 乘法因子测试（测试 100M, 200M, 400M）
-b 100M -e 400M -f 2

# 完整性能测试（推荐）
-b 8K -e 1G -f 2
```

---

## 6. Results

### 6.1 输出格式

```
data_size      avg_time(us)    alg_bandwidth(GB/s)    check_result
8192           125.3           0.065                  success
16384          132.1           0.124                  success
...
```

| 字段 | 说明 |
|------|------|
| `data_size` | 单个 NPU 上参与集合通信的数据量（Bytes） |
| `avg_time` | 集合通信算子执行耗时（us） |
| `alg_bandwidth` | 算法带宽（GB/s），计算方式：集合通信数据量/耗时 |
| `check_result` | 结果校验标识：success/failed/NULL |

### 6.2 结果解析

```bash
# 解析结果文件
./scripts/parse-hccl-result.py output.log

# 输出 Markdown 表格
./scripts/parse-hccl-result.py output.log -f markdown
```

---

## 7. Actual Test Results

### 7.1 双机 16×910B3 测试数据

我们在双机 16×910B3 环境测试了全部 10 种算子：

| 算子 | 结果 | 备注 |
|------|------|------|
| AllReduce | ✅ PASS | 最高带宽 48.59 GB/s (32MB) |
| AllGather | ✅ PASS | - |
| AllGatherV | ❌ FAIL | retcode: 5（变长参数问题） |
| AlltoAll | ✅ PASS | - |
| AlltoAllV | ✅ PASS | - |
| Broadcast | ✅ PASS | - |
| Reduce | ✅ PASS | - |
| ReduceScatter | ✅ PASS | - |
| ReduceScatterV | ❌ FAIL | retcode: 5（变长参数问题） |
| Scatter | ✅ PASS | - |

**通过率：8/10 (80%)**，核心算子全部通过。

### 7.2 测试环境参考

- **测试节点**：175.100.2.3, 175.100.2.4
- **NPU**：910B3 × 8 每节点（共 16 卡）
- **CANN**：25.3.rc1
- **通信网卡**：enp189s0f0
- **MPI**：MPICH 3.2.1

---

## 8. Common Issues

### 8.1 前置条件问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| SSH 免密登录失败 | 未配置 SSH 密钥 | 执行 ssh-copy-id 配置免密登录 |
| CANN 版本不一致 | 多机 CANN 版本不同 | 统一所有节点的 CANN 版本 |
| NPU Alarm 状态 | NPU 硬件故障 | 检查 npu-smi info -t health，排除故障卡 |

### 8.2 运行时问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| gethostbyname failed | 主机名无法解析 | 配置 /etc/hosts |
| retcode: 7 | 残余进程干扰 | 执行清理命令：`mpirun -f hostfile -n 16 pkill -9 -f "all_reduce_test"` |
| retcode: 5 | 变长参数配置错误 | AllGatherV/ReduceScatterV 需要特殊参数配置 |

### 8.3 其他注意事项

1. **Docker 容器**：如果使用 Docker 容器进行测试，需要使用 host 网络模式
2. **日志查看**：测试失败时可查看 `~/ascend/log/debug/plog` 中的最新日志
3. **进程清理**：测试前检查卡上是否有其他进程占用，如有需要手动清理

> 详细故障排除见 [references/common-issues.md](references/common-issues.md)

---

## 9. Scripts

### 9.1 前置检查脚本

```bash
# 多机测试前置检查
./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3
```

检查内容：
- SSH 免密登录
- CANN 版本一致性
- NPU 健康状态
- 网络连通性

### 9.2 快速验证脚本

```bash
# 测试核心算子
./scripts/quick-verify.sh 8

# 完整性能测试
./scripts/quick-verify.sh 8 full
```

### 9.3 多机测试脚本

```bash
# 一键多机测试（自动前置检查 + 测试）
./scripts/multi-node-test.sh --nodes 175.99.1.2,175.99.1.3 --npus 8 --mode full
```

---

## Official References

- **CANN 8.3.RC1**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/hccltool/HCCLpertest_16_0001.html
- **CANN 8.5**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/hccltool/HCCLpertest_16_0001.html
- **HCCL 常见问题**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/hccltool/HCCLpertest_16_0008.html
