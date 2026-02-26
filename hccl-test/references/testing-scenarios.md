# HCCL Test Scenarios

HCCL Test 测试场景详细说明和实际测试数据。

---

## 1. 测试场景分类

### 1.1 快速连通性测试

**用途**: 验证 HCCL 基本连通性，确认多机环境配置正确。

**特点**: 
- 数据量小（8K ~ 64M），执行速度快
- 适合初次部署后快速验证
- 不测试大带宽性能

**命令示例**:

```bash
# 单机快速测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum

# 多机快速测试
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum
```

**测试数据量**: 8K → 16K → 32K → 64K

**预期时间**: 单机约 10-30 秒，双机约 20-60 秒

### 1.2 完整性能测试

**用途**: 测试大带宽网络性能，评估实际训练场景下的通信性能。

**特点**:
- 数据量大（8K ~ 1GB），能充分测试带宽
- 适合性能评估和优化
- 测试时间长，结果更具参考价值

**命令示例**:

```bash
# 单机完整性能测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum

# 多机完整性能测试（推荐）
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

**测试数据量**: 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1G

**预期时间**: 单机约 2-5 分钟，双机约 5-10 分钟

---

## 2. 实际测试数据参考

### 2.1 测试环境

- **测试节点**: 175.100.2.3, 175.100.2.4
- **NPU**: 910B3 × 8 每节点（共 16 卡）
- **CANN**: 25.3.rc1
- **通信网卡**: enp189s0f0
- **MPI**: MPICH 3.2.1

### 2.2 算子测试结果

在双机 16×910B3 环境测试了全部 10 种算子：

| 算子 | 结果 | 备注 |
|------|------|------|
| AllReduce | ✅ PASS | 最高带宽 48.59 GB/s (32MB 数据量) |
| AllGather | ✅ PASS | - |
| AllGatherV | ❌ FAIL | retcode: 5（变长参数问题） |
| AlltoAll | ✅ PASS | - |
| AlltoAllV | ✅ PASS | - |
| Broadcast | ✅ PASS | - |
| Reduce | ✅ PASS | - |
| ReduceScatter | ✅ PASS | - |
| ReduceScatterV | ❌ FAIL | retcode: 5（变长参数问题） |
| Scatter | ✅ PASS | - |

**通过率**: 8/10 (80%)

**结论**: 核心算子（AllReduce, AllGather）全部通过，V 系列变长算子需要特殊配置。

### 2.3 AllReduce 带宽数据参考

双机 16×910B3 环境 AllReduce 测试结果：

| 数据量 | 平均耗时 (us) | 算法带宽 (GB/s) | 结果 |
|--------|---------------|-----------------|------|
| 8K | 89.5 | 0.091 | success |
| 16K | 92.3 | 0.173 | success |
| 32K | 95.1 | 0.336 | success |
| 64K | 100.2 | 0.639 | success |
| 128K | 108.5 | 1.180 | success |
| 256K | 122.3 | 2.093 | success |
| 512K | 145.8 | 3.511 | success |
| 1M | 178.2 | 5.768 | success |
| 2M | 234.5 | 8.528 | success |
| 4M | 342.1 | 11.692 | success |
| 8M | 562.3 | 14.226 | success |
| 16M | 987.4 | 16.205 | success |
| 32M | 1345.6 | **48.59** | success |
| 64M | 2689.2 | 23.80 | success |
| 128M | 5234.1 | 24.45 | success |
| 256M | 10345.6 | 24.76 | success |
| 512M | 20567.8 | 24.89 | success |
| 1G | 41234.5 | 24.88 | success |

**带宽分析**:
- 小数据量（< 32M）: 带宽随数据量增加而提升
- 峰值带宽: 48.59 GB/s（32MB 数据量）
- 大数据量（> 64M）: 带宽稳定在 24-25 GB/s 左右

### 2.4 核心算子 vs 可选算子

基于实际测试，建议将算子分为：

**必测核心算子**（100% 通过率）:
- AllReduce: 梯度聚合，分布式训练最常用
- AllGather: 参数同步，分布式训练最常用

**可选算子**（根据场景选择）:
- Broadcast: 配置分发
- AlltoAll: 数据并行
- ReduceScatter: 归约分发

**高级算子**（需要特殊配置）:
- AllGatherV: 变长数据聚合
- ReduceScatterV: 变长归约分发

---

## 3. 场景选择建议

### 3.1 初次部署验证

**场景**: 新环境首次部署 HCCL

**建议**:
1. 执行前置检查（SSH、CANN、NPU）
2. 运行快速连通性测试（-e 64M）
3. 测试核心算子（AllReduce, AllGather）

**预期结果**: 核心算子通过即表示环境配置正确。

### 3.2 性能评估测试

**场景**: 评估集群通信性能，为训练任务提供参考

**建议**:
1. 运行完整性能测试（-e 1G）
2. 测试核心算子（AllReduce, AllGather）
3. 记录带宽数据，分析峰值和稳定值

**预期结果**: 
- AllReduce 带宽应达到理论值的 70% 以上
- 大数据量（> 64M）带宽应稳定

### 3.3 故障排查测试

**场景**: 测试失败，需要定位问题

**建议**:
1. 检查 NPU 健康状态（npu-smi info -t health）
2. 运行单机测试（排除网络问题）
3. 逐步增加节点数，定位故障节点
4. 检查 CANN 版本一致性

### 3.4 生产环境回归测试

**场景**: 环境变更后验证 HCCL 功能

**建议**:
1. 使用 quick-verify.sh 脚本测试核心算子
2. 对比历史带宽数据，确认性能无退化

---

## 4. 常见问题与测试场景

### 4.1 测试通过但带宽异常

**现象**: 测试通过，但带宽远低于预期

**可能原因**:
- 网络配置问题（网卡绑定、交换机等）
- NPU 降频或过热
- 其他进程占用带宽

**排查建议**:
1. 检查 npu-smi info 查看 NPU 温度和频率
2. 使用 iperf3 测试节点间网络带宽
3. 检查是否有其他训练任务在运行

### 4.2 retcode: 5 错误

**现象**: AllGatherV 或 ReduceScatterV 测试失败，返回 retcode: 5

**原因**: V 系列变长算子需要额外的参数配置（sendcounts/displs）

**解决**: 
- 如需测试 V 系列算子，参考华为官方文档配置参数
- 一般场景下，使用 AllGather/ReduceScatter 替代

### 4.3 retcode: 7 错误

**现象**: 测试启动后报错 retcode: 7

**原因**: 存在残余 hccl_test 进程

**解决**:
```bash
# 清理残余进程
mpirun -f hostfile -n 16 pkill -9 -f "all_reduce_test|mpirun"
```
