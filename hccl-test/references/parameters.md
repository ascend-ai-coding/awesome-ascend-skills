# HCCL Test Parameters

HCCL Test 工具的详细参数说明。

---

## 1. MPI 参数

### 1.1 MPICH 参数

| 参数 | 必选 | 说明 | 示例 |
|------|------|------|------|
| `-f <hostfile>` | 多机必填 | Hostfile 节点列表文件 | `-f hostfile` |
| `-n <number>` | 是 | 启动的 NPU 总数 | `-n 16` |

**示例**：

```bash
# 单机测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 1G

# 多机测试（2 节点 × 8 卡 = 16）
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G
```

### 1.2 Open MPI 参数

| 参数 | 必选 | 说明 | 示例 |
|------|------|------|------|
| `--prefix <path>` | 多机建议 | Open MPI 安装路径 | `--prefix /usr/local/openmpi` |
| `-hostfile <file>` | 多机必填 | Hostfile 节点列表文件 | `-hostfile hostfile` |
| `-n <number>` | 是 | 启动的 NPU 总数 | `-n 16` |
| `-x <env>` | 是 | 传递给远程节点的环境变量 | `-x LD_LIBRARY_PATH` |
| `--allow-run-as-root` | 可选 | 允许 root 用户执行 | `--allow-run-as-root` |
| `--mca btl_tcp_if_include <nic>` | 可选 | 指定通信网卡 | `--mca btl_tcp_if_include eth0` |
| `--mca opal_set_max_sys_limits 1` | 可选 | 大规模集群建议配置 | `--mca opal_set_max_sys_limits 1` |

**示例**：

```bash
mpirun --prefix /usr/local/openmpi -hostfile hostfile \
  -x LD_LIBRARY_PATH -x HCCL_SOCKET_FAMILY -x HCCL_SOCKET_IFNAME \
  --allow-run-as-root --mca btl_tcp_if_include eth0 \
  -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

---

## 2. HCCL Test 参数

### 2.1 核心参数

| 参数 | 长格式 | 必选 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `-p <npus>` | `--npus` | 可选 | 节点 NPU 总数 | 单节点参与训练的 NPU 个数 |
| `-b <size>` | `--minbytes` | 可选 | 64M | 测试数据量起始值（单位：K/M/G） |
| `-e <size>` | `--maxbytes` | 可选 | 64M | 测试数据量结束值（单位：K/M/G） |
| `-i <bytes>` | `--stepbytes` | 可选 | 计算 | 增量步长（单位：Bytes） |
| `-f <factor>` | `--stepfactor` | 可选 | - | 乘法因子 |
| `-o <op>` | `--op` | 可选 | sum | 操作类型：sum/prod/max/min |
| `-r <root>` | `--root` | 可选 | 0 | 根节点 Device ID（broadcast/reduce/scatter） |
| `-d <type>` | `--datatype` | 可选 | fp32 | 数据类型 |
| `-n <iters>` | `--iters` | 可选 | 20 | 迭代次数 |
| `-w <warmup>` | `--warmup_iters` | 可选 | 10 | 预热迭代次数 |
| `-c <0/1>` | `--check` | 可选 | 1 | 是否开启结果校验 |
| `-z <0/1>` | `--zero_copy` | 可选 | 0 | 是否开启零拷贝 |

### 2.2 数据量配置示例

```bash
# 固定数据量测试（只测试 100M）
-b 100M -e 100M

# 步长增量测试（100M, 100.5M, 101M, ...）
-b 100M -e 400M -i 500

# 乘法因子测试（100M, 200M, 400M）
-b 100M -e 400M -f 2

# 持续测试（使用起始值）
-b 100M -e 400M -i 0  # 只测试 100M

# 完整性能测试（推荐）
-b 8K -e 1G -f 2  # 测试 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1G
```

### 2.3 操作类型说明

| 操作类型 | 适用算子 | 说明 |
|----------|----------|------|
| sum | AllReduce, Reduce, ReduceScatter | 求和 |
| prod | AllReduce, Reduce, ReduceScatter | 求积 |
| max | AllReduce, Reduce, ReduceScatter | 求最大值 |
| min | AllReduce, Reduce, ReduceScatter | 求最小值 |

### 2.4 数据类型支持

| 算子 | 支持的数据类型 |
|------|---------------|
| all_reduce_test, reduce_scatter_test, reduce_test | int8, int16, int32, int64, fp16, fp32, bfp16 |
| broadcast_test, all_gather_test, alltoall_test, scatter_test | int8, uint8, int16, uint16, int32, uint32, int64, uint64, fp16, fp32, fp64, bfp16 |

---

## 3. 完整算子列表

HCCL Test 工具支持 10 种集合通信算子：

| 算子 | 可执行文件 | 通信模式 | 适用场景 | 推荐度 |
|------|-----------|---------|---------|--------|
| **AllReduce** | `all_reduce_test` | 多对多 | 梯度聚合、参数同步 | ⭐⭐⭐ 必测 |
| **AllGather** | `all_gather_test` | 多对多 | 数据聚合、参数收集 | ⭐⭐⭐ 必测 |
| AllGatherV | `all_gatherv_test` | 多对多 | 变长数据聚合 | ⭐⭐ 高级 |
| AlltoAll | `alltoall_test` | 多对多 | 数据重排、负载均衡 | ⭐⭐ 可选 |
| AlltoAllV | `alltoallv_test` | 多对多 | 变长数据全对全通信 | ⭐⭐ 高级 |
| Broadcast | `broadcast_test` | 一对多 | 配置分发、初始化 | ⭐⭐ 可选 |
| Reduce | `reduce_test` | 多对一 | 结果收集到根节点 | ⭐ 可选 |
| ReduceScatter | `reduce_scatter_test` | 多对多 | 归约后分发 | ⭐⭐ 可选 |
| ReduceScatterV | `reduce_scatterv_test` | 多对多 | 变长归约后分发 | ⭐⭐ 高级 |
| Scatter | `scatter_test` | 一对多 | 数据分发 | ⭐ 可选 |

### 3.1 核心算子测试（推荐）

```bash
# AllReduce（最常用）
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum

# AllGather（最常用）
mpirun -n 8 ./bin/all_gather_test -p 8 -b 8K -e 1G -f 2 -d fp32
```

### 3.2 可选算子测试

```bash
# Broadcast
mpirun -n 8 ./bin/broadcast_test -p 8 -b 8K -e 1G -f 2 -d fp32 -r 0

# AlltoAll
mpirun -n 8 ./bin/alltoall_test -p 8 -b 8K -e 1G -f 2 -d fp32

# ReduceScatter
mpirun -n 8 ./bin/reduce_scatter_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

### 3.3 高级算子（需要特殊配置）

V 系列变长算子（AllGatherV, ReduceScatterV）需要额外的参数配置，在基础测试中容易失败：

```bash
# AllGatherV 示例（需要配置 sendcounts/displs）
# 详见华为官方文档
mpirun -n 8 ./bin/all_gatherv_test -p 8 -b 8K -e 64M -d fp32
```

> **实际测试结果**: 在双机 16×910B3 环境中，AllGatherV 和 ReduceScatterV 的通过率为 0%（retcode: 5），建议非必要不测。

---

## 4. NPU 数量约束

### 4.1 每节点 NPU 数 (`-p` 参数)

| 产品系列 | `-p` 范围 | Device ID |
|----------|-----------|-----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 1~16 | [0, p-1] |
| Atlas A2 训练系列产品 | 1~8 | [0, p-1] |
| Atlas 训练系列产品 | 1, 2, 4, 8 | [0, p-1] |

### 4.2 Atlas 300I Duo 特殊约束

| 测试命令 | 最大 `-p` 值 |
|----------|-------------|
| all_gather_test | 32 |
| all_gatherv_test | 4 |
| all_reduce_test | 32 |
| alltoall_test | 4 |
| alltoallv_test | 4 |
| reduce_scatter_test | 32 |
| reduce_scatterv_test | 4 |

### 4.3 Zero Copy 约束

零拷贝功能（`-z 1`）生效条件：
- 仅支持 Atlas A3 训练系列产品/Atlas A3 推理系列产品
- 仅支持 reduce_scatter_test、all_gather_test、all_reduce_test、broadcast_test
- 仅支持通信算法编排展开位置在 AI CPU 的场景
