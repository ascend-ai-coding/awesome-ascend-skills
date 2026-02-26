# HCCL Test Pre-test Checklist

多机测试前置检查详细指南。

---

## 1. SSH 免密登录配置

### 1.1 生成 SSH 密钥

如果还没有 SSH 密钥，需要先生成：

```bash
# 检查是否已有密钥
ls ~/.ssh/id_rsa

# 如不存在，生成新密钥（一路回车使用默认配置）
ssh-keygen -t rsa -b 4096
```

### 1.2 分发公钥到所有节点

```bash
# 获取本机 IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

# 分发到所有节点（包括本机）
for node in 175.99.1.2 175.99.1.3 $LOCAL_IP; do
    ssh-copy-id -i ~/.ssh/id_rsa.pub root@$node
done
```

### 1.3 验证免密登录

```bash
# 测试所有节点
for node in 175.99.1.2 175.99.1.3; do
    ssh root@$node "hostname" && echo "$node SSH 免密登录正常"
done
```

### 1.4 常见问题

**问题**: `ssh-copy-id: command not found`

**解决**: 手动复制公钥

```bash
# 在目标节点上执行
cat >> ~/.ssh/authorized_keys << 'EOF'
# 粘贴本地 ~/.ssh/id_rsa.pub 的内容
EOF

# 设置权限
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

---

## 2. CANN 版本一致性检查

### 2.1 检查所有节点版本

```bash
#!/bin/bash
# check-cann-version.sh

NODES=("175.99.1.2" "175.99.1.3")
CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"

echo "检查 CANN 版本一致性..."
echo ""

versions=()
for node in "${NODES[@]}"; do
    version=$(ssh root@$node "grep runtime_running_version ${CANN_PATH}/version.cfg 2>/dev/null || echo 'NOT_FOUND'")
    echo "$node: $version"
    versions+=("$version")
done

# 检查一致性
first_ver="${versions[0]}"
consistent=true
for v in "${versions[@]}"; do
    if [[ "$v" != "$first_ver" ]]; then
        consistent=false
        break
    fi
done

if [[ "$consistent" == true ]]; then
    echo ""
    echo "✅ CANN 版本一致"
else
    echo ""
    echo "❌ CANN 版本不一致！请统一版本"
    exit 1
fi
```

### 2.2 版本升级/降级

如果版本不一致，需要统一版本：

```bash
# 建议统一为最新 RC 版本
# 1. 下载目标版本 CANN 包
# 2. 在所有节点上安装相同版本
# 3. 重启 NPU 驱动

# 重启驱动命令
/usr/local/Ascend/driver/tools/ascend_toolkit restart
```

---

## 3. NPU 健康状态检查

### 3.1 检查单个 NPU

```bash
# 检查 NPU 0 的健康状态
npu-smi info -t health -i 0

# 输出示例：
# NPU ID                         : 0
# Health                         : OK
# ...
```

### 3.2 批量检查所有 NPU

```bash
#!/bin/bash
# check-npu-health.sh

NODES=("175.99.1.2" "175.99.1.3")

for node in "${NODES[@]}"; do
    echo "=== $node ==="
    
    # 获取 NPU 数量
    npu_count=$(ssh root@$node "npu-smi info -t info | grep -c 'NPU Name'")
    
    for ((i=0; i<npu_count; i++)); do
        health=$(ssh root@$node "npu-smi info -t health -i $i | grep Health | awk '{print \$2}'")
        
        if [[ "$health" == "OK" ]]; then
            echo "  NPU $i: ✅ OK"
        elif [[ "$health" == "Alarm" ]]; then
            echo "  NPU $i: ⚠️  Alarm (需要排查)"
        elif [[ "$health" == "Offline" ]]; then
            echo "  NPU $i: ❌ Offline (不可用)"
        else
            echo "  NPU $i: ? Unknown ($health)"
        fi
    done
    echo ""
done
```

### 3.3 NPU 状态说明

| 状态 | 说明 | 操作建议 |
|-----|------|---------|
| OK | NPU 工作正常 | ✅ 可以使用 |
| Alarm | NPU 存在告警 | ⚠️ 需要排查故障原因 |
| Offline | NPU 离线 | ❌ 不可用，需要维修 |

### 3.4 处理故障 NPU

如果某个 NPU 状态为 Alarm 或 Offline，需要在 hostfile 中排除该卡：

```bash
# 假设 175.99.1.2 的 NPU 0 故障
# 原 hostfile:
# 175.99.1.2:8
# 175.99.1.3:8

# 修改后 hostfile（排除故障卡）：
# 175.99.1.2:7
# 175.99.1.3:8

# 同时需要调整 -p 参数
mpirun -f hostfile -n 15 ./bin/all_reduce_test -p 7 -b 8K -e 1G -f 2
```

---

## 4. 网络连通性检查

### 4.1 基础连通性测试

```bash
# 测试节点间网络连通性
for node in 175.99.1.2 175.99.1.3; do
    ping -c 3 $node && echo "$node 网络正常" || echo "$node 网络不通"
done
```

### 4.2 测试 HCCL 通信网卡

```bash
# 查看通信网卡
npu-smi info -t board -i 0

# 查看网卡 IP
ip addr show eth0

# 测试网卡间连通性（使用 HCCL 通信网卡）
iperf3 -c 175.99.1.3 -t 10
```

### 4.3 防火墙检查

```bash
# 检查防火墙状态
systemctl status firewalld

# 建议测试时关闭防火墙
systemctl stop firewalld
systemctl disable firewalld

# 或者开放 HCCL 通信端口
firewall-cmd --add-port=50000-60000/tcp --permanent
firewall-cmd --reload
```

---

## 5. 一键检查脚本

使用提供的 `pre-test-check.sh` 脚本一键完成所有检查：

```bash
# 基础用法
./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3

# 指定 CANN 路径
CANN_PATH=/opt/ascend-toolkit/latest ./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3

# 指定 SSH 用户
SSH_USER=admin ./scripts/pre-test-check.sh 175.99.1.2 175.99.1.3
```

---

## 6. 检查清单汇总

多机测试前请确认：

- [ ] SSH 免密登录配置完成（所有节点）
- [ ] CANN 版本一致（所有节点）
- [ ] NPU 状态为 OK（所有 NPU）
- [ ] 节点间网络连通（ping 测试通过）
- [ ] 防火墙已关闭或端口已开放
- [ ] HCCL Test 工具已编译
- [ ] MPI 环境已配置
