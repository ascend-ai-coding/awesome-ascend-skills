# 故障排查

## GPU 资源检测失败

**错误**: `ValueError: Total available GPUs 0 is less than total desired GPUs 8`

**原因**: NPU 环境下 Ray 无法自动检测 GPU，需要显式声明。

**解决**:
```bash
# Ray 启动时显式指定 GPU 数
ray start --head --num-gpus=8 --dashboard-port=8265

# 或在代码中声明
ray.init(num_gpus=8)
```

## OOM 错误

脚本会自动处理 OOM：
1. 第一次 OOM → 开启 `offload` 重试
2. 第二次 OOM → 开启 `recompute` 重试
3. 仍失败 → 建议：
   - 减小 batch size
   - 使用更小模型
   - 减少 `max_response_length`

## Ray 进程冲突

```bash
ray stop --force
rm -rf /tmp/ray* ~/.ray /root/.ray /dev/shm/ray*
```

## 网络问题

```bash
# 设置代理（根据实际环境配置）
export http_proxy=http://your-proxy:8080
export https_proxy=http://your-proxy:8080
```

## SwanLab 未登录

```bash
# 设置环境变量后登录
swanlab login --host $SWANLAB_HOST
```

## 查看日志

```bash
# 查看日志
docker exec {container} tail -100 /verl/train.log

# 检查 Ray 状态
docker exec {container} ray status

# 清理 Ray session
docker exec {container} bash -c "ray stop --force && rm -rf /tmp/ray*"

# 检查是否出错
docker exec {container} grep -E "Error|error|OOM|Exception" /verl/train.log | tail -10
```