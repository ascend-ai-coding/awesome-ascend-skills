# 完整工作流程

## 1. 环境准备

### 1.1 创建 Docker 容器

```bash
# 拉取镜像 (根据 NPU 类型)
# A2 (910B/910B3)
docker pull quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0

# A3 (Ascend 910)
docker pull quay.io/ascend/verl:verl-8.3.rc1-a3-ubuntu22.04-py3.11-v0.7.0

# 启动容器
docker run -d --name verl_container \
    --network host \
    --privileged \
    -v /usr/local/dcm:/usr/local/dcm \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt:/mnt \
    -v /mnt2:/mnt2 \
    quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0 \
    tail -f /dev/null
```

### 1.2 容器内环境配置

```bash
docker exec -it verl_container bash

# 设置代理（根据实际环境配置）
export http_proxy=http://your-proxy:8080
export https_proxy=http://your-proxy:8080

# 安装 SwanLab
pip install swanlab

# 登录 SwanLab（首次运行时会提示输入 API KEY）
swanlab login --host http://your-swanlab-host:8000
```

## 2. 训练前检查

```bash
# 在 skill 目录下执行
bash scripts/preflight_check.sh

# 指定 0.6B 模型
bash scripts/preflight_check.sh --model-size 0.6B

# 自定义路径
bash scripts/preflight_check.sh \
    --model /path/to/model \
    --train /path/to/train.parquet \
    --val /path/to/val.parquet
```

**检查内容**:
1. NPU 类型检测 → 推荐镜像
2. 模型/训练数据/验证数据路径
3. Docker 容器查找

## 3. 生成配置

### 3.1 使用 Python 配置生成器

```bash
# 查看帮助
python config_generator.py --help

# 生成默认脚本
python config_generator.py --output run.sh

# 自定义配置
python config_generator.py \
    --framework megatron \
    --model /mnt2/metis/huggface_models/Qwen/Qwen3-8B \
    --steps 10 \
    --feature offload \
    --output run.sh
```

### 3.2 输出格式

**Shell 脚本** (默认):
```bash
python config_generator.py --output run.sh
# 生成可执行的 shell 脚本
```

**命令行参数**:
```bash
python config_generator.py --format args
# 直接打印 Hydra 命令行参数
# 可用于复制粘贴或管道
```

**YAML 配置**:
```bash
python config_generator.py --format yaml
# 输出 YAML 格式的配置
```

## 4. 启动训练

### 方式1: 运行生成的脚本

```bash
bash run.sh
```

### 方式2: 使用启动脚本

```bash
# 设置环境变量
export FRAMEWORK=megatron
export TRAIN_STEPS=10
export ENABLE_OFFLOAD=true

# 运行
bash scripts/run_dapo.sh
```

### 方式3: 直接执行命令

```bash
# 生成命令并直接执行
python config_generator.py --format args | bash
```

## 5. 监控训练

### 5.1 查看日志

```bash
# 实时查看
tail -f logs/DAPO-Qwen3-8b-megatron-async_*.log

# 搜索错误
grep -i "error" logs/*.log
```

### 5.2 SwanLab Dashboard

访问: `{SWANLAB_HOST}/@{SWANLAB_USER}/verl_hlm`

### 5.3 训练进度

```
Training Progress:  20%|██        | 2/10 [05:10<20:40, 155.81s/it]
```

## 6. Checkpoint

### 6.1 保存位置

```
/mnt/project/jins/ckpt/{exp_name}/global_step_N/
```

### 6.2 恢复训练

设置 `trainer.resume_mode=auto`

## 常见问题

### Ray 进程冲突

```bash
ray stop --force
rm -rf /tmp/ray* ~/.ray /root/.ray /dev/shm/ray*
```

### 网络问题

```bash
# 设置代理（根据实际环境配置）
export http_proxy=http://your-proxy:8080
export https_proxy=http://your-proxy:8080
```

### SwanLab 未登录

```bash
# 设置环境变量后登录
swanlab login --host $SWANLAB_HOST
```