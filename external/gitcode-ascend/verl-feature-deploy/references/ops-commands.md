# 常用运维命令

## Docker 管理

```bash
docker stop <容器名>
docker kill <容器名>
docker rm <容器名>
docker ps -a                          # 查看所有容器
docker exec -it <容器名> /bin/bash    # 进入运行中的容器
```

## 容器迁移

```bash
# 打包容器为镜像
docker commit <容器名> <镜像名>:latest

# 导出镜像
docker save -o /path/to/save/image.tar <镜像名>:latest

# 在目标机器上加载镜像
docker load -i /path/to/image.tar
```

## NPU 监控

```bash
npu-smi info                          # 查看 NPU 状态
pip install ascend-nputop && nputop   # 实时监控
```

## 进程管理

```bash
pkill -9 python                       # 杀掉所有 python 进程
ray stop --force                      # 停止 Ray 集群
rm -rf /tmp/ray                       # 清理 Ray 临时文件
```

## 指定 NPU 卡号

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # 仅使用 0-3 号卡
```

## 日志查看

```bash
tail -f logs/verl_qwen3_8b_megatron_*.log  # 实时查看训练日志
```

## CANN 环境激活

容器重启后需重新执行：

```bash
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh
```

## 后台运行

```bash
nohup bash start_verl.sh &
jobs -l                    # 查看后台任务
tail -f nohup.out          # 实时查看输出
```
