# 环境检查常见问题

## NPU 设备问题

### 1. npu-smi 不可用

**症状**：执行 `npu-smi` 提示命令未找到

**排查步骤**：
1. 检查 CANN 是否安装：`ls /usr/local/Ascend/`
2. 检查环境变量：`echo $ASCEND_HOME_PATH`
3. 如果未设置，执行：`source /usr/local/Ascend/cann/set_env.sh`

### 2. 设备未被识别

**症状**：`npu-smi info` 没有输出设备信息，或 `npu-smi info -m` 显示空

**可能原因**：
- 硬件未正确连接
- 驱动未安装或未加载
- 设备被占用

**排查步骤**（按优先级）：
```bash
# 1. 检查设备映射表（确认设备是否被系统识别）
npu-smi info -m

# 2. 检查驱动模块是否加载
lsmod | grep -E 'ascend|davinci'

# 3. 查看系统日志中的 NPU 相关信息
dmesg | grep -i -E 'npu|ascend|davinci' | tail -20

# 4. 检查设备健康状态
npu-smi info -t health -i <device_id>
```

### 3. 设备资源被占满

**症状**：算子运行提示设备忙

**排查命令**：
```bash
# 查看设备使用情况
npu-smi info -t usages -i <device_id>

# 强制释放（如确认可释放）
# 注：进程释放方式请查阅当前版本 npu-smi --help，不同版本命令不同
```

## CANN 环境问题

### 1. ASCEND_HOME_PATH 未设置

**症状**：
```
ERROR: ASCEND_HOME_PATH not set
```

**解决方案**：
```bash
# root 用户
source /usr/local/Ascend/cann/set_env.sh

# 非 root 用户
source $HOME/Ascend/cann/set_env.sh

# 验证
echo $ASCEND_HOME_PATH
```

### 2. ASCEND_OPP_PATH 未设置

**症状**：
```
ERROR: ASCEND_OPP_PATH not set
```

**说明**：编译时可跳过，运行算子时必需

**解决方案**：
1. 安装 CANN Ops 包
2. source set_env.sh

### 3. 自定义算子未找到

**症状**：
```
ERROR: 561003 - Kernel lookup failed
```

**排查**：
1. 检查算子包是否安装
2. 检查 LD_LIBRARY_PATH
3. 运行 `bash scripts/check_env.sh`

### 4. 算子运行失败（环境配置问题）

**症状**：开发的算子在昇腾 NPU 上运行失败

**排查步骤**（按优先级）：

**第一步：优先运行环境检查脚本**
```bash
bash scripts/check_env.sh
```
该脚本会自动检测：
- `ASCEND_HOME_PATH` 是否已设置
- `ASCEND_OPP_PATH` 是否已设置
- CANN 版本是否完整
- 自定义算子包是否安装

**第二步：手动检查关键环境变量**
```bash
# 检查 CANN 基础环境变量
echo $ASCEND_HOME_PATH
echo $ASCEND_OPP_PATH

# 检查库路径
echo $LD_LIBRARY_PATH | grep ascend

# 如未设置，执行 source
source /usr/local/Ascend/cann/set_env.sh
```

**第三步：检查 NPU 设备状态**
```bash
npu-smi info -m
npu-smi info -t health -i <device_id>
```

### 5. 运行时库依赖问题

**症状**：
```
libascend_*.so: cannot open shared object file
```

**解决方案**：
```bash
# 添加库路径
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/runtime/lib64:$LD_LIBRARY_PATH

# 或重新 source 环境
source $ASCEND_HOME_PATH/set_env.sh
```

## 混合部署问题

### 1. 多卡环境配置

**设置单卡**：
```bash
export ASCEND_DEVICE_ID=0  # 使用 0 号卡
```

**查看卡数**：
```bash
npu-smi info -m | grep -c "Ascend"
```

### 2. 容器环境

**确保**：
- 容器已映射 NPU 设备（--device）
- 容器内已安装 CANN
- 已正确 source 环境变量