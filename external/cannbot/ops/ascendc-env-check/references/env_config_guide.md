# 环境变量配置指南

本文档基于官方资料整理，提供 Ascend C 算子开发的环境变量配置方法。

## 配置流程

### 场景1: 仅编译算子（无需 NPU）

```bash
source /usr/local/Ascend/cann/set_env.sh  # root
# 或
source $HOME/Ascend/cann/set_env.sh       # 非root
```

### 场景2: 运行内置算子

```bash
# 先 source CANN Toolkit
source /usr/local/Ascend/cann/set_env.sh

# 安装 CANN Ops（如未安装）
./Ascend-cann-${soc_name}-ops_*.run --install --install-path=${install_path}
source ${install_path}/cann/set_env.sh  # 重新 source 使 ASCEND_OPP_PATH 生效
```

### 场景3: 运行自定义算子

```bash
source /usr/local/Ascend/cann/set_env.sh
./build_out/<your_operator_package>.run  # 安装自定义算子包

# 配置动态库路径（vendor_name 为编译时的 --vendor_name 参数）
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}/op_api/lib:$LD_LIBRARY_PATH
```

## 环境变量说明

| 环境变量 | 所属包 | 配置方式 | 必需场景 |
|---------|--------|---------|---------|
| ASCEND_HOME_PATH | CANN Toolkit | `source set_env.sh` | 编译+运行 |
| ASCEND_OPP_PATH | CANN Ops | 由 set_env.sh 自动设置 | 仅运行 |
| LD_LIBRARY_PATH | 自定义算子包 | `export ...` | 运行自定义算子 |
| ASCEND_CUSTOM_OPP_PATH | asc-devkit 算子包 | `source set_env.bash` | asc-devkit 生成 |

**注意**：不要手动 export ASCEND_OPP_PATH，应使用官方 set_env.sh 脚本。

## 常见错误

### 错误1: 手动 export ASCEND_OPP_PATH
**问题**: 遗漏其他必要环境变量，导致错误 561107。  
**解决**: `source /usr/local/Ascend/cann/set_env.sh`

### 错误2: 混淆包类型
- **CANN Toolkit**: 基础包，编译+运行必需
- **CANN Ops**: 运行态依赖，提供算子库
- **自定义算子包**: 开发者编译生成，追加安装到 vendors 目录

### 错误3: 未配置 LD_LIBRARY_PATH
**问题**: 运行自定义算子时报错 561003（Kernel查找失败）。  
**解决**: `export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}/op_api/lib:$LD_LIBRARY_PATH`

## 验证环境

```bash
bash scripts/check_env.sh
```

检查项：ASCEND_HOME_PATH、ASCEND_OPP_PATH、算子安装、调试配置

