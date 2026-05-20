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

## CANN 版本兼容性

CANN 版本需与固件驱动、子包版本配套使用，无通用最低/最高版本要求。具体配套关系以官方 Release Notes 为准。

### 版本信息来源

| 信息 | 来源路径 |
|------|---------|
| CANN 版本号 | `$ASCEND_HOME_PATH/compiler/version.info` 中的 `Version` 字段 |
| 运行时依赖基线 | 同文件中的 `required_package_runtime_version` 字段 |
| 全量子包依赖 | 同文件中的 `required_package_*` 系列字段 |

可通过 `check_env.sh` 自动检测并展示上述信息。

### 官方配套关系查询

- CANN 与 Ascend HDK 版本配套关系、组合包配套关系等，请查阅官方 Release Notes：https://www.hiascend.com/cann/document
- 版本下载与快速安装：https://www.hiascend.com/cann/download

### 常见版本问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| CANN 与驱动不配套 | 固件驱动版本与 CANN 版本不匹配 | 查阅 Release Notes 配套关系表，升级驱动或 CANN |
| 子包版本不一致 | runtime/metadef/opbase 等子包版本不匹配 | 检查 `version.info` 中 `required_package_*` 字段，重新安装配套版本 |
| API 不可用 | 当前 CANN 版本不支持该 API | 查阅 API 文档中的版本要求说明 |

## 验证环境

```bash
bash scripts/check_env.sh
```

检查项：ASCEND_HOME_PATH、CANN 版本、ASCEND_OPP_PATH、算子安装、调试配置

