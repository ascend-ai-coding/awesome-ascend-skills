---
name: external-gitcode-ascend-cann-operator-env-config
description: 提供昇腾NPU的CANN安装指导。当用户需要安装CANN、配置昇腾环境或解决安装问题时调用。
original-name: cann-operator-env-config
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# CANN开发环境配置

此技能为在昇腾NPU平台上安装CANN（神经网络计算架构）提供全面指导。

## 调用时机

在以下情况下调用此技能：
- 用户询问CANN安装
- 用户需要帮助设置昇腾开发环境
- 用户遇到CANN安装问题
- 用户想了解CANN安装方法

## 安装步骤

### 第1步：检查驱动是否安装

**硬件要求**
- 已安装昇腾NPU设备
- 设备状态正常

**驱动检查命令**
```bash
npu-smi info
```

**检查结果**
- 如果显示NPU设备信息，说明驱动已安装
- 记录驱动版本用于后续CANN版本兼容性检查
- 如果提示"command not found"，需要先安装昇腾驱动

### 第2步：检查CANN安装依赖
检查CANN安装依赖（Python和pip）是否已安装：
```bash
# 检查Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Python not found, please install from env_dependence.md"
fi

# 检查pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "pip not found, please install from env_dependence.md"
fi
```

如果任何依赖缺少，会显示提示信息，请参考 [环境依赖安装指南](./reference/env_dependence.md) 进行安装。

### 第3步：安装CANN

#### 方法一：离线安装

**步骤3.1：检查当前文件夹是否存在安装包**
```bash
# 检查当前文件夹是否存在CANN安装包
if ls Ascend-cann-toolkit_*.run 2>/dev/null | grep -q .; then
    echo "找到CANN Toolkit安装包"
    TOOLKIT_PACKAGE=$(ls Ascend-cann-toolkit_*.run)
else
    echo "未找到CANN Toolkit安装包，请使用在线安装方式"
    # 退出离线安装，建议使用在线安装
    exit 1
fi

# 检查当前文件夹是否存在ops安装包
if ls Ascend-cann-*-ops_*.run 2>/dev/null | grep -q .; then
    echo "找到CANN ops安装包"
    OPS_PACKAGE=$(ls Ascend-cann-*-ops_*.run)
else
    echo "未找到CANN ops安装包，请使用在线安装方式"
    # 退出离线安装，建议使用在线安装
    exit 1
fi
```

**步骤3.2：安装CANN**

**重要：必须先安装Toolkit，再安装ops**

**run格式安装：**
```bash
# 安装 Toolkit (默认路径 /usr/local/Ascend)
bash "$TOOLKIT_PACKAGE" --install --quiet
# 安装 ops
bash "$OPS_PACKAGE" --install --quiet
```

#### 方法二：在线安装（conda方式）

**注意：conda主要用于创建Python虚拟环境，CANN工具包仍需通过离线安装包安装。**

**步骤3.1：创建conda虚拟环境**
```bash
# 创建Python环境（CANN支持Python 3.7.x - 3.13.11）
conda create -n cann_env python
# 激活环境
conda activate cann_env
```

**步骤3.2：配置昇腾conda官方源**
```bash
# 添加昇腾conda源
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda
```

**步骤3.3：安装CANN包**
```bash
# 使用conda安装CANN工具包
conda install ascend::cann-toolkit
# 安装CANN ops包（根据芯片类型选择，如910、910b、310等）
conda install ascend::cann-{芯片类型}-ops
```

#### 方法三：在线安装（yum方式 - CentOS/RHEL）

**步骤3.1：配置华为官方源**
```bash
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
```

**步骤3.2：安装CANN包**
```bash
yum install -y Ascend-cann-toolkit
yum install -y Ascend-cann-{芯片类型}-ops
```

### 第4步：环境变量配置

**临时设置（仅当前终端生效）**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**永久设置（写入Shell配置文件）**
```bash
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
```

### 第5步：安装验证

**使用Python验证ACL接口**
```bash
python3 -c "import acl;acl.init();acl.rt.set_device(0)"
```
**验证结果**
- 如果执行成功，说明CANN Python接口正常工作
- 如果执行失败，检查环境变量是否配置正确

### 第6步：安装运行时依赖
```bash
pip3 install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
pip3 install protobuf==3.20.0
```

## 注意事项

1. **版本兼容性**：确保CANN版本与驱动版本匹配
2. **安装顺序**：必须先安装Toolkit，再安装ops
3. **环境变量**：安装后必须设置环境变量
4. **权限问题**：某些操作可能需要root权限
5. **依赖包**：确保所有必要的依赖包已安装

## 故障排查

如果安装过程中遇到问题：
- 检查驱动版本是否与CANN版本兼容
- 检查依赖包是否安装完整
- 查看安装日志获取详细错误信息
- 参考昇腾社区文档或寻求技术支持

## 参考资料

详细参考资料请查看 [reference](./reference/) 目录，包含：
- [环境依赖安装指南](./reference/env_dependence.md)：必需软件安装命令和验证方法
