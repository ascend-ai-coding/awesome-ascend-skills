---
name: migration-ascend-torchnpu-skills-environment-setup
description: Provides detailed procedures for setting up Ascend NPU development environment. Invoke when installing Ascend drivers, CANN toolkit, torch\_npu, or resolving environment compatibility issues.
---


# Skill: 昇腾NPU环境搭建

你是一位专注于昇腾NPU环境配置的工程师。本Skill提供从驱动安装到Python环境配置的完整操作流程。

<constraints>
- MUST 优先采用昇腾官方Docker镜像搭建环境，避免手动安装带来的依赖问题
- MUST 优先保持原项目PyTorch版本不变；若与昇腾不兼容，则采用昇腾支持的最新版本
- MUST 保持torch_npu版本与PyTorch版本严格对应
- NEVER 在未验证版本兼容性的情况下安装软件包
- MUST 记录每一步操作命令，用于迁移报告
- MUST 使用pip安装时优先指定第三方镜像源（如阿里源、清华源），仅在镜像源失败时回退至pip默认源
- MUST 优先使用镜像站或ModelScope获取模型/数据集资源，NEVER默认使用HuggingFace官网；仅在用户明确确认网络可达HuggingFace时方可使用HuggingFace官网
</constraints>

<version_notice>
本Skill中的版本配套信息基于编写时的最新数据。昇腾软件栈版本迭代较快，后续出现新版本时，MUST以昇腾官网发布的版本配套表为准：

- torch\_npu版本配套表：<https://gitcode.com/Ascend/pytorch>
- CANN版本说明（含HDK版本配套关系）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/release-notes.md>
- CANN安装指南：<https://www.hiascend.com/document/detail/zh/canncommercial/>
- 昇腾驱动下载：<https://www.hiascend.com/developer/download/community>
- 昇腾镜像仓库：<https://www.hiascend.com/developer/ascendhub>

**⚠ 版本会持续更新，MUST实时访问上述官网链接获取最新版本配套关系，不可依赖本Skill中已固化的版本信息。** 每次执行迁移任务时，应首选查阅官方文档获取最新版本信息，仅在无法访问官网时回退使用本Skill中的参考数据。
</version_notice>

## 一、版本兼容性矩阵

### 1.1 核心组件版本对应关系

> 以下数据来源于 torch\_npu 官方仓库（gitcode.com/Ascend/pytorch），仅为编写时快照，新版本请查阅官方文档。

| CANN版本       | PyTorch版本 | torch\_npu版本 | Python版本 | 备注               |
| ------------ | --------- | ------------ | -------- | ---------------- |
| CANN 9.0.0   | 2.10.0    | 2.10.0       | 3.9-3.13 | 最新版，支持Python3.13 |
| CANN 9.0.0   | 2.9.0     | 2.9.0.post2  | 3.9-3.12 | <br />           |
| CANN 9.0.0   | 2.8.0     | 2.8.0.post2  | 3.9-3.12 | <br />           |
| CANN 9.0.0   | 2.7.1     | 2.7.1.post3  | 3.9-3.11 | <br />           |
| CANN 8.5.2   | 2.9.0     | 2.9.0.post1  | 3.9-3.12 | <br />           |
| CANN 8.5.2   | 2.8.0     | 2.8.0.post1  | 3.9-3.11 | <br />           |
| CANN 8.5.2   | 2.7.1     | 2.7.1.post2  | 3.9-3.11 | <br />           |
| CANN 8.5.2   | 2.6.0     | 2.6.0.post6  | 3.9-3.11 | <br />           |
| CANN 8.5.1   | 2.9.0     | 2.9.0.post1  | 3.9-3.12 | <br />           |
| CANN 8.5.1   | 2.8.0     | 2.8.0.post1  | 3.9-3.11 | <br />           |
| CANN 8.5.1   | 2.7.1     | 2.7.1.post2  | 3.9-3.11 | <br />           |
| CANN 8.5.1   | 2.6.0     | 2.6.0.post6  | 3.9-3.11 | <br />           |
| CANN 8.5.0   | 2.9.0     | 2.9.0        | 3.9-3.12 | <br />           |
| CANN 8.5.0   | 2.8.0     | 2.8.0        | 3.9-3.11 | <br />           |
| CANN 8.5.0   | 2.7.1     | 2.7.1        | 3.9-3.11 | <br />           |
| CANN 8.5.0   | 2.6.0     | 2.6.0.post5  | 3.9-3.11 | <br />           |
| CANN 8.3.RC1 | 2.8.0     | 2.8.0        | 3.9-3.11 | <br />           |
| CANN 8.3.RC1 | 2.7.1     | 2.7.1        | 3.9-3.11 | <br />           |
| CANN 8.3.RC1 | 2.6.0     | 2.6.0.post3  | 3.9-3.11 | <br />           |
| CANN 8.2.RC1 | 2.6.0     | 2.6.0        | 3.9-3.11 | <br />           |
| CANN 8.2.RC1 | 2.5.1     | 2.5.1.post1  | 3.9-3.11 | <br />           |
| CANN 8.2.RC1 | 2.1.0     | 2.1.0.post13 | 3.8-3.11 | <br />           |
| CANN 8.1.RC1 | 2.5.1     | 2.5.1        | 3.9-3.11 | <br />           |
| CANN 8.1.RC1 | 2.4.0     | 2.4.0.post4  | 3.8-3.11 | <br />           |
| CANN 8.1.RC1 | 2.3.1     | 2.3.1.post6  | 3.8-3.11 | <br />           |
| CANN 8.1.RC1 | 2.1.0     | 2.1.0.post12 | 3.8-3.11 | <br />           |
| CANN 8.0.0   | 2.4.0     | 2.4.0.post2  | 3.8-3.11 | <br />           |
| CANN 8.0.0   | 2.3.1     | 2.3.1.post4  | 3.8-3.11 | <br />           |
| CANN 8.0.0   | 2.1.0     | 2.1.0.post10 | 3.8-3.11 | <br />           |
| CANN 7.0.0   | 2.1.0     | 2.1.0        | 3.8-3.10 | <br />           |
| CANN 7.0.0   | 2.0.1     | 2.0.1.post1  | 3.7-3.10 | <br />           |

> **小版本迭代说明**：实际使用时，CANN版本号可能会有小迭代（如 CANN 9.0.0 → CANN 9.0.1），此类小版本迭代通常保持与主版本的兼容性，可按照主版本（如9.0.0）的兼容性进行判断。但MUST在安装后通过 `torch.npu.is_available()` 验证实际可用性。

### 1.1b CANN与Ascend HDK（驱动固件）版本对应关系

> 以下数据来源于CANN社区版各版本说明文档（hiascend.com），仅为编写时快照。**版本持续更新，MUST实时查阅官网获取最新对应关系**：
> <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/release-notes.md>

| CANN版本 | 配套 Ascend HDK 版本 | 备注 |
|----------|---------------------|------|
| CANN 9.0.0 | 26.0.RC1 / 25.5.2 / 25.5.1 | 最新版 |
| CANN 8.5.2 | 25.5.x | — |
| CANN 8.5.1 | 25.5.x | — |
| CANN 8.5.0 | 25.5.x | — |
| CANN 8.3.RC1 | 25.x.x | — |
| CANN 8.2.RC1 | 25.2.0 | — |
| CANN 8.1.RC1 | 24.x.x | — |
| CANN 8.0.0 | 24.x.x | — |
| CANN 7.0.0 | 23.x.x | — |

> **说明**：
> - Ascend HDK 版本号即昇腾驱动固件版本号，驱动安装包文件名格式为 `Ascend-hdk-{HDK版本}-npu-driver_linux-{arch}.run`
> - `26.0.RC1` 中的 RC 表示 Release Candidate（发布候选版）
> - 同一CANN版本可能兼容多个HDK版本，以官网版本说明中的配套关系为准
> - HDK版本与CANN版本之间的兼容性是单向的：必须先确定CANN版本，再确定配套的HDK版本
> - **⚠ 版本会更新，上表仅作参考，MUST在官网实时查询：** 通过访问对应CANN版本的版本说明页面（将URL中的版本号替换为目标版本），可查到该版本的精确配套HDK版本

### 1.2 版本选择原则

1. 确定原项目使用的PyTorch版本
2. 检查该版本是否在昇腾支持列表中
3. 若兼容：保持原项目PyTorch版本，查找对应的torch\_npu版本
4. 若不兼容：采用昇腾支持的最新PyTorch版本，查找对应的torch\_npu版本
5. 根据torch\_npu版本查找对应的CANN版本
6. 根据CANN版本查找对应的昇腾驱动版本（查阅 [1.1b CANN与Ascend HDK版本对应关系](#11b-cann与ascend-hdk驱动固件版本对应关系) 和CANN版本说明中的配套HDK版本，**MUST实时查阅官网获取最新对应关系**）
7. **无强制版本要求时**：MUST查阅最新的版本配套关系，优先采用最新版本，以获得更好的算子支持和性能优化

### 1.3 版本选择示例
   ```

场景：原项目使用 PyTorch 2.1.0
1. 查表：PyTorch 2.1.0 在多个CANN版本中均有支持
2. 选择策略：选择最新的CANN版本（8.2.RC1），对应的torch_npu为 2.1.0.post13
3. 查CANN-HDK对应关系：CANN 8.2.RC1 配套 Ascend HDK 25.2.0（见 1.1b 节表）
4. 确认驱动：需要安装 25.2.0 版本的昇腾驱动

场景：原项目使用 PyTorch 2.9.0
1. 查表：PyTorch 2.9.0 在 CANN 9.0.0/8.5.2/8.5.1/8.5.0 中均有支持
2. 选择策略：选择最新的CANN版本（9.0.0），对应的torch_npu为 2.9.0.post2
3. 查CANN-HDK对应关系：CANN 9.0.0 配套 Ascend HDK 26.0.RC1 / 25.5.2 / 25.5.1（见 1.1b 节表）
4. 确认驱动：需要安装 26.0.RC1 或 25.5.x 版本的昇腾驱动

场景：原项目使用 PyTorch 2.6.0
1. 查表：PyTorch 2.6.0 在 CANN 8.5.2/8.5.1/8.5.0/8.3.RC1/8.2.RC1 中均有支持
2. 选择策略：选择最新的CANN版本（8.5.2），对应的torch_npu为 2.6.0.post6
3. 查CANN-HDK对应关系：CANN 8.5.2 配套 Ascend HDK 25.5.x（见 1.1b 节表）
4. 确认驱动：需要安装 25.5.x 版本的昇腾驱动
```

## 二、环境搭建（首选Docker镜像方式）

> **MUST优先使用Docker镜像方式搭建环境**。昇腾官方镜像已预集成CANN toolkit、kernels、NNAL等软件包和系统依赖，可大幅减少手动安装的工作量和出错概率。建议以最新版本的CANN镜像作为基础进行搭建。

### 2.1 镜像选择

从昇腾镜像仓库选择镜像：<https://www.hiascend.com/developer/ascendhub>

**CANN基础镜像**（推荐，最灵活）：

镜像仓库地址：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann`

> 具体Tag请查阅以下官方渠道（Tag随CANN版本持续更新）：
>
> - 昇腾镜像仓库：<https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884>
> - Dockerfile与Tag列表：<https://github.com/Ascend/cann-container-image/tree/main/cann>

Tag命名规则：`{CANN版本}-{芯片型号}-{OS}-py{Python版本}`

> 镜像内已集成 Toolkit开发套件包、Kernels算子包、NNAL加速库，无需额外安装。

示例（以编写时可查到的Tag为例，新版本请查阅上述官方渠道）：

| Tag组成部分  | 可选值                              | 说明                     |
| -------- | -------------------------------- | ---------------------- |
| CANN版本   | `8.5.0`、`8.1.rc1`、`9.0.0` 等      | 选择与torch\_npu匹配的CANN版本 |
| 芯片型号     | `910b`、`910c`、`310p`、`910`、`950` | 910C对应A3系列，其他请查阅昇腾官网   |
| OS       | `ubuntu22.04`、`openeuler24.03`   | 建议选择Ubuntu版本           |
| Python版本 | `py3.10`、`py3.11`                | 选择与项目匹配的Python版本       |

> 芯片型号与产品系列的对应关系请查阅昇腾官方文档，不要自行推断

### 2.2 拉取镜像

```bash
# 根据实际需求替换Tag，示例：CANN 8.5.0 + 910B + Ubuntu22.04 + Python3.10
# 具体Tag请查阅：https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884

# ARM架构（鲲鹏，推荐）
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10

# x86_64架构
docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10
```

### 2.3 启动容器

> 宿主机MUST已安装昇腾驱动和固件，且Docker版本 >= 1.11.2。

```bash
# 查看镜像ID
docker images | grep ascendhub/cann

# 启动容器（MUST挂载NPU设备相关路径）
# 参考官方镜像说明：https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884
docker run --name npu_dev -dit --privileged --net=host --shm-size=500g \
  -w /home \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  -v /data:/data \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  {镜像ID} \
  bash
```

**设备挂载参数说明**：

| 参数                                         | 说明                           |
| ------------------------------------------ | ---------------------------- |
| `--privileged`                             | 开启特权模式，确保NPU设备完整可用           |
| `--device=/dev/davinci0~7`                 | 映射所有NPU卡（davinci0\~davinci7） |
| `--device=/dev/davinci_manager`            | 映射NPU设备管理接口                  |
| `--device=/dev/devmm_svm`                  | 映射设备内存管理接口                   |
| `--device=/dev/hisi_hdc`                   | 映射主机与设备间通信接口                 |
| `-v /usr/local/dcmi`                       | 挂载DCMI工具和库                   |
| `-v /usr/local/Ascend/driver/lib64/`       | 挂载驱动库                        |
| `-v /usr/local/Ascend/driver/version.info` | 挂载驱动版本信息                     |
| `-v /etc/ascend_install.info`              | 挂载驱动安装信息                     |

### 2.4 容器内环境配置

> 容器内不建议使用conda或venv，直接使用pip管理Python环境即可，避免与容器内已有环境冲突。

```bash
# 先启动容器（后台运行）
docker start npu_dev

# 再进入容器
docker exec -it npu_dev bash

# 配置CANN环境变量（镜像内已安装CANN）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 配置NNAL环境变量（CANN 8.1.RC1及以后版本需要）
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装PyTorch和torch_npu
# 优先从官方安装指南获取对应版本的下载链接
# 官方安装指南（MUST实时查阅昇腾官网最新版本文档，以下链接仅供参考）：
# 昇腾官网 → 文档 → PyTorch框架适配 → 软件安装指南 → 安装 → 使用二进制包安装
# 参考链接：https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md
# 根据目标PyTorch版本、架构(aarch64/x86_64)、Python版本，从上述文档查找对应的wget下载链接

# 方式1：wget下载whl包安装（推荐，优先使用）
# 示例：PyTorch 2.6.0 + aarch64 + Python 3.10 + CANN 8.5.0
pip install pyyaml setuptools -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
wget https://download.pytorch.org/whl/cpu/torch-2.6.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch-2.6.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post5-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.6.0.post5-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 方式2：pip直接安装（备选，网络通畅时使用）
# aarch64架构
# pip install torch==2.6.0
# x86_64架构
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# pip install torch-npu==2.6.0.post5

# 验证
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 2.5 常见容器问题

- **问题**：容器启动后立即退出
  → 使用 `-it` 参数以交互模式启动，或使用 `docker start` 重新启动已停止的容器
- **问题**：容器内 `npu-smi info` 无输出
  → 检查宿主机驱动是否安装，检查 `--device=/dev/davinci_manager` 是否添加
- **问题**：容器内 `torch.npu.is_available()` 返回 False
  → 检查驱动库是否正确挂载：`ls /usr/local/Ascend/driver/lib64/`

## 三、手动安装方式（备选）

> 仅在无法使用Docker时采用手动安装方式。手动安装需依次完成：驱动安装 → CANN安装 → Python环境配置。

### 3.1 昇腾驱动安装

#### 检查硬件环境

```bash
npu-smi info
uname -m
# aarch64 = ARM架构（鲲鹏）
# x86_64 = x86架构
```

#### 安装驱动

> 首次安装按"驱动 > 固件"顺序；覆盖安装按"固件 > 驱动"顺序。

```bash
# 1. 创建驱动运行用户（如尚未创建）
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

# 2. 下载对应版本的驱动包
# https://www.hiascend.com/developer/download/community

# 3. 安装驱动（需要root权限）
# 根据CANN版本选择对应的驱动版本，查阅 1.1b CANN与Ascend HDK版本对应关系 和CANN版本说明中的配套HDK版本
chmod +x Ascend-hdk-xxx-npu-driver_linux-aarch64.run
./Ascend-hdk-xxx-npu-driver_linux-aarch64.run --full --install-for-all

# 4. 安装固件
chmod +x Ascend-hdk-xxx-npu-firmware_linux-aarch64.run
./Ascend-hdk-xxx-npu-firmware_linux-aarch64.run --full

# 5. 重启系统
reboot

# 6. 验证安装
npu-smi info
```

#### 常见驱动问题

- **问题**：`npu-smi info` 无输出
  → 检查驱动是否安装成功，检查 `/usr/local/Ascend/driver` 目录是否存在
- **问题**：权限不足无法访问NPU
  → 将当前用户加入 `HwHiAiUser` 组：`usermod -aG HwHiAiUser $USER`
- **问题**：内核升级后驱动不可用
  → 关闭内核自动更新：`apt-mark hold linux-image-generic linux-headers-generic`

### 3.2 CANN软件栈安装

#### 安装系统依赖

```bash
# Ubuntu/Debian
apt-get update
apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3

# openEuler/EulerOS
yum install -y gcc gcc-c++ make cmake zlib zlib-devel openssl-devel sqlite-devel libffi-devel unzip pciutils net-tools blas-devel gfortran
```

#### 安装CANN toolkit

> 安装toolkit前确保安装目录可用空间大于9G。MUST先安装toolkit再安装kernels。

```bash
# 下载CANN toolkit
# https://www.hiascend.com/developer/download/community

# 安装（根据实际CANN版本替换文件名）
chmod +x Ascend-cann-toolkit_linux-aarch64.run
./Ascend-cann-toolkit_linux-aarch64.run --install --install-for-all --quiet
```

#### 安装CANN kernels（推荐）

```bash
chmod +x Ascend-cann-kernels_linux-aarch64.run
./Ascend-cann-kernels_linux-aarch64.run --install --install-for-all --quiet
```

#### 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

#### 验证CANN安装

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

### 3.3 Python环境配置

#### 创建虚拟环境

> 裸机环境建议使用conda管理Python环境；容器环境直接使用pip即可，不需要conda或venv。

```bash
# 裸机环境：使用conda创建环境（推荐）
conda create -n npu_env python=3.10 -y
conda activate npu_env

# 容器环境：直接使用pip，无需conda或venv
```

#### 安装PyTorch

> MUST优先从官方安装指南获取对应版本的wget下载链接：
> https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md
>
> 根据目标PyTorch版本、架构(aarch64/x86_64)、Python版本，从上述文档查找对应的下载链接。

```bash
# 方式1：wget下载whl包安装（推荐，优先使用）
# 示例：PyTorch 2.6.0 + aarch64 + Python 3.10
wget https://download.pytorch.org/whl/cpu/torch-2.6.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch-2.6.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 示例：PyTorch 2.6.0 + x86_64 + Python 3.10
# wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl
# pip install torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl

# 方式2：pip直接安装（备选，网络通畅时使用）
# aarch64架构（ARM/鲲鹏）
# pip install torch==2.6.0

# x86_64架构
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 安装torch\_npu

> MUST优先从官方安装指南获取对应版本的wget下载链接：
> https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md
>
> torch_npu版本必须与PyTorch版本和CANN版本严格对应，从上述文档查找对应的下载链接。

```bash
pip install pyyaml setuptools -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 方式1：wget下载whl包安装（推荐，优先使用）
# 示例：torch_npu 2.6.0.post5 + aarch64 + Python 3.10
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post5-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.6.0.post5-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 方式2：pip直接安装（备选，网络通畅时使用）
# pip install torch-npu==2.6.0.post5
```

#### 验证torch\_npu安装

```python
import torch
import torch_npu

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU device count: {torch.npu.device_count()}")
    print(f"NPU device name: {torch.npu.get_device_name(0)}")
    x = torch.randn(3, 3).npu()
    y = torch.randn(3, 3).npu()
    z = x + y
    print(f"NPU compute test passed: {z.shape}")
```

## 四、环境变量配置

### 4.1 必要环境变量

```bash
# CANN环境（MUST）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# torch_npu日志级别（可选，调试时设置）
export ASCEND_LOG_LEVEL=3  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# NPU可见设备（可选）
export ASCEND_RT_VISIBLE_DEVICES="0"  # 只使用NPU:0
```

### 4.2 性能优化环境变量

```bash
# 算子编译优化
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit
export TASK_QUEUE_ENABLE=1

# 内存优化
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

# HCCL通信超时（分布式训练时建议设置）
export HCCL_CONNECT_TIMEOUT=1800
```

## 五、常用第三方库安装

> 昇腾NPU服务器通常为ARM（aarch64）架构，部分第三方库的预编译wheel包可能不兼容，优先建议通过编译安装。

### 5.0 昇腾已支持的第三方库（优先使用）

> 安装第三方库前，MUST从昇腾官网实时查阅最新版本的第三方库支持列表，优先使用昇腾已适配的版本，并严格按照其版本要求安装。昇腾已对部分主流第三方库进行了NPU适配，使用已适配的库可获得更好的性能和兼容性。
>
> 查阅路径：昇腾官网 → 文档 → PyTorch框架适配 → 第三方库适配 → 支持的套件和第三方库
>
> 参考链接（版本迭代较快，MUST以官网最新版本文档为准）：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/modparts/FrameworkPTAdapter/26.0.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md>
>
> 已知昇腾原生支持的第三方库包括（以官方文档为准）：
>
> - **transformers**：`pip install transformers`，原生支持Ascend NPU
> - **accelerate**：`pip install accelerate`，原生支持Ascend NPU
> - **peft**：`pip install peft`，原生支持Ascend NPU
> - **trl**：`pip install trl`，原生支持Ascend NPU
> - **deepspeed**：需安装昇腾适配版本，参见官方文档
>
> 安装原则：
>
> 1. 优先查阅上述官方文档，确认目标库是否已有昇腾适配版本
> 2. 如有适配版本，严格按照官方文档的版本要求和安装指导操作
> 3. 如无适配版本，再采用编译安装或pip安装方式

### 5.1 FFmpeg（视频处理）

```bash
# 编译安装（ARM架构下pip可能无预编译包）

# 1. 下载FFmpeg源码
wget https://ffmpeg.org/releases/ffmpeg-8.1.tar.gz
tar -zxvf ffmpeg-8.1.tar.gz

# 2. 编译
cd ffmpeg-8.1
./configure --enable-shared
make -j$(nproc)
sudo make install

# 3. 验证
ffmpeg -version
```

### 5.2 OpenCV（图像处理）

```bash
git clone https://github.com/opencv/opencv.git -b 4.8.0 --depth 1
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      ..
make -j$(nproc)
sudo make install
```

### 5.3 其他常见库

| 库名称          | aarch64安装建议                                 |
| ------------ | ------------------------------------------- |
| numpy        | `pip install numpy` 通常可直接安装                 |
| scipy        | `pip install scipy` 通常可直接安装；如失败需编译安装        |
| pillow       | `pip install pillow` 通常可直接安装                |
| scikit-learn | `pip install scikit-learn` 通常可直接安装；如失败需编译安装 |
| onnx         | `pip install onnx` 通常可直接安装                  |
| onnxruntime  | 需检查是否有aarch64预编译包；如无可从源码编译                  |
| tokenizers   | `pip install tokenizers` 通常可直接安装            |
| transformers | `pip install transformers` 通常可直接安装          |

### 5.4 编译安装通用模板

```bash
git clone <repo_url> -b <version_tag> --depth 1
cd <repo_name>
pip install -e .
# 或
python setup.py build_ext --inplace
pip install .
```

### 5.5 镜像站与模型资源获取

NPU服务器通常位于内网环境，部分网站如HuggingFace、GitHub等可能无法访问或网速较慢。MUST优先使用镜像站或国内替代平台获取资源。

#### 模型与数据集获取

**MUST优先使用以下方式获取模型/数据集，NEVER默认使用HuggingFace官网**（仅在用户明确确认网络可达HuggingFace时方可使用）：

| 优先级 | 方式 | 说明 |
|--------|------|------|
| 1 | ModelScope（<https://www.modelscope.cn>） | 国内平台，网络稳定，大量模型与HuggingFace等价 |
| 2 | HuggingFace镜像站（<https://hf-mirror.com>） | 设置 `export HF_ENDPOINT=https://hf-mirror.com` 后正常使用huggingface\_hub |
| 3 | HuggingFace官网（<https://huggingface.co>） | 仅在用户明确确认网络可达时使用 |

**ModelScope使用示例**：

```python
# 安装modelscope
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 从ModelScope下载模型（替代HuggingFace的from_pretrained）
from modelscope import snapshot_download
model_dir = snapshot_download('模型ID')

# 使用transformers加载ModelScope下载的模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_dir)
```

**HuggingFace镜像站使用方式**：

```bash
# 设置环境变量后，huggingface_hub会自动使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或在Python代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### 代码仓库获取

| 原始站点        | 镜像站                     | 用法示例                                                                             |
| ----------- | ----------------------- | -------------------------------------------------------------------------------- |
| GitHub      | <https://bgithub.xyz>   | 将 `github.com` 替换为 `bgithub.xyz`，如 `git clone https://bgithub.xyz/user/repo.git` |
| GitHub      | <https://gitcode.com>   | 部分仓库在gitcode上有镜像 |

## 六、项目依赖安装

### 6.1 安装策略

根据代码仓的 `requirements.txt` 安装依赖时，遵循以下原则：

1. **尽量不影响原有库**：在独立虚拟环境中安装，避免污染系统环境
2. **版本要求严格时**：先尝试放宽版本要求安装，并记录变更
3. **实际运行异常时**：回退到严格版本，记录冲突原因

### 6.2 安装步骤

```bash
# 1. 先尝试使用阿里源安装requirements.txt
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 如阿里源失败，尝试清华源
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如镜像源均失败，回退至pip默认源
# pip install -r requirements.txt

# 2. 如遇版本冲突，逐个安装并记录
# 记录格式：包名 | 原要求版本 | 实际安装版本 | 是否有变更 | 变更原因
pip install package_name==specific_version -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 3. 如严格版本与torch_npu环境冲突，尝试放宽版本
# 示例：原要求 numpy==1.24.0，假设不兼容
# 放宽为 numpy>=1.24.0
pip install "numpy>=1.24.0" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 记录：numpy | ==1.24.0 | >=1.24.0 | 是 | 与某某版本冲突
```

### 6.3 依赖冲突处理

- **冲突1**：项目要求CUDA版PyTorch，但NPU需要CPU版或aarch64版
  → aarch64架构直接安装PyTorch官方包；x86\_64安装CPU版PyTorch + torch\_npu
- **冲突2**：某些包依赖 `nvidia-*` 库
  → 检查是否为硬依赖，如仅为可选加速则可忽略
- **冲突3**：numpy/scipy版本与torch\_npu不兼容
  → 参照torch\_npu官方文档的推荐版本，放宽requirements中的版本约束
- **冲突4**：requirements中指定了 `torch==x.y.z+cu118` 等CUDA后缀
  → 替换为 `torch==x.y.z`（无CUDA后缀），安装后由torch\_npu提供NPU支持

## 七、环境验证清单

<verification>
环境搭建完成后逐项检查：
- [ ] `npu-smi info` 正常显示NPU设备信息
- [ ] CANN环境变量已配置（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`）
- [ ] Python虚拟环境已创建并激活
- [ ] `import torch` 无报错，版本正确
- [ ] `import torch_npu` 无报错，版本正确
- [ ] `torch.npu.is_available()` 返回 True
- [ ] NPU简单计算测试通过
- [ ] 项目依赖全部安装成功（如有放宽版本，已记录变更）
- [ ] CPU基线脚本可正常运行
</verification>

## 八、常见陷阱

- **陷阱1**：手动安装时安装了CUDA版PyTorch再安装torch\_npu，导致冲突
  → aarch64架构安装PyTorch官方包；x86\_64架构MUST安装CPU版PyTorch
- **陷阱2**：CANN版本与torch\_npu版本不匹配，运行时报错
  → 严格按照版本兼容性矩阵匹配，以昇腾官网发布的配套表为准
- **陷阱3**：忘记 `source set_env.sh`，导致找不到CANN库
  → 将 `source` 命令写入 `~/.bashrc` 或虚拟环境激活脚本
- **陷阱4**：Docker容器启动后立即退出
  → 使用 `-dit` 参数以交互+后台模式启动，先 `docker start` 再 `docker exec -it` 进入
- **陷阱5**：Docker容器内缺少NPU设备映射，`torch.npu.is_available()` 返回 False
  → 启动容器时MUST添加 `--device=/dev/davinci_manager` 等参数，并挂载驱动库
- **陷阱6**：ARM架构下直接pip安装第三方库失败
  → 优先通过编译安装，参见第五章
- **陷阱7**：手动安装时先安装kernels再安装toolkit导致安装失败
  → MUST先安装toolkit再安装kernels
- **陷阱8**：内核自动升级后NPU驱动不可用
  → Ubuntu：`apt-mark hold linux-image-generic linux-headers-generic linux-image-extra`
  → openEuler：`yum install yum-plugin-versionlock && yum versionlock add kernel-$(uname -r)`
  → 出处：Atlas 800T A2 安装指南 EDOC1100349380、Atlas 800I A3 安装指南 EDOC1100471070
- **陷阱9**：Docker镜像Tag选择错误（芯片型号不匹配）
  → 芯片型号与产品系列的对应关系请查阅昇腾官方文档

