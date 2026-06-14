---
name: external-gitcode-ascend-os_adapter
description: 操作系统适配工具，用于生成和管理不同操作系统的适配方案。当需要为新 OS 创建适配或更新 OS 配置时调用。
original-name: os_adapter
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-06-14'
synced-commit: 62f55373f87e10e1d64f3f82f369be625c56b9fc
license: UNKNOWN
---

# OS Adapter

## 功能概述
操作系统适配工具，用于自动化处理新 OS 的适配流程，包括：
- 更新 OS 信息到代码库
- 创建 OS 相关配置文件
- 生成依赖和源信息
- 更新构建配置

## 使用场景
- 当需要为新的操作系统版本创建适配时
- 当需要更新 OS 相关配置信息时
- 当需要为新 OS 配置硬件支持时
- 当需要初始化 OS 的依赖和源信息时

## 适配步骤

### 步骤 1：更新 OS 基本信息

1. 通过 `/etc/os-release` 获取 OS 信息
2. 参考 `BCLINUX_21_10_AARCH64 = "BCLinux_21.10_aarch64"` 格式
3. 在 `module_utils/common_info` 中的 `OSName` 添加对应数据
4. 如果已存在，则忽略

### 步骤 2：更新 common_info 数据

需要更新的数据：
- `OSName`：OS 名称常量
- `dl_os_list`：下载 OS 列表
- 注意：重复数据需忽略

### 步骤 3：更新硬件兼容性配置

更新 `module_utils/compatibility_config` 中的数据：
- `HardwareOSTags` 中的 `OS_TO_CARD_TAG_MAP`
- 需要用户提供该 OS 支持的硬件类型（I2、A2、A3 等）

参考：`compatibility_config.py` 第 167-177 行

### 步骤 4：创建 OS 配置文件

在 `downloader_config` 目录中：
1. 创建同名文件夹
2. 创建以下文件：
   - `installed.txt`：已安装库信息
   - `pkg_info.json`：包版本信息（JSON 数组格式）
   - `source.repo` 或 `source.list`：系统源配置（根据 OS 类型）

### 步骤 5：读取已安装库信息

读取系统已安装的库，将相关信息写入 `installed.txt`

### 步骤 6：配置系统源

根据 OS 类型生成不同的源配置文件：
- **RPM 系列（CentOS/RHEL/openEuler 等）**：生成 `source.repo`
  - 包含 base、docker-ce、everything、update、EPOL 等源
- **DEB 系列（Debian/Ubuntu 等）**：生成 `source.list`
  - 包含 debian main、updates、security、docker-ce 等源

### 步骤 7：梳理依赖并生成包信息

使用依赖分析脚本梳理所有依赖（CANN 和 NPU），生成 `pkg_info.json` 文件。

#### RPM 系列依赖分析（使用 analyze_dep_tree.py）

**依赖列表**：
- **cann 组**：gcc, gcc-c++, make, cmake, unzip, zlib-devel, libffi-devel, openssl-devel, pciutils, net-tools, sqlite-devel, lapack-devel, gcc-gfortran
- **npu 组**：make, dkms, gcc, kernel-headers, kernel-devel

**分析命令**：
```bash
python3 scripts/analyze_dep_tree.py --show_version --show_pkg_name \
  --group cann --pkgs gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran \
  --group npu --pkgs make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r)
```

#### DEB 系列依赖分析（使用 apt_analyzer.py）

**依赖列表**：
- **cann 组**：gcc, g++, make, cmake, libsqlite3-dev, zlib1g-dev, libssl-dev, libffi-dev, net-tools
- **npu 组**：dkms, gcc, linux-headers

**分析命令**：
```bash
python3 scripts/apt_analyzer.py --show_version --show_pkg_name \
  --group cann --pkgs gcc g++ make cmake libsqlite3-dev zlib1g-dev libssl-dev libffi-dev net-tools \
  --group npu --pkgs dkms gcc linux-headers-$(uname -r)
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| --show_version | 在输出中显示包版本号 |
| --show_pkg_name | 在输出中显示完整包名 |
| --group | 指定依赖组名称（如 cann、npu） |
| --pkgs | 指定该组包含的包列表 |

#### 输出格式

脚本会输出 JSON 格式的依赖树和包列表，生成的 `pkg_info.json` 格式：
```json
[
    { "name": "gcc", "version": "9.3.0-1.el7" },
    { "name": "make", "version": "3.82-24.el7" },
    ...
]
```

### 步骤 8：更新构建配置

更新 `scripts/nexus_config.json`：
- 根据 OS 类型（rpm_os 或 deb_os）分类
- 将 OS 信息添加到对应分类
- 注意去重

## 目录结构

```
skills/os_adapter/
├── SKILL.md                    # 技能定义文件
├── scripts/                    # 脚本目录
│   ├── os_adapter.py          # 主适配脚本
│   ├── analyze_dep_tree.py    # RPM 系列依赖分析脚本
│   ├── apt_analyzer.py        # DEB 系列依赖分析脚本
│   └── utils.py               # 工具函数
└── templates/                  # 模板目录
    ├── installed.txt.tpl      # installed.txt 模板
    ├── pkg_info.json.tpl      # pkg_info.json 模板
    ├── source.repo.tpl        # RPM 系列源配置模板
    └── source.list.tpl        # DEB 系列源配置模板
```

## 使用方法

### 通过 SSH 连接远程 OS（推荐）
```bash
# 通过 SSH 连接远程 OS，自动获取 OS 信息、repo 信息和依赖信息
python scripts/os_adapter.py --ssh-host 192.168.1.100 --ssh-user root --ssh-key ~/.ssh/id_rsa --hardware "I2,A2,A3"

# 指定 SSH 端口
python scripts/os_adapter.py --ssh-host 192.168.1.100 --ssh-port 2222 --ssh-user root --hardware "I2,A2"

# 如果 SSH 无法获取 repo 信息，会提示用户手动配置
```

### 本地执行
```bash
# 执行完整适配流程（本地系统）
python scripts/os_adapter.py --os-name "NewOS_22.04_x86_64"

# 仅更新 OS 基本信息
python scripts/os_adapter.py --step basic-info --os-name "NewOS_22.04_x86_64"

# 仅创建配置文件
python scripts/os_adapter.py --step create-config --os-name "NewOS_22.04_x86_64"

# 指定硬件支持
python scripts/os_adapter.py --os-name "NewOS_22.04_x86_64" --hardware "I2,A2,A3"
```

### 交互式使用
```bash
# 交互式录入信息（支持 SSH 连接）
python scripts/os_adapter.py --interactive
```

## SSH 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --ssh-host | SSH 主机地址 | 无 |
| --ssh-port | SSH 端口 | 22 |
| --ssh-user | SSH 用户名 | root |
| --ssh-key | SSH 私钥文件路径 | 无 |

## 相关文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| OSName | module_utils/common_info | OS 名称常量定义 |
| dl_os_list | module_utils/common_info | 下载 OS 列表 |
| HardwareOSTags | module_utils/compatibility_config | 硬件兼容性配置 |
| OS 配置目录 | downloader_config/{os_name}/ | OS 配置文件目录 |
| installed.txt | downloader_config/{os_name}/installed.txt | 已安装库信息 |
| pkg_info.json | downloader_config/{os_name}/pkg_info.json | 包版本信息（JSON 数组） |
| source.repo | downloader_config/{os_name}/source.repo | RPM 系列系统源配置 |
| source.list | downloader_config/{os_name}/source.list | DEB 系列系统源配置 |
| nexus_config.json | scripts/nexus_config.json | 构建配置 |

## 注意事项

1. **SSH 连接**：推荐通过 SSH 连接远程 OS，可自动获取 OS 信息和 repo 信息
2. **项目检测**：脚本执行前会自动检测是否为 ascend-deployer 项目，只有检测通过才会执行
3. **去重处理**：所有添加操作都需要检查是否已存在，避免重复
4. **Repo 信息获取**：优先通过 SSH 获取，如果获取失败会提示用户手动配置
5. **硬件支持**：需要用户明确指定 OS 支持的硬件类型
6. **OS 类型判断**：根据包管理器判断是 rpm_os 还是 deb_os
7. **版本号获取**：通过调用源 API 获取准确的包版本号

## 项目检测

脚本执行前会检测以下标志文件/目录，确认是否为 ascend-deployer 项目：
- `module_utils/`
- `downloader_config/`
- `scripts/nexus_config.json`

如果检测失败，脚本会输出错误信息并退出。

## 示例

### 通过 SSH 适配远程 OS

假设要适配远程服务器上的 `openEuler 24.03 aarch64`：

1. 执行适配脚本：
```bash
python scripts/os_adapter.py --ssh-host 192.168.1.100 --ssh-user root --ssh-key ~/.ssh/id_rsa --hardware "I2,A2"
```

2. 脚本会自动：
   - 通过 SSH 连接到远程服务器
   - 获取 OS 信息（/etc/os-release）
   - 获取已安装包列表
   - 获取 repo 信息（/etc/yum.repos.d/*.repo 或 /etc/apt/sources.list）
   - 在 `OSName` 中添加常量
   - 在 `dl_os_list` 中添加条目
   - 在 `OS_TO_CARD_TAG_MAP` 中添加硬件映射
   - 创建配置文件目录和文件
   - 更新 `nexus_config.json`

### 适配 RPM 系列 OS 示例

假设要适配 `BCLinux 21.10 aarch64`：

1. 执行适配脚本：
```bash
python scripts/os_adapter.py --os-name "BCLinux_21.10_aarch64" --hardware "I2,A2"
```

2. 脚本会自动：
   - 在 `OSName` 中添加 `BCLINUX_21_10_AARCH64 = "BCLinux_21.10_aarch64"`
   - 在 `dl_os_list` 中添加对应条目
   - 在 `OS_TO_CARD_TAG_MAP` 中添加硬件映射
   - 创建 `downloader_config/BCLinux_21.10_aarch64/` 目录
   - 生成 `installed.txt`、`pkg_info.json`、`source.repo`
   - 更新 `nexus_config.json` 中的 `rpm_os` 列表

### 适配 DEB 系列 OS 示例

假设要适配 `Debian 10 x86_64`：

1. 执行适配脚本：
```bash
python scripts/os_adapter.py --os-name "Debian_10_x86_64" --hardware "I2,A2"
```

2. 脚本会自动：
   - 在 `OSName` 中添加 `DEBIAN_10_X86_64 = "Debian_10_x86_64"`
   - 在 `dl_os_list` 中添加对应条目
   - 在 `OS_TO_CARD_TAG_MAP` 中添加硬件映射
   - 创建 `downloader_config/Debian_10_x86_64/` 目录
   - 生成 `installed.txt`、`pkg_info.json`、`source.list`
   - 更新 `nexus_config.json` 中的 `deb_os` 列表

## 技术要点

- **跨平台兼容**：支持 rpm 和 deb 系列操作系统
- **自动化处理**：尽可能自动获取系统信息
- **交互式支持**：对于无法自动获取的信息，提供交互式录入
- **幂等性**：重复执行不会产生重复数据
- **可扩展性**：易于添加新的 OS 类型和硬件支持
