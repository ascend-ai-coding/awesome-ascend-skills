# OS Adapter Skill

操作系统适配工具，用于自动化处理新 OS 的适配流程。

## 功能特性

- 自动检测系统信息并生成 OS 名称
- 更新 `module_utils/common_info` 中的 OSName 和 dl_os_list
- 更新 `module_utils/compatibility_config` 中的硬件兼容性配置
- 创建 OS 配置文件目录和文件
- 读取系统已安装包信息
- 配置系统源信息
- 更新构建配置

## 目录结构

```
skills/os_adapter/
├── SKILL.md                    # 技能定义文件
├── README.md                   # 本文件
├── scripts/                    # 脚本目录
│   ├── os_adapter.py          # 主适配脚本
│   └── utils.py               # 工具函数
└── templates/                  # 模板目录
    ├── installed.txt.tpl      # installed.txt 模板
    ├── pkg_info.json.tpl      # pkg_info.json 模板
    └── source.repo.tpl        # source.repo 模板
```

## 安装

将此技能目录复制到项目的 `skills/` 目录下：

```bash
cp -r skills/os_adapter /path/to/your/project/skills/
```

## 使用方法

### 基本用法

```bash
# 执行完整适配流程（自动检测当前系统）
python skills/os_adapter/scripts/os_adapter.py

# 指定 OS 名称
python skills/os_adapter/scripts/os_adapter.py --os-name "BCLinux_21.10_aarch64"

# 指定硬件支持
python skills/os_adapter/scripts/os_adapter.py --os-name "BCLinux_21.10_aarch64" --hardware "I2,A2,A3"

# 指定项目根目录
python skills/os_adapter/scripts/os_adapter.py --project-root /path/to/project
```

### 分步执行

```bash
# 仅更新 OS 基本信息
python skills/os_adapter/scripts/os_adapter.py --step basic-info --os-name "NewOS_22.04_x86_64"

# 仅创建配置文件
python skills/os_adapter/scripts/os_adapter.py --step create-config --os-name "NewOS_22.04_x86_64"
```

### 交互式使用

```bash
# 交互式录入信息
python skills/os_adapter/scripts/os_adapter.py --interactive
```

## 适配流程

### 1. 更新 OS 基本信息

- 读取 `/etc/os-release` 获取系统信息
- 生成 OS 名称常量（如 `BCLINUX_21_10_AARCH64`）
- 更新 `module_utils/common_info.py` 中的 `OSName` 类
- 更新 `dl_os_list` 列表

### 2. 更新硬件兼容性配置

- 更新 `module_utils/compatibility_config.py` 中的 `OS_TO_CARD_TAG_MAP`
- 需要用户提供支持的硬件类型（I2、A2、A3 等）

### 3. 创建 OS 配置文件

在 `downloader_config/{os_name}/` 目录下创建：
- `installed.txt`：已安装包列表
- `pkg_info.json`：包版本信息
- `source.repo`：系统源配置

### 4. 更新构建配置

- 更新 `scripts/nexus_config.json`
- 根据 OS 类型（rpm_os 或 deb_os）分类添加

## 相关文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| OSName | module_utils/common_info.py | OS 名称常量定义 |
| dl_os_list | module_utils/common_info.py | 下载 OS 列表 |
| OS_TO_CARD_TAG_MAP | module_utils/compatibility_config.py | 硬件兼容性配置 |
| OS 配置目录 | downloader_config/{os_name}/ | OS 配置文件目录 |
| nexus_config.json | scripts/nexus_config.json | 构建配置 |

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| --os-name | OS 名称 | BCLinux_21.10_aarch64 |
| --hardware | 硬件支持列表 | I2,A2,A3 |
| --step | 执行步骤 | basic-info, create-config, full |
| --interactive | 交互模式 | - |
| --project-root | 项目根目录 | /path/to/project |

## 注意事项

1. **去重处理**：所有添加操作都会检查是否已存在，避免重复
2. **系统源检查**：如果系统源为空，需要手动配置 `source.repo`
3. **硬件支持**：需要明确指定 OS 支持的硬件类型
4. **包版本信息**：`pkg_info.json` 需要手动补充或通过 API 获取

## 示例

### 适配 BCLinux 21.10 aarch64

```bash
python skills/os_adapter/scripts/os_adapter.py \
    --os-name "BCLinux_21.10_aarch64" \
    --hardware "I2,A2" \
    --project-root /path/to/project
```

执行后会：
1. 在 `OSName` 中添加 `BCLINUX_21_10_AARCH64 = "BCLinux_21.10_aarch64"`
2. 在 `dl_os_list` 中添加 `"BCLinux_21.10_aarch64"`
3. 在 `OS_TO_CARD_TAG_MAP` 中添加硬件映射
4. 创建 `downloader_config/BCLinux_21.10_aarch64/` 目录
5. 生成 `installed.txt`、`pkg_info.json`、`source.repo`
6. 更新 `nexus_config.json` 中的 `rpm_os` 列表

## 开发

### 扩展功能

可以通过修改 `os_adapter.py` 和 `utils.py` 来扩展功能：

- 添加新的包管理器支持
- 扩展硬件类型验证
- 添加更多的配置文件模板

### 贡献

欢迎提交 Issue 和 Pull Request。
