---
name: swanlab-setup
description: SwanLab 实验追踪平台配置与登录管理。触发场景：(1) 配置 SwanLab 登录凭据 (2) 在容器内安装/登录 SwanLab (3) 为指定容器配置 SwanLab (4) 检查 SwanLab 连接状态。支持多种配置获取方式：环境变量、配置文件、交互式输入。可被其他 skill 通过 source scripts/functions.sh 调用。
---

# SwanLab 配置管理

## 快速使用

```bash
# source 共享函数库
source scripts/functions.sh

# 在当前环境配置 SwanLab
swanlab_setup

# 为指定容器配置
swanlab_setup_for_container my_container

# 检查连接状态
swanlab_check
```

## 配置获取优先级

**环境变量 > 配置文件 > 交互式输入**

| 方式 | 变量/路径 | 说明 |
|------|-----------|------|
| 环境变量 | `SWANLAB_HOST`, `SWANLAB_API_KEY` | 推荐方式 |
| 配置文件 | `~/.verl/swanlab.conf` | 首次交互后自动保存 |
| 交互式输入 | 终端提示 | 仅在交互式终端下触发 |

## API

脚本 `scripts/functions.sh` 提供 sourceable 函数：

| 函数 | 用途 |
|------|------|
| `swanlab_clear_proxy` | 清除代理设置（SwanLab 连接需要） |
| `swanlab_load_config` | 加载配置（环境变量 > 文件） |
| `swanlab_save_config HOST KEY` | 保存配置到 `~/.verl/swanlab.conf` |
| `swanlab_prompt_config` | 交互式获取配置 |
| `swanlab_init` | 按优先级初始化配置 |
| `swanlab_login` | 在当前环境执行 pip install + swanlab login |
| `swanlab_setup` | 完整流程：clear_proxy + init + login |
| `swanlab_setup_for_container NAME` | 在指定容器内执行 swanlab_setup |
| `swanlab_check` | 检查登录状态和网络连通性 |

## 其他 Skill 调用方式

```bash
# 1. 发现 skill 路径（推荐通过环境变量）
SWANLAB_SETUP_PATH="${SWANLAB_SETUP_PATH:-$(dirname "$(find ~/.claude/skills/swanlab-setup -name functions.sh 2>/dev/null || echo "")")}"

# 2. source 函数库
source "$SWANLAB_SETUP_PATH/scripts/functions.sh"

# 3. 调用
swanlab_setup
```

## 脚本结构

```
swanlab-setup/
├── SKILL.md              # 本文档
└── scripts/
    ├── functions.sh      # 共享函数库（被其他 skill source）
    └── swanlab_check.sh  # 独立检查脚本
```