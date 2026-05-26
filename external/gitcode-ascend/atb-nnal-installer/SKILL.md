---
name: external-gitcode-ascend-atb-nnal-installer
description: '昇腾 NPU NNAL（ATB 加速库）安装技能。依赖 cann-operator-env-config 提供 Toolkit+Kernels
  环境，本技能仅负责 NNAL 包的安装、环境变量配置与验证。

  '
keywords:
- cann
- nnal
- atb
- installer
- 昇腾
- ascend
- 加速库
metadata:
  author: ascend-transformer-boost-team + Claude Code + opus4.7
  version: 2.1.0
  created: '2026-04-17'
  updated: '2026-04-28'
  skill-type: env-setup
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: '[ -z "$INSTALL_PATH" ] && echo ''[PATH CHECK] INSTALL_PATH 未设置，请先向用户获取
        NNAL 安装路径（打断并等待用户提供）'' >&2 || true'
original-name: atb-nnal-installer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# CANN NNAL 安装部署

## 功能概述

在昇腾 NPU 环境中安装 NNAL（ATB 加速库）。**本技能假设 Toolkit+Kernels 已由 `cann-operator-env-config` 技能安装完毕。**

## 调用时机

- 用户已完成 Toolkit+Kernels 安装（通过 `cann-operator-env-config`），需要安装 NNAL
- 用户在 Docker 容器中需要安装 NNAL
- 用户需要安装或升级特定版本的 NNAL

## 前置条件

**必须已执行 `cann-operator-env-config` 技能**，确保：
- Toolkit 已安装，`/usr/local/Ascend/ascend-toolkit/set_env.sh` 存在
- Kernels（ops）已安装

## ⚠️ 路径约束（必须执行）

执行此技能前，**必须从用户处获取以下路径**：

- `<INSTALL_PATH>`: NNAL 安装目标路径（如 `/usr/local/Ascend`）

**若用户未提供 INSTALL_PATH，立即停止并打断，要求用户提供安装路径。**

脚本校验示例：
```bash
if [ -z "$INSTALL_PATH" ]; then
    echo "ERROR: INSTALL_PATH 未设置，请提供 NNAL 安装路径。"
    exit 1
fi
```

---

## 详细内容

<details>
<summary>🔍 查看详细安装步骤</summary>

### 第1步：检查 NPU 驱动

```bash
npu-smi info
```

若 `npu-smi` 不可用或未检测到 NPU 设备，**停止安装并报错**。

### 第2步：调用 cann-operator-env-config（如尚未安装 Toolkit+Kernels）

如果 Toolkit+Kernels 尚未安装，首先调用 `cann-operator-env-config` 技能安装 Toolkit 和 Kernels：
- 该技能位于 `/home/s30073260/swiglu_quant/agent-skills-ivan-fork/skills/cann-operator-env-config/`
- 提供 3 种安装方式：离线 run 包、conda 在线、yum 在线
- 安装完成后，Toolkit 和 Kernels 将位于 `/usr/local/Ascend/`

### 第3步：Source Toolkit 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**关键**：NNAL 安装脚本依赖 `ASCEND_TOOLKIT_HOME` 等环境变量，必须先 source。

验证环境变量：
```bash
echo $ASCEND_TOOLKIT_HOME
# 应输出: /usr/local/Ascend/ascend-toolkit/latest 或自定义路径
```

### 第4步：安装 NNAL

> **注意**：以下命令使用用户提供的 `<INSTALL_PATH>`，如未提供则已在路径约束步骤中打断。

#### 方式一：run 包安装（推荐）

```bash
chmod +x Ascend-cann-nnal_*.run
./Ascend-cann-nnal_*.run --install --install-path="${INSTALL_PATH}"
```

#### 方式二：从 Docker 镜像提取

```bash
CANN_IMAGE="quay.io/ascend/cann:<版本>-<芯片>-<OS>-py<Python版本>"

# 示例
# CANN_IMAGE="quay.io/ascend/cann:9.0.0-beta.2-910-openeuler24.03-py3.11"

docker create --name cann_nnal_temp "${CANN_IMAGE}"
docker cp cann_nnal_temp:/usr/local/Ascend/nnal "${INSTALL_PATH}/"
docker rm cann_nnal_temp
```

### 第5步：加载环境变量

```bash
source "${INSTALL_PATH}/nnal/atb/set_env.sh"
```

持久化到 `~/.bashrc`：
```bash
cat >> ~/.bashrc << EOF
source ${INSTALL_PATH}/nnal/atb/set_env.sh
EOF
```

### 第6步：安装验证

```bash
# 检查 NNAL 目录结构
ls -lh ${INSTALL_PATH}/nnal/atb/

# 验证 Toolkit 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || \
    cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info

# 验证 NNAL 版本
cat ${INSTALL_PATH}/nnal/atb/version 2>/dev/null || echo "检查 NNAL 安装目录"

# 验证关键命令
which bisheng && bisheng --version
which atc && atc --version

# 验证 NPU 状态
npu-smi info
```

### 环境变量汇总

安装完成后，以下环境变量应正确设置：

| 变量 | 值 | 来源 |
|------|-----|------|
| `ASCEND_TOOLKIT_HOME` | `/usr/local/Ascend/ascend-toolkit/latest` | `set_env.sh` (Toolkit) |
| `ATB_HOME_PATH` | `${INSTALL_PATH}/nnal/atb` | `set_env.sh` (NNAL) |
| `LD_LIBRARY_PATH` | 含 `${INSTALL_PATH}/nnal/atb/lib` | `set_env.sh` (NNAL) |

</details>

---

## 参考信息

<details>
<summary>📦 NNAL 安装包规模与耗时</summary>

| 项目 | 参考值 |
|------|--------|
| NNAL run 包大小 | ~500-600MB |
| 安装耗时 | 约 5-10 分钟 |
| 总计（含 Toolkit+Kernels） | 约 15-20 分钟（3 包 ~3.6GB） |

</details>

<details>
<summary>🔧 非交互式安装</summary>

部分 run 包会询问确认，使用 `yes |` 管道实现非交互安装：
```bash
yes | ./Ascend-cann-nnal_*.run --install --install-path="${INSTALL_PATH}"
```

</details>

<details>
<summary>🐳 Docker 容器环境提示</summary>

安装时若出现 `"driver package is not installed"` 警告，属于正常现象——Docker 容器使用宿主机驱动，无需安装驱动包。

</details>

---

## 常见问题

<details>
<summary>❓ npu-smi 未检测到 NPU 设备</summary>

**原因**：NPU 驱动未安装或 Docker 容器未挂载 NPU 设备。

**解决**：
1. 确认宿主机 NPU 驱动正常：宿主机执行 `npu-smi info`
2. Docker 容器启动时添加 `--device=/dev/davinciX` 参数
</details>

<details>
<summary>❓ source set_env.sh 后 ASCEND_TOOLKIT_HOME 仍为空</summary>

**原因**：Toolkit 未安装或 set_env.sh 路径不正确。

**解决**：
1. 确认 `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh` 存在
2. 如不存在，返回执行 `cann-operator-env-config` 安装 Toolkit+Kernels
</details>

<details>
<summary>❓ NNAL 安装报权限错误</summary>

**原因**：安装路径需 root 权限。

**解决**：
```bash
# 使用 root 权限安装
sudo ./Ascend-cann-nnal_*.run --install --install-path=/usr/local/Ascend
```
</details>

---

## 执行结果

> ⚠️ **执行后填写**：技能执行完成后，参照下方格式填写实际执行结果。

### 检查点检查表

| 步骤 | 检查点描述 | 状态 |
|------|-----------|------|
| 1 | NPU 驱动正常 (`npu-smi info`) | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | Toolkit+Kernels 已安装（通过 cann-operator-env-config） | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 3 | `ASCEND_TOOLKIT_HOME` 已设置 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 4 | NNAL 安装成功 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 5 | 环境变量已加载 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 6 | NNAL 版本验证通过 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 6 | `bisheng` / `atc` 可用 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 6 | `npu-smi info` 正常 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |

**VERDICT: ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED / ⏭️ SKIPPED**

### 问题列表（若有）

| 等级 | 检查点 | 问题描述 | 建议 |
|------|--------|---------|------|
| 🔴 CRITICAL | - | - | - |
| 🟡 WARNING | - | - | - |

### 执行摘要

- **执行时间**：
- **执行环境**：
- **NNAL 版本**：
- **持续时间**：
- **通过率**：
