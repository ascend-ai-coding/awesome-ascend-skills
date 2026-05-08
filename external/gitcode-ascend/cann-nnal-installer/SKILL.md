---
name: external-gitcode-ascend-cann-nnal-installer
description: 昇腾NPU CANN Toolkit+Kernels+NNAL安装部署技能。支持从官网下载run包安装和从Docker镜像提取两种方式，覆盖驱动检查、包下载、安装、环境变量配置与验证全流程。当用户需要安装CANN全套组件或指定版本CANN到自定义路径时调用。
keywords:
- cann
- nnal
- toolkit
- kernels
- installer
- 昇腾
- ascend
- 环境搭建
metadata:
  short-description: 安装CANN Toolkit、Kernels、NNAL到指定路径，支持run包和Docker镜像两种安装方式。
  why: CANN是昇腾NPU开发的基础软件栈，但安装流程涉及多个组件和严格顺序，容易出错。
  what: 提供CANN Toolkit+Kernels+NNAL的端到端安装指导，支持离线run包安装和Docker镜像提取两种方式。
  how: 按照驱动检查→下载→安装(Toolkit→Kernels→NNAL)→环境变量配置→验证的固定流程执行。
  results: 产出安装验证报告，确认Toolkit/Kernels/NNAL版本正确、环境变量生效、bisheng/atc等工具可用。
  version: 1.0.0
  updated: '2026-04-16T00:00:00Z'
  jtbd-1: 当用户需要在新环境或Docker容器中安装指定版本的CANN Toolkit+Kernels+NNAL时。
  jtbd-2: 当用户需要将CANN安装到自定义路径（非默认/usr/local/Ascend）时。
  jtbd-3: 当用户因官网下载链接需要登录认证而无法直接wget获取CANN安装包时。
  skill-type: env-setup
allowed-tools: Bash(*)
original-name: cann-nnal-installer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# CANN Toolkit+Kernels+NNAL 安装部署

在昇腾NPU环境中安装CANN完整组件（Toolkit、Kernels、NNAL）到指定路径。

**核心原则**：安装顺序必须为 Toolkit → Kernels → NNAL，不可颠倒。NNAL依赖Toolkit的环境变量。

## 调用时机

- 用户要求安装CANN（含toolkit、kernels、nnal）
- 用户需要将CANN安装到自定义路径
- 用户在Docker容器中搭建昇腾开发环境
- 用户需要安装特定版本的CANN（如9.0.0-beta.2）

## 前置检查

### 第1步：检查NPU驱动

```bash
npu-smi info
```

**预期结果**：输出NPU设备信息（芯片型号、显存等）。
**异常处理**：如果 `command not found`，需先安装昇腾驱动和固件，参考 [ascend-npu-driver-install](../ascend-npu-driver-install/) 技能。

### 第2步：确认芯片型号

从 `npu-smi info` 输出中获取芯片型号，用于选择对应的Kernels包：

| 芯片型号 | Kernels包名后缀 |
|---------|----------------|
| 910B / 910B3 | `910b` |
| 910 | `910` |
| 310P | `310p` |
| 310B | `310b` |

### 第3步：确认系统架构

```bash
uname -m
```

- `aarch64`：ARM架构，选择 `_linux-aarch64.run`
- `x86_64`：x86架构，选择 `_linux-x86_64.run`

## 安装方式一：从官网下载run包安装

### 第4步：获取CANN安装包

访问昇腾社区下载中心获取安装包：

```
https://www.hiascend.com/developer/download/community/result?module=cann&cann=<版本号>
```

**需要下载的包**（3个）：

| 包名 | 格式 | 说明 |
|------|------|------|
| `Ascend-cann-toolkit_<版本>_linux-<架构>.run` | run包 | 开发工具包（必须先安装） |
| `Ascend-cann-kernels-<芯片>_<版本>_linux-<架构>.run` | run包 | 算子库（依赖Toolkit） |
| `Ascend-cann-nnal_<版本>_linux-<架构>.run` | run包 | 加速库（依赖Toolkit环境变量） |

**注意**：官网下载链接需要登录认证，无法直接 `wget`。替代方案：
1. 浏览器登录后下载，再上传到服务器
2. 使用下方的"方式二：从Docker镜像提取"

### 第5步：安装Toolkit

```bash
INSTALL_PATH="<用户指定安装路径>"

chmod +x Ascend-cann-toolkit_*.run
./Ascend-cann-toolkit_*.run --install --install-path="${INSTALL_PATH}"
```

**默认路径**：`/usr/local/Ascend`
**自定义路径示例**：`/home/user/run/cann900beta2`

### 第6步：安装Kernels

```bash
chmod +x Ascend-cann-kernels-*.run
./Ascend-cann-kernels-*.run --install --install-path="${INSTALL_PATH}"
```

### 第7步：配置Toolkit环境变量（NNAL安装前置条件）

```bash
source "${INSTALL_PATH}/ascend-toolkit/set_env.sh"
```

**关键**：NNAL安装脚本依赖 `ASCEND_TOOLKIT_HOME` 等环境变量，必须先source。

### 第8步：安装NNAL

```bash
chmod +x Ascend-cann-nnal_*.run
./Ascend-cann-nnal_*.run --install --install-path="${INSTALL_PATH}"
```

## 安装方式二：从Docker镜像提取

当无法直接下载run包时，可从华为官方CANN Docker镜像中提取已安装的CANN文件。

### 第4步：确认本地镜像

```bash
docker images | grep cann
```

华为官方CANN镜像标签格式：`quay.io/ascend/cann:<版本>-<芯片>-<OS>-py<Python版本>`

示例：`quay.io/ascend/cann:9.0.0-beta.2-910-openeuler24.03-py3.11`

如果没有本地镜像，先拉取：

```bash
docker pull quay.io/ascend/cann:9.0.0-beta.2-910-openeuler24.03-py3.11
```

### 第5步：创建临时容器并提取文件

```bash
INSTALL_PATH="<用户指定安装路径>"
CANN_IMAGE="quay.io/ascend/cann:9.0.0-beta.2-910-openeuler24.03-py3.11"

docker create --name cann_temp "${CANN_IMAGE}"

docker cp cann_temp:/usr/local/Ascend/cann-9.0.0-beta.2 "${INSTALL_PATH}/"

docker cp cann_temp:/usr/local/Ascend/nnal "${INSTALL_PATH}/"

docker rm cann_temp
```

**镜像内目录结构**：

| 镜像内路径 | 内容 | 目标路径 |
|-----------|------|---------|
| `/usr/local/Ascend/cann-<版本>/` | Toolkit + Kernels | `${INSTALL_PATH}/cann-<版本>/` |
| `/usr/local/Ascend/nnal/` | NNAL (atb加速库) | `${INSTALL_PATH}/nnal/` |

### 第6步：修正set_env.sh中的硬编码路径

Docker镜像中的 `set_env.sh` 包含硬编码路径 `/usr/local/Ascend/cann-<版本>`，需替换为实际安装路径：

```bash
sed -i "s|/usr/local/Ascend/cann-<版本>|${INSTALL_PATH}/cann-<版本>|g" \
  "${INSTALL_PATH}/cann-<版本>/set_env.sh"
```

同时检查其他脚本中的硬编码路径：

```bash
grep -rl '/usr/local/Ascend/cann-<版本>' "${INSTALL_PATH}/" --include='*.sh' | \
  xargs sed -i "s|/usr/local/Ascend/cann-<版本>|${INSTALL_PATH}/cann-<版本>|g"
```

## 环境变量配置

### 第9步：加载环境变量

```bash
source "${INSTALL_PATH}/cann-<版本>/set_env.sh"
source "${INSTALL_PATH}/nnal/atb/set_env.sh"
```

**持久化配置**（写入Shell配置文件）：

```bash
cat >> ~/.bashrc << EOF
source ${INSTALL_PATH}/cann-<版本>/set_env.sh
source ${INSTALL_PATH}/nnal/atb/set_env.sh
EOF
```

### 环境变量说明

| 变量名 | 来源 | 说明 |
|--------|------|------|
| `ASCEND_TOOLKIT_HOME` | toolkit set_env.sh | Toolkit安装根路径 |
| `ASCEND_HOME_PATH` | toolkit set_env.sh | 同ASCEND_TOOLKIT_HOME |
| `ASCEND_OPP_PATH` | toolkit set_env.sh | OPP算子路径 |
| `ATB_HOME_PATH` | nnal atb set_env.sh | ATB加速库路径 |
| `LD_LIBRARY_PATH` | 两个set_env.sh | 动态库搜索路径 |
| `PYTHONPATH` | toolkit set_env.sh | Python模块搜索路径 |

## 安装验证

### 第10步：验证安装完整性

```bash
source "${INSTALL_PATH}/cann-<版本>/set_env.sh"
source "${INSTALL_PATH}/nnal/atb/set_env.sh"

echo "=== Toolkit Version ==="
cat "${INSTALL_PATH}/cann-<版本>/<架构>-linux/ascend_toolkit_install.info" | grep version

echo "=== Kernels Version ==="
cat "${INSTALL_PATH}/cann-<版本>/<架构>-linux/ascend_ops_install.info" | grep version

echo "=== NNAL Version ==="
cat "${INSTALL_PATH}/nnal/atb/latest/version.info"

echo "=== Key Binaries ==="
which bisheng && echo "bisheng OK"
which atc && echo "atc OK"

echo "=== Environment Variables ==="
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME}"
echo "ATB_HOME_PATH=${ATB_HOME_PATH}"

echo "=== NPU Status ==="
npu-smi info | head -5
```

### 验证检查清单

- [ ] Toolkit版本信息正确
- [ ] Kernels(ops)版本信息正确
- [ ] NNAL(atb)版本信息正确
- [ ] `bisheng` 命令可用
- [ ] `atc` 命令可用
- [ ] `ASCEND_TOOLKIT_HOME` 指向正确路径
- [ ] `ATB_HOME_PATH` 指向正确路径
- [ ] `npu-smi info` 正常输出设备信息

## 安装验证报告格式

```
CANN INSTALLATION REPORT
========================
Install Path:  <安装路径>
Toolkit:       <版本号>
Kernels:       <版本号>
NNAL:          <版本号>
Architecture:  <aarch64/x86_64>
Chip:          <芯片型号>

VERDICT: SUCCESS / PARTIAL / FAILED

CHECKS:
  [1] Toolkit installed:    PASS / FAIL
  [2] Kernels installed:    PASS / FAIL
  [3] NNAL installed:       PASS / FAIL
  [4] Environment vars:     PASS / FAIL
  [5] bisheng available:    PASS / FAIL
  [6] atc available:        PASS / FAIL
  [7] NPU accessible:       PASS / FAIL

ISSUES: <count>
  [CRITICAL] <finding>
  [WARNING]  <finding>
```

## 常见问题

### 1. run包安装时报权限不足

**原因**：安装目录权限不够。

**解决方案**：
```bash
chmod 755 <安装路径的父目录>
```

### 2. NNAL安装失败，提示找不到Toolkit环境变量

**原因**：未先source Toolkit的set_env.sh。

**解决方案**：
```bash
source "${INSTALL_PATH}/ascend-toolkit/set_env.sh"
# 或（Docker镜像提取方式）
source "${INSTALL_PATH}/cann-<版本>/set_env.sh"
```

### 3. Docker镜像提取后bisheng/atc命令找不到

**原因**：set_env.sh中路径未修正。

**解决方案**：执行第6步的sed命令替换硬编码路径。

### 4. 官网下载链接无法直接wget

**原因**：昇腾社区下载链接需要登录认证，OBS存储的URL带有签名和过期时间。

**解决方案**：使用"方式二：从Docker镜像提取"。

### 5. Kernels包名与芯片型号不匹配

**原因**：910B芯片应选择 `910b` 后缀的kernels包，不是 `910`。

**解决方案**：根据第2步确认的芯片型号选择正确的kernels包。

## 注意事项

1. **安装顺序**：必须 Toolkit → Kernels → NNAL，不可颠倒
2. **环境变量**：NNAL安装前必须先source Toolkit的set_env.sh
3. **路径一致性**：三个组件必须安装到同一路径下
4. **版本匹配**：Toolkit、Kernels、NNAL版本必须一致
5. **驱动兼容**：CANN版本必须与宿主机NPU驱动版本兼容

## 参考链接

- [CANN软件安装指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0003.html)
- [CANN社区版下载](https://www.hiascend.com/developer/download/community/result?module=cann)
- [CANN Docker镜像](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)
