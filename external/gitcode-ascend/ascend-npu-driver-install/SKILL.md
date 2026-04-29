---
name: external-gitcode-ascend-ascend-npu-driver-install
description: 能完成昇腾NPU驱动和固件安装部署，实现安装包正则匹配提取、按需添加可执行权限、Python+Shell双重包校验、系统依赖先验后装、适配CentOS/RHEL/Ubuntu/Debian系统，适用于昇腾NPU驱动和固件安装部署。
metadata:
  author: ascend-deploy-team
  version: 1.0.0
  supported-chip: Ascend310P/Ascend910A/Ascend910B
  core-scripts: check_package.py,install_npu_driver.sh
  skill-type: hardware-deploy
allowed-tools: Bash(*) Python3(*)
original-name: ascend-npu-driver-install
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---
# Ascend NPU Driver & Firmware Auto-Install
该技能为提供**端到端自动化的NPU驱动和固件安装能力**，覆盖从安装包校验到驱动生效验证的全流程，无需手动分步操作，严格遵循昇腾官方安装规范，适配主流Linux发行版。

## 核心能力
1. 基于官方正则自动提取指定文件夹内的驱动/固件包，强制校验包数量唯一性，仅允许1个驱动包+1个固件包；
2. 安装包可执行权限按需赋权，无权限时自动执行`chmod +x`并二次校验赋权结果，避免权限问题导致安装失败；
3. Python+Shell双重包校验，提前验证包格式、路径、文件有效性，拦截无效安装包；
4. 系统依赖先验后装
5. 严格按昇腾官方**先驱动后固件**顺序安装，安装后提供交互式重启选项+官方`npu-smi`原生命令验证驱动状态。

## 前置准备
- 该版本无需校验系内核，直接进行部署即可
### 1. 脚本文件准备
将核心脚本`check_package.py`（Python包校验）和`install_npu_driver.sh`（Shell主安装）放在**同一目录**，本技能的根目录建议命名为`ascend-npu-driver-install`，与name字段保持一致。

### 2. 安装包要求
指定的安装包文件夹内**仅存放1个**符合昇腾官方命名格式的驱动.run包和1个固件.run包，无其他无关文件，包名格式严格遵循：
- 驱动包：`Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`
- 固件包：`Ascend-hdk-<chip_type>-npu-firmware_<version>.run`

### 3. 系统环境要求
- 权限：必须拥有**ROOT管理员权限**，可通过`sudo -i`命令切换；
- 系统：适配CentOS/RHEL、Ubuntu/Debian系列Linux发行版，支持yum/apt包管理器；
- 基础环境：系统已预装Python3，离线无网络环境需**手动提前安装**gcc、make、dkms核心依赖；
- 硬件：适配昇腾Ascend310P、Ascend910A、Ascend910B系列NPU芯片。

## 快速使用步骤
### 限制要求
- 安装包的参数仅支持 --full、--install、--install-for-all

### 步骤1：为脚本添加可执行权限
进入脚本所在的技能根目录，执行以下命令为两个核心脚本赋予可执行权限：
```
chmod +x ./scripts/install_npu_driver.sh ./scripts/check_package.py
```
### 步骤2：执行自动化安装脚本
**命令格式**：`./scripts/install_npu_driver.sh <NPU包文件夹完整路径> <驱动运行用户>`
**推荐示例**（使用root用户进行安装）：
```bash
./scripts/install_npu_driver.sh /opt/ascend/npu_pkgs root
```

### 步骤3：系统重启（驱动生效必做）
安装完成后脚本会弹出交互式重启提示，**NPU驱动和固件生效必须重启系统**，无重启则无法完成驱动加载：
- 输入`y`：系统立即重启，完成NPU驱动内核加载；
- 输入`n`：跳过立即重启，后续需手动执行`reboot`命令完成系统重启。
### 步骤 4：验证安装结果
系统重启后，执行昇腾官方原生命令验证 NPU 驱动加载状态：
```
npu-smi info
```
**安装成功标识**：命令输出内容包含NPU 芯片型号、Driver Version（驱动版本）、Firmware Version（固件版本），无任何报错信息。
### 核心脚本说明
**check_package.py（Python 包校验脚本）**
由 Shell 主脚本自动调用，无需手动执行，核心完成以下包校验工作：
检查安装包所在文件夹是否存在；
按昇腾官方正则匹配驱动 / 固件包，校验包数量唯一性；
验证包为有效文件（非目录）；
检测包的可执行权限，无权限时给出警告提示；
输出校验通过的驱动 / 固件包完整绝对路径，供 Shell 脚本调用。
**install_npu_driver.sh（Shell 主安装脚本）**
技能核心执行脚本，按固定流程自动化运行，全程无需人工干预，执行流程为：ROOT/Python3环境检查 → 包路径正则提取 → 包可执行权限按需赋权 → Python包二次校验 → 运行用户/组自动创建 → 系统依赖先验后装 → NPU驱动安装 → NPU固件安装 → 交互式重启确认 → npu-smi原生验证
