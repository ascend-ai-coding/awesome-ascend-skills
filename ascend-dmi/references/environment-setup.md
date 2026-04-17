# Ascend DMI 环境配置

## 前提条件

| 软件包 | 说明 | 必选 |
|--------|------|------|
| npu-driver | NPU 驱动程序 | 是 |
| npu-firmware | NPU 固件 | 是 |
| CANN 开发套件包 | 性能测试和诊断依赖 | 是 |
| Ascend-cann-ops | 算子库（CANN 8.5.0 前为 Ascend-cann-kernels） | 训练场景必选 |

---

## ToolBox 环境配置

ascend-dmi 包含在 MindCluster ToolBox 中。

### 配置流程

```bash
# 1. 检查 ascend-dmi 是否可用
which ascend-dmi

# 2. 如不可用，source ToolBox 环境
source /usr/local/Ascend/toolbox/set_env.sh

# 3. 验证
which ascend-dmi && ascend-dmi -v
```

### 如果仍不可用：安装 MindCluster ToolBox

```bash
# 下载（以 7.3.0 / aarch64 为例，按实际版本和架构调整）
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/MindX%207.3.0/Ascend-mindx-toolbox_7.3.0_linux-aarch64.run

# 安装（需要 root 权限）
chmod u+x Ascend-mindx-toolbox_*.run
./Ascend-mindx-toolbox_*.run --check    # 校验完整性
./Ascend-mindx-toolbox_*.run --install  # 安装

# 配置环境变量
source /usr/local/Ascend/toolbox/set_env.sh
```

安装成功标志：`[INFO] xxx install success`

---

## CANN 环境配置

多数 ascend-dmi 功能依赖 CANN。检查方法：

```bash
python -c "import acl; print(acl.get_soc_name())"
```

### 如果 CANN 不可用

**步骤 1：询问用户 CANN 路径**
- 直接问："CANN 安装在哪个路径？"
- 用户提供路径后：`source <路径>/set_env.sh`
- 必须获得用户同意后才能 source

**步骤 2：如用户不知道路径，征得同意后自动查找**

CANN 8.5.0+ 目录名为 `cann`，8.5.0 前为 `ascend-toolkit`。按优先级查找：

```bash
# 8.5.0+ 版本
ls /usr/local/Ascend/cann/set_env.sh 2>/dev/null
ls /home/miniconda3/Ascend/cann/set_env.sh 2>/dev/null
find /home -path "*/Ascend/cann/set_env.sh" 2>/dev/null

# 8.5.0 前版本
ls /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
find /home -path "*/Ascend/ascend-toolkit/set_env.sh" 2>/dev/null
```

**步骤 3：找到后确认并 source**
- 告知用户找到的路径，获得同意后 source
- source 后验证：`python -c "import acl; print(acl.get_soc_name())"`
- 如验证失败，继续查找下一个位置（不要执着于修复当前目录）

**步骤 4：全部失败时**
- 不要无限循环查找
- 告知用户："自动查找 CANN 失败，请提供准确的 CANN 安装路径"

---

## 环境检查清单

```bash
# 驱动
npu-smi info

# ascend-dmi
which ascend-dmi && ascend-dmi -v

# CANN
python -c "import acl; print(acl.get_soc_name())"

# 环境变量（可选）
echo $ASCEND_TOOLBOX_PATH
```

---

## 故障排查

### ascend-dmi 命令未找到

1. 检查是否已安装：`ls /usr/local/Ascend/toolbox/latest/Ascend-DMI/bin/ascend-dmi`
2. 已安装但未找到：`source /usr/local/Ascend/toolbox/set_env.sh`
3. 未安装：按上方步骤安装 MindCluster ToolBox

### CANN 不可用

按上方「CANN 环境配置」步骤处理。如果 source 后仍报错，检查驱动和固件是否安装：`npu-smi info`
