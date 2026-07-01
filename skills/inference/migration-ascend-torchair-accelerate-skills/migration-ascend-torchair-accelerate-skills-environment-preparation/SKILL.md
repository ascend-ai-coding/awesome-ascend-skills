---
name: migration-ascend-torchair-accelerate-skills-environment-preparation
description: Provides guidance on acquiring CANN Docker images, setting up the Ascend NPU environment, configuring mirrors, and performing pre-migration checks. Invoke during steps 1-2 of the torchair migration workflow.
---

# Skill: 环境准备 — CANN镜像获取与容器搭建

你是一名昇腾环境配置工程师。本 Skill 提供迁移步骤1~2所需的环境准备知识：资源获取方式、资源选择原则、镜像源配置、容器搭建和前置检查。

## 零、目录与工作区约束

**必须遵守以下目录访问原则：**

1. 所有文件操作（代码仓 clone、模型权重下载、脚本创建、日志输出等）**严格限制在用户指定的工作目录内**，禁止访问或修改指定范围之外的任何目录
2. 当 AI agent 不在 NPU 服务器本地运行时（通过 SSH 远程操作），须明确区分两类目录：
   - **本地目录**（agent 所在机器）：仅用于存放分析文档草稿、临时文件等非执行产物
   - **远端目录**（NPU 服务器/容器内）：代码仓 clone、模型权重下载、脚本执行、日志输出等所有实际操作均在此完成
3. 远端 NPU 环境的操作路径须在操作前与用户确认，禁止向未经确认的远端路径写入文件
4. Docker 容器内的工作目录建议统一规划（如 `/workspace`），避免文件散落导致后续复现困难

## 零.五、远端SSH执行建议

> ⚠ 当 AI agent 不在 NPU 服务器本地运行时（Windows 客户端通过 SSH 操作远端 Linux 服务器），常见问题：
> - paramiko `get_pty=True` + 大超时导致命令返回后仍然空转等待
> - PowerShell 内联 Python 的引号转义层层嵌套极易出错

**推荐方式**：使用统一的 SSH 执行器（参见本 Skill 同级目录下的 `ssh_runner.py`）：

```python
from ssh_runner import run_remote

# 远端执行命令，实时流式输出，进程退出后立即返回，deadline 兜底
r = run_remote(
    "docker exec container_name python3 /workspace/script.py 2>&1",
    deadline=600,   # 总时间上限，防止真卡死
)
print(r["stdout"])
```

**相比手写 paramiko 的优势**：
- `channel.settimeout(2)` 短超时 → 每次读完立即检查 `exit_status_ready()` → 进程一退出立刻返回
- 不再需要空等 600 秒超时
- 不再需要手动处理 `recv_ready()` / `recv_stderr_ready()` 循环

**传递脚本到远端**：先通过 sftp 上传，再 docker exec 执行，禁止在 SSH 命令中内联大量 Python 代码（引号嵌套必然出错）。

```python
# 正确：上传脚本后执行
sftp = client.open_sftp()
sftp.put(local_script, remote_path)
sftp.close()
run_remote(f"docker exec container_name python3 {remote_path}", deadline=600)

# 错误：内联长脚本
run_remote("docker exec xxx python3 -c '...几百行...'")  # 必出错
```

## 一、CANN 镜像获取

### 1.1 获取路径

优先基于昇腾官方提供的 CANN 镜像进行环境搭建：

1. 访问昇腾官网：<https://www.hiascend.com>
2. 进入"昇腾镜像仓库"
3. 选择 CANN 相关镜像

**镜像选择原则：**
- **必须基于最新商发（商用发布）版本的 CANN 镜像进行适配**，以确保后续所有操作可复现
- 严格禁止使用在研版本（如 master 分支）的镜像进行正式迁移
- 最终报告中必须写明镜像的完整路径和 tag

### 1.2 Docker 容器启动

```bash
# 基础启动命令（根据实际镜像路径调整）
docker run -it --rm \
    --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    <CANN镜像路径:tag> \
    /bin/bash
```

**关键配置说明：**
- `--privileged`：NPU 驱动访问的必要权限。缺少则出现 `"Can't get ascend_hal device count"` 错误
- `--device=/dev/davinci*`：映射 NPU 设备到容器内
- 驱动挂载路径须与宿主机实际安装路径一致

### 1.3 NPU 可用性验证

```bash
# 容器内验证 NPU 数量
npu-smi info

# 验证 PyTorch NPU 扩展
python -c "import torch; print(torch.npu.is_available(), torch.npu.device_count())"
```

## 二、镜像源配置

执行迁移时必须优先使用国内可访问的资源：

### 2.1 pip 源配置

```bash
# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 清华源（备选）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2.2 模型与数据集获取

| 资源类型 | 优先来源 | 备选来源 |
|---------|---------|---------|
| 模型权重 | ModelScope (<https://modelscope.cn>) | hf-mirror.com |
| 数据集 | ModelScope | 项目自带样例数据 |
| HuggingFace 模型 | hf-mirror.com（通过 `HF_ENDPOINT` 环境变量） | HuggingFace 官网（仅当镜像不可用） |

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.3 代码仓获取

| 代码仓 | 优先镜像地址 |
|--------|-------------|
| Ascend/torchair | <https://gitcode.com/Ascend/torchair> |
| Ascend/pytorch (torch_npu) | <https://gitcode.com/Ascend/pytorch> |
| GitHub 仓库 | bgithub.xyz 镜像 |

仅在镜像不可用时回退至 GitHub 官网。

## 三、前置五项检查（步骤1必须完成）

在 eager 基线跑通之前，必须先完成以下 5 项检查：

| 检查项 | 为什么必须 | 不检查的后果 | 检查方法 |
|--------|----------|------------|---------|
| **权重可达性** | 模型文件可能只在 GCS/内网 | 代码分析完才发现权重下不来，反复切换下载源 | 尝试下载或 `wget --spider` 探测 |
| **依赖版本兼容** | transformers/numpy/triton 版本冲突 | 反复安装/卸载/降级，每次都重来 | 对比 requirements.txt vs CANN 镜像内置版本 |
| **模型创建耗时** | 大模型构造可能数十秒至数分钟，非"挂起" | 误判为 OOM 或死循环，反复 kill 进程重试 | 在 CPU 上先创建一次，记录耗时 |
| **显存预估** | 确认 NPU 显存是否足够 | 误判为 OOM 放弃全模型 forward | 参数量 × dtype 字节数 × 倍数（含中间激活） |
| **Docker 权限** | NPU 驱动访问需 `--privileged` | "Can't get ascend_hal device count"反复排查 | 启动容器时检查 --privileged 是否配置 |

## 四、版本选择决策

### 4.1 torchair 版本选择流程

```
PyTorch 版本 → 查版本配套表 → 确定 torchair 版本范围
  ├── 版本表中存在 → 选最新稳定 Release
  ├── 版本表中不存在 → 能否调整 PyTorch？
  │   ├── 能 → 调整到最近的支持版本
  │   └── 不能 → 评估相近版本兼容性（需实际验证）
  └── 最终版本 → 尝试获取（内置/pip/源码）
        ├── 成功 → 使用该版本
        └── 失败 → 回退到版本配套表手动匹配下一个可用的 Release
```

### 4.2 环境版本对比

遇到编译失败 / 性能无提升 / 双模式均无法工作时：
- **必须对比至少 2 个 CANN/PyTorch 版本组合**
- 快速验证：复用已有容器快速筛选
- 最终确认：基于 CANN 官方镜像新建容器，报告中写明镜像路径和 tag
- **每次切换版本后必须完整重跑：eager 基线 → torchair 编译 → 精度 → 性能**
- **不可断言"某版本一定有问题"——同一模型在不同小版本上的行为可能完全不同，必须以实测为准**

### 4.3 torchair 获取方式（三选一）

**方式1：torch_npu 内置（首选，无需额外安装）**

torch_npu 2.9.0 及以上版本内置 torchair：

```python
from torch_npu.dynamo.torchair import get_npu_backend, CompilerConfig
```

**方式2：独立 torchair 包**

```bash
pip install torchair -i https://mirrors.aliyun.com/pypi/simple/
```

**方式3：源码编译（兜底）**

```bash
git clone https://gitcode.com/Ascend/torchair.git
cd torchair
git checkout <版本tag>
git submodule update --init --recursive
cd ./torchair
bash ./configure
mkdir build && cd build
cmake .. && make torchair -j8
make install_torchair
```

> 内置版本和独立包版本不可同时安装（会有命名空间冲突）。优先使用内置版本。

## 五、环境信息记录模板

步骤2完成后，必须以以下格式记录环境信息（用于迁移报告）：

```markdown
| 项目 | 值 |
|------|-----|
| NPU 型号 | Ascend 910B / 910B2 / ... |
| NPU 数量 | N |
| CANN 镜像 | <完整路径:tag> |
| CANN 版本 | <cann_version> |
| PyTorch 版本 | <torch_version> |
| torch_npu 版本 | <torch_npu_version> |
| torchair 版本 | <torchair_version>（获取方式：内置/pip/源码） |
| Python 版本 | <python_version> |
| Driver 版本 | <driver_version> |
| Docker 启动参数 | --privileged / 设备映射等 |
```

## 六、常见问题

### Q1: `Can't get ascend_hal device count`

**根因**：Docker 未配置 `--privileged` 或设备映射不完整。

**解决**：重新启动容器并添加 `--privileged --device=/dev/davinci0 --device=/dev/davinci_manager`。

### Q2: pip 安装 torch_npu 失败

**根因**：pip 源未配置或版本不兼容。

**解决**：使用阿里源，或从昇腾官方获取对应 CANN 版本的 torch_npu whl 包。

### Q3: `HF_ENDPOINT` 设置后仍无法下载模型

**解决**：切换至 ModelScope 下载模型权重：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('模型id')
```
