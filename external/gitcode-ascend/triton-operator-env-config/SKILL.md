---
name: external-gitcode-ascend-triton-operator-env-config
description: 在 Ascend 昇腾平台上校验并构建triton算子开发所需环境,包括CANN、Python/torch/torch_npu/triton-ascend依赖和PATH环境变量等设置。当用户需要配置triton算子开发环境、检查CANN/torch/triton-ascend安装、验证环境是否可用时使用。
original-name: triton-operator-env-config
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Triton 算子开发环境配置

## 核心原则

**必须按顺序执行环境检查，每个步骤依赖前一步的成功。**

## 前置步骤：获取最新配套要求（MANDATORY）

**配套版本可能更新，必须先获取官方最新文档。**

**MANDATORY - READ ENTIRE PAGE**：访问并完整阅读官方在线文档 https://triton-ascend.readthedocs.io/zh-cn/latest/quick_start.html 和 https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html，获取最新的：
- Python 版本要求
- torch / torch_npu 版本要求
- triton-ascend 版本要求
- CANN 版本要求
- 各组件版本对应关系

以文档中的版本要求为准，更新后续步骤中的版本号。

**以下版本对应关系仅供参考（截止 2026-03-26），以在线文档为准：**

| Triton-Ascend 版本 | CANN 版本 | 发布日期 |
|-------------------|----------|---------|
| 3.2.0 | 8.5.0（推荐） | 2026/01/16 |
| 3.2.0rc4 | 8.3.RC2 | 2025/11/20 |

**torch_npu 版本（仅供参考）：2.7.1**

⚠️ 如果在线文档中的版本与上表不一致，**必须以在线文档为准**。

## 环境检查与配置流程

### 1. CANN 环境配置（第一步）

**必须最先检查 CANN 环境**

1. 执行`npu-smi info`，检查是否成功加载驱动
2. 再执行`which bisheng`，检查是否成功加载CANN环境获取到npuir编译器，应该输出路径
3. 如果没有输出，尝试加载 CANN 环境：
   - 优先：`source /usr/local/Ascend/cann/set_env.sh`
   - 备选：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
4. 再次执行第1步和第2步检查，如果不成功，需要等待用户去检查解决CANN的环境配置，可以提醒去https://www.hiascend.com/cann/download 上下载安装CANN

### 2. Python 版本检查

**如果遇到python问题，最优先使用miniconda创建环境解决**

1. 检查当前 python 的路径：`which python3`
2. 若失败尝试执行：`export PATH="/usr/bin:$PATH"`
3. 再次执行第1步检查，如果不成功，则需要提醒用户安装python，安装方式优先使用下面miniconda的方法
4. 如果存在python3，检查python版本：`python3 --version`，以官方 quick_start.md 文档中的要求为准

### 3. Python 环境安装（按需）

1. 如果用户需要安装python环境，先使用`conda init bash`检查是否有conda环境，有则跳转第5步，如果没有则执行第2步安装
2. 执行`uname -m`确认当前系统架构
   如果系统架构为aarch64，执行：`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`
   如果系统架构为x86_64，执行：`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
3. 执行安装脚本：`bash Miniconda3-latest-Linux-x86_64.sh`或`bash Miniconda3-latest-Linux-aarch64.sh`，按照提示安装后激活miniconda环境
4. 检查是否成功安装miniconda环境：`conda init bash`
5. 创建一个python环境，执行：`conda create -n triton python=<官方要求版本>`
6. 激活python环境，执行：`conda activate triton`

### 4. torch 配置

**必须保证前面的步骤都成功**

1. 执行：`pip list | grep "torch"`
2. 检查 torch / torch_npu 版本，以官方文档中的要求为准
3. 如果版本不符合要求，按文档中的版本安装：
   - 先安装 torch：`pip install torch==<官方要求版本>`
   - 若报错 `ERROR: No matching distribution found for torch==2.7.1+cpu`，尝试：
     ```bash
     pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
     ```
   - 再安装 torch_npu：`pip install torch_npu==<官方要求版本>`
4. 检查torch环境是否配置成功：`python3 -c "import torch; print(torch.__version__)"`
5. 运行一个简单的torch代码，检查是否成功加载npu设备：`python3 -c "import torch; a = torch.randn(2, 3); print(a)"`，应该输出类似结果：`tensor([[2.86, 1.0406, 1.5811], [0.8329, 1.0024, 1.3639]])`

### 5. triton-ascend 配置

1. 执行：`pip list | grep "triton"`
2. **社区 Triton 和 Triton-Ascend 不能同时存在**：
   - 如果安装了原生的 triton，必须先卸载：`pip uninstall triton`
   - 如果同时安装了 triton-ascend，也需要先卸载，再重新安装
3. 安装最新的 triton-ascend 包：`pip install triton-ascend`
4. 如需安装 nightly 版本，参考：https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#nightly-build

## 环境验证（MANDATORY）

用于确认当前终端环境可以正常执行triton算子。

**MANDATORY - READ ENTIRE FILE**：在执行验证前，必须完整阅读 [`scripts/01-vector-add.py`](scripts/01-vector-add.py)。

执行验证：
```bash
python3 <skill-dir>/scripts/01-vector-add.py
```

执行该算子样例后，如出现类似结果则表明其计算符合预期，difference为0.0则视为验证通过：
```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## 故障处理

| 现象 | 动作 |
|------|------|
| **torch 安装失败：No matching distribution found for torch==2.7.1+cpu** | 尝试从 PyTorch 官方源安装：`pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu` |
| **同时安装了原生 triton 和 triton-ascend** | 先卸载原生 triton：`pip uninstall triton`，再重新安装 triton-ascend |
| **找不到 C++ compiler** | 安装编译器：`apt-get install g++` |
| **需要从源码编译时缺少依赖** | 安装系统库：`sudo apt install zlib1g-dev clang-15 lld-15`（推荐 clang >= 15, lld >= 15） |
| **已按上表重试仍失败** | 保留完整终端报错与已执行的命令序列，便于本地或后续排查 |

## 反模式清单（NEVER）

- ❌ 跳过阅读官方在线文档获取最新配套要求
- ❌ 跳过 CANN 环境检查直接配置 Python
- ❌ 使用不符合官方要求的 Python / torch 版本
- ❌ 同时安装原生 triton 和 triton-ascend
- ❌ 不进行环境验证就开始开发
- ❌ 在不同终端会话中分步执行（必须在同一终端）

## 检查清单

- [ ] 已阅读官方在线文档获取最新配套要求？
- [ ] CANN 环境加载成功（npu-smi info 和 which bisheng 有输出）？
- [ ] Python 版本符合官方要求？
- [ ] torch / torch_npu 版本符合官方要求？
- [ ] triton-ascend 已安装（原生 triton 已卸载）？
- [ ] 环境验证通过（01-vector-add.py 运行成功）？

## 注意事项

- 如果上述流程后还有问题，可以到官方在线文档查看是否有新的配套要求：https://triton-ascend.readthedocs.io/zh-cn/latest/quick_start.html
