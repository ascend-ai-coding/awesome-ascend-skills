---
name: ai-for-science-ai4s-basic
description: AI for Science 通用昇腾 NPU 模型迁移 Skill，适用于将基于 PyTorch、TensorFlow、vLLM 等框架的 CUDA 项目迁移到华为 Ascend NPU，覆盖环境检查、代码分析、自动迁移、手动适配、分布式改造、第三方库替换、验证与专属 skill 沉淀全流程。
keywords:
   - ai-for-science
   - migration
   - cuda-to-npu
   - pytorch
   - tensorflow
   - ascend
---

 
 # 通用昇腾 NPU 模型迁移 Skill
 
 本 Skill 提供一套通用的、从 CUDA 迁移到昇腾 NPU 的标准化流程，
 适用于 PyTorch、TensorFlow、vLLM 等主流框架的模型项目。
 迁移完成后，应根据实际跑通步骤生成该模型专属的迁移 Skill，以便复用。
 
 ## 前置条件
 
 执行迁移前确认以下环境就绪：
 
 | 项目 | 要求 |
 |------|------|
 | 硬件 | Ascend910 系列（至少 1 卡） |
 | OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
 | CANN | ≥ 8.0（推荐 8.2+ 或 8.3.RC1） |
 | Python | 3.8 – 3.10（推荐 3.10） |
 | PyTorch | 与 CANN 版本匹配（参考华为版本配套表） |
 | torch_npu | 与 PyTorch 版本一致 |
 
 ## 迁移流程总览
 
 ```
 0. 常用技巧与环境初始化（高优先级）
 → 1. 环境检查与代码分析
 → 2. 自动迁移注入
 → 3. 手动修改 CUDA 依赖
 → 4. 分布式适配
 → 5. 非 torch 框架兼容（如有）
 → 6. CUDA 内核算子 .cu 文件处理
 → 7. 第三方依赖库适配
 → 8. 适配验证
 → 9. 生成模型专属迁移 Skill
 ```
 
 按以下各节顺序执行，每步完成后再进入下一步。
 
 ---
 
 ## 0. 常用技巧与环境初始化（在所有步骤前高优先级执行）
 
 ### 0.1 设备与 CANN 环境
 
 ```bash
 # 指定可见 NPU 卡
 export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
 
 # CANN 默认环境脚本
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 
 # 查看 CANN 版本
 cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
 
 # 查看 NPU 设备状态
 npu-smi info
 ```
 
 ### 0.2 torch_npu 基础校验
 
 ```bash
 python3 -c "import torch; import torch_npu; a = torch.randn(3, 4).npu(); print(a + a)"
 ```
 
 若可正常输出 `device='npu:0'` 的 Tensor，说明 PyTorch + NPU 运行时可用。
 
 若报错，排查顺序：
 1. `set_env.sh` 是否已 source
 2. `decorator` 等运行时依赖是否安装（见 0.3）
 3. CANN 与 torch_npu 版本是否匹配
 
 ### 0.3 依赖与镜像源
 
 ```bash
 # 华为镜像
 export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
 
 # numpy 使用 1.26.4（规避 numpy 2.x 与低版本 CANN 兼容问题）
 pip install numpy==1.26.4
 
 # torch_npu 运行常见依赖
 pip install decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado
 ```
 
 ### 0.4 文本换行与仓库清理
 
 ```bash
 find ./ -type f -exec dos2unix {} \;
 ```
 
 用于修复从 Windows 环境带来的换行符问题。
 
 ### 0.5 日志与调试环境变量
 
 ```bash
 # 开启 CANN 详细日志（调试时使用，正常运行关闭以避免性能损失）
 export ASCEND_GLOBAL_LOG_LEVEL=1   # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
 export ASCEND_SLOG_PRINT_TO_STDOUT=1
 
 # 算子编译缓存，加速二次运行
 export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
 ```
 
 ---
 
 ## 1. 环境检查与代码分析
 
 在动手修改之前，对项目进行全面分析，明确迁移工作量。
 
 ### 1.1 框架识别
 
 判断项目所用框架，不同框架迁移策略不同：
 
 | 框架 | 迁移方式 |
 |------|----------|
 | PyTorch | `torch_npu` + `transfer_to_npu`（自动迁移优先） |
 | TensorFlow | `npu_device` 插件或 `npu_bridge` |
 | vLLM | 需确认昇腾版 vLLM 支持情况，替换 backend |
 | JAX | 当前昇腾不直接支持，需改写为 PyTorch 或评估可行性 |
 | PaddlePaddle | 昇腾原生支持，使用 `paddle_npu` 插件 |
 
 ### 1.2 CUDA 依赖分析
 
 在项目中搜索以下 CUDA 特有 API，标记需要手动处理的位置：
 
 ```bash
 # 搜索 CUDA 特有 API
 rg -n "torch\.cuda\.get_device_capability" .
 rg -n "torch\.cuda\.get_device_properties.*\.major" .
 rg -n "torch\.cuda\.amp" .
 rg -n "@torch\.cuda\.amp\.autocast" .
 rg -n "torch\.cuda\.mem_get_info" .
 rg -n "cuda\.CUDAGraph\|torch\.cuda\.graph" .
 
 # 搜索 .cu / .cuh 文件
 find . -name "*.cu" -o -name "*.cuh" | head -20
 ```
 
 ### 1.3 依赖包分析
 
 检查 `requirements.txt` / `setup.py` / `pyproject.toml` 中是否包含 CUDA 特有包：
 
 | CUDA 特有包 | 昇腾替代方案 |
 |-------------|-------------|
 | `flash_attn` | `torch_npu.npu_fusion_attention` |
 | `jax[cuda]` | 暂不支持，需框架替换 |
 | `xformers` | 部分算子可用 `torch_npu` 原生注意力替代 |
 | `apex` | `torch_npu` 内置 AMP 支持，或用 PyTorch 原生 AMP |
 | `triton` | 昇腾当前不支持 Triton，需改写为标准 PyTorch 算子或 AscendC |
 | `bitsandbytes` | 昇腾暂无等价实现，需禁用量化或换用昇腾量化工具 |
 | `deepspeed` | 需使用昇腾适配版 DeepSpeed |
 | `cupy` | 需替换为 numpy/scipy 或 AscendCL API |
 
 ### 1.4 分布式分析
 
 ```bash
 # 检查分布式相关代码
 rg -n "DataParallel\b" .           # DP 模式，需改为 DDP
 rg -n "DistributedDataParallel" .   # DDP 模式，改 backend 为 hccl
 rg -n "nccl" .                      # 需替换为 hccl
 rg -n "init_process_group" .        # 检查 backend 参数
 ```
 
 **关键**：若项目使用 `DataParallel`（DP），必须改为 `DistributedDataParallel`（DDP），
 因为昇腾 NPU 不支持 DP 模式。
 
 ### 1.5 生成分析报告
 
 建议将分析结论整理为迁移清单，记录：
 - 框架类型
 - 需手动修改的文件及行号
 - 不兼容的第三方库及替代方案
 - 分布式模式及所需改动
 - .cu 文件数量及复杂度评估
 
 ---
 
 ## 2. 自动迁移注入（优先执行）
 
 在入口脚本**顶部**（所有其他 import 之前）添加：
 
 ```python
 import torch_npu
 from torch_npu.contrib import transfer_to_npu
 ```
 
 `transfer_to_npu` 会自动完成以下映射，无需手动逐行替换：
 
 | 原始 CUDA API | 自动映射目标 |
 |---------------|-------------|
 | `torch.cuda.is_available()` | 返回 True（NPU 可用时） |
 | `torch.Tensor.cuda()` / `nn.Module.cuda()` | `.npu()` |
 | `torch.device('cuda')` | `torch.device('npu')` |
 | DDP backend `nccl` | `hccl` |
 | `torch.cuda.*` 系列 API | `torch.npu.*` |
 
 ### 注入位置选择
 
 - 若项目有**单一入口**（如 `main.py`、`run.py`），只需在该文件注入
 - 若有**多个入口**（如 `train.py`、`inference.py`、`test.py`），每个入口文件都需注入
 - 也可在项目根 `__init__.py` 中注入，确保任何 import 路径都能生效
 
 ---
 
 ## 3. 手动修改 CUDA 依赖
 
 `transfer_to_npu` 无法覆盖所有场景，以下 API 需要手动处理。
 
 ### 3.1 `torch.cuda.get_device_properties(...).major`
 
 NPU 没有 `major` 属性（CUDA compute capability 概念），直接访问会抛出 `AttributeError`。
 
 **修复模式**：用 try-except 包裹：
 
 ```python
 # 原始代码
 if stage == "predict" and not (
     torch.cuda.is_available()
     and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
 ):
     self.use_kernels = False
 
 # 修复：try-except 包裹
 if stage == "predict":
     try:
         has_capability = (
             torch.cuda.is_available()
             and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
         )
     except (AttributeError, RuntimeError):
         has_capability = torch.cuda.is_available()
     if not has_capability:
         self.use_kernels = False
 ```
 
 ### 3.2 `torch.cuda.get_device_capability()`
 
 同理，NPU 无此概念，需包裹或替换为固定值：
 
 ```python
 # 原始
 cap = torch.cuda.get_device_capability()
 if cap[0] >= 8:
     use_bf16 = True
 
 # 修复：NPU 默认支持 bf16（Ascend910 系列）
 try:
     cap = torch.cuda.get_device_capability()
     use_bf16 = cap[0] >= 8
 except (AttributeError, RuntimeError):
     use_bf16 = True  # Ascend910 支持 bf16
 ```
 
 ### 3.3 `torch.cuda.amp` / `autocast`
 
 NPU 支持 `torch.npu.amp`，但建议统一使用 PyTorch 原生 AMP：
 
 ```python
 # 推荐写法（兼容 CUDA 和 NPU）
 from torch.amp import autocast
 with autocast(device_type="npu"):  # 或根据实际设备动态设置
     output = model(input)
 ```
 
 ### 3.4 `torch.cuda.mem_get_info()`
 
 替换为 `torch_npu` 等价 API：
 
 ```python
 # 原始
 free, total = torch.cuda.mem_get_info()
 
 # 修复
 try:
     free, total = torch.cuda.mem_get_info()
 except (AttributeError, RuntimeError):
     total = torch.npu.get_device_properties(0).total_memory
     free = total - torch.npu.memory_allocated(0)
 ```
 
 ---
 
 ## 4. 分布式适配
 
 ### 4.1 DP → DDP 改造
 
 昇腾 NPU **不支持** `torch.nn.DataParallel`（DP），必须改为 `DistributedDataParallel`（DDP）。
 
 ```python
 # 原始 DP 写法
 model = torch.nn.DataParallel(model)
 
 # 改为 DDP
 import torch.distributed as dist
 dist.init_process_group(backend="hccl")  # 昇腾使用 hccl
 local_rank = int(os.environ.get("LOCAL_RANK", 0))
 torch.npu.set_device(local_rank)
 model = model.npu(local_rank)
 model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
 ```
 
 ### 4.2 Backend 替换
 
 所有 `backend="nccl"` 需改为 `backend="hccl"`。
 `transfer_to_npu` 通常会自动处理，但在显式配置文件（如 DeepSpeed config、
 Accelerate config）中需手动修改。
 
 ### 4.3 分布式启动命令
 
 ```bash
 # 单机多卡
 torchrun --nproc_per_node=8 train.py
 
 # 多机多卡（需配置 HCCL 通信）
 torchrun --nnodes=2 --nproc_per_node=8 \
     --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     train.py
 ```
 
 ---
 
 ## 5. 非 torch 框架兼容
 
 ### 5.1 TensorFlow 项目
 
 ```python
 # 安装昇腾 TF 插件
 # pip install npu_device  (或 npu_bridge，取决于 TF 版本)
 
 from npu_device.compat.v1.npu import npu_config
 npu_config()
 ```
 
 ### 5.2 vLLM 项目
 
 确认是否有昇腾适配版 vLLM，若有则替换安装；若无则需要评估改用其他推理框架
 （如 MindIE、vLLM 昇腾 fork）。
 
 ### 5.3 混合框架项目
 
 部分项目可能混用 PyTorch + 其他库（如数据处理用 JAX/TF），需分别处理各框架的设备映射，
 确保所有路径都路由到 NPU 或 CPU。
 
 ---
 
 ## 6. CUDA 内核算子 .cu 文件处理
 
 ### 6.1 评估策略
 
 先检查 `.cu` 文件中的逻辑复杂度：
 
 | 复杂度 | 判断标准 | 处理方式 |
 |--------|----------|----------|
 | 低 | 简单逐元素运算、reduction | 改写为 PyTorch 原生算子或 AscendC |
 | 中 | 涉及 shared memory、warp 操作 | 评估 AscendC 改写成本，或 fallback CPU |
 | 高 | 深度依赖 CUDA 生态（cuBLAS 等） | 优先 fallback CPU，必要时用 AscendCL |
 
 ### 6.2 CPU Fallback 模式
 
 若难以适配 NPU，让该算子切到 CPU 上执行，执行完毕后立刻切回 NPU：
 
 ```python
 def cuda_kernel_fallback(func, *args, **kwargs):
     # 将输入移到 CPU
     cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
     cpu_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
     # 在 CPU 上执行
     result = func(*cpu_args, **cpu_kwargs)
     # 结果移回 NPU
     if isinstance(result, torch.Tensor):
         return result.npu()
     return result
 ```
 
 **注意**：CPU fallback 会带来数据搬运开销，仅适合非热点路径。对性能敏感的算子应优先
 考虑 AscendC 改写或寻找等价的 torch_npu 原生算子。
 
 ### 6.3 禁用编译型扩展
 
 如果 `.cu` 文件通过 `setup.py` 的 `CUDAExtension` 构建，可在安装时跳过：
 
 ```bash
 # 设置环境变量跳过 CUDA 扩展编译
 export FORCE_CUDA=0
 export TORCH_CUDA_ARCH_LIST=""
 pip install -e . --no-build-isolation
 ```
 
 然后确保代码中有 fallback 路径（通常是 `try: import cuda_ext except: use_fallback = True`）。
 
 ---
 
 ## 7. 第三方依赖库适配
 
 ### 7.1 Flash Attention
 
 昇腾上使用 `torch_npu.npu_fusion_attention` 替代：
 
 ```python
 # 原始
 from flash_attn import flash_attn_func
 out = flash_attn_func(q, k, v, causal=True)
 
 # 替换
 import torch_npu
 out = torch_npu.npu_fusion_attention(
     q, k, v,
     head_num=num_heads,
     input_layout="BNSD",  # 根据实际 layout 调整
     atten_mask=causal_mask,
 )[0]
 ```
 
 **注意**：`npu_fusion_attention` 的输入 layout 和参数与 `flash_attn` 不完全一致，
 需要根据项目实际的 tensor shape 调整 `input_layout`（支持 `"BSH"`, `"BNSD"` 等）。
 
 ### 7.2 PyG 扩展库（torch_scatter / torch_sparse / torch_cluster）

 **强制要求：所有 PyG 扩展库必须从源码编译安装，禁止使用 `pip install torch_scatter -f https://data.pyg.org/whl/...` 等预编译 wheel 方式。**
 PyPI / PyG 预编译包基于 CUDA 或 x86 CPU 构建，在 aarch64 + NPU 环境下可能存在 ABI 不兼容或算子缺失问题，必须从源码编译：

```bash
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 
 git clone https://github.com/rusty1s/pytorch_scatter.git
 git clone https://github.com/rusty1s/pytorch_sparse.git
 git clone https://github.com/rusty1s/pytorch_cluster.git
 
 # 分别进入目录执行
 cd pytorch_scatter && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
 cd pytorch_sparse  && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
 cd pytorch_cluster && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
 ```
 
 ## 8. 适配验证
 
 ### 8.1 验证策略
 
 验证优先级：
 1. **用户指定**：若 prompt 中明确了验证方式，优先执行用户要求
 2. **官方 README**：按模型 GitHub 官网 README 中的步骤进行验证
 3. **最小推理测试**：运行最简单的推理示例，确认端到端无报错
 
 ### 8.2 通用验证步骤
 
 ```bash
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 export ASCEND_RT_VISIBLE_DEVICES=0
 
 # 运行推理/测试命令（根据项目实际情况调整）
 python inference.py --num_workers 0
 ```
 
 `--num_workers 0` 可避免部分环境下多进程与 NPU 驱动冲突。
 
 ### 8.3 验证通过标准
 
 - 程序正常退出（exit code 0），无 NPU 相关报错
 - 输出结果文件非空且格式正确
 - 结果与 CPU/CUDA 基线的精度偏差在合理范围内（fp32 场景差异 < 1e-5，
   fp16/bf16 场景差异 < 1e-3）
 
 ### 8.4 精度注意事项
 
 - Ascend910 **不支持 fp64（double）精度**，torch_npu 自动降级为 fp32
 - 若模型含随机采样（如扩散模型），逐样本对比无意义，应比较分布级指标
 - 建议在验证阶段固定随机种子以提高可比性：
 
 ```python
 import torch
 import numpy as np
 import random
 torch.manual_seed(42)
 np.random.seed(42)
 random.seed(42)
 ```
 
 ---
 
 ## 9. 生成模型专属迁移 Skill
 
 迁移跑通后，根据实际步骤生成一份模型完整的迁移适配 Skill，要求：

- **可复现**：任何一台配置相同的服务器上，任何 Agent 都可以根据该 Skill 快速完成迁移
- **自包含**：包含完整的环境要求、依赖安装命令、代码修改和验证步骤
- **格式统一**：遵循本 Skill 的 YAML frontmatter + Markdown 结构
- **记录差异**：明确标注该模型特有的适配点（如特殊算子替换、特殊配置修改）
 - **华为镜像源**：生成的 Skill 中所有 pip install 步骤必须配置华为镜像源，确保在国内环境下可快速安装依赖：

   ```bash
   export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
   ```

   或在每条 pip install 命令中追加 `-i https://repo.huaweicloud.com/repository/pypi/simple/`。
   生成的 Skill 应在依赖安装章节最前面设置此镜像源。
 - **PyG 库源码安装**：若项目依赖 PyG 扩展库（torch_scatter / torch_sparse / torch_cluster 等），生成的 Skill 必须使用源码编译安装，禁止使用预编译 wheel。参照本 Skill 第 7.2 节的源码编译步骤。

生成的 Skill 应放置在 `~/.kernelcat/skills/<model-name>/SKILL.md`。
 
 ---
 
 ## 迁移检查清单
 
 迁移到 NPU 时至少检查以下项目：
 
 - [ ] 入口脚本是否已注入 `transfer_to_npu`
 - [ ] 是否规避 `torch.cuda.get_device_properties(...).major` 的 NPU 不兼容访问
 - [ ] 是否规避 `torch.cuda.get_device_capability()` 的 NPU 不兼容访问
 - [ ] `torch.cuda.amp` / `autocast` 是否已适配
 - [ ] CUDA 专有 fused kernel 是否有 NPU fallback
 - [ ] `.cu` 文件是否已处理（改写 / CPU fallback / 跳过编译）
 - [ ] 分布式模式是否已从 DP 改为 DDP（如适用）
 - [ ] 分布式 backend 是否已从 nccl 改为 hccl
 - [ ] `flash_attn` 等第三方库是否已替换
 - [ ] 推理框架是否已注册 `npu` accelerator
 - [ ] 推理命令是否使用 `--num_workers 0`（避免部分环境多进程问题）
 - [ ] `__init__.py` 是否完整（避免 `ModuleNotFoundError`）
 - [ ] CUDA Graph 相关代码是否已禁用
 - [ ] 环境变量（`ASCEND_RT_VISIBLE_DEVICES`、`set_env.sh`）是否已配置
 
 ---
 
 ## 常见问题
 
 | 问题 | 原因 | 解决方案 |
 |------|------|----------|
 | `No module named 'decorator'` | torch_npu 运行时依赖缺失 | `pip install decorator` |
 | `SetPrecisionMode ... error code 500001` | CANN 环境未加载 | `source set_env.sh` |
 | `ModuleNotFoundError: xxx` | 缺少 `__init__.py` | 在对应目录添加空 `__init__.py` |
 | `torch_scatter` 编译失败 | 缺少编译工具 | `pip install setuptools wheel` |
 | double 精度警告 | Ascend910 不支持 fp64 | 无需处理，自动降级为 fp32 |
 | 多卡训练 hang | HCCL 通信问题 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 和网络配置 |
 | `RuntimeError: ... FORCE_CUDA` | 试图编译 CUDA 扩展 | 设置 `export FORCE_CUDA=0` |
 | OOM（内存不足） | NPU 显存管理不同 | 减小 batch_size 或启用梯度检查点 |
 | 算子不支持报错 `not supported on NPUAscend` | 特定算子未适配 | 使用 CPU fallback 或寻找等价算子 |
 | `HCCL ... timeout` | 多卡通信超时 | 增大 `HCCL_CONNECT_TIMEOUT`，检查卡间通信 |
 | 性能远低于预期 | 算子频繁 fallback CPU | 用 `torch_npu.profiler` 定位热点，优化算子 |

## 配套脚本

- 基础 NPU 环境预检：`python scripts/check_npu_basics.py --device npu:0`

## 参考资料

- 迁移前分析模板：[`references/pre-migration-analysis.md`](references/pre-migration-analysis.md)
- PyTorch 迁移 API 替换表：[`references/pytorch-migration-api-replacement.md`](references/pytorch-migration-api-replacement.md)
- 迁移完成定义与检查清单：[`references/validation-checklist.md`](references/validation-checklist.md)
