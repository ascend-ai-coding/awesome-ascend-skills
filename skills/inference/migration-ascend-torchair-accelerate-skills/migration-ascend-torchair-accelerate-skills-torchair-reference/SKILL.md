---
name: migration-ascend-torchair-accelerate-skills-torchair-reference
description: Provides torchair version compatibility, operator support matrix, CompilerConfig reference, and source code navigation guidance. Invoke when selecting torchair version, checking operator support, configuring compiler options, or solving issues by reading torchair source code.
---

# Skill: torchair 版本、算子、配置参考

你是一位精通 torchair 架构和接口的工程师。本 Skill 提供 torchair 的版本匹配关系、算子支持度查询方法、编译配置项和源码导航参考。服务于迁移步骤4（torchair 迁移）。

> 官方参考：
> - torchair 仓库：<https://gitcode.com/Ascend/torchair>
> - torchair 图模式使用指南：<https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00003.html>
> - 昇腾文档中心：<https://www.hiascend.com/document/>
>
> **⚠ 必须实时从官方仓库查阅最新的版本信息和 API 文档，以下内容仅供参考。**
> **⚠ 版本选择时优先使用最新稳定 Release，避免使用 master 在研分支。**

<constraints>
- 必须从 torchair 官方代码仓获取最新的版本配套信息，严格禁止凭记忆判断版本兼容性
- 必须在 torchair 代码仓中确认算子的 converter 实现状态，严格禁止假设某个算子已支持
- 必须优先使用镜像站或 ModelScope 获取模型/数据集资源
- 必须获取代码仓时优先使用国内可访问的代码托管平台或镜像（gitcode.com、bgithub.xyz 等）
- 必须使用 pip 安装时优先指定第三方镜像源（阿里源、清华源）
- **必须在遇到编译/性能阻塞时，参考多 CANN 版本兼容性差异表选择替代环境进行对比测试**
</constraints>

---

## 一、torchair 简介

torchair 是昇腾提供的 PyTorch 图模式推理加速扩展库，提供三种运行模式（`CompilerConfig.mode`）：

| 模式 | 后端 | 适用场景 |
|------|------|----------|
| **max-autotune** | GE Graph Engine | 下发瓶颈（算子融合减少下发次数）+ 计算瓶颈（深层融合+tiling） |
| **npugraph_ex** | ACL Graph | 下发瓶颈（Capture&Replay 消除重复下发）+ 推理，支持编译缓存 |
| reduce-overhead | ACL Graph | 已废弃，请用 npugraph_ex |

**模式选择核心逻辑：** 图模式编译的核心价值在于消除 Host→NPU **下发开销**。两种模式从不同路径解决：
- **max-autotune**：GE 算子融合 → 减少总算子数 → 减少总下发次数 → 适合算子数量多但可融合的场景
- **npugraph_ex**：Capture&Replay → 一次捕获多次重放 → 消除重复下发 → 适合同一子图被反复调用的场景
- 计算瓶颈（大矩阵饱和）→ max-autotune 融合+tiling 收益最显著
- 访存瓶颈 → npugraph_ex 优先（减少调度抖动），但需实测验证，纯访存瓶颈的编译收益有限

torchair 优先检查 `torch_npu.dynamo.torchair`（内置），次选 pip install，最后源码编译。

---

## 二、torchair 版本配套关系（完整版）

> **⚠ 编写时快照。以下版本表仅作兜底参考。**

### 2.1 版本命名体系说明

torchair 经历了两次版本命名体系变更：

| 命名体系 | 版本举例 | 匹配来源 | 时间范围 |
|----------|----------|----------|----------|
| **新体系** | 26.0.0, 26.1.0 | 版本号 = PyTorch 版本号 × 10 | 2026年起 |
| **旧体系** | 7.0.0 ~ 7.3.0 | 版本号 = torch_npu 版本号 | 2024-2025年 |
| **更早体系** | 6.0.0 | 版本号 = torch_npu 版本号 | 2024年 |

**版本名推导规则**：
- 新体系：PyTorch X.Y → torchair `{X}{Y}.0.0`（如 2.6 → 26.0.0, 2.10 → 210.0.0）
- 旧体系：跟随 torch_npu 版本号

### 2.2 完整版本配套表

| torchair | PyTorch | torch_npu | CANN | Python | 备注 |
|------------|------------------------|------------------|--------|------|
| **master**（在研） | 2.6.0+ | 在研 | 在研 | 3.9-3.13 | 在研分支，严格禁止用于正式迁移 |
| **26.1.0** | 2.10.0 / 2.12.0 | 待从官方仓库确认 | 待从官方仓库确认 | 3.9-3.13 | 最新稳定版，支持 inductor_npu_ext、cat lowering。**实际使用时须从官方仓库获取准确的 torch_npu 和 CANN 配套版本** |
| **26.0.0** | 2.6.0 | 7.3.0 | 8.5.0 | 3.9-3.11 | 新命名体系首个正式版 |
| 7.3.0 | 2.6.0 / 2.7.1 / 2.8.0 | 7.3.0 | 8.5.0 | 3.9-3.11 | |
| 7.3.0 | 2.9.0 | 7.3.0 | 8.5.0 | 3.9-3.12 | Python 3.12 支持 |
| 7.2.0 | 2.1.0 | 7.2.0 | 8.3.RC1 | 3.8-3.11 | |
| 7.2.0 | 2.6.0 / 2.7.1 / 2.8.0 | 7.2.0 | 8.3.RC1 | 3.9-3.11 | |
| 7.1.0 | 2.1.0 | 7.1.0 | 8.2.RC1 | 3.8-3.11 | |
| 7.1.0 | 2.5.1 / 2.6.0 | 7.1.0 | 8.2.RC1 | 3.9-3.11 | |
| 7.0.0 | 2.1.0 / 2.3.1 / 2.4.0 | 7.0.0 | 8.1.RC1 | 3.8-3.11 | |
| 7.0.0 | 2.5.1 | 7.0.0 | 8.1.RC1 | 3.9-3.11 | |
| 6.0.0 | 旧版 | 6.0.0 | 旧版 CANN | 3.8-3.11 | 最早可用 release |

### 2.3 各版本主要差异

| 版本跨越 | 关键变化 |
|----------|----------|
| **6.0.0 → 7.0.0** | 升级 CANN 8.1.RC1、PyTorch 扩展到 2.5.1、AclGraph 基础框架、集合通信 zero-copy |
| **7.0.0 → 7.1.0** | 升级 CANN 8.2.RC1、PyTorch 扩展到 2.6.0、新增 MoE v2 接口、编译缓存增强 |
| **7.1.0 → 7.2.0** | 升级 CANN 8.3.RC1、PyTorch 扩展到 2.8.0、AclGraph tiling 下沉+micro-batch、`dynamic=True` 支持 |
| **7.2.0 → 7.3.0** | 升级 CANN 8.5.0、PyTorch 扩展到 2.9.0、Python 3.12、triton 兼容 |
| **7.3.0 → 26.0.0** | 新命名体系（对齐 PyTorch 2.6.0）、新增量化算子、原生多流 API、run_eagerly 静态 kernel |
| **26.0.0 → 26.1.0** | PyTorch 扩展到 2.10/2.12、新增 inductor_npu_ext、cat lowering、super kernel canfuse |

### 2.4 不同 CANN 版本间兼容性差异提示

> **⚠ 以下为经验积累的常见差异模式，非绝对规则。遇到编译/性能阻塞时，必须尝试不同版本进行对比验证，不可假设某个版本"一定"有问题或没问题。**

**已观察到的版本差异倾向**（仅供参考，每次迁移需实际验证）：

| 方面 | 较新版本（如 9.x） | 较旧版本（如 8.5.x） |
|------|-------------------|---------------------|
| `npu_fusion_attention_v3` 对 attention mask 格式的兼容性 | 可能更严格，部分 mask shape 组合可能不支持 | 可能更宽松 |
| max-autotune GE 编译器 shape 推断能力 | 不同版本对复杂 shape 的处理能力可能有差异 | 不同版本对复杂 shape 的处理能力可能有差异 |
| npugraph_ex 模式 | 较新版本才支持 | 可能不支持或仅有 max-autotune / reduce-overhead |
| torchair 获取方式 | 较新版本倾向于内置在 torch_npu 中 | 可能需要独立安装 |
| `TORCH_NPU_FUSED_ATTENTION=0` 有效性 | 不同版本对 NPU 融合注意力的拦截行为可能不同 | 不同版本对 NPU 融合注意力的拦截行为可能不同 |

**版本选择启发式：**
1. 优先使用用户提供的或服务器上已有的最新稳定版本
2. 若遇到 GE 编译器报错（如 InferShape 失败）→ 尝试其他可用 CANN 版本
3. 若遇到 FlashAttention 兼容问题 → 尝试其他可用 CANN 版本或调整 attention 实现
4. 若 npugraph_ex 不可用 → 尝试较新版本或使用 max-autotune
5. **每次切换版本后必须完整重跑：eager 基线 → torchair 编译 → 精度 → 性能**
6. **不可断言"某版本一定有问题"——同一模型在不同小版本上的行为可能完全不同，必须以实测为准**

### 2.5 版本选择决策流程

```
PyTorch 版本 → 版本配套表匹配 → 获取 torchair（内置/pip/源码）
  ├── 成功 → 使用获取到的版本
  └── 失败 → 回退到版本配套表手动匹配
       ├── 在表中 → 选最新稳定版本
       └── 不在表中 → 能否调整 PyTorch？
           ├── 能 → 调整到最近的支持版本
           └── 不能 → 评估相近版本兼容性
```

---

## 三、torchair 代码仓结构导航

```
torchair/
├── python/torchair/                          # Python 前端包
│   ├── __init__.py                           # 对外 API 入口
│   ├── npu_fx_compiler.py                    # 主编译器（_NpuFxCompiler）和编译流程
│   ├── configs/                              # 编译器配置项
│   │   └── compiler_config.py                # CompilerConfig 完整定义
│   ├── _ge_concrete_graph/                   # GE 图后端实现（max-autotune 模式）
│   │   ├── fx2ge_converter.py                # FX → GE IR 核心转换逻辑
│   │   ├── ge_converter/                     # ATen 算子 → GE IR 转换器目录
│   │   │   ├── aten/                         # ATen 算子转换器（每个算子一个 .py 文件）
│   │   │   ├── prim/                         # Prim 算子转换器
│   │   │   ├── builtin_converters.py         # Python 内建函数转换器
│   │   │   ├── converter_utils.py            # 转换器公共工具函数
│   │   │   └── experimental/                 # 实验性转换器
│   │   ├── auto_generated_ge_raw_ops.py      # 自动生成的 GE 原生算子封装
│   │   └── ge_ir_pb2.py                      # GE IR Protobuf 定义
│   ├── _acl_concrete_graph/                  # ACL 图后端实现（reduce-overhead/npugraph_ex 模式）
│   │   └── fx2acl_converter.py               # FX → ACL 图转换
│   ├── ge/                                   # GE 图操作 Python 封装
│   │   └── _ge_graph.py                      # GE Graph/Tensor/Op 定义
│   ├── patterns/                             # FX 图 pattern 级优化 Pass
│   │   └── pattern_pass_manager.py           # Pattern Pass 管理器
│   └── inference/                            # 推理相关工具
│       └── _gear_utils.py                    # 动态分档工具
├── torchair/                                 # C++ 核心库
│   ├── core/
│   │   ├── torchair.h                        # C++ 核心类定义
│   │   └── torchair.cpp                      # C++ GE 图 Load/Compile/Run 入口
│   ├── concrete_graph/
│   │   └── concrete_graph.cpp                # GE 图编译执行实现
│   └── npu_graph_executor/                   # NPU 执行器
│       ├── static_npu_graph_executor.cpp      # 静态 shape 执行器
│       ├── dynamic_npu_graph_executor.cpp     # 动态 shape 执行器
│       └── muti_gear_npu_graph_executor.cpp   # 多分档执行器
├── codegen/                                   # 自动代码生成工具
├── tests/                                     # 测试用例
└── docs/                                      # 文档
```

### 3.1 关键查找路径

| 要查找的内容 | 在代码仓中的位置 |
|-------------|-----------------|
| 某 ATen 算子的 converter 实现 | `python/torchair/_ge_concrete_graph/ge_converter/aten/算子名.py` |
| CompilerConfig 全部配置项 | `python/torchair/configs/compiler_config.py` |
| GE 图构建入口 | `python/torchair/npu_fx_compiler.py` : `_NpuFxCompiler` |
| FX → GE IR 转换核心 | `python/torchair/_ge_concrete_graph/fx2ge_converter.py` |
| GE 原生算子定义 | `python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py` |
| 自定义 converter 注册 | `python/torchair/_ge_concrete_graph/fx2ge_converter.py` : `register_fx_node_ge_converter` |
| 模式选择逻辑 | `python/torchair/npu_fx_compiler.py` : `_NpuFxCompiler._gen_compiled_gm()` |

---

## 四、torchair 核心 API 说明

### 4.1 主要入口

```python
import torch

# 方式A：torch_npu 内置 torchair（优先级最高，torch_npu ≥ 2.9.0）
from torch_npu.dynamo.torchair import get_npu_backend, CompilerConfig

# 方式B：独立 torchair 包
import torchair
get_npu_backend = torchair.get_npu_backend
CompilerConfig = torchair.CompilerConfig

# 1. 获取 NPU 后端（最常用入口）
config = CompilerConfig()
npu_backend = get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False)

# 2. 获取编译器实例（底层入口，高级场景使用）
compiler = torchair.get_compiler(compiler_config=config)

# 3. 图导出（将模型编译并导出为 air 格式文件）
output = torchair.dynamo_export(model, *inputs, compiler_config=config)

# 4. 注册自定义 ATen 算子 converter（扩展不支持算子）
@torchair.register_fx_node_ge_converter(torch.ops.aten.my_op.default)
def my_converter(converter, node, inputs):
    pass

# 5. 注册自定义 FX 图替换 Pass
torchair.register_replacement(pattern, replacement, match_fn)

# 6. 启用集合通信入图（多卡推理）
torchair.patch_for_hcom()
```

### 4.2 CompilerConfig 完整配置项树

```
CompilerConfig
├── mode                    # "max-autotune" | "reduce-overhead"（已废弃）| "npugraph_ex"
├── post_grad_custom_pre_pass    # 用户自定义 FX 图前处理 Pass（callable）
├── post_grad_custom_post_pass   # 用户自定义 FX 图后处理 Pass（callable）
├── debug                   # 调试配置
│   ├── graph_dump.enabled      # 导出编译后 GE 图（.pbtxt 格式）
│   ├── graph_dump.path         # 图导出路径
│   ├── fx_summary.enabled      # 导出 FX 图摘要 CSV
│   ├── fx_summary.full_path    # FX 摘要保存路径
│   ├── data_dump.enabled       # 精度数据 dump
│   ├── data_dump.path          # 数据 dump 路径
│   └── run_eagerly             # debug 模式：强制 eager 执行
├── export                 # 图导出配置
│   ├── export_mode             # 导出 air 格式图
│   └── experimental.*         # 实验性导出选项
├── fusion_config           # 融合配置
│   └── fusion_switch_file     # 自定义融合策略 JSON 文件
├── experimental_config     # 实验性功能
│   ├── pattern_fusion_pass    # Pattern 级融合
│   ├── enable_view_optimize   # View 算子优化
│   ├── remove_noop_ops        # 消除空操作
│   ├── npu_fx_pass            # NPU 特定 FX 图 Pass
│   ├── topology_sorting_strategy # 拓扑排序策略
│   ├── cc_parallel_enable     # CC 并行
│   ├── enable_ref_data        # 引用数据
│   └── static_model_ops_lower_limit # 静态模型算子下限
├── inference_config       # 推理配置
│   └── use_internal_format_weight # 使用 NPU 私有格式权重
├── ge_config              # GE 引擎配置
│   ├── enable_single_stream    # 单流模式
│   ├── oo_level               # 优化级别（O0/O1）
│   ├── oo_constant_folding    # 常量折叠
│   └── oo_dead_code_elimination # 死代码消除
├── aclgraph_config        # ACL 图配置（reduce-overhead/npugraph_ex 模式）
│   └── use_custom_pool        # 使用自定义内存池
├── dump_config            # 数据 dump 配置
│   ├── enable_dump             # 开启 dump
│   └── data_dump_stage        # dump 阶段（"original" 等）
└── aoe_config             # AOE 调优配置
```

---

## 五、从源码获取解决方案

### 5.1 确定算子是否有 converter

```bash
# 在 torchair 代码仓中搜索算子名
cd torchair
ls python/torchair/_ge_concrete_graph/ge_converter/aten/ | grep -i "target_op_name"

# 或全局搜索
grep -r "target_op_name" python/torchair/_ge_concrete_graph/ge_converter/
```

### 5.2 查看某个 converter 的实现细节

阅读 `python/torchair/_ge_concrete_graph/ge_converter/aten/` 下对应的 `.py` 文件，了解：
- 输入输出如何映射
- GE 算子如何选择和调用
- 参数如何传递

### 5.3 理解编译错误

torchair 的编译错误通常包含：
- **不支持的算子**：形如 `"No converter found for XXX"` → 需要添加 converter
- **GE 图编译错误**：形如 `"GE compile failed XXX"` → 查看 GE 引擎日志
- **算子参数不匹配**：形如 `"Mismatch in op XXX args"` → 检查 converter 实现的参数传递

### 5.4 从 torchair 仓库拉取指定版本

```bash
# 查看所有可用版本分支
git branch -a

# 切换到需要的版本
git checkout -b 26.0.0 remotes/origin/26.0.0

# 或重新克隆
git clone -b 26.0.0 https://gitcode.com/Ascend/torchair.git
```

---

## 六、模型级别优化参考

### 6.1 静态 shape vs 动态 shape

| 场景 | 推荐配置 | 原因 |
|------|----------|------|
| 固定分辨率输入 | `dynamic=False` | 编译优化最充分，性能最优 |
| 变化分辨率但种类有限 | `dynamic=False` + dynamic gears | 兼顾灵活性和性能 |
| 完全动态 shape | `dynamic=True` | 灵活但性能略有损失 |

### 6.2 大模型推理优化

- **分页 KV Cache**：torchair 支持分页注意力（PagedAttention）图模式
- **量化推理**：torchair 支持动态量化（DynamicQuant、DynamicBlockMxQuant）
- **Super Kernel**：torchair 支持将多个算子融合为一个超级算子减少调度开销

### 6.3 多模型部署优化

- **编译缓存**：npugraph_ex 模式支持将编译结果缓存，避免重复编译
- **图导出加载**：使用 `dynamo_export` 导出 air 图文件，后续直接加载运行
- **静态 Kernel 预编译**：纯静态 shape 网络可预编译静态 Kernel 包

---

## 七、NPUGraph — torch_npu 原生图捕获（torchair 备选方案）

> **NPUGraph 不是 torchair / torch.compile 的一部分。** 它属于 `torch_npu`（`torch_npu.npu.NPUGraph`），是 torch_npu 提供的 Stream 级图捕获机制，与 CUDA Graphs 理念一致。当 torchair 图模式编译遇到无法解决的阻塞时，NPUGraph 可作为备选入图方案。

### 7.1 NPUGraph vs torchair 对比

| 维度 | **NPUGraph**（torch_npu） | **torch.compile + torchair** |
|------|--------------------------|------------------------------|
| **机制** | Stream 级算子序列捕获 → replay | Dynamo trace FX 图 → GE/ACL 编译 → 执行 |
| **入图方式** | `capture_begin()` / `capture_end()` 或 `with torch_npu.npu.graph(g)` | `torch.compile(model, backend=npu_backend)` |
| **粒度** | 裸算子序列（直接操作 Stream） | FX 计算图（经 Dynamo 自动 trace） |
| **算子要求** | 仅支持 aclnn 算子 | max-autotune 需 GE converter；npugraph_ex 兼容性更好 |
| **动态 shape** | 不支持（捕获时 shape 固定） | max-autotune + dynamic=True 支持 |
| **图优化（融合）** | 无 | GE 引擎自动融合 |
| **编译缓存** | 无需编译（直接捕获） | npugraph_ex 支持 cache_compile |
| **API 层次** | 底层手动控制 | 中高层自动 |
| **适用场景** | 静态子图、短算子密集序列、自回归逐 token 解码 | 复杂模型的全图/子图编译优化 |

### 7.2 何时应尝试 NPUGraph

NPUGraph 在以下场景尤为重要：

- **自回归解码**：每 decode 步操作序列固定（Q/K/V projection → attention → FFN），但 torch.compile 难以处理逐 token 变化的 seq_len → 用 NPUGraph 每步捕获一张图，replay 消除 Host 调度开销
- **编译失败后的兜底**：torchair 因 Dynamo trace 失败（如 hook、控制流、namedtuple 输入）无法编译 → NPUGraph 不经过 Dynamo，直接在 Stream 上捕获
- **短核密集型网络**：大量小算子导致 Host→Device 调度成为瓶颈 → NPUGraph replay 将 N 次下发合并为 1 次

### 7.3 三种使用方式

**方式一：NPUGraph 类（底层精细控制）**

```python
import torch
import torch_npu

s = torch_npu.npu.Stream()
with torch_npu.npu.stream(s):
    a = torch.full((1000,), 1, device="npu")
    g = torch_npu.npu.NPUGraph()
    torch_npu.npu.empty_cache()
    g.capture_begin()
    b = a
    for _ in range(10):
        b = b + 1
    g.capture_end()
torch_npu.npu.current_stream().wait_stream(s)

g.replay()  # 可多次调用
```

**方式二：graph 上下文管理器（简化）**

```python
g = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(g):
    b = a
    for _ in range(10):
        b = b + 1
g.replay()
```

**方式三：make_graphed_callables（自动处理不可捕获部分）**

```python
# 自动识别安全子图并封装，动态控制流/CPU 同步部分保持 eager
module1 = torch_npu.npu.make_graphed_callables(module1, (static_input,))
# 后续正常调用 module1(input)，内部自动 replay
```

### 7.4 NPUGraph 约束

- **仅支持 aclnn 算子**，不支持 GE 编译后的融合算子
- **捕获时 shape 固定**，不支持动态 shape（每次 shape 变化需重新捕获）
- **不支持动态控制流**（条件分支、循环结构需在图外）
- **不支持需要 CPU-NPU 同步的操作**（如 `.item()` 在图内触发同步会失败）
- **图内不能创建新 tensor**：token buffer、KV buffer 须图外预分配
- **仅支持单卡**（每张卡独立捕获独立的图）

### 7.5 与 torchair 的配合策略

```
模型推理管线
├── 优先尝试 torchair 全图/子图编译（max-autotune 或 npugraph_ex）
├── 编译失败或性能不达预期
│   ├── 静态子图（如 encoder、单层 attention）→ 尝试 NPUGraph 捕获
│   ├── 自回归 decode 循环 → NPUGraph 逐步捕获 + KV-cache 外部管理
│   └── 含动态控制流的部分 → 保持 eager
└── 报告中对 torchair 和 NPUGraph 的结果分别记录，取最优
```

> 官方文档：<https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/framework_feature_guide_pytorch/pytorch_npugraph_desc.md>
