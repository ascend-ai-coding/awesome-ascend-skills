---
name: migration-ascend-torchair-accelerate-skills-migration-execution
description: Provides step-by-step procedures for torch_npu eager migration and torchair graph-mode migration on Ascend NPU. Invoke when performing actual code migration, configuring torch.compile, debugging unsupported operators, or tuning graph-mode performance.
---

# Skill: torchair 图模式代码迁移执行

你是一位专注于 PyTorch 图模式推理加速的工程师。本 Skill 提供步骤3（torch_npu 迁移）和步骤4（torchair 迁移）的具体执行步骤、代码模板、问题定位方法和性能调优策略。

> 官方参考文档：
> - torchair 图模式使用指南：<https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00003.html>
> - torchair 官方仓库：<https://gitcode.com/Ascend/torchair>
> - torch_npu 官方仓库：<https://gitcode.com/Ascend/pytorch>
> - torch_npu 迁移指南：<https://gitcode.com/Ascend/docs/tree/master/FrameworkPTAdapter>
> - torchair converter 扩展指南：torchair 代码仓 `docs/zh/ascend_ir/` 目录
> - torch_npu 算子支持清单：<https://www.hiascend.com/document/detail/zh/Pytorch/60.0.0.1/apiref/apilist/ptaoplist_0001.html>（版本号替换为实际版本）
>
> **⚠ 版本持续更新，必须实时查阅官方仓库获取最新文档。**

<constraints>
- 必须先在小规模输入上验证迁移代码，再扩展到全量数据
- 必须保留原始 torch_npu eager 推理脚本，新建独立脚本用于 torchair 迁移
- 必须编译范围按优先级递进：全模型 → 拆分子模型 → split-compile → 逐子模块，优先尝试最大编译范围
- 必须每步编译后立即验证精度（eager vs graph tensor 误差），不可累积到最后
- **必须使用真实模型+真实权重+真实推理管线进行最终验证**。简化模型或替代实现仅可用于编译通路验证，其数据不可作为最终结果
- **必须测量端到端全流程性能**（如 `transcribe()`、`generate()`、模型全链路 forward）。严格禁止仅测量 encoder/decoder 子模块的纯 forward 时间作为最终加速比
- 允许 patch 第三方库源码，但必须在报告中明确标注每次 patch 的位置、原因和前后对比
- 必须记录每一步的修改内容和原因，用于迁移报告
- 必须优先使用镜像站或 ModelScope 获取模型/数据集资源
- 必须使用 pip 安装时优先指定第三方镜像源（如阿里源、清华源）
- 必须遇到阻塞时穷举替代方案（≥3种，含环境版本切换）→ 逐一试验 → 全部失败后才可标记不可解决
- **必须遇到编译/性能阻塞时，主动对比至少 2 个 CANN/PyTorch 版本组合**
- **必须标准化 benchmark**：所有性能测量统一使用 warmup=3, measure=5, 取中位数, 标注标准差。严格禁止使用单次运行时间或无 warmup 的首次运行时间
- **必须测量接口一致性**：最终性能对比必须使用与 eager 基线完全相同的推理接口。子模块 forward 时间仅作补充分析，不可替代端到端数据
- **必须真实模型自检**：进入性能测量阶段前，确认当前模型是代码仓指定的完整真实模型（非简化版、非子模块独立构造），且加载了真实权重
</constraints>

---

## 一、迁移流程

以下为步骤3（torch_npu 迁移）和步骤4（torchair 迁移）的核心操作。

### 1.1 步骤3：torch_npu 迁移

> 本子步骤是主 Skill 步骤3（torch_npu 迁移）的具体执行手册。迁移方式选择、自动/手工迁移原理、CUDA→NPU API 映射等概念性知识参见 [torch_npu基础子Skill](../migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)（§三）。
>
> 官方参考：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/mig_methods_comp.md>

#### 3.1 NPU 环境验证

迁移代码添加完成后，在容器内确认 NPU 驱动与 PyTorch NPU 扩展可用：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__version__)"
python -c "import torch; print(torch.npu.is_available(), torch.npu.device_count())"
```

#### 3.2 模型在 NPU 上运行验证

```python
import torch
from torch_npu.contrib import transfer_to_npu

device = torch.device("npu:0")
model = YourModel().to(device)
model.eval()

# 确认所有参数和 buffer 已在 NPU
for name, p in model.named_parameters():
    assert p.device.type == 'npu', f"Parameter '{name}' on {p.device}, expected NPU"
for name, b in model.named_buffers():
    assert b.device.type == 'npu', f"Buffer '{name}' on {b.device}, expected NPU"

# 小规模 forward 验证
test_input = torch.randn(1, ...).to(device)
with torch.no_grad():
    output = model(test_input)
print("Model forward on NPU: OK")
```

#### 3.3 NPU 覆盖率检查与优化

遍历步骤1中分析出的推理调用栈热路径，检查是否存在 CPU 上的计算，对已识别的问题按以下优先级处理：

1. **直接创建在 NPU**：`torch.ones(..., device='npu')` 取代 `torch.ones(...).to('npu')`
2. **避免 NPU→CPU 搬出**：尽量减少 `.item()`、`.cpu()`、`.numpy()` 等操作
3. **无法迁移的操作**（如 `.item()` 控制流）→ 规划为步骤4 torchair 编译时的 eager 区域

> 详细检查方法与代码模板参见 §二，概念原理参见 [torch_npu基础子Skill](../migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)（§六）。

#### 3.4 Eager 基线记录

必须用标准化 benchmark 记录一次 eager 全链路推理的精度和性能基线。

> Benchmark 模板参见 [torch_npu基础子Skill](../migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)（§七）。

**关键要求：**
- **warmup=3, measure=5, 取中位数, 标注标准差**
- 记录推理接口签名（如 `model.transcribe(audio_path)`）—— 后续所有对比必须使用此接口
- 记录完整的环境版本信息

---

### 1.2 步骤4：torchair 迁移操作

#### 4.0 性能剖分（编译前必做，不可跳过）

> ⚠ **未完成性能剖分不得执行任何 torch.compile 调用。**

编译前对上一步跑通的 eager 模型执行子模块级 benchmark。目的：锁定**下发瓶颈（dispatch-bound）模块**——即 Host CPU 向 NPU 下发算子的调度开销占端到端耗时比例高的模块。图模式编译的核心价值在于消除此类下发开销（max-autotune 通过 GE 算子融合减少总下发次数，npugraph_ex 通过 Capture&Replay 消除重复下发），dispatch-bound 的判断是选择编译目标的最高优先级依据。

```python
# 对每个子模块做 eager benchmark，输出耗时和占 E2E 比例
def profile_submodules(model, inputs):
    results = []
    e2e_total = _bench_e2e(model, inputs)
    # 按调用栈逐层 benchmark
    submodules = [
        ("backbone", model.backbone),
        ("bert", model.bert),
        ("transformer.encoder", model.transformer.encoder),
        ("transformer.decoder", model.transformer.decoder),
        # ... 根据模型结构调整
    ]
    for name, sub in submodules:
        t, std = _bench_module(sub, inputs)
        results.append((name, t, std, t / e2e_total * 100))
    return results
```

产物：每个子模块的 eager 耗时、占 E2E 百分比、瓶颈类型标注：
- **下发瓶颈（dispatch-bound）**：大量小算子、Host 调度开销占比高 → 图模式加速的核心目标，编译收益最大
- **计算瓶颈（compute-bound）**：大矩阵乘法/attention 为主、NPU 计算单元近饱和 → max-autotune GE 融合优化收益显著
- **访存瓶颈（memory-bound）**：显存带宽为限制因素 → npugraph_ex 的 Capture&Replay 可减少 Host 调度开销，但需实测验证

决策：按"下发瓶颈程度 × 算子可编译性预估"排序编译优先级。任何占比的模块均须逐一评估——编译开销仅在首次编译时发生（后续可缓存），而编译收益在每次推理中累积。除非实测编译后该模块端到端性能出现负收益，否则不应跳过。

#### 4.1 模型分析（编译前必须完成）

**(a) 架构拆解** — 列出所有子模块、参数量，标注每步 tensor 操作的设备和特殊用法。

**结构特征识别**（决定后续可选策略）：
- **解码模式**：并行解码（一次 forward 产出全部输出）还是自回归解码（token-by-token）？
- **注意力结构**：Q/K/V 是否使用相同的归一化输入？Pre-LN 还是 Post-LN？
- **KV-cache 支持**：源码中是否已有 KV-cache 实现？hook 机制如何工作？
- **封装层级**：是否有 AutoModel 等封装层拦截了内部 tensor？

| 维度 | 分析内容 | 必须标注 |
|------|---------|----------|
| 模型结构 | 列出所有 module/layer 及其参数量 | 参数量 |
| 输入输出 | tensor 的 shape、dtype、动态/静态 | shape/dtype |
| **设备位置** | 推理热路径上每步 tensor 创建/运算的设备 | CPU 还是 NPU？`.to("npu")` 调用标注为"CPU→NPU copy" |
| **外部接口依赖** | 外部代码如何与该模块交互？ | 是否依赖__getitem__/__len__/isinstance/自定义属性等编译后会失效的接口 |
| 控制流 | if/for、while 循环 | 是否依赖运行时 tensor 值 |
| 特殊机制 | KV-cache、hook、beam search 等 | 对图编译的影响 |

**(b) NPU 覆盖率检查** — 参见 §二，在步骤3中完成，步骤4确认。

**(c) 算子兼容性预扫描** — 编译前用 FX graph 导出 + 模式匹配发现不兼容算子：

```python
import torch

def scan_unsupported_ops(model, example_inputs):
    """几秒完成，不需真正编译"""
    aten_ops = set()
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception:
        try:
            gm = torch.export.export(model, example_inputs).module()
        except Exception as e:
            return [], [("export", str(e)[:300])]

    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method"):
            aten_ops.add(str(node.target))

    # 不兼容模式分类（分类而非穷举，适用任何模型）
    # 核心方法论：遇到未知算子时，去 torchair 源码目录查 converter 是否存在
    #   ls torchair/python/torchair/_ge_concrete_graph/ge_converter/aten/ | grep <op_name>
    #   找不到 → 该算子不支持，需要替代方案或隔离在 eager
    KNOWN_PATTERNS = [
        # 类别1：常见无 GE converter 的算子（持续补充）
        ("aten::roll", "cat+slice 等价替换"),
        ("aten::nonzero", "动态输出 shape，需 dynamic=True 或用 mask 替代"),
        ("grid_sample", "目前无替代方案，含此算子的模块必须保持 eager"),

        # 类别2：Python 语句被 Dynamo 捕获后无法编译
        # 检测：搜索源码中的 assert/print/isinstance/hasattr，这些都会被 Dynamo 捕获
        ("aten::_assert_async", "Python assert 被捕获，替换为 if/raise"),
        ("__ior__", "in-place 操作 Dynamo 无法 trace，改为 out-of-place"),

        # 类别3：CUDA/NPU 特定 API 在编译图中异常
        # 检测：搜索源码中的 torch.cuda/torch.npu/autocast，这些可能在 transfer_to_npu 后行为异常
        ("autocast", "amp.autocast 在编译图中行为异常，enabled=False 时直接移除"),
        ("_C\.", "CUDA 自定义扩展不存在于 NPU，需提供纯 PyTorch fallback"),

        # 类别4：Dynamo trace 结构性问题
        # 判定：编译报错中不含 "EZ"/"GE" 错误码 → Dynamo 层面问题，非后端
        # 检测：搜索源码中的 checkpoint/hook/闭包
        ("checkpoint", "torch.utils.checkpoint 与编译 graph 冲突，需 monkey-patch 为 no-op"),
        ("register_forward_hook", "forward hook 副作用被编译图捕获，需模块级跳过"),

        # 类别5：第三方依赖 API 变更
        # 检测：对比代码仓 requirements.txt 与实际安装版本
        ("get_head_mask", "transformers 5.x 已删除 BertModel.get_head_mask"),
    ]

    issues = []
    for op in aten_ops:
        for pattern, suggestion in KNOWN_PATTERNS:
            if pattern in op:
                issues.append((op, suggestion))

    try:
        torch.export.export(model, example_inputs)
    except Exception as e:
        msg = str(e)
        if "not implemented" in msg.lower() or "unsupported" in msg.lower():
            issues.append(("export", msg[:300]))
        # Dynamo 失败判定：不含 EZ/GE 错误码 → 非后端问题，需缩小编译边界
        if "EZ" not in msg and "GE" not in msg:
            issues.append(("ALERT:Dynamo", "Dynamo trace 失败，非后端问题。缩小编译边界排查"))

    return list(aten_ops), issues
```

**FX 扫描的局限**：复杂模型（含动态控制流/NestedTensor/闭包）FX trace 可能失败。此时须配合 **(d) 源码关键词扫描**。对已识别的算子，**主动查阅 torchair converter 目录**确认支持状态：

```bash
# 确认某算子是否有 GE converter
ls torchair/python/torchair/_ge_concrete_graph/ge_converter/aten/ | grep <op_name>
# 若找不到对应文件 → 该算子不支持 → 等价替换或模块保持 eager
```

#### 4.2 torchair 导入

```python
import torch

# 路径A：torch_npu 内置 torchair（优先级最高）
try:
    from torch_npu.dynamo.torchair import get_npu_backend, CompilerConfig
    print("Using torchair bundled in torch_npu")
except ImportError:
    # 路径B：独立 torchair 包
    import torchair
    get_npu_backend = torchair.get_npu_backend
    CompilerConfig = torchair.CompilerConfig
    print("Using standalone torchair")
```

#### 4.3 编译策略与渐进编译

**编译策略优先级：优先尝试最大范围的编译，失败了再缩小范围。** 即：全模型 → 拆分子模型（如 encoder/decoder）→ split-compile → 逐子模块独立编译。找到可成功编译的最大范围后，逐步扩展直至覆盖全部可编译模块。

##### 4.3.1 编译模式选择（CompilerConfig.mode）

torchair 通过 `CompilerConfig.mode` 提供以下运行模式（来源：<https://gitcode.com/Ascend/torchair>）：

| 模式 | 后端 | 特性 | 适用场景 |
|------|------|------|---------|
| **max-autotune** | GE Graph Engine | 最高级别图优化（算子融合、tiling、常量折叠），首次编译耗时较长，运行时性能最优。融合后算子数量减少 → 总下发次数减少 → 消除下发瓶颈 | 下发瓶颈 + 计算瓶颈场景（大矩阵乘法、attention 等计算密集型算子占比高；或大量小算子需要融合减少下发次数） |
| **npugraph_ex** | ACL Graph | 编译速度快，支持编译缓存（cache_compile）。Capture&Replay 将多次下发合并为一次 → 消除重复下发开销 | 下发瓶颈场景（大量小算子、频繁推理调用、Host 调度开销占主导）；纯访存瓶颈场景 |
| reduce-overhead | ACL Graph | 已废弃 | 请使用 npugraph_ex 替代 |

##### 4.3.2 编译范围选择（compile scope）

根据 §4.1 的模型分析结果，按优先级选择编译范围：

| 优先级 | 策略 | 说明 | 适用场景 | 限制 |
|--------|------|------|---------|------|
| **P0（优先尝试）** | **全模型编译** | 对整个模型执行 `torch.compile(model, backend=...)` | 纯前馈网络、所有算子均已支持、无动态控制流 | Dynamo trace 可能因不兼容操作（namedtuple 输入、hook、Python 控制流等）失败。即使失败也应先尝试后再降级 |
| **P1（全模型失败后）** | **拆分子模型编译** | 将模型拆分为若干大的子模型（如 encoder、decoder）分别执行 `torch.compile` | 模型包含独立的功能模块，模块间有清晰的边界 | 各子模型需独立分析瓶颈类型（下发/计算/访存） |
| **P2（子模型仍失败）** | **split-compile** | 将纯计算子模块入图编译，动态控制流或含不兼容算子的部分保持 eager | 模型含动态控制流、不兼容算子、CPU 操作 | 需仔细设计编译边界，避免频繁的图内/图外切换 |
| **P3（兜底）** | **逐子模块独立编译** | 对每个子模块分别执行 `torch.compile` | 子模块间独立性强、输入输出类型统一；前面均失败后使用 | 多个独立编译单元增加首次编译总耗时 |

##### 4.3.3 编译目标粒度规则

`torch.compile(module)` 编译的是 `module.forward()` **内部全部操作**，不仅是核心计算。选定编译目标时须检查三项：

**规则A：编译叶节点计算模块，不编译包装器**

若 forward 内包含多种操作混合（如 embedding 查找 → mask 生成 → 主计算 → pooler），编译整个 module 会将不同生命周期的 tensor 打包进同一个 graph，导致 "stale tensors" 等 tensor 生命周期冲突。

```
错误：torch.compile(model.bert)  ← forward() 内含:
        self.embeddings → mask生成 → self.encoder → self.pooler
      全部打包在一个 graph → tensor 生命周期冲突

正确：torch.compile(model.bert.encoder)  ← 12 层纯 transformer，无副作用操作
      embeddings/pooler/mask 生成保持 eager
```

判定：阅读目标模块的 `forward()` 源码。若内部调用 ≥2 个不同类型的子模块且夹杂 tensor 创建/状态修改，优先编译最深的 compute-heavy 子模块。

**规则B：检查外部代码对模块的接口依赖**

`torch.compile` 后类型变为 `OptimizedModule`，不再继承原始类型。编译前检查外部代码是否依赖了以下将失效的接口：

| 接口 | 典型场景 | 检测方法 |
|------|---------|---------|
| `__getitem__` | `self.backbone[idx]`（nn.Sequential） | `grep 'self\.<name>\['` |
| `__len__`/`__iter__` | `for layer in module` | `grep 'for .* in.*<name>'` |
| `isinstance` | 类型检查分支 | `grep 'isinstance.*<类型名>'` |
| 自定义属性 | 外部访问 `module.custom_attr` | `grep '\.<name>\.'` |

若存在上述依赖且模块必须编译，使用包装器维护对外接口：

```python
class CompilableWrapper(nn.Module):
    def __init__(self, compiled_core, *eager_parts):
        super().__init__()
        for i, m in enumerate([compiled_core] + list(eager_parts)):
            self.add_module(str(i), m)
    def __getitem__(self, idx): return self._modules[str(idx)]
    def __len__(self): return len(self._modules)
    def forward(self, *a, **kw): ...  # 按原始调用链组合
```

**规则C：按剖分数据排编译优先级**

基于 §4.0 的剖分结果，按下发瓶颈程度排序（非拓扑顺序）：

| 优先级 | 条件 | 典型占比 |
|--------|------|---------|
| P0 | 下发瓶颈（dispatch-bound，大量小算子），无预扫描阻塞算子 | >5% |
| P1 | 下发瓶颈，含少量可替换算子（如 roll→cat） | >5% |
| P2 | 计算瓶颈（compute-bound），算子兼容性良好 | >10% |
| P3 | 占比 <5% 但下发瓶颈明显，或含少量不可替换算子 | 先尝试，实测负收益再跳过 |

##### 4.3.4 选择决策流程

基于 §4.1 模型分析结果进行匹配，按优先级递进：

```
1. 模型性能剖分 → 标注瓶颈类型
   ├── 下发瓶颈（dispatch-bound）→ 图模式加速的核心目标
   │   ├── 大量小算子 + 首次编译耗时可接受 → 优先尝试 max-autotune（GE 融合减少下发次数）
   │   └── 频繁重复推理 + 编译缓存需求 → 优先尝试 npugraph_ex（Capture&Replay 消除重复下发）
   ├── 计算瓶颈（compute-bound）→ 优先尝试 max-autotune（GE 深层融合 + tiling 收益大）
   └── 访存瓶颈（memory-bound）→ 优先尝试 npugraph_ex（Capture&Replay 减少调度抖动）
       └── 注意：纯访存瓶颈的加速收益通常有限，需实测验证

2. 编译范围选择（按优先级递进）
   ├── P0：优先尝试全模型编译
   │   └── 成功 → 完成，记录结果
   │   └── 失败 → 进入 P1
   ├── P1：拆分子模型编译（如 encoder/decoder）
   │   └── 成功 → 完成，记录结果
   │   └── 部分失败 → 成功的保留，失败的进入 P2
   ├── P2：split-compile（编译纯计算子模块，其余保持 eager）
   │   └── 成功 → 完成，记录结果
   │   └── 部分失败 → 进入 P3
   └── P3：逐子模块独立编译（如单层 attention/MLP）
       └── 找到可成功编译的最大范围后，逐步扩展

3. 每步编译后立即验证精度 → 精度通过后扩展范围
   ├── 精度或性能未达预期 → 回退至备选模式（如 max-autotune ↔ npugraph_ex 切换）
   └── 两种模式均无法达到预期 → 降级至下一优先级编译范围
```

> **最终须对所有可行组合逐一尝试，取性能最优者作为交付配置。**

##### 4.3.5 渐进编译代码模板

编译策略按优先级递进：全模型 → 子模型 → 子模块。找到可成功编译的最大范围后，逐步扩展。

```python
# 配置编译选项（根据 §4.3.1 选择）
config = CompilerConfig()
config.mode = "max-autotune"  # 初始选择：下发/计算瓶颈 → max-autotune；重复推理下发瓶颈 → npugraph_ex
config.debug.graph_dump.enabled = True
config.debug.graph_dump.path = "./torchair_dump_graph"

npu_backend = get_npu_backend(compiler_config=config)

# 加载模型
model = YourModel().to("npu")
model.eval()

# ===== 第1步：优先尝试全模型编译 =====
try:
    compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    with torch.no_grad():
        _ = compiled_model(test_input)  # 触发 JIT 编译
    # 验证精度
    with torch.no_grad():
        eager_out_full = model(test_input)
        graph_out_full = compiled_model(test_input)
    max_diff = torch.max(torch.abs(eager_out_full - graph_out_full) / (torch.abs(eager_out_full) + 1e-8))
    assert max_diff < 1e-2, f"精度超标: {max_diff.item()}"
    print(f"[Step 1] Full model compile: OK, max_diff={max_diff.item():.6f}")
except Exception as e:
    print(f"[Step 1] Full model compile failed ({e}), trying sub-model...")

    # ===== 第2步：拆分子模型（如 encoder + decoder）=====
    try:
        original_encoder = model.encoder
        model.encoder = torch.compile(model.encoder, backend=npu_backend, dynamic=False)
        with torch.no_grad():
            _ = model.encoder(encoder_input)
        # 验证精度
        with torch.no_grad():
            eager_enc = original_encoder(encoder_input)
            graph_enc = model.encoder(encoder_input)
        max_diff = torch.max(torch.abs(eager_enc - graph_enc) / (torch.abs(eager_enc) + 1e-8))
        assert max_diff < 1e-2, f"精度超标: {max_diff.item()}"
        print(f"[Step 2] Sub-model (encoder) compile: OK, max_diff={max_diff.item():.6f}")
    except Exception as e2:
        print(f"[Step 2] Sub-model failed ({e2}), trying sub-module...")

        # ===== 第3步：拆分子模块（如单层 attention）=====
        original_block = model.encoder.layers[0].self_attn
        model.encoder.layers[0].self_attn = torch.compile(
            original_block, backend=npu_backend, dynamic=False
        )
        # 验证精度
        test_input = torch.randn(1, 128, 512, device="npu")
        with torch.no_grad():
            eager_out = original_block(test_input)
            graph_out = model.encoder.layers[0].self_attn(test_input)
        max_diff = torch.max(torch.abs(eager_out - graph_out) / (torch.abs(eager_out) + 1e-8))
        assert max_diff < 1e-2, f"精度超标: {max_diff.item()}"
        print(f"[Step 3] Sub-module OK, max_diff={max_diff.item():.6f}")
```

#### 4.4 自回归解码的入图策略

> 自回归模型（token-by-token 生成）的 decoder 在循环中被反复调用，每步序列长度递增。以下是可选入图策略，按推荐程度排序。

**策略 A（推荐）：参考 vllm-ascend 的 batch 分档**

拉取 vllm-ascend 代码库，分析其图捕获和 replay 流程：
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
# 关键文件：
#   vllm_ascend/attention/attention_v1.py   — 注意力后端实现
#   vllm_ascend/compilation/acl_graph.py     — NPUGraph 封装
#   vllm_ascend/worker/model_runner_v1.py    — 分档和执行调度
```

核心思路：KV-cache 预分配到最大长度 → batch 维度分档 → `graph_task_update` 在 replay 前更新动态参数（seq_len, block_table）→ 一张图复用于所有 decode 步 → O(n)。

**策略 B：KV-cache 手动管理 + torch_npu NPUGraph 精确分档（graph_task_update 不可用时）**

适用条件：模型权重为 FP32，NPU 融合注意力不可用；或 graph_task_update 在当前 CANN 版本不支持。

```
每 decode 步捕获一张 NPUGraph → 精确匹配当前 seq_len (无 padding)
→ KV 在图外预分配 buffer 中管理 → N 张图覆盖 N 个 decode 步
→ 适用于短序列场景 (<50 tokens)
```

> NPUGraph 详细说明与 API 参见 [torchair参考子Skill](../migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)（§七）。

**KV-cache 关键约束（通用，与策略无关）：**

prefill 和 decode 步写入 KV 的函数和输入形式必须一致。否则 cache 值无法在 decode 步正确使用。

验证方法：
```python
# prefill 后验证 cache
k_cached = kv_cache[key_module]              # hook 存的
k_recomputed = key_module(same_input)         # 用 prefill 时的输入重算
assert (k_cached - k_recomputed).abs().max() < 1e-5, "KV cache mismatch"
```

常见陷阱：
- **Q 和 K/V 使用不同输入**：如 attention 内部 `q = query(ln(x))` 但 `k = key(x)`（Pre-LN 模型通常是相同的）
- **prefill 后二次调用 decoder**：hook 会 `torch.cat` 拼接新旧 KV，导致长度翻倍
- **NPUGraph 内不能创建 tensor**：token 和 KV buffer 必须图外预分配。NPUGraph 仅支持 aclnn 算子，参见 [torchair参考子Skill](../migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)（§七）

**注意力实现切换的提示：**

模型原有注意力实现（如 PyTorch SDPA、手动 matmul+softmax）替换为 NPU 融合注意力时：
- 遇到不熟悉的 API，先查阅昇腾文档：https://www.hiascend.com/document/
- `actual_seq_lengths` 参数表示 Q token 的累计长度，`actual_seq_lengths_kv` 表示 KV token 的累计长度——单 token decode 时两者值不同
- workspace 需通过 `_npu_fused_infer_attention_score_get_max_workspace` 预申请，按参数缓存避免 OOM
- 图模式必须使用 `.out()` 变体

#### 4.5 故障决策树

```
错误类型？
├── "No converter for aten::XXX" → 等价替换 → fallback → 自定义 converter → 标记跳过
├── CPU→NPU memcpy / stream capture 失败 → split-compile 策略（详见 §3.1）
├── shape/dtype 不匹配 → dynamic=True / dynamic gears / 消除 None 返回
├── 编译超时 → 缩小单元 / dynamic=True / 换 npugraph_ex
├── 精度超标 → data dump 逐层定位 → 该层保持 eager
├── 性能无提升/下降 → 确认瓶颈类型（下发/计算/访存）→ 切换编译模式重试 → 换 CANN 版本（≥2 个）
└── GE 编译错误（InferShape/EZ9999）→ CANN 版本问题 → 必须换版本重试
```

> **⚠ 每次切换版本后必须完整重跑：eager 基线 → torchair 编译 → 精度 → 性能。**

#### 4.6 性能测量

**进入性能测量前必须执行真实模型自检：**

```
□ 当前编译的模型是否为代码仓指定的完整模型（非简化版、非子模块独立构造）？
□ 是否已加载代码仓提供的真实权重（非随机初始化）？
□ 推理接口是否与步骤3基线接口完全一致？
□ 如任一为"否" → 立即停止当前测量流程，切换为代码仓指定的真实模型，重新执行步骤4
```

标准化 benchmark（必须同时测端到端和入图部分）：

```python
import torch, time, numpy as np

WARMUP, MEASURE = 3, 5
device = torch.device("npu:0")

# ===== 1. 入图部分性能（编译的子模块 forward）=====
for _ in range(WARMUP):
    with torch.no_grad():
        _ = compiled_module(test_input)
    torch.npu.synchronize()

times = []
for _ in range(MEASURE):
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = compiled_module(test_input)
    torch.npu.synchronize()
    times.append((time.time() - t0) * 1000)

times.sort()
print(f"入图部分: median={times[2]:.2f}ms, std={np.std(times):.2f}ms")

# ===== 2. 端到端全流程性能（必须，用户实际调用的接口）=====
# 例如 whisper: model.transcribe(audio_path)
# 例如 Paraformer: model.generate(input=audio)
for _ in range(WARMUP):
    result = model.full_pipeline(test_input)
    torch.npu.synchronize()

e2e_times = []
for _ in range(MEASURE):
    torch.npu.synchronize()
    t0 = time.time()
    result = model.full_pipeline(test_input)
    torch.npu.synchronize()
    e2e_times.append((time.time() - t0) * 1000)

e2e_times.sort()
print(f"端到端全流程: median={e2e_times[2]:.2f}ms, std={np.std(e2e_times):.2f}ms")
```

#### 4.7 优化雷达

跑通后扫描所有优化维度，参见主 Skill 步骤4.4。

---

## 二、NPU 覆盖率检查（步骤3详细操作）

**在尝试 torchair 之前，必须确认模型推理的热路径上所有主要计算都已在 NPU 上执行。**

> 概念原理与优化优先级参见 [torch_npu基础子Skill](../migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)（§六）。以下为可执行的检查代码模板。

```python
def check_npu_coverage(model, example_input):
    """检查模型的 NPU 覆盖率，输出未在 NPU 上的参数和可能的 CPU tensor 创建"""

    # 1. 遍历参数和 buffer，确认全部在 NPU
    for name, p in model.named_parameters():
        if p.device.type != 'npu':
            print(f"WARNING: Parameter '{name}' on {p.device}")
    for name, b in model.named_buffers():
        if b.device.type != 'npu':
            print(f"WARNING: Buffer '{name}' on {b.device}")

    # 2. 确认输入在 NPU
    assert example_input.device.type == 'npu', "Input tensor must be on NPU"

    # 3. 使用 profiler 追踪 CPU→NPU 数据传输
    #    以下为精简示例，实际场景可使用 torch.profiler.profile() 获取详细 trace
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        with torch.no_grad():
            _ = model(example_input)
    cpu_ops = [e for e in prof.function_events if 'cpu' in str(e.device_type).lower()]
    if cpu_ops:
        print(f"WARNING: {len(cpu_ops)} operations detected on CPU")
        for op in cpu_ops[:5]:  # 展示前5个
            print(f"  - {op.name} ({op.cpu_time_total:.3f}ms)")

    print("NPU coverage check completed.")
```

---

## 三、常见问题与解决方案

### 3.1 Split-Compile 策略

**当部分算子或模块无法入图编译时，可使用 split-compile 策略将模型拆分为编译区域和 eager 区域。**

核心思想：将模型拆分为"可编译的纯计算部分"和"必须保持 eager 的控制流/特殊操作部分"，两者之间通过 eager 执行桥接。

**典型场景：**
- CPU→NPU memcpy 或 stream capture 失败（`aclrtMemcpy error 107030`）→ 产生 CPU tensor 的操作移到编译区域外
- 动态控制流（`ValuePack`、`Tensor.item()`、`if seq_len >= N`）→ 包含控制流的模块保持 eager
- 特定算子无 GE converter → 该算子所在子模块保持 eager

**实现模板：**
```python
class CompiledCore(torch.nn.Module):
    """将可编译的纯计算层封装，便于整体编译"""
    def __init__(self, model):
        super().__init__()
        self.layer1 = model.layer1  # 可编译层
        self.layer2 = model.layer2  # 可编译层

    def forward(self, x, mask):
        x = self.layer1(x, mask)
        x = self.layer2(x, mask)
        return x, mask

core = CompiledCore(model)
compiled_core = torch.compile(core, backend=npu_backend, dynamic=False)

class FastPipeline(torch.nn.Module):
    """完整推理管线：eager 前后处理 + 编译核心"""
    def __init__(self, compiled_core, model):
        super().__init__()
        self.compiled_core = compiled_core
        self.after_norm = model.after_norm

    def forward(self, input):
        # Step 1: eager — 准备输入（避免 CPU→NPU memcpy）
        mask = compute_mask(input).to(input.device)

        # Step 2: torchair 加速 — 纯计算部分
        output, _ = self.compiled_core(input, mask)

        # Step 3: eager — 后处理
        output = self.after_norm(output)
        return output
```

### 3.2 精度调试：逐层对比法

**问题**：torchair 推理输出与 torch_npu eager 输出精度差异较大

**解决步骤**：
1. 启用 data dump 功能：`config.debug.data_dump.enabled = True`
2. 对比每一层的中间输出，定位首次出现精度偏差的算子
3. 检查该算子在 torchair 中的 converter 实现是否正确
4. 尝试调整编译选项（如关闭特定融合）：
   ```python
   config.fusion_config.fusion_switch_file = "./fusion_switch.json"
    # fusion_switch.json 格式：{"关闭的融合名": "off"}
    ```

### 3.3 动态 shape 问题

**问题**：`torch.compile(dynamic=False)` 要求输入 shape 固定，但实际推理中 shape 可能变化。

**解决**：
- 如果 shape 确实固定 → 保持 `dynamic=False`
- 如果 shape 可变 → 设置 `dynamic=True`（需 torchair 版本支持）
- 如果 shape 变化有规律（如只有几个固定分辨率）→ 配置 dynamic gears：
  ```python
  from torchair.inference._gear_utils import set_dim_gears
  set_dim_gears(input_tensor, [[1, 3, 224, 224], [1, 3, 384, 384]])
  ```

### 3.4 编译输出 shape 不匹配

**问题**：GE graph 的 NetOutput shape 与 FX graph 不一致（如 `[0]` vs `[0,0,0,0]`）

**根因**：模型中 None 返回值或 hook 副作用被编译图捕获

**解决**：
1. 检查模型代码中是否有返回 `None` 的分支（如某些模型的 attention 在 SDPA 模式下 `qk=None`）
2. 检查是否有 `register_forward_hook` 修改了输出
3. 修改模型代码消除 None 返回或 hook 副作用
4. 如无法消除，编译更小的子模块（跳过该层）

### 3.5 首次编译耗时长

**问题**：首次运行 `compiled_model(input)` 编译时间过长

**说明**：torch.compile 是 JIT（Just-In-Time）编译，首次编译包含图捕获、优化、GE 编译等过程，耗时较长是正常现象。

**优化**：
- 渐进编译：先编译最小单元验证，成功后再扩展，避免全模型一次编译超时
- 使用 warmup：预热一次后再计时
- 对于 npugraph_ex 模式，torchair 支持编译缓存（cache_compile）
- 对于 max-autotune 模式，首次编译耗时长，但运行时性能最优

### 3.6 运行时 OOM

**问题**：torchair 编译后运行 OOM（显存不足）

**解决**：
- 检查是否因图模式导致显存复用策略变化
- 尝试减小 batch size
- 检查是否有不必要的 tensor 拷贝

### 3.7 多卡推理

torchair 支持多卡推理，通过 `patch_for_hcom()` 启用集合通信入图：

```python
import torchair
torchair.patch_for_hcom()
# 后续正常使用 torch.distributed 进行多卡通信即可
```

---

## 四、CompilerConfig 常用配置

```python
config = CompilerConfig()

# --- 运行模式 ---
config.mode = "max-autotune"  # GE Graph Engine，算子融合+tiling，减少下发次数，适合下发瓶颈/计算瓶颈场景
# config.mode = "npugraph_ex"  # ACL Graph，Capture&Replay 消除重复下发，适合下发瓶颈/重复推理场景
# config.mode = "reduce-overhead"  # 已废弃，请使用 npugraph_ex

# --- 调试配置 ---
config.debug.graph_dump.enabled = True     # 导出编译后的 GE 图（.pbtxt 格式）
config.debug.graph_dump.path = "./torchair_dump_graph"
config.debug.fx_summary.enabled = True     # 导出 FX 图摘要（含算子数量统计）
config.debug.fx_summary.full_path = "./torchair_fx_summary.csv"
config.debug.data_dump.enabled = True      # 导出中间层精度数据，用于定位精度问题
config.debug.data_dump.path = "./torchair_dump_data"
config.debug.run_eagerly = True            # debug 模式：跳过编译，回退到 eager 执行

# --- 图导出配置 ---
config.export.export_mode = True           # 导出 air 格式图文件

# --- 融合配置 ---
config.fusion_config.fusion_switch_file = "./fusion_switch.json"  # 自定义融合策略文件

# --- 实验性配置 ---
config.experimental_config.pattern_fusion_pass = True  # 启用 pattern 级融合
config.experimental_config.enable_view_optimize = True # 启用 view 算子优化

# --- 推理配置 ---
config.inference_config.use_internal_format_weight = False  # 是否使用 NPU 私有格式存储权重
```

---

## 五、算子支持度验证

### 5.1 FX graph 预扫描（推荐，编译前执行）

在真正编译前，用 FX graph 导出 + 模式匹配提前发现不兼容算子，详见 §4.1(c) 的 `scan_unsupported_ops()` 脚本。几秒即可完成，不需真正编译。

### 5.2 在 torchair 代码仓中查找 converter（手动确认）

```bash
# 查找某个算子是否有 converter 实现
ls torchair/python/torchair/_ge_concrete_graph/ge_converter/aten/ | grep -i "算子名"
```

### 5.3 处理不支持的算子

遇到不支持的算子时，必须按问题穷尽原则处理（列出 ≥3 种替代方案 → 逐一试验）：

1. **等价算子替换**：查找 torchair 支持的等价算子（如 `aten.roll` → `cat+slice`, `__ior__` → `__or__`）
2. **API 降级**：FlashAttention → math SDPA，SDPA → manual matmul
3. **模块级跳过**：编译能编译的部分，其余保持 eager（不影响整体图编译）
4. **torch.compile 的 fallback 机制**：部分不支持的算子会自动 fallback 到 eager 执行
5. **自定义 converter**（最后手段）：
   ```python
   from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter

   @register_fx_node_ge_converter(torch.ops.aten.your_op.default)
   def your_converter(converter, node, inputs):
       pass
   ```
6. **标记为不支持**：全部方案失败后，在报告中说明该算子的所有已尝试方案及失败原因

---

## 六、迁移脚本模板（可复用）

```python
"""
torchair 迁移脚本模板
注意事项：
1. 优先使用 torch_npu 内置 torchair
2. 编译策略按优先级递进：全模型 → 子模型 → 子模块，找到最大可编译范围
3. 每步编译后立即验证精度
4. 标准化 benchmark：warmup=3, 测量=5, 取中位数
5. 必须同时测量端到端和入图部分
"""
import torch
import time
import numpy as np

# ========== 1. 环境配置 ==========
use_npu = torch.npu.is_available()
assert use_npu, "NPU not available, cannot proceed"
device = torch.device("npu:0")

# ========== 2. torchair 导入 ==========
try:
    from torch_npu.dynamo.torchair import get_npu_backend, CompilerConfig
except ImportError:
    import torchair
    get_npu_backend = torchair.get_npu_backend
    CompilerConfig = torchair.CompilerConfig

# ========== 3. torchair 配置 ==========
config = CompilerConfig()
config.mode = "max-autotune"  # 根据瓶颈类型选择：下发瓶颈 → max-autotune 或 npugraph_ex；计算瓶颈 → max-autotune
npu_backend = get_npu_backend(compiler_config=config)

# ========== 4. 加载模型 ==========
model = YourModel().to(device)
model.eval()

# ========== 5. 渐进编译（从最小单元开始）==========
# 第1步：编译最小单元
model.smallest_block = torch.compile(model.smallest_block, backend=npu_backend, dynamic=False)

# 验证 + 精度检查
test_input = torch.randn(B, C, H, W, device=device)
with torch.no_grad():
    eager_out = original_block(test_input)
    graph_out = model.smallest_block(test_input)
max_diff = torch.max(torch.abs(eager_out - graph_out) / (torch.abs(eager_out) + 1e-8))
assert max_diff < 1e-2, f"精度超标: {max_diff.item()}"
print(f"[Step 1] Smallest block OK, max_diff={max_diff.item():.6f}")

# 第2步：逐模块扩展（示例：编译 encoder）
model.encoder = torch.compile(model.encoder, backend=npu_backend, dynamic=False)
# ... 重复精度验证 ...

# 第3步：全模型（如前面都通过）

# ========== 6. 性能测试（标准化，必须同时测端到端和入图部分）==========
WARMUP = 3
MEASURE = 5

# ---------- 6a. 入图部分性能（编译的子模块 forward）----------
for _ in range(WARMUP):
    with torch.no_grad():
        _ = compiled_model(test_input)
    torch.npu.synchronize()

times = []
for _ in range(MEASURE):
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = compiled_model(test_input)
    torch.npu.synchronize()
    times.append((time.time() - t0) * 1000)

times.sort()
median_ms = times[len(times) // 2]
std_ms = np.std(times)
print(f"入图部分: {median_ms:.2f} ms (±{std_ms:.2f})")

# ---------- 6b. 端到端全流程性能（必须，用户实际调用的接口）----------
# 例如 whisper: model.transcribe(audio_path)
# 例如 Paraformer: model.generate(input=audio)
for _ in range(WARMUP):
    result = model.full_pipeline(test_audio)
    torch.npu.synchronize()

e2e_times = []
for _ in range(MEASURE):
    torch.npu.synchronize()
    t0 = time.time()
    result = model.full_pipeline(test_audio)
    torch.npu.synchronize()
    e2e_times.append((time.time() - t0) * 1000)

e2e_times.sort()
e2e_median_ms = e2e_times[len(e2e_times) // 2]
e2e_std_ms = np.std(e2e_times)
print(f"端到端全流程: {e2e_median_ms:.2f} ms (±{e2e_std_ms:.2f})")

# ========== 7. 精度验证（最终确认）==========
with torch.no_grad():
    eager_output = original_model(test_input)
    graph_output = compiled_model(test_input)
max_diff = torch.max(torch.abs(eager_output - graph_output) / (torch.abs(eager_output) + 1e-8))
print(f"Final max relative error: {max_diff.item():.8f}")
assert max_diff < 1e-2, f"最终精度不达标: {max_diff.item()}"
```
