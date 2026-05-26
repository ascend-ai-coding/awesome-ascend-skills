---
name: external-cannbot-model-model-infer-graph-mode
description: 基于 PyTorch 框架的昇腾 NPU 模型推理图模式适配技能。将模型适配到 torch.compile 图模式以加速推理性能。触发场景：npugraph_ex
  或 GE 图模式适配、torch.compile 在昇腾 NPU 上的使用、图中断（Graph Break）修复、aclgraph 图编译问题。LLM 模型的图模式适配优先阅读
  LLM 模型改造指南。
original-name: model-infer-graph-mode
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# 图模式适配优化技能

提供 npugraph_ex 和 GE 两种图模式的适配指南，覆盖方案设计、图中断修复、FA 参数配置和验证测试。

---

## 重要原则

- **前置条件**：模型必须在 NPU 上运行且已导入 `torch_npu`
- **图模式仅适用于 Decode 阶段**：Prefill 阶段输入长度动态变化，不适合图模式，保持 eager 执行
- **保持模型完整性**：不为适配图模式而简化模型逻辑
- **NPU 不支持的后端**：`aot_eager`、`inductor`、`cudagraphs` 等不可用
- **固定 tensor 图外预创建**：atten_mask、KV cache 等推理全程不变的 tensor 在图外预创建，用 `torch._dynamo.mark_static()` 标记
- **LLM Decode 重编译注意**：`kv_len`/`actual_seq_lengths_kv` 每步变化，npugraph_ex 可能触发重编译，遇到 `hit recompile_limit` 参考重编译解决方案排查
- **精度问题调试**：遇到精度问题可调用 `model-infer-precision-debug` skill 进行排查

---

## 工作流程

### 第一步：方案设计

1. **分析模型结构**：识别模型中可能阻碍图模式的代码模式
2. **识别问题点**：找出 Graph Break、重编译、动态 shape 等问题
3. **设计改造方案**：
   - 需要修改哪些文件
   - 修改的具体内容
   - 对现有功能的影响评估
4. **输出设计方案文档**：以 Markdown 格式呈现

### 第二步：方案确认

等待用户确认方案后再进行开发。有修改意见则返回第一步。

### 第三步：实施开发

按照确认的设计方案逐步实施代码修改。

### 第四步：验证测试

由 Agent 实际执行以下测试，记录真实结果：

- **编译验证**：运行 `torch.compile`，检查是否成功，记录编译日志
- **功能验证**：运行模型，对比图模式前后输出
- **性能验证**：记录 Prefill/Decode 阶段耗时
- **测试报告**：整理测试环境、测试用例、对比数据

---

## 图模式选择

在开始适配前先确定使用哪种模式。如果用户未指定，默认采用 npugraph_ex。

| 场景 | 推荐模式 | 详细文档 |
|------|---------|---------|
| **LLM 大语言模型** | npugraph_ex（优先阅读 LLM 指南） | `references/llm-model-guide.md` |
| **通用模型** | npugraph_ex | `references/npugraph_ex-guide.md` |
| **需要 GE 图模式** | GE（Ascend IR） | `references/ge-graph-guide.md` |

| 特性 | npugraph_ex 后端 (aclgraph) | GE 图模式 (Ascend IR) |
|------|----------------------------|----------------------|
| **启用方式** | `backend="npugraph_ex"` | `torchair.get_npu_backend()` |
| **实现原理** | 捕获模式 (Capture & Replay) | FX 图转换为 Ascend IR，GE 引擎编译执行 |
| **成熟度** | 试验特性，暂不支持商用 | 更成熟稳定 |
| **PyTorch版本** | 需要 2.6.0+ | 无特殊要求 |
| **支持场景** | 在线推理 | 通用场景 |
| **类似技术** | torch.cuda.CUDAGraph | 传统图编译 |

### npugraph_ex 快速示例

```python
import torch
import torch_npu

model = YourModel().to("npu")
opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
# 注：LLM Decode 场景 actual_seq_lengths 每步变化时需 dynamic=True
output = opt_model(input_tensor)
```

**关键约束**：PyTorch 2.6.0+、仅支持在线推理、不支持随机数算子和动态控制流、forward 中不可使用 `.item()`

### GE 图模式快速示例

```python
import torch
import torch_npu
import torchair
from torchair import patch_for_hcom

patch_for_hcom()  # 集合通信入图（有 TP/EP 并行时需调用）

model = YourModel().to("npu")
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
output = opt_model(input_tensor)
```

> 详细文档：npugraph_ex 见 `references/npugraph_ex-guide.md`，GE 见 `references/ge-graph-guide.md`

---

## LLM 模型适配要点

> 对于 LLM 推理模型，**必须严格区分 prefill 和 decode 阶段**。

### Prefill 与 Decode 阶段限制

| 阶段 | 是否支持图模式 | 原因 |
|------|---------------|------|
| **Prefill** | **禁止使用** | 输入长度动态变化、首 token 生成逻辑复杂、shape 不固定 |
| **Decode** | **推荐使用** | 输入长度固定（通常为 1）、shape 稳定、适合图捕获 |

**实现建议**：
- 将 prefill 和 decode 的 forward 逻辑分离成不同方法
- 仅对 decode 方法应用 `torch.compile` 图模式
- prefill 阶段使用 eager 模式执行

```python
class YourModel:
    def prefill(self, input_ids, ...):
        """Prefill 阶段：使用 eager 模式"""
        # 输入长度动态变化，不适合图模式
        return self._forward(input_ids, ...)

    def decode(self, input_ids, ...):
        """Decode 阶段：可使用图模式"""
        # 输入长度固定（通常为 1），适合图捕获
        return self._forward(input_ids, ...)

# 仅对 decode 方法应用图模式, model.prefill保持 eager
model.decode = torch.compile(model.decode, backend="npugraph_ex", ...)
```

### 核心改造原则

**核心原则**：将动态变化的东西提取为模型输入，模型内部尽量保证静态。

| 动态因素 | 问题表现 | 解决思路 |
|---------|---------|---------|
| **内存地址变化** | Guard 失败、重编译 | 预分配固定大小，原地更新 |
| **Shape 变化** | 图中断、多次编译 | 固定 shape 或通过参数控制 |
| **Python 控制流** | Graph Break | 使用 Tensor 操作或模式参数 |
| **`.item()` 调用** | 强制 Graph Break | 保持 Tensor 或外部传入 |

> 详细改造指南：`references/llm-model-guide.md`

---

## 问题定界流程

> 问题定界优先基于本 skill 内置的知识进行独立分析。

```
问题发生
    │
    ├─→ aot_eager 验证 ──失败──→ 修复用户脚本
    │       ↓ 正常
    │
    ├─→ force_eager/run-eagerly ──失败──→ 修复用户脚本
    │       ↓ 正常
    │
    └─→ 图模式问题
            ├── 重编译问题 → 阅读 LLM 指南 + npugraph_ex 指南
            ├── Graph Break 问题 → 阅读 TorchAir 在线文档中的典型案例
            └── 其他问题 → 阅读对应模式文档（npugraph_ex-guide.md 或 ge-graph-guide.md）
```

**调试知识来源（按优先级）**：
1. 本 SKILL.md 中的方法、原则和检查清单
2. `references/npugraph_ex-guide.md` - npugraph_ex 详细指南
3. `references/llm-model-guide.md` - LLM 模型改造指南
4. `references/ge-graph-guide.md` - GE 图模式指南
5. [TorchAir 官方文档](https://gitcode.com/Ascend/torchair/tree/master/docs/zh)

**注意**：避免盲目复制其他模型的图模式配置，应基于当前模型结构独立分析。

---

## 图模式 + FA 融合算子快速 Debug

> 图模式与 FA 融合算子结合时，`actual_seq_lengths` 参数的处理是最常见的出错点。
> 本节提供快速诊断和解决方案。

### 问题现象

图模式 + FA 场景下的典型问题：
- 编译报错：`actual_seq_lengths` 类型不匹配
- 运行时报错：重编译（recompile）触发
- 性能问题：动态 shape 导致无法充分优化

### 关键参数：actual_seq_lengths

FA 算子的 `actual_seq_lengths` / `actual_seq_qlen` / `actual_seq_kvlen` 参数在不同图模式下有不同的要求：

| 图模式 | FA 接口来源 | actual_seq_lengths 类型 | dynamic 设置 | 执行模式约束 | 说明 |
|--------|------------|------------------------|-------------|-------------|------|
| **GE 模式（推荐）** | torchair FA 接口 | **Tensor** | `dynamic=False` | **仅支持 GE 图模式** | 最佳方案，静态图 |
| **GE 模式（不推荐）** | torch_npu FA 接口 | list[int] | `dynamic=True` + `mark_static` | 无限制 | 需额外配置，易出错 |
| **npugraph_ex 模式** | torch_npu FA 接口 | list[int] | `dynamic=True` | 无限制 | 动态捕获模式 |
| **npugraph_ex 模式** | torch_npu FA 接口 | Tensor（如有） | `dynamic=False` | 无限制 | 需确认接口是否支持 |

### GE 模式配置指南

#### 方案一：torchair FA 接口（推荐）

```python
import torch
import torch_npu
import torchair
from torchair.ge_concrete_graph.ge_graph import mark_static

# 使用 torchair 提供的 FA 接口
# actual_seq_lengths 为 Tensor 类型
attn_output = torchair.ops.npu_fused_infer_attention_score(
    query, key, value,
    actual_seq_qlen=actual_seq_qlen_tensor,   # Tensor 类型
    actual_seq_kvlen=actual_seq_kvlen_tensor, # Tensor 类型
    # ... 其他参数
)

# 编译配置
opt_model = torch.compile(model, backend=npu_backend, dynamic=False)
```

**优点**：
- `dynamic=False` 可获得更好的静态图优化
- 无需额外的 `mark_static` 配置
- 图编译更稳定

**约束**：
- TorchAir FA 接口**仅支持 GE 图模式**，不支持 Eager 模式和 npugraph_ex 模式调用
- 必须在 GE 图模式（`torchair.get_npu_backend()`）下使用，不可在 Eager 或 npugraph_ex 模式下调用

#### 方案二：torch_npu FA 接口（不推荐）

```python
import torch
import torch_npu
import torchair
from torchair.ge_concrete_graph.ge_graph import mark_static

# 使用 torch_npu 的 FA 接口
# actual_seq_lengths 为 list[int] 类型
attn_output = torch.ops.npu.npu_fused_infer_attention_score(
    query, key, value,
    actual_seq_lengths=[seq_len],  # list[int] 类型
    actual_seq_lengths_kv=[kv_len],
    # ... 其他参数
)

# 必须配置 dynamic=True
# 并使用 mark_static 标记除 actual_seq_lengths 外的静态输入
mark_static(input_ids)          # 静态输入
mark_static(position_ids)       # 静态输入
mark_static(attention_mask)     # 静态输入
# actual_seq_lengths 保持动态

# 编译配置
opt_model = torch.compile(model, backend=npu_backend, dynamic=True)
```

**缺点**：
- 需要配置 `dynamic=True`，性能略逊于静态图
- 需要手动调用 `mark_static` 标记所有静态输入
- 配置繁琐，易遗漏导致问题

### npugraph_ex 模式配置指南

#### 方案一：list[int] 类型 + dynamic=True

```python
import torch
import torch_npu

# 使用 torch_npu 的 FA 接口
# actual_seq_lengths 为 list[int] 类型
attn_output = torch.ops.npu.npu_fused_infer_attention_score(
    query, key, value,
    actual_seq_lengths=[seq_len],  # list[int] 类型
    actual_seq_lengths_kv=[kv_len],
    # ... 其他参数
)

# 必须配置 dynamic=True
opt_model = torch.compile(model, backend="npugraph_ex", dynamic=True)
```

#### 方案二：Tensor 类型 + dynamic=False（如有接口支持）

```python
import torch
import torch_npu

# 查询是否有 Tensor 类型的 actual_seq_lengths 接口
# 通过 subagent 调用 /model-infer-fusion 查询

# 如果有支持的接口：
attn_output = torch.ops.npu.npu_fused_infer_attention_score(
    query, key, value,
    actual_seq_lengths=actual_seq_lengths_tensor,  # Tensor 类型
    actual_seq_lengths_kv=actual_seq_kvlen_tensor,
    # ... 其他参数
)

# 可配置 dynamic=False
opt_model = torch.compile(model, backend="npugraph_ex", dynamic=False)
```

### FA 接口查询

如需查询具体的 FA 接口参数和版本支持，使用 subagent 调用 `/model-infer-fusion` 技能：

```
启动 subagent（类型：general-purpose），提示词包含：
1. 明确指示调用 /model-infer-fusion 技能
2. 询问具体问题，例如：
   - "查询 npu_fused_infer_attention_score 的 actual_seq_lengths 参数是否支持 Tensor 类型"
   - "查询 npu_fused_infer_attention_score_v2 在图模式下的推荐配置"
   - "查询 torchair 提供的 FA 接口及其参数要求"
3. 提供当前使用的图模式（GE / npugraph_ex）
```

### 常见错误与修复

| 错误现象 | 根因 | 修复方案 |
|---------|------|---------|
| 编译报错：actual_seq_lengths 类型错误 | GE 模式下 torch_npu FA 接口传 Tensor | 改用 torchair FA 接口，或改用 list[int] + dynamic=True |
| 运行时频繁重编译 | dynamic=False 但 actual_seq_lengths 为 list[int] | 改用 Tensor 类型 + torchair 接口，或设置 dynamic=True |
| 性能不达预期 | dynamic=True 导致无法充分优化 | 尽量使用 torchair FA 接口 + dynamic=False |
| npugraph_ex 模式报错 | actual_seq_lengths 为 Tensor 但接口不支持 | 确认接口支持情况，或改用 list[int] + dynamic=True |
| Eager 或 npugraph_ex 模式调用 TorchAir FA 报错 | TorchAir FA 接口仅支持 GE 图模式 | 改用 torch_npu FA 接口，或切换到 GE 图模式 |

### Debug 检查清单

在图模式 + FA 场景下，按以下清单逐一排查：

- [ ] 1. 确认使用的图模式：GE 还是 npugraph_ex
- [ ] 2. 确认 FA 接口来源：torch_npu 还是 torchair
- [ ] 3. 若使用 torchair FA 接口，确认当前为 GE 图模式（不支持 Eager 和 npugraph_ex）
- [ ] 4. 检查 actual_seq_lengths 类型：GE+torchair→Tensor；GE+torch_npu→list[int]+dynamic=True；npugraph_ex→list[int]+dynamic=True
- [ ] 5. 检查 dynamic 配置是否与 actual_seq_lengths 类型匹配
- [ ] 6. 若使用 GE + torch_npu FA + list[int]，检查是否已 mark_static 标记所有静态输入
- [ ] 7. 如有疑问，调用 `model-infer-fusion` skill 查询接口详情

---

## TorchAir 官方文档索引

> 在线文档：[gitcode.com/Ascend/torchair/docs/zh](https://gitcode.com/Ascend/torchair/tree/master/docs/zh)

### 入门必读

| 主题 | 在线链接 |
|-----|---------|
| 简介 | [overview.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/overview.md) |
| 支持的 ATen API 清单 | [aten_api.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/aten_api.md) |

### GE 图模式

| 主题 | 在线链接 |
|-----|---------|
| GE 图模式总览 | [ascend_ir.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/ascend_ir.md) |
| GE 图模式快速上手 | [quick_start.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/quick_start.md) |
| GE 图模式功能（基础） | [basic/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/features/basic) |
| GE 图模式功能（进阶） | [advanced/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/features/advanced) |
| GE 图模式 API 参考 | [api/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api) |

### 自定义算子入图

| 主题 | 在线链接 |
|-----|---------|
| 概述 | [overview.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/custom_op_graph/overview.md) |
| In-place 算子开发和入图样例 | [in_place_op_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/custom_op_graph/in_place_op_cases.md) |
| 非 In-place 算子开发和入图样例 | [non_in_place_op_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/custom_op_graph/non_in_place_op_cases.md) |
| 算子插件化适配 | [op_plugin_adapt_torchair.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/custom_op_graph/op_plugin_adapt_torchair.md) |

### 常见案例和定位方法

| 主题 | 在线链接 |
|-----|---------|
| 大模型图模式推理案例 | [infer_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases/infer_cases.md) |
| 动/静态图概念及典型问题定位 | [dynamic_static_graph/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases/dynamic_static_graph) |
| 入图失败定界与定位 | [graph_failed_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases/graph_failed_cases.md) |

### 性能分析与精度比对

| 主题 | 在线链接 |
|-----|---------|
| 图模式性能分析 | [perfermance_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases/perfermance_cases.md) |
| 图模式精度比对 | [accuracy_cases.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases/accuracy_cases.md) |
| FAQ | [faq.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/faq.md) |
