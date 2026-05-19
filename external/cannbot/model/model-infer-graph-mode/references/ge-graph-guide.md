# GE 图模式使用指南

> GE图模式通过 TorchAir 的 CompilerConfig 开启，将 FX 图转换为 Ascend IR 图，并通过 GE 图引擎实现图编译和执行。

---

## 适用场景

- **生产环境**：稳定性优先
- **通用场景**：功能丰富，支持广泛
- **复杂模型**：需要更多配置选项

---

## 与 npugraph_ex 对比

| 特性 | npugraph_ex 后端 | GE 图模式 |
|------|------------------|-----------|
| **启用方式** | `backend="npugraph_ex"` | `torchair.get_npu_backend()` |
| **实现原理** | 捕获模式 (Capture & Replay) | FX图转换为Ascend IR，GE引擎编译执行 |
| **默认模式** | 需显式指定 | 系统默认模式 `mode="max-autotune"` |
| **成熟度** | 试验特性，暂不支持商用 | 更成熟稳定 |
| **PyTorch版本** | 需要 2.6.0+ | 无特殊要求 |
| **支持场景** | 在线推理 | 通用场景 |
| **配置方式** | `options={}` 参数 | `CompilerConfig` 对象 |

---

## 快速上手

```python
# 导包（必须先导 torch_npu 再导 torchair）
import torch
import torch_npu
import torchair

# Patch方式实现集合通信入图（可选）
from torchair import patch_for_hcom
patch_for_hcom()  # 集合通信入图（有 TP/EP 并行时需调用）

model = YourModel().npu()

# 图执行模式默认为 max-autotune
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 基于 TorchAir backend 进行 compile
opt_model = torch.compile(model, backend=npu_backend)

# 执行编译后的 Model
output = opt_model(input_tensor)
```

> 完整示例见 [GE 图模式快速上手](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/quick_start.md)

---

## CompilerConfig 配置

```python
config = torchair.CompilerConfig()

# debug 类功能
config.debug.xxx = ...

# export 类功能（离线导图）
config.export.xxx = ...

# dump_config 类功能
config.dump_config.xxx = ...

# fusion_config 类功能
config.fusion_config.xxx = ...

# experimental_config 类功能
config.experimental_config.xxx = ...

# inference_config 类功能
config.inference_config.xxx = ...

# ge_config 类功能
config.ge_config.xxx = ...
```

---

## 核心 API

| API | 用途 | 文档路径 |
|-----|------|---------|
| `CompilerConfig类` | 配置图模式功能 | [compiler_config.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/compiler_config.md) |
| `get_npu_backend()` | 获取 NPU 后端 | [get_npu_backend.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/get_npu_backend.md) |
| `get_compiler()` | 获取编译器 | [get_compiler.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/get_compiler.md) |
| `dynamo_export()` | 导出模型 | [dynamo_export.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/dynamo_export.md) |
| `register_fx_node_ge_converter()` | 注册转换器 | [register_fx_node_ge_converter.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/register_fx_node_ge_converter.md) |
| `register_replacement()` | 自定义算子融合 | [register_replacement.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair/register_replacement-0.md) |

---

## 功能文档索引

> 在线文档：[gitcode.com/Ascend/torchair/docs/zh/ascend_ir](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir)

### 概述

| 功能 | 在线链接 |
|-----|---------|
| GE 图模式总览 | [ascend_ir.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/ascend_ir.md) |
| GE 图模式快速上手 | [quick_start.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/quick_start.md) |

### 基础功能

| 功能 | 在线链接 |
|-----|---------|
| 基础功能总览 | [basic/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/features/basic) |

### 高级功能

| 功能 | 在线链接 |
|-----|---------|
| 进阶功能总览 | [advanced/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/features/advanced) |

### API 参考

| API | 在线链接 |
|-----|---------|
| API 总览 | [api/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api) |
| API 列表 | [api_list.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/api_list.md) |
| torchair 核心 | [torchair/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/torchair) |
| torchair.ge | [ge/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/ge) |
| torchair.inference | [inference/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/inference) |
| torchair.ops | [ops/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/ops) |
| torchair.scope | [scope/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/api/scope) |

---

## 相关文档

- **LLM 模型改造指南**：`llm-model-guide.md`（LLM 适配优先阅读）
- **npugraph_ex 指南**：`npugraph_ex-guide.md`