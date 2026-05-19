# npugraph_ex 后端使用指南

> npugraph_ex 是基于 torch.compile 的 aclgraph 图模式加速方案，通过捕获模式 (Capture & Replay) 构建模型运行实例。

---

## 适用场景

- **在线推理场景**：追求简单快速适配
- **熟悉 CUDAGraph 模式**：使用习惯类似
- **LLM Decode 阶段**：固定 shape 的单 token 输入

---

## 使用约束

| 约束项 | 说明 |
|-------|------|
| PyTorch版本 | 需要 2.6.0 及以上版本 |
| 支持场景 | 在线推理场景，不支持反向流程 capture |
| 随机数算子 | 不支持 capture（randn、dropout 等） |
| 动态控制流 | 不支持，需保证图静态 |
| Stream同步 | 不支持 stream sync 操作 |
| 成熟度 | 试验特性，暂不支持商用产品 |

---

## 快速上手

```python
import torch
import torch_npu

model = YourModel().npu()
opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
output = opt_model(input_tensor)
```

> 详细说明见 [npugraph_ex 快速上手](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/quick_start.md)

---

## 问题定界流程

```
问题发生
    │
    ├─→ aot_eager 验证 ──失败──→ 修复用户脚本
    │       ↓ 正常
    │
    ├─→ force_eager=True ──失败──→ 修复用户脚本
    │       ↓ 正常
    │
    └─→ npugraph_ex/FX图问题
```

---

## options 配置速查

```python
opt_model = torch.compile(
    model, backend="npugraph_ex", fullgraph=True,
    options={
        # ========== 调试 ==========
        "force_eager": False,          # 强制 eager 模式调试

        # ========== FX图优化 ==========
        "inplace_pass": True,          # 原地操作优化
        "input_inplace_pass": True,    # 输入原地优化
        "pattern_fusion_pass": True,   # 算子融合

        # ========== 内存优化 ==========
        "reuse_graph_pool_in_same_fx": True,  # 图池复用
        "clone_input": True,           # 克隆输入
        "clone_output": False,         # 克隆输出
        "use_graph_pool": None,        # 图池配置

        # ========== 性能优化 ==========
        "static_kernel_compile": False,  # 静态Kernel编译
        "remove_noop_ops": True,         # 移除空操作
        "frozen_parameter": False,       # 冻结参数

        # ========== 捕获控制 ==========
        "capture_limit": 64,             # 重捕获次数限制
    }
)
```

---

## 核心 API

| API | 用途 | 文档路径 |
|-----|------|---------|
| `compile_fx()` | 自定义 backend | [compile_fx.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/npugraph_ex/compile_fx.md) |
| `register_replacement()` | 自定义算子融合 | [register_replacement.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/npugraph_ex/register_replacement.md) |
| `cache_compile()` | 编译缓存 | [cache_compile.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/inference/cache_compile.md) |
| `limit_core_num()` | 限核功能 | [limit_core_num.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/scope/limit_core_num.md) |

---

## 功能文档索引

> 在线文档：[gitcode.com/Ascend/torchair/docs/zh/npugraph_ex](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex)

### 基础功能

| 功能 | 在线链接 |
|-----|---------|
| npugraph_ex 后端概述 | [npugraph_ex.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/npugraph_ex.md) |
| 快速上手 | [quick_start.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/quick_start.md) |
| 基础功能总览 | [basic/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic) |
| force_eager | [force_eager.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/force_eager.md) |
| FX图优化Pass配置 | [inplace_pass.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/inplace_pass.md) |
| 内存复用 | [memory_reuse.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/memory_reuse.md) |
| 静态Kernel编译 | [static_kernel_compile.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md) |
| 冗余算子消除 | [remove_noop_ops.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/remove_noop_ops.md) |
| 重捕获次数限制 | [capture_limit.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/capture_limit.md) |
| 集合通信入图 | [communication_graph.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/basic/communication_graph.md) |

### 进阶功能

| 功能 | 在线链接 |
|-----|---------|
| 进阶功能总览 | [advanced/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/advanced) |
| 模型编译缓存 | [compile_cache.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/advanced/compile_cache.md) |
| 多流表达功能 | [multi_stream.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/advanced/multi_stream.md) |
| AI Core/Vector Core 限核 | [limit_cores.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/advanced/limit_cores.md) |
| 自定义FX图优化Pass | [post_grad_custom_pass.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/advanced/post_grad_custom_pass.md) |

### DFX 功能

| 功能 | 在线链接 |
|-----|---------|
| DFX 总览 | [dfx/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/dfx) |
| 图编译Debug信息保存 | [debug_save.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/dfx/debug_save.md) |
| 算子Data Dump | [data_dump.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/dfx/data_dump.md) |

### API 参考

| API | 在线链接 |
|-----|---------|
| API 总览 | [api/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api) |
| API 列表 | [api_list.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/api_list.md) |
| torch.npu.npugraph_ex | [torch-npu-npugraph_ex.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/npugraph_ex/torch-npu-npugraph_ex.md) |
| compile_fx | [compile_fx.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/npugraph_ex/compile_fx.md) |
| register_replacement | [register_replacement.md](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/npugraph_ex/register_replacement.md) |
| inference (cache_compile, readable_cache) | [inference/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/inference) |
| scope (limit_core_num) | [scope/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/api/scope) |

---

## 常见问题

### 1. 如何判断是否应该使用 npugraph_ex？

- **适合**：LLM decode 阶段、固定 shape 推理、简单快速适配
- **不适合**：需要动态 shape、生产环境稳定性优先、训练场景

### 2. 报错 "不支持 capture" 怎么办？

检查代码中是否包含：
- 随机数算子（randn、dropout）
- 动态控制流（if/while 基于 tensor 值）
- `.item()` 调用

### 3. 性能劣化怎么办？

1. 开启重编译日志：`torch._logging.set_logs(recompiles=True)`
2. 检查是否发生重编译
3. 参考 `llm-model-guide.md` 中的改造指南

---

## 相关文档

- **LLM 模型改造指南**：`llm-model-guide.md`（LLM 适配优先阅读）
- **GE 图模式指南**：`ge-graph-guide.md`