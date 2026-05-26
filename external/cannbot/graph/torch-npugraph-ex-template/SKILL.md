---
name: external-cannbot-graph-torch-npugraph-ex-template
description: npugraph_ex 模式的 MRE（最小可复现示例）代码模板。包含标准 npugraph_ex 编译模板和 npugraph_ex 编译缓存（cache_compile）模板。触发：当用户需要从零生成
  npugraph_ex 模式代码、做概念解释、对比分析或配置指导时加载。
original-name: torch-npugraph-ex-template
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# npugraph_ex 模式 MRE 模板

从零生成 npugraph_ex 模式代码时，使用以下模板。所有模板遵循 agent 定义的 MRE 三段式结构（`import` → `def infer_one_step()` → `if __name__ == "__main__"`）和固定变量名（`config`、`compiled_model`、`dummy_input`、`output`）。

## MRE 模板 — npugraph_ex 模式

```python
import torch

class <模型名>(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # <模型层定义>

    def forward(self, x):
        # <前向逻辑>
        return x

def infer_one_step():
    """
    <功能描述（一句话）>

    关键配置：
    - backend="npugraph_ex" — Capture & Replay，任务下沉 Device，降低 Host 调度开销
    - fullgraph=True — 整图捕获，不允许断图
    - dynamic=False — 固定 shape（npugraph_ex底层调用的aclgraph不支持动态 shape，开启dynamic=True执行过程中可能会有多张aclgraph）
    - <其他核心配置项> — <作用说明>
    """
    model = <模型名>().npu().eval()
    compiled_model = torch.compile(
        model,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
    )
    dummy_input = torch.randn(<shape>).npu()
    output = compiled_model(dummy_input)

if __name__ == "__main__":
    infer_one_step()
```

## MRE 模板 — npugraph_ex 编译缓存（cache_compile）

```python
import torch

class <模型名>(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # <模型层定义>
        self.cached_forward = torch.npu.npugraph_ex.inference.cache_compile(
            self._forward,
            dynamic=False,
            # cache_dir=".torchair_cache",  # 可选，指定缓存路径
        )

    def forward(self, dummy_input):
        """
        <功能描述（一句话）>

        关键配置：
        - torch.npu.npugraph_ex.inference.cache_compile — 替代 torch.compile，缓存 Dynamo 编译结果
        - dynamic=False — 固定 shape（npugraph_ex底层调用的aclgraph不支持动态 shape，开启dynamic=True执行过程中可能会有多张aclgraph）
        - cache_dir — 缓存文件保存路径，默认 .torchair_cache
        - <其他核心配置项> — <作用说明>
        """
        return self.cached_forward(dummy_input)

    def _forward(self, dummy_input):
        # <实际推理逻辑>
        pass

def infer_one_step():
    model = <模型名>().npu().eval()
    dummy_input = torch.randn(<shape>).npu()
    output = model(dummy_input)

if __name__ == "__main__":
    infer_one_step()
```

## 自定义算子代码块（按需嵌入 MRE 或迁移脚本）

当用户需求涉及自定义算子 + 图模式时，将下方对应的代码块嵌入代码中：从零生成代码时插入 MRE 的 `import` 区域之后、`def infer_one_step()` 之前；脚本迁移时在用户脚本的适当位置插入。模型内通过 `torch.ops.<namespace>.<op_name>(x)` 调用该算子。算子实现体用 `# <用户实现>` 占位，不替用户编写算子内部逻辑。**生成前须先读取 `custom_op_graph/non_in_place_op_cases.md` 或 `in_place_op_cases.md` 验证写法。**

### Out-of-place 算子注册骨架

结果写入新 tensor，不修改输入：

```python
@torch.library.custom_op("mylib::my_op", mutates_args=(), device_types="npu")
def my_op(x: torch.Tensor) -> torch.Tensor:
    # <用户实现>
    ...

@my_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### In-place 算子注册骨架

原地修改输入，如 kv_cache 更新：

```python
@torch.library.custom_op("mylib::my_op_", mutates_args=("x",), device_types="npu")
def my_op_(x: torch.Tensor) -> None:
    # <用户实现：原地修改 x>
    ...

@my_op_.register_fake
def _(x: torch.Tensor) -> None:
    return None
```

> 关键点：`mutates_args` 必须准确声明被修改的参数名；`device_types="npu"` 在 NPU 场景下必须指定；`register_fake`（Meta 推导函数）负责告知编译器输出 tensor 的 shape/dtype/device，不执行实际计算。

## 自定义算子代码块（op-plugin 注册风格）

对齐 torchair 官方文档（`docs/zh/custom_op_graph/non_in_place_op_cases.md` / `in_place_op_cases.md`）的生产级注册路径：算子由 Ascend C 工程化方式实现，通过 op-plugin yaml 完成 PyTorch 算子原型注册和 Eager 适配，Python 侧用 `torch.library.Library` + `@impl` 注册 Meta 推导函数。调用方式为 `torch.ops.npu.<op_name>(...)`。

> 前置依赖（本模板不覆盖，需用户自行完成）：
> 1. 用 msOpGen 创建 Ascend C 算子工程，编译并部署 `custom_opp_<os>_<arch>.run`
> 2. 在 `third_party/op-plugin/op_plugin/config/op_plugin_functions.yaml` 的 `custom:` 字段下添加算子 schema，并配置 `gen_opapi`（`out.size` / `out.dtype` / `exec: aclnnXxx`）
> 3. 重新编译安装 torch_npu（`bash ci/build.sh --python=<ver>`）
>
> 本模板只生成 Python 侧的 Meta 注册骨架（生产建议落在 `op_plugin/python/meta/_meta_registrations.py`，此处以 inline 形式给出便于自洽 MRE）。

### Out-of-place 算子注册骨架（op-plugin 风格）

```python
import torch
from torch.library import Library, impl

# 对应 op_plugin_functions.yaml 中的 schema：
# custom:
#   - func: my_op(Tensor x, Tensor? y, Tensor[] z, float attr1, int attr2) -> Tensor
#     op_api: all_version
#     gen_opapi:
#       out:
#         size: x
#         dtype: x
#       exec: aclnnMyOp

m = Library("npu", "IMPL", "Meta")

@impl(m, "my_op")
def my_op_meta(x, y, z, attr1, attr2):
    # <根据算子语义推导输出 shape/dtype>
    return torch.empty_like(x)

# 调用方式：torch.ops.npu.my_op(x, y, z, attr1, attr2)
```

### In-place 算子注册骨架（op-plugin 风格）

```python
import torch
from torch.library import Library, impl

# 对应 op_plugin_functions.yaml 中的 schema：
# custom:
#   - func: my_inplace(Tensor(a!) x, Tensor y) -> Tensor
#     op_api: all_version
#     gen_opapi:
#       out:
#         size: y
#         dtype: y
#       exec: aclnnMyInplace

m = Library("npu", "IMPL", "Meta")

@impl(m, "my_inplace")
def my_inplace_meta(x, y):
    # 输出 shape/dtype 与 y 相同；不返回被修改的输入 x
    return torch.empty_like(y)

# 调用方式：torch.ops.npu.my_inplace(x, y)
```

> 关键点：
> - In-place 算子 schema 中被修改的输入必须用 `Tensor(a!)` 标注，且**不返回**任何被修改的输入（PyTorch 2.6+ 自动函数化的硬性约束）
> - Meta 推导函数必须在 `torch.compile` 执行前完成注册
> - npugraph_ex 模式下完成 Meta 注册即可入图，无需额外 Converter（aclgraph 直接 capture & replay Eager 算子调用，不走 Ascend IR）
> - 函数化转换由 TorchAir 自动完成，无需手动实现
