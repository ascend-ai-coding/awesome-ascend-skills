---
name: external-cannbot-graph-torch-custom-ops-guide
description: 自定义算子入图完整指南。覆盖从零开发、Eager 算子适配 npugraph_ex 图模式（torch.library.custom_op
  / torch.library.Library）、Meta 推导函数编写等全流程。适用于两种纯 Python 自定义算子注册场景。关键词：custom_op、torch.library.Library、register_fake、meta、mutates_args。
original-name: torch-custom-ops-guide
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# 自定义算子入图

## 算子状态确认

若用户未说明算子状态，使用交互式提问工具发起问题：`您的算子处于什么状态？`

路由：
- 还没开发，从零开始 → 按 npugraph_ex 代码生成场景生成包含算子注册骨架（参考 `torch-npugraph-ex-template` skill 中的「自定义算子代码块」）+ 图模式编译调用的完整 MRE，算子实现用 `# <用户实现>` 占位。In-place 与 Out-of-place 的区别在 `custom_op` 路径上体现在 `mutates_args` 与 `register_fake` 返回值，在 `torch.library.Library` 路径上体现在 schema 的 alias 标注和 Meta 返回值；函数化转换由 TorchAir 自动完成，无需手动实现
- 已经能在 Eager 模式下运行，需要适配图模式 → 进入已有 Eager 算子适配流程
- 已经入图但遇到问题 → 转到问题定位（加载 `torch-npugraph-ex-dfx-triage` skill）

---

## 已有 Eager 算子适配图模式

### 确认算子注册方式

若用户未说明注册方式，使用交互式提问工具发起问题：`您的算子是通过哪种方式注册的？`

路由：
- `torch.library.custom_op`（Python 层，在用户脚本中注册）→ `register_fake` 写在**用户脚本内**（与算子定义同文件）。参考 `docs/zh/custom_op_graph/op_adapt_torchair.md`，以及 `non_in_place_op_cases.md` / `in_place_op_cases.md`（取决于算子类型）
- 纯 Python `torch.library.Library`（Python 层，在用户脚本中通过 `Library(..., "FRAGMENT")` + `define(...)` + `@impl(...)` 注册 schema 和 Eager 实现，再通过 `Library(..., "IMPL", "Meta")` 注册 Meta）→ schema、Eager 实现和 Meta 注册都写在**用户脚本内**；Eager 实现通常使用 `@impl(mylib, "op_name", "PrivateUse1")`，Meta 使用 `meta_lib = Library("mylib", "IMPL", "Meta")` + `@impl(meta_lib, "op_name")`
- 不确定 → 引导用户检查：如果算子通过 `@torch.library.custom_op` 装饰器定义则为 `torch.library.custom_op`；如果脚本中显式创建 `Library("<namespace>", "FRAGMENT")`，并调用 `define(...)`、`@impl(..., "PrivateUse1")`、`Library(..., "IMPL", "Meta")` 则为纯 Python `torch.library.Library`

### 编写 Meta 推导函数（`register_fake` / Meta impl）

**agent 行为规则**：
1. 请求用户提供**算子签名和语义描述**（输入/输出 tensor 的 shape、dtype 映射关系）。Meta 函数只关心操作的数学语义，不关心具体计算实现
2. 若是 `torch.library.custom_op` 路径 → 在用户脚本内使用 `@my_op.register_fake` 编写 Meta 推导函数
3. 若是纯 Python `torch.library.Library` 路径 → 在用户脚本内使用 `meta_lib = Library("<namespace>", "IMPL", "Meta")` + `@impl(meta_lib, "op_name")` 编写 Meta 推导函数
4. 若用户能提供算子签名和语义 → 帮助编写完整 Meta 骨架，遵循对应 template skill 中的「自定义算子代码块」格式
5. 若用户无法提供 → 给出骨架代码，用 `# TODO: 根据算子语义推导输出 shape/dtype` 标注需用户填充的部分

> 关键点：Meta 推导函数只做 shape/dtype/device 推导，不执行实际计算。写 Meta 函数只需理解操作的数学语义和 input→output 的映射关系。In-place 算子返回 `None`，Out-of-place 算子返回与输出同 shape/dtype 的空 tensor。

### 后续步骤

完成 Meta 推导函数后，**即可支持 npugraph_ex 入图，无需额外步骤**。

下一步：加载 `torch-npugraph-ex-knowledge` skill，继续处理编译调用、调试定位或性能优化。

---

## 纯 Python `torch.library.Library` 写法对照

用途：为不使用 `@torch.library.custom_op` 的用户提供一份对照表，帮助他们在同一脚本内完成 schema、Eager 实现和 Meta 注册。

| 阶段 | 常见写法 | 说明 |
|------|----------|------|
| Schema 定义 | `mylib = Library("mylib", "FRAGMENT")` + `mylib.define("my_op(Tensor x) -> Tensor")` | 定义算子 schema，In-place 算子需在 schema 中显式标注 alias，如 `Tensor(a!)` |
| Eager 实现 | `@impl(mylib, "my_op", "PrivateUse1")` | 为 NPU 后端注册实际执行逻辑 |
| Meta 注册 | `meta_lib = Library("mylib", "IMPL", "Meta")` + `@impl(meta_lib, "my_op")` | 注册 Meta 推导函数，用于 `torch.compile` 入图 |

> **agent 使用流程**：收到纯 Python `torch.library.Library` 算子需求 → 先确认 schema 和算子语义 → 再按 `define(...)`、`@impl(..., "PrivateUse1")`、`@impl(..., "Meta")` 三步生成对应骨架。
