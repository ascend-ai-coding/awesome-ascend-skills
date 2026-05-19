---
name: external-cannbot-graph-torch-npugraph-ex-compile-error-diagnosis
description: PyTorch 昇腾 NPU npugraph_ex 编译期报错诊断。覆盖 torch.compile 触发后 TorchDynamo /
  FX / AOTAutograd / npugraph_ex backend / ACL graph capture 阶段的报错排查，包括 Unsupported
  / graph break / BackendCompilerFailed / Meta 推导失败 / capture 失败等场景。本 skill 由 dfx-triage
  路由进入。触发：当用户遇到 npugraph_ex 入图失败、graph break、BackendCompilerFailed、Meta 推导失败或 capture
  失败时加载。关键词：入图失败、断图、graph break、BackendCompilerFailed、Meta、FakeTensor、aot_eager、capture。
original-name: torch-npugraph-ex-compile-error-diagnosis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# npugraph_ex 编译期报错诊断

> 本 skill 处理「模型还没真正跑一步」就崩溃的场景：`torch.compile` 触发后到 ACL graph capture 完成前的所有报错。
> 由 `torch-npugraph-ex-dfx-triage` 路由进入；如果尚未做首轮日志收集，先回到 triage skill 完成采集再进入本 skill。

## 适用范围

- `torch._dynamo` 追踪报错、`Unsupported`、`graph break`
- `BackendCompilerFailed`、AOTAutograd / Functorch 栈帧报错
- `npugraph_ex` backend 编图层报错（`npu_fx_compiler`、Meta 推导失败、FakeTensor 类型 / shape 错误）
- ACL graph capture 阶段失败（capture 时算子不支持、动态分支、host 操作等）
- 用户复现脚本「第一次调用编译后模型时就崩」

不属于本 skill 的场景：capture 已成功、replay 时 kernel / 通信 / 内存等运行时报错 → 转 `torch-npugraph-ex-runtime-error-diagnosis`。

## 总原则：先找首个致因报错

看 compile-error 相关日志时，优先锚定**首个致因报错**，不要被最后统一抛出的包装异常带偏。

- 优先找第一个 `torch._dynamo` / `BackendCompilerFailed` / `Unsupported` / Meta / capture 相关致因报错
- 不要只盯住尾部 summary、统一 rethrow、上层包装栈
- 若同一轮日志里同时出现多个异常，优先选那个足以解释后续 graph break、编图失败或 capture 失败的源头错误

## 证据不足时的补充 DFX 工具

`torch-npugraph-ex-dfx-triage` 已默认收集全量日志；本节不是继续堆更多日志开关，而是在证据不足时补充更贴近 `npugraph_ex` 编图链路的 DFX 工具：

1. 优先读取 `torch-npugraph-ex-knowledge` 中的 `docs/zh/npugraph_ex/dfx/dfx.md`，确认当前可用的 DFX 能力与采集入口。
2. 使用 `docs/zh/npugraph_ex/dfx/debug_save.md` 保存编图 / ACL graph capture 相关的 debug 产物，优先拿这些直接证据做归因。
3. 只有在需要对比具体节点、张量或生成物证据时，再按需使用 `docs/zh/npugraph_ex/dfx/data_dump.md`；它不是 compile-error 场景的默认首选。
4. 若仍缺少 PyTorch 编译前半程证据，再补充打开：
   - `TORCH_LOGS=+dynamo,+aot,+graph_breaks` 或 `TORCH_LOGS=+all`
   - `TORCH_COMPILE_DEBUG=1`（dump FX graph、AOT graph）

## 诊断执行规则

1. **先定界阶段**：
   - Eager 失败 → 归类到 Eager / `torch_npu` / NPU API 层问题（脱离图模式范畴，先修 Eager）
   - `backend="aot_eager"` 失败 → 归类到 Dynamo / FX / AOTAutograd 层问题
   - `aot_eager` 成功但 `npugraph_ex` 失败 → 归类到 npugraph_ex backend / Meta 推导 / ACL graph capture 层问题
2. **先锚定首个致因报错**：先从完整 traceback、相邻日志和 debug 产物里找出第一个足以解释后续失败的异常，再围绕它展开定位。
3. **就近读取源码**：若报错栈、日志、调用栈或调试输出已指向具体文件、pass、函数、模块或生成代码，先读取该上下文附近源码；不要先机械地浏览整个仓库。
4. **直接证据不足时再补查**：只有当前上下文不足以解释问题时，才按下方「问题定位时的源码检索」进行三仓兜底。
5. **固定输出格式**：问题归类 → 证据 → 最可能根因 → 下一步最小动作。若信息不足，输出「还缺什么信息」和采集方法，而不是直接给大段 MRE。
6. **默认不回到模式选型**：除非用户明确要求切换模式，或根因已经明确证明当前模式不适配，否则不要把问题定位直接转成模式推荐。

## 问题定位时的源码检索

采用「**就近定位优先，三仓兜底**」的规则，**不**按固定仓库顺序机械检索。

1. **直接证据优先**：若报错栈、日志、调用栈、dump、生成代码或调试输出已经给出文件名、pass 名、函数名、模块名或具体代码片段，先直接读取该上下文附近的源码并归因。
2. **本地源码补查**：若 workspace 中没有对应文件，优先读取本地 Python 环境中的 installed source，例如 `site-packages/torchair/`、`site-packages/torch_npu/`、`site-packages/torch/`。
3. **三仓兜底**：只有当直接证据不足以落到具体实现时，才按问题阶段补查下列仓库。**下表中的地址仅用于标识源码来源**；若本地已有 workspace 源码或 installed source，直接读取；本地不存在时再拉取到缓存目录。

| 仓库 | 地址 | 适用阶段 |
|------|------|---------|
| upstream PyTorch | `https://github.com/pytorch/pytorch.git` | `torch.compile` 前半程语义：Dynamo、FX、AOTAutograd、Functorch、graph break |
| TorchAir | `https://gitcode.com/Ascend/torchair.git` | `npugraph_ex` backend、`npu_fx_compiler`、ACL graph capture、相关 pass |
| Ascend / PyTorch | `https://gitcode.com/Ascend/pytorch.git` | `torch_npu` 桥接、Meta / FakeTensor 注册、自定义算子注册 |

**本地拉取示例**（仅在本地无对应源码时）：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git ~/.cache/pytorch-src
cd ~/.cache/pytorch-src && git sparse-checkout set torch/_dynamo torch/fx torch/_functorch

git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/Ascend/torchair.git ~/.cache/torchair-src
cd ~/.cache/torchair-src && git sparse-checkout set npugraph_ex docs/zh

git clone --depth 1 --filter=blob:none --sparse -b v2.7.1 https://gitcode.com/Ascend/pytorch.git ~/.cache/torch-src
cd ~/.cache/torch-src && git sparse-checkout set torch_npu docs/zh
```

4. **按阶段补查，而非按仓库先后**：
   - `torch._dynamo` / `torch.fx` / FakeTensor / AOTAutograd 报错 → 优先 upstream PyTorch
   - `npugraph_ex` pass、`npu_fx_compiler`、ACL graph capture 失败 → 优先 TorchAir
   - Meta / FakeTensor 注册缺失、`torch_npu` 桥接异常 → 优先 Ascend / PyTorch

## 兜底文档

- `torch-npugraph-ex-knowledge` 中的「调试定位 / 入图失败」小节
- TorchAir 仓库 `npugraph_ex` 文档与样例
- 案例集：`appendix/cases/graph_failed_cases.md`（若仓库中存在）
