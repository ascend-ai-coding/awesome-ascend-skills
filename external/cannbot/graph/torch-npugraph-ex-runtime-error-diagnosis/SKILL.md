---
name: external-cannbot-graph-torch-npugraph-ex-runtime-error-diagnosis
description: PyTorch 昇腾 NPU npugraph_ex 运行时报错诊断。覆盖 ACL graph 已 capture 成功之后，replay
  / kernel launch / 通信 / 内存 / device API 阶段的报错排查，包括 aclnnXxx 算子失败、HCCL 错误、stream/event
  同步、segfault、device side assert、OOM 等场景。本 skill 由 dfx-triage 路由进入。触发：当用户遇到 npugraph_ex
  replay / aclnn / HCCL / stream / OOM 等运行时报错时加载。关键词：runtime、ACL、aclnn、HCCL、stream、event、device
  side assert、segfault、OOM、CANN plog。
original-name: torch-npugraph-ex-runtime-error-diagnosis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# npugraph_ex 运行时报错诊断

> 本 skill 处理「模型已在执行」阶段的崩溃：ACL graph capture 已成功、replay 后才出现的报错。
> 由 `torch-npugraph-ex-dfx-triage` 路由进入；如果尚未做首轮日志收集，先回到 triage skill 完成采集再进入本 skill。

## 适用范围

- `aclnnXxx` 算子执行失败、ACL runtime 报错、`acl error`
- HCCL 通信错误（`hccl error`、ranktable、ring 异常）
- `stream` / `event` 同步错误、跨流依赖丢失
- `device side assert`、`segfault`、kernel illegal memory access
- device OOM、内存碎片、workspace 不足
- 用户脚本「训练 / 推理跑了 N 步后才崩」

不属于本 skill 的场景：编译期 / capture 期就报错 → 转 `torch-npugraph-ex-compile-error-diagnosis`。

> 边界情况：capture 阶段无报错、replay 算子才 fail，**默认归本 skill**；若证据不足以判定，可同时参考 compile-error skill 的 Meta / capture 部分。

## 总原则：先找首个致因报错

看 runtime-error 相关日志时，优先锚定**首个致因报错**，不要被后续 task fail、stream 同步失败、graph replay 失败等次生报错带偏。

- 优先找第一个 ACL / HCCL / `aclnnXxx` / OOM / stream-event 相关致因报错
- 不要只根据尾部 device error、统一 runtime 报错或 replay 失败做归因
- 若后面是一串级联失败，优先选择最早那个足以解释后续连锁错误的 runtime 异常

## 推荐启用的日志开关

在已收集的全量日志基础上，若证据不足，可补充打开：

- `ASCEND_GLOBAL_LOG_LEVEL=1`（INFO）或 `=0`（DEBUG，谨慎）
- `ASCEND_SLOG_PRINT_TO_STDOUT=1`
- `HCCL_LOGIC_SUPERPOD_ID`、`HCCL_BUFFSIZE`、`HCCL_EXEC_TIMEOUT` 等通信相关变量按需
- 收集 CANN plog（默认在 `~/ascend/log/`），定位时间戳与崩溃栈对齐

## 诊断执行规则

1. **先定界子系统**：
   - 报错关键字含 `aclnn` / `aclop` → 算子层（kernel / tiling / shape）
   - 报错关键字含 `HCCL` / `hccl` → 通信层（ranktable、链路、buffer）
   - 报错关键字含 `stream` / `event` / `synchronize` → 流 / 事件 / 同步层
   - 报错关键字含 `OOM` / `out of memory` / `allocate` → 内存层
   - `segfault` / `device side assert` / illegal memory access → 先看是否落在算子层，再看是否为 host 侧指针 / 索引越界
2. **先锚定首个致因报错**：对齐 stdout/stderr、CANN plog、HCCL log 的时间戳，优先找出第一个足以解释后续 task / replay / device 失败的 runtime 异常。
3. **就近读取源码**：若报错栈已指向 `torch_npu` 中具体文件 / 函数（如 stream / event / NPU API），先读取该上下文附近源码。
4. **直接证据不足时再补查**：按下方「问题定位时的本地 CANN / 桥接层检索」继续缩小范围，不要直接套用 compile-error 的源码检索思路。
5. **固定输出格式**：问题归类（算子 / 通信 / 流事件 / 内存 / device）→ 证据 → 最可能根因 → 下一步最小动作（包含建议的环境变量或最小化复现）。
6. **默认不回到模式选型**：除非根因已经明确证明 npugraph_ex 不适配该模型，否则不直接建议换 backend；可建议局部 fallback eager 作为规避。

## 问题定位时的本地 CANN / 桥接层检索

runtime-error 的定位策略**不同于** compile-error：优先看本地 CANN 安装产物、plog、错误码、算子规格和 HCCL/ACL 运行时线索；`torch_npu` / TorchAir 仓库源码只用于补桥接层证据，不作为第一入口。

1. **直接证据优先**：报错栈、CANN plog、HCCL log 已给出 error code、算子名、stream id、task id、rank、具体 API（如 `aclnnXxx` / `aclrt`）时，先围绕这些直接证据定位，不要先泛读仓库源码。
2. **本地 CANN 安装产物优先**：优先检查本机 CANN 安装目录（如 `$ASCEND_HOME_PATH` 或 `/usr/local/Ascend/`）下可直接支撑归因的内容，例如：
   - runtime / operator / compiler 相关日志与 plog
   - ACL / HCCL 错误码说明、头文件、API 声明
   - 算子规格、配置、约束说明，以及与报错算子对应的文档
   - 已安装 `.so`、符号名、版本信息、依赖关系
3. **再看本地 Python 桥接层**：若需要确认 PyTorch 调 CANN 的桥接路径、参数传递、stream/event 封装，再读取本地 `site-packages/torch_npu/`、`site-packages/torchair/`。
4. **仓库源码兜底**：只有当本地日志和安装产物仍不足以解释问题时，再补查下列仓库：

| 仓库 | 地址 | 适用阶段 |
|------|------|---------|
| Ascend/pytorch | `https://gitcode.com/Ascend/pytorch.git` | `torch_npu` runtime、NPU API、stream / event、多流、内存分配器、自定义算子桥接 |
| TorchAir | `https://gitcode.com/Ascend/torchair.git` | `npugraph_ex` replay 链路、capture 后执行入口、`replace_stream_event` 等 pass |
| upstream PyTorch | `https://github.com/pytorch/pytorch.git` | `torch.cuda` 风格的 stream / event 抽象（仅作为 `torch_npu` 桥接对照） |

> 若本地安装的 CANN 不包含可读源码，不要假设一定能看到“CANN 内部实现”；优先依赖 plog、错误码、算子规格、头文件、符号和版本信息做归因。

**本地拉取示例**（仅在本地日志 / 安装产物不足，且确需看桥接层源码时）：

```bash
git clone --depth 1 --filter=blob:none --sparse -b v2.7.1 https://gitcode.com/Ascend/pytorch.git ~/.cache/torch-src
cd ~/.cache/torch-src && git sparse-checkout set torch_npu docs/zh

git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/Ascend/torchair.git ~/.cache/torchair-src
cd ~/.cache/torchair-src && git sparse-checkout set npugraph_ex docs/zh
```

5. **按子系统补查**：
   - 算子失败 → 先看 CANN plog、算子名、错误码、算子规格 / 约束；必要时再看 `torch_npu` 算子桥接
   - 通信失败 → 先看 HCCL 日志、ranktable、网络 / 拓扑 / 环境变量；必要时再看 Ascend/pytorch HCCL 桥接
   - 流 / 事件 → 先看 ACL / CANN runtime 日志和 API 语义；若怀疑桥接封装，再看 `torch_npu` stream/event 实现与 TorchAir `replace_stream_event`
   - 内存 → 先看 CANN/ACL 内存报错、workspace 分配信息、OOM 上下文；必要时再看 `torch_npu` allocator 与 `PYTORCH_NPU_ALLOC_CONF`

## 兜底文档

- `torch-npugraph-ex-knowledge` 中「调试定位 / 运行时报错」小节
- 案例集：`appendix/cases/runtime_cases.md`（若仓库中存在）
- PyTorch NPU 参考文档：stream / event / multi-stream / 内存管理章节
