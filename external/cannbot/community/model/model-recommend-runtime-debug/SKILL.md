---
name: external-cannbot-community-model-model-recommend-runtime-debug
description: 运行时问题诊断与调试保障（工作流条目 13，贯穿全流程）。GE Session 推理运行时错误系统化排查：GE 初始化失败、onnx 解析失败、aicore
  timeout 507014、OOM、卡死、aclnn/GE 错误码、plog 解析。触发：运行时错误、错误码、推理卡住、卡死、hang、aicore timeout、OOM、段错误、崩溃、GEInitialize
  failed、onnx 解析失败。
original-name: model-recommend-runtime-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# 运行时问题诊断与调试保障

## Prerequisites

- 任一工作流环节报错 / 卡死 / 崩溃即进入本 skill
- 保留故障现场：`models/M<id>/report/` 日志、`error.log`（GE_ERROR_LOG）、npu-smi 快照、进程状态

## 目标

任一环节报错/卡死时，按标准路径快速定界修复，不阻塞主流程。核心理念：**先定位再修复**（二分法逐步缩小故障范围）。不覆盖精度问题（精度走 `model-recommend-precision-comparison`）。

## 诊断路径总览

```
问题发生
  │
  ├── 有明确错误信息 → error.log + plog（run.sh --info / --debug，
  │     日志自动落 models/M<id>/report/run_debug_<时间戳>.log）
  │     ├── GEInitialize failed → PYTHONPATH 含 numpy/tbe（test.sh 已设置）；检查 CANN set_env
  │     ├── onnx 解析失败 → 非标 domain / 自定义算子插件缺失
  │     │     → 非标 domain 折叠改写（model-recommend-plugin-dev）/ 自定义算子接入（model-recommend-custom-op-dev）
  │     ├── Invalid tensor data type → dtype 支持集合 → inspect_config.py
  │     ├── aicore timeout (507014) → 二分法定位
  │     │     ├── 定位到阶段（建图/编译/执行）→ 节点（dump_graph 对照）→ 算子
  │     │     └── 查算子约束（dtype/shape/layout）→ 修复策略 → 验证
  │     ├── OOM / HBM 超限 → 权重 + workspace + 并发图实例
  │     │     → 降 GRAPH_MAX_PARALLEL / THREAD_NUM / batch，或转 all_static 收敛 workspace
  │     ├── Shape 不匹配 → parser_shape vs runtime_shape / output_overrides / 动态维解析
  │     └── aclnn / GE 错误码 → 按环境配置 / Tiling / Kernel 三类归因
  │
  └── 无明确错误 → npu-smi 状态检查（进程 × HBM × AICore 判定表）
        ├── 有进程，HBM 不变，CPU 100% → 建图/编译阶段（kernel_meta 生成中，大图编译耗时）
        ├── 有进程，HBM 不变，CPU 0% → hang 诊断
        │     ├── 检查残留进程（kill 后重跑）
        │     ├── plog 定位卡在哪个阶段（--debug 等级 0）
        │     └── 图执行死锁 → dump_graph 定位可疑节点 → 算子级复现
        ├── 有进程，HBM 持续增长 → 权重加载/编译中 → 等待或评估是否越界
        └── 0 进程 → 查 error.log → 通用诊断流程
```

### GE 自定义交付相关报错

**1. 解析期算子不支持**（插件未安装 / 未生效）：

```
[ERROR] ... [pre_checker.cc] CheckTypeSupported: ErrorNo: 4294967295(failed) Check op[<node>]'s type[<domain>::<version>::<OpType>] failed, it is not supported.
[ERROR] ... [onnx_parser.cc] ModelParseToGraphImpl: ... [Invoke][Prechecker] failed.
[ERROR] parse model failed, status=4294967295
```

处置：取 `type[<domain>::<version>::<OpType>]`；能由现有算子组合表达 → ONNX 插件翻译
（`model-recommend-plugin-dev` 的等价改写）；
否则转自定义算子（`model-recommend-custom-op-dev`）。

**2. 无匹配内核 / dtype 不支持**：

```
[ERROR] ... [dnnengine_manager.cc] GetDNNEngineName: ErrorNo: 1343242282(assign engine failed)
  [Check][OpSupported] Op type <OpType> of ops kernel AIcoreEngine is unsupported, reason :
  [tbe-custom]: op type <OpType> is not found in this op store.
  [Dynamic shape check]: data type <dtype> of input [<input>] is not supported.
```

处置：先判断"算子完全缺失"还是"仅当前 dtype/format 不支持"；后者优先用等价改写/Pass 在前后加
Cast/Transpose 规避（`model-recommend-fusion-pass-dev`），
再转自定义算子（`model-recommend-custom-op-dev`）。

## npu-smi 状态判定表（单卡单进程）

| npu-smi 表现 | 含义 | 诊断路径 |
|-------------|------|---------|
| 有进程，AICore 0%，HBM 稳定 | 未进入执行（编译/等待/死锁） | 对照 CPU 占用分流 |
| 有进程，AICore 有占用 | 正在执行 | 时长异常才介入（507014 二分） |
| 有进程，HBM 逼近上限 | OOM 风险 | 降并发图实例/batch；转静态 shape 收敛 workspace |
| 0 进程 | 全部 crash | 查 `error.log`（GE_ERROR_LOG）与 report/ 日志 |
| 多进程残留 | 上次运行未清干净 | kill 残留后重跑（fail-fast 下异常退出可能残留） |

## 单卡部署模式要点（卡住/crash 时先确认执行形态）

| 差异点 | 说明 |
|--------|------|
| 进程模型 | 单进程多线程（`THREAD_NUM` 个执行线程），共享同一 `graph_id`；fail-fast 任一线程失败置 stop 全退出 |
| 初始化顺序 | `GEInitializeV2` → `aclInit` → `GeSession` → `AddGraph`（编译）→ 线程循环；卡在不同阶段症状不同 |
| 设备 | 单设备（`DEVICE_NUM=1`），`aclrtSetDevice` 按线程；`START_DEVICE` 经 `TOTAL_DIE` 取模 |
| 模型环境 | `M<id>_env.sh` 每次 source；环境类错误先确认该文件内容 |

## 昇腾 NPU 专属要点

- **日志双通道**：`--info`（plog 等级 2）/ `--debug`（等级 0）开启后，进程 stdout/stderr 自动落 `models/M<id>/report/run_debug_<时间戳>.log`；设备侧 plog 由 CANN 写默认日志目录。
- **GE 产物目录**：`.cache/run`（fusion_result.json / kernel_meta）、`.cache/dump_graph`（最多 3 份）、`.cache/dump_data`（最多 3 份）——排查时先看产物是否存在判断卡在哪一阶段。
- **自定义算子**（`model-recommend-custom-op-dev` 产物）的卡死/崩溃/内存越界：plog 解析 + memcheck 主动检测；aclnn 错误码（161xxx/361xxx/561xxx）按环境配置 / Tiling / Kernel 三类归因。
- **onnx 带外部权重**：运行时权重文件必须与 `model.onnx` 同目录（prepare.sh 已校验；手动挪动后报错先查此项）。
- **设备占用冲突**：npu-smi 进程表有他任务时换 `START_DEVICE` 或等待；sweep.py 内置占用检测逻辑可复用。
- 直接执行二进制（绕过 test.sh/run.sh）导致的 GE 产物散落/路径错误：重走脚本入口。

## 输出契约

问题概要 → 定位过程（阶段 → 节点 → 算子）→ 根因 → 修复 → 验证 → 同类错误二次出现时的直达路径记录。

**修复的落地纪律**：修复若改动模型执行（onnx/配置/GE 选项/算子实现），与优化项同等过双门禁——只确认"报错消失"不算修复完成；规则见 `model-recommend-ascend-env-baseline` 全局约束。

## 产物目录约束

- **文档格式**：禁止裸单波浪线（GFM 删除线），规则见 `model-recommend-ascend-env-baseline` 全局约束。
- **文档类交付件** → `docs/{model_name}/runtime-debug/`：排查记录与沉淀文档（问题概要/定位过程/根因/修复/验证、同类错误直达路径）
- **数据类交付件** → 无（现场数据均属中间产物）
- **中间产物** → `tmp/{model_name}/runtime-debug/`：plog、memcheck 输出、现场日志快照、临时复现脚本
- 全局约定与判定标准见 `model-recommend-ascend-env-baseline`；违反目录约定的产物视为未交付

## 验收标准

- 故障案例沉淀为可检索的排查记录
- 同类错误二次出现时按路径直达根因
- 修复若改动模型执行（onnx/配置/选项/算子），**已过「改动校验双门禁」**（串行 verify 全 pass + 同口径性能不劣化）后才可关闭案例
- 修复后全量回归通过（`model-recommend-regression-baseline-mgmt`）

## 依据

【框架】`run.sh --info/--debug`（plog 与日志落盘）、故障排查表（`README.md` §12「故障排查」）、fail-fast 机制、`.cache` 产物治理、`M<id>_env.sh`。
【实测】npu-smi 25.5.t6.b001 可用；本机单卡容器形态。
【技能库经验】二分定位、npu-smi 判定表、aclnn 错误码三类归因、plog/memcheck 方法论。
