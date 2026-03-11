---
name: npu-op-benchmark
description: 昇腾 NPU 单算子性能基准测试 Skill；当前版本只做现有环境检查、CANN 版本识别、用户确认后执行 benchmark，不负责修复或安装环境。
keywords:
  - Ascend
  - NPU
  - torch_npu
  - CANN
  - benchmark
  - operator
  - repeat_interleave
  - transpose
  - docker
  - ssh
---

# NPU Operator Benchmark

这个 Skill 用于在昇腾 NPU 环境中对单个算子做可复现的延迟测试。

## 执行策略

1. 先检查现有环境。优先运行 `scripts/cann_detect.sh`，并结合 [cann](references/cann.md) 判断当前 CANN 布局；同时检查 `pip list | grep torch_npu`。
2. 判断 CANN 版本时，先区分版本布局：
   - `CANN < 8.5` 常见路径：`/usr/local/Ascend/ascend-toolkit/latest`
   - `CANN >= 8.5` 常见路径：`/usr/local/Ascend/cann/latest`
3. 如果用户要求特定 CANN 版本，例如 `8.2` 或 `8.3`，优先查看对应 toolkit 目录下是否存在目标版本目录，再看 `latest` 实际指向；如果 `latest` 不是软链接，只能据目录结构和运行环境保守判断。
4. Docker 场景优先运行 `scripts/find_docker_cann.sh`，并结合 [docker](references/docker.md) 选择满足要求的容器。
5. 发现满足要求的容器后，只报告容器名，等待用户明确同意再执行 benchmark。
6. 执行 benchmark 前，先询问使用者是“提供测试 demo”还是“由 AI 根据目标算子生成 demo”。如果使用者提供 demo，优先使用使用者的 demo。
7. 真正执行 benchmark 时，优先复用容器原本已有的 `torch` 和 `torch_npu`，不要重复安装。
8. 如果当前环境缺少目标 CANN、`torch_npu` 不可用、或算子执行报环境错误，立即停止，不修环境；直接要求使用者提供新的可用环境。
9. 不自动安装 CANN，不自动修复容器，不启用任何回退安装流程。
10. 如果测试过程中必须创建文件，都放到隔离测试目录，结束后删除。
11. benchmark 结束后，直接返回完整结果信息，不额外生成报告文件。

## 返回结果

返回内容至少包括：

- `CANN 版本`
- `torch 版本`
- `torch_npu 版本`
- `测试环境`
- `测试方法`
- `tensor shape`
- `demo 代码`
- `性能数据`

如果环境不满足或算子执行失败，统一按下面模板返回：

```text
环境不满足，已停止当前测试。请提供新的可用环境。

torch 版本:
<torch_version 或 unknown>

torch_npu 版本:
<torch_npu_version 或 unknown>

当前引用的 CANN 版本:
<根据 /usr/local/Ascend/ascend-toolkit/latest 和同级目录判断出的版本，若无法确定则写 unknown，并附目录信息>

demo.py 内容:
<实际执行的 demo 代码；优先使用使用者提供的 demo，否则使用 AI 生成的 demo>

input/output tensor shape:
<实际测试使用的输入 shape，以及必要时的输出 shape>

执行失败报错:
<原始错误摘要或关键报错>
```

## 入口

- CANN 检测：`scripts/cann_detect.sh`
- Docker 容器检查：`scripts/find_docker_cann.sh`
- 通用算子基准：`scripts/bench_op.py`
- repeat_interleave 示例：`scripts/bench_repeat_interleave.py`

更多说明见：
- [usage](references/usage.md)
- [docker](references/docker.md)
- [conda](references/conda.md)
- [cann](references/cann.md)
- [troubleshooting](references/troubleshooting.md)
