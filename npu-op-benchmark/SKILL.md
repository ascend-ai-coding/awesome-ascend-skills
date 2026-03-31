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

1. 首次向使用者收集信息时，用自然语言说明需要这些内容，不要直接丢固定模板。推荐表述为：我需要服务器信息（IP、账号、密码）；版本要求（CANN 和 torch，没有可以不用写）；测试环境（conda 环境名、docker 容器名、CANN 目录，没有可以不写）；以及 demo 方案（我生成 / 我提供）。
2. 如果使用者没有提供服务器 IP、账号、密码，直接中断，不猜测环境。
3. CANN 检查优先使用使用者提供的 CANN 目录；未提供或填写 `无` 时，再按 [cann](references/cann.md) 检索常见路径并判断当前版本布局：
   - `CANN < 8.5` 常见路径：`/usr/local/Ascend/ascend-toolkit/latest`
   - `CANN >= 8.5` 常见路径：`/usr/local/Ascend/cann/latest`
4. 如果使用者填写了 conda 环境名，优先按 [conda](references/conda.md) 走 conda 方案。
5. 如果使用者填写了 docker 容器名，优先按 [docker](references/docker.md) 检查该容器。
6. 如果 `conda 环境名` 和 `docker 容器名` 都是 `无`，则先查 conda 候选环境；用户不接受 conda 时再查容器候选环境。
7. 无论 conda 还是容器，都必须先把环境名、torch/torch_npu 版本、CANN 版本返回给使用者确认，再执行 benchmark。
8. `demo 方案` 只接受两种值：`我生成` 或 `我提供`。如果使用者选择 `我提供`，优先使用使用者提供的 demo 或算子补充信息。
9. 当前仓库里的 `assets/config_template.yaml`、`scripts/bench_repeat_interleave.py` 和 `scripts/bench_op.py` 中 `repeat_interleave` 相关内容只是示例，不代表这个 Skill 只支持测试 `repeat_interleave`。
10. 如果使用者未提供额外用例，agent 可以根据目标算子的性能测试原则自行设计 demo；设计完成后先返回给使用者。
11. 真正执行 benchmark 时，优先复用目标环境原本已有的 `torch` 和 `torch_npu`，不要重复安装。
12. 如果当前环境缺少目标 CANN、`torch_npu` 不可用、或算子执行报环境错误，立即停止，不修环境；直接要求使用者提供新的可用环境。
13. 不自动安装 CANN，不自动修复 conda 或容器，不启用任何回退安装流程。
14. 如果测试过程中必须创建文件，都放到隔离测试目录，结束后删除。
15. benchmark 结束后，直接返回完整结果信息，不额外生成报告文件。

## 返回结果

返回内容至少包括：

- `CANN 版本`
- `torch 版本`
- `torch_npu 版本`
- `测试环境`
- `环境名称（conda 名或容器名）`
- `测试方法`
- `目标算子`
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
<根据用户提供目录或常见路径判断出的版本；若无法确定则写 unknown，并附 `latest` 布局和候选目录信息>

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
- repeat_interleave 示例包装脚本：`scripts/bench_repeat_interleave.py`

更多说明见：
- [usage](references/usage.md)
- [docker](references/docker.md)
- [conda](references/conda.md)
- [cann](references/cann.md)
- [troubleshooting](references/troubleshooting.md)
