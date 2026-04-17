# Usage Guide

执行顺序必须按下面流程进行：

1. 首次收集信息时，用自然语言一次性说明需要这些内容，不要直接丢固定模板。推荐表述为：我需要服务器信息（IP、账号、密码）；版本要求（CANN 和 torch，没有可以不用写）；测试环境（conda 环境名、docker 容器名、CANN 目录，没有可以不写）；以及 demo 方案（我生成 / 我提供）。
2. 如果 `IP`、`账号`、`密码` 缺失，直接中断。
3. 如果 `conda 环境名` 有值且不是 `无`，优先检查该 conda 环境。
4. 如果 `docker 容器名` 有值且不是 `无`，优先检查该容器。
5. 如果 `CANN 目录` 有值且不是 `无`，优先使用该目录判断 CANN 版本；否则按常见路径自动检索。
6. 如果 `conda 环境名` 和 `docker 容器名` 都是 `无`，则先找 conda 候选环境；若用户不接受，再找容器候选环境。
7. 如果用户给出了 `CANN 要求`、`torch 要求`、`torch_npu 要求`，则按这些要求筛选 conda 或容器；如果没给要求，则返回候选环境和版本信息供用户确认。
8. `demo 方案` 为 `我提供` 时，优先使用用户补充的 demo、算子名或相关用例。
9. `demo 方案` 为 `我生成` 时，由 agent 根据目标算子的性能测试原则设计 demo；当前仓库中的 `repeat_interleave` 相关脚本和配置只是示例，不代表只支持测试 `repeat_interleave`。
10. demo 设计完成后，先把 demo 方案返回给使用者。
11. 在用户确认的 conda 或容器环境中执行测试。
12. 如果环境不满足要求，或算子执行时报环境错误，立即停止并要求使用者提供新的环境。

结果返回应包含：

- `CANN version`
- `torch version`
- `torch_npu version`
- `test environment`
- `test target`
- `test method`
- `tensor shape`
- `demo code`
- `performance data`
