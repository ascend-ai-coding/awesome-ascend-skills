# 算子实现边界

设计阶段必须明确算子的实现边界，避免下游实现阶段把复杂逻辑放到错误的位置。

## 各层职责

| 位置 | 职责 |
| --- | --- |
| Device kernel | 算子语义中的计算和通信（RMA、collective）、Device 侧数据搬运和格式转换、Phase 间同步 |
| `main.cpp` | ACL/SHMEM 初始化和清理、读写输入输出文件、Host-Device 数据拷贝、Kernel 调用、资源释放；单进程单 PE |
| Host helper `.cpp/.h` | 运行时 shape/layout 推导、tiling/launch 参数构造、输入 packing/输出 unpacking、route/payload 计划 |
| Python scripts | 输入数据生成、测试用 Route/Payload 计算、Golden 计算、精度验证 |

## 禁止的实现

- Host RMA 作为算子主要通信路径（即使是 correctness-first）
- Device kernel 中用 `DataCopy` 直接访问远端 PE 地址完成跨 PE 搬运
- `main.cpp` 中包含复杂逻辑（route、编码、解码、tiling、packing、业务预处理等）
- `main.cpp` 中包含 golden 生成或精度验证
- `main.cpp` fork/spawn 多个子进程分别负责多个 PE
- 以"先实现 Host 版本"为由跳过 Device kernel

## 判断标准

- 只服务测试数据或 oracle 的逻辑 -> Python
- 运行时 Host 计划逻辑 -> 独立 Host `.cpp/.h`
- 算子通信或计算语义 -> Device kernel
- `main.cpp` 应该是胶水代码，只串联阶段

## Kernel 实现策略（统一路径 — MUST）

设计阶段 **MUST** 默认 **single path**：一套 Device kernel 逻辑覆盖 correctness case matrix 与性能平台区（S/L 档），UB 内 `while` 分块是内部细节，**不是** `small`/`large` 两套并行实现。

| 原则 | 说明 |
| --- | --- |
| 默认 unified | `schedule` / `implementation` 描述一条主路径 + UB 分块 |
| 外层 tile | 仅因 `GVA_BUFF_MAX_SIZE`、对称堆容量触发的 GM 分块循环可保留 |
| 禁止默认分叉 | **不得**在设计中预先规划 `*_small_data`/`*_big_data`、`*_small`/`*_large` 双路径，除非用户明确要求且附收益假设 |
| 证明后分拆 | 若 Phase 6.5 profiling 证明 size 分支 **kernel_bus_bandwidth_GBps 或时延 ≥5%** 收益，才在 design 修订中记录分拆理由与对比数据 |

DSL `schedule` 中若出现多路径，**MUST** 在 Design Review 标注「需 profiling 证明」或合并为 unified。代码生成见 [code-patterns.md §6.4](../../shmem-ops-code-gen/references/code-patterns.md)；优化阶段见 [agent-execution-contract.md §5](../../shmem-ops-dev/references/agent-execution-contract.md)。

## 禁止的实现（补充）

- 未证明收益即在 kernel 中维护 `if (0)` / `INT64_MAX` 禁用的 dead `large` 路径与 unified 路径长期并存（合并后 **MUST** 删除死代码）

## 数据放置方式选择

远端 PE 能访问的只有对称内存。用户的 `input` 在普通 Device GM（`aclrtMalloc`），需要搬到对称内存后才能被远端 PE 读取。搬运方式取决于通信引擎对地址类型的支持：

| 引擎 | put_nbi src 能否是用户 GM | get_nbi dst 能否是用户 GM | 放置方式 |
| --- | --- | --- | --- |
| MTE | 能 | 能 | **做法 A**：kernel 内直接用 `put_nbi(symm, user_gm, ...)` 搬运 |
| SDMA | 能 | 能 | 同上 |
| UDMA | 能 | 能 | 同上 |
| RDMA | 不能 | 不能 | **做法 B**：Host 侧 `aclrtMemcpy(user_gm → symm)`，再进 kernel |

**约束**：
- MTE/SDMA/UDMA 下 **SHOULD** 选用做法 A——直接在 kernel 内把用户 GM 搬到对称内存，避免无意义的 kernel 外搬运
- RDMA 下 **MUST** 选用做法 B——必须先 Host 侧搬运，双方数据都在对称内存上才能通信
- 设计 DSL `schedule` 中 **MUST** 标注使用的引擎和对应的放置方式（A 或 B）
- 做法 A 下 e2e_us≈kernel_us 是正常的，不是问题；做法 B 下两者 MUST 有明显差值
