# Tensor API 路线验收与检视清单

> 供功能精度、性能和代码检视等验收节点按 implementation_route=tensor_api_custom 条件加载。

## 功能精度验收

| 检查 | 判定 |
|---|---|
| API 边界 | tensor.h 主体 + experimental/vector_ops.h Reg Vector，C API 只做辅助控制 |
| 禁用 API | 无 TPipe/TQue/TQueBind/TBufPool |
| Experimental Reg Vector | 设计记录 devkit commit、架构条件、dtype/mask 语义和验证证据 |
| 初始化与收尾 | asc_init 与最终 asc_sync_pipe(PIPE_ALL) 存在 |
| 同步 | lock/unlock、notify/wait 配对；事件 ID 不使用 6/7 |
| 尾块 | M/N/K 显式处理，padding 不参与 golden |
| mmad | 对齐、范围、首轮累加、K 累加同步符合当前手册 |
| Golden | 独立实现，使用最终实际 H2D 物理 bytes |
| device error | 全部 case 为零 |
| 证据 | 命令、返回码、stdout/stderr、输入输出、验证记录齐全 |

## 性能验收

1. 性能合同记录 shape、命令、有效字节数、目标、基线和采样次数。
2. 报告包含耗时分布、带宽、流水利用率、核间负载和冲突证据。
3. 跨轮波动跨越门槛的 case 记录多轮中位数与离散度。
4. 每次优化写入实验账本，负结果也保留。
5. 改变数值路径的优化必须附权威精度回归结果。

## 代码检视验收

1. 片上内存预算与代码常量一致且不超限。
2. 输出 writer 唯一。
3. 所有 device/host 资源释放。
4. 文档、设计合同与实现一致。
5. 构建产物符号/schema 可验证。
6. 无证据时结论只能是 BLOCKED，不得给“有条件通过”。
