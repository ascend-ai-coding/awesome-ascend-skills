# Tensor API DESIGN 模板

> 每栏填什么、怎么填、填写顺序见 [design-guidance.md](design-guidance.md)。

    # <operator_name> Tensor API DESIGN

    ## 1. 用户合同
    | 项目 | 内容 |
    |---|---|
    | 数学定义 | |
    | 逻辑输入/输出 | |
    | dtype/shape/layout 支持域 | |
    | 目标芯片 / NPU Arch | |
    | implementation_route | tensor_api_custom |
    | 用户硬约束 | 单 Kernel、数据驻留、精度、性能等 |

    ## 2. API 分工
    | 能力 | API | 证据 |
    |---|---|---|
    | Tensor/Layout/Copy/Mmad | Tensor API | 头文件/手册/样例 |
    | 初始化/同步/锁/事件/原子 | C API | 头文件/手册 |

    ## 3. 数据流与 Layout
    | Tensor | 逻辑 shape | 物理 shape | Pattern | dtype | 存储 | writer | reader | 同步 owner |
    |---|---|---|---|---|---|---|---|---|

    ## 4. 内存预算
    | Storage | Capacity | Alignment | Buffers | Elements | dtype bytes | Count | Offset | Total |
    |---|---:|---:|---|---:|---:|---:|---:|---:|

    ## 5. Tiling 与尾块
    | Axis | tile | tail | alignment | policy |
    |---|---:|---:|---:|---|

    ## 6. ABI Crosswalk
    | crosswalk_row_id | 逻辑对象 | 物理表示 | byte span | offset/stride | host 接线 | device 消费者 | 证据 |
    |---|---|---|---|---|---|---|---|

    ## 7. 同步与生命周期
    | 依赖 | producer | consumer | 原语 | event/mutex ID | 首轮 | 循环复用 | 尾轮 drain |
    |---|---|---|---|---|---|---|---|

    ## 8. Golden 合同
    | 输出 | checker | threshold | dtype 链 | 累加顺序 | 非有限值 | mismatch_count 语义 |
    |---|---|---|---|---|---|---|

    ## 9. 构建合同
    | 项目 | 值 |
    |---|---|
    | target_chip | |
    | npu_arch | dav-3510 |
    | CANN version | |
    | include 策略 | |
    | link set | |
    | run modes | sim / demo_only / npu |

    ## 10. 验证与性能
    | case | shape | dtype | 目标 | 基线 | 采样 | 精度 |
    |---|---|---|---|---|---|---|

    ## 11. 阻塞与不支持点
    | point | evidence | impact | next action |
    |---|---|---|---|
