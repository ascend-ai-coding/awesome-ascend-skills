# NPU 执行模型与 Tensor API 概念

> 本 Skill 默认业务目标是 Ascend 950 / DAV_3510。asc-devkit master 同时包含 3510 与 9201 相关能力；任何非 3510 目标必须逐 API 查看当前头文件注释和实现特化。容量、流水和派发事实必须以当前手册、头文件或官方样例复核。

## 1. 存储层级与数据流

Cube 典型数据流：

    GM
      -> L1
      -> L0A / L0B
      -> MMAD
      -> L0C
      -> Fixpipe
      -> GM

Vector 典型数据流：

    GM -> UB -> Vector -> UB -> GM

| 存储 | 容量基线 | 用途 |
|---|---:|---|
| UB | 256KB | Vector 工作区 |
| L1 | 512KB | A/B tile 暂存，支持 ping-pong |
| L0A | 64KB | Cube 输入 A |
| L0B | 64KB | Cube 输入 B |
| L0C | 256KB | Cube 累加器，通常 FP32 |

## 2. 传输引擎与核心类型

| 组件 | 功能 |
|---|---|
| MTE1 | L1 与 L0A/L0B/UB 之间搬运 |
| MTE2 | GM 与 L1 之间搬运 |
| MTE3 / Fixpipe | L0C、UB 或 GM 输出路径 |
| Cube | 矩阵乘累加 |
| Vector | 向量、激活、Cast、量化等计算 |

AIC 包含 Cube、Vector 和搬运引擎；AIV 以 Vector 为主。混合执行时，task type、核数比例和同步 owner 必须由当前源码或设备证据冻结。

## 3. 抽象层

    AscendC 原语
      -> Tensor API
      -> 算子项目实现

Tensor API 提供强类型 Tensor、Layout、Copy Atom 和 Mmad Atom；它不自动管理事件同步，也不提供完整调度循环。同步、tile 循环、资源生命周期和输出唯一 writer 由项目设计合同负责。

## 4. Tensor API 核心概念

| 概念 | 说明 |
|---|---|
| Shape | 维度大小，可嵌套表达分形 |
| Stride | 每维步长 |
| Coord | 多维坐标 |
| Layout | Shape + Stride + Pattern/Trait 信息 |
| Pattern | 编译期路由标签，例如 NDExt、NZ、ZN |
| Trait | dtype、C0 等编译期属性 |
| Atom | Copy/Mmad 操作抽象 |

Pattern 决定派发：Routing 表按目标/源位置、架构、Pattern 等条件做编译期选择。Pattern 不在白名单时触发 static_assert。

同步是用户责任：Tensor API 不接管 SetFlag/WaitFlag 或 C API 事件，生产者/消费者配对由项目维护。

## 5. 常用 Pattern

| Pattern | 形态 | 主要用途 |
|---|---|---|
| NDExtLayoutPtn | 外层 ND | GM 输入/输出 |
| DNExtLayoutPtn | 外层 DN | GM 转置入口 |
| NDLayoutPtn | 普通 ND | UB/内部 Tensor |
| DNLayoutPtn | 普通 DN | UB/内部 Tensor |
| NZLayoutPtn | NZ 分形 | L1/L0A/GM 预重排 |
| ZNLayoutPtn | ZN 分形 | L1/L0B/GM 转置预重排 |
| ZZLayoutPtn | ZZ 分形 | ScaleA 等量化数据 |
| NNLayoutPtn | NN 分形 | ScaleB 等量化数据 |

Ext 表示外层入口的双层嵌套 shape；GM 与片上之间必须区分 Ext 与非 Ext。

## 6. C0 速查

| dtype | C0 |
|---|---:|
| fp16 / bf16 | 16 |
| fp32 | 8，L0C 上仍按 16 处理 |
| int8 / fp8 | 32 |
| fp4x2 | 64 |
| fp8_e8m0 scale | 2 |

L0C cube 边长恒为 16，不写成 32/sizeof(L0CType)。C0 和分形规则必须以当前头文件复核。
