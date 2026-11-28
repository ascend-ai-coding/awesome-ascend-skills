# MC2 通算融合算子生成能力域

MC2（Matrix Computation & Communication）算子生成能力域的领域术语表。覆盖需求分析、方案设计、开发、审查、性能优化全链路。

## Language

### MC2

多卡间集合通信 + 单卡内计算融合，通过通算流水掩盖通信开销。算子类型本质不随芯片或框架变化。

### 决策栈

实现一个通算融合算子需要逐层决策的五层模型。上层选择约束下层的合法集合：

- **L0 需求层**：做什么、在哪跑、怎么被调用。算子类型 × 芯片 × 调用形态。这是输入而非选择。
- **L1 通信路径层**：数据怎么跨卡搬（引擎 + 协议）。取值：UDMA（URMA 协议的同义称呼，大块搬运，计算密集型）/ MTE通信（AIV+UBMEM，细粒度 + 状态位协议，路由通信密集型）/ CCU（CCU+URMA，尚无直调参考工程）/ HCCL 高阶集合通信（服务端调度黑盒，仅注册场景）。
- **L2 编程抽象层**：用什么写（buy vs build）。取值即**编码底座**：blaze-shmem（SHMEM 通信库 + Blaze 计算模板，手工组装）/ apace（APACE 模板库，通信+计算+工程组织全包）/ ascendc-api（裸 Ascend C API 全自建，如 MTE通信场景的 compat 层）/ hccl-matmul（HCCL 高阶 + `AscendC::Matmul` 高阶，官方路径，仅注册，知识目录 `references/foundations/hccl-matmul/`）。L1 与 L2 多对多：UDMA 上可用 blaze-shmem 或 apace 底座；apace 底座可跑 UDMA（直调）或 HCCL windows（注册）。
- **L3 工程组织层**：代码怎么摆。独立 CMake 工程 / 框架共享层 `kernel/<op>/` / 样例工程 + compat 分层。基本被 L2 选定。
- **L4 流水编排层**：通信与计算怎么重叠。GET/PUT、flag 编排、tileCnt、localMatmul 模式。部分被 L2 给定。

_Avoid_: 把决策栈理解为轴的笛卡尔积——L1×L2 多对多，合法组合只能逐行登记，不能自由组合推导。

### 通信引擎

通信的驱动/执行单元。取值：AIV（向量核直驱，触发 MTE/UDMA 执行跨卡搬运）、AICPU（传统直驱方案）、CCU（专用通信引擎，配对 URMA 协议）。MTE（Memory Transfer Engine）是核内搬运硬件，由 AIV 触发执行，"MTE通信"以此命名。

### 通信协议

跨卡互连协议。取值：URMA（Unreliable Remote Memory Access，与 UDMA 同义）、UBMEM（UB 内存域，超节点/单机内全互联）。

### 通信库

编程接口层，屏蔽通信引擎与协议差异，归属 L2 编程抽象。取值：

- **HCOMM 通信基础库**：最底层通信基础库（控制面管通信域/资源，数据面提供本地操作/同步/通信操作），屏蔽多引擎多协议。HCCL 集合通信库与 APACE 通信基础 API 均基于 HCOMM 构建。
- **HCCL 集合通信库**：Ascend C 高阶 API 的一部分，基于 HCOMM 构建，服务端调度的集合通信原语（`Hccl::*`）。Kernel 直调场景不可用（依赖框架注入上下文）。
- **APACE 通信基础 API**：APACE 自建，基于 HCOMM 构建，与 HCCL 集合通信库无关。
- **SHMEM**：独立通信库（cann/shmem），与 HCOMM 无关，驱动 UDMA/URMA。

_Avoid_: "HCCL 的通信基础库"（HCOMM 与 HCCL 集合通信库是两个东西，应分开说）；把 hcomm 当通信引擎（hcomm 是库，CCU 才是引擎）。

### 编码底座（Foundation）

L2 编程抽象层的实体——编码时立于其上的技术底座。当前四个：`blaze-shmem`、`apace`、`ascendc-api`、`hccl-matmul`，知识库中对应 `references/foundations/{底座名}/` 目录。注意它们不是并列的软件层：apace 是基于 Ascend C 基础 API 构建的模板库（其通信基础 API 基于 HCOMM，与 HCCL 集合通信库无关），ascendc-api 就是裸基础 API 本身，blaze-shmem 是"SHMEM 通信库 + Blaze 计算模板"的手工组装（SHMEM 与 HCOMM 无关），hccl-matmul 是官方 HCCL 高阶集合通信 + `AscendC::Matmul` 高阶（仅注册）——它们抽象层级不同，但在"选什么写代码"这个决策点上是并列选项。

### Ascend C API

最底层编程底座（基础 API + 高阶 API，均经 `kernel_operator.h` 调用），所有 kernel 都基于它书写。高阶 API 含 HCCL 集合通信库（基于 HCOMM 构建，服务端调度）。APACE 是基于 Ascend C 基础 API 构建的模板库（其通信基础 API 基于 HCOMM，与 HCCL 集合通信库无关），但"用 APACE"≠"用裸 Ascend C"——两者约束集、工程边界、可修改范围完全不同，在 L2 层是并列选项。

### 路线（Route）

决策栈上的一条**一致路线**（L0→L1→L2 的具体组合 + 对应 L3/L4 形态），是知识库的组织单元。**路线按底座命名**。当前四条 supported 路线：blaze-shmem 路线（AIV+URMA × blaze-shmem 底座 → `references/foundations/blaze-shmem/`）、apace 路线（AIV+URMA × apace 底座 → `references/foundations/apace/`）、ascendc-api 路线（HCCL window+MTE × ascendc-api 底座 → `references/foundations/ascendc-api/moe-dispatch-combine/`）、hccl-matmul 路线（HCCL 高阶 × hccl-matmul 底座 → `references/foundations/hccl-matmul/`，仅注册）。注意 ascendc-api 路线特指"裸 Ascend C API 全自建"——apace 模板库虽基于 Ascend C API 构建，但属独立底座，与 hccl-matmul（官方高阶 API 组合）也不混。
_Avoid_: 把"MTE通信"当路线名（它是 L1 通信路径名；该路线按底座命名为 ascendc-api 路线）

### 能力声明（路线登记表）

`references/capability-declaration.md`。登记决策栈上**已验证/规划/明确不支持**的路径，每行 = chip × op_type × 调用形态 × 通信路径 × 编程抽象 + status + reference_impl + 知识目录。含**否定行**（unsupported + 原因），否决项有据可查。供 Architect 在需求分析阶段查询：可行性确认 → 编程抽象选择 → 知识目录路由 → 参考实现定位。
_Avoid_: 路由表（不仅路由，还声明可用性状态与否定项）；能力矩阵（旧称，三维轴模型已被路径登记模型替代，见 ADR-0002）。

### 调用形态

算子的工程调用方式：Kernel 直调（`<<<>>>`，走 ops-direct-invoke）/ 注册算子（op_host + op_kernel，走 ops-registry-invoke）。L0 的一维，直接砍掉 L1/L2 的部分选项（直调禁 HCCL 高阶、HCCL windows、CCU——后两者无 `__global__` 入口，高阶 API 依赖框架注入上下文）。

### Greenfield

从零开始创建新算子（新建工程、新写代码）。两个现有 plugin 的默认模式。

### Brownfield

基于已有算子做功能拓展或性能优化（修改现有代码，不新建工程）。需要从代码推断路线坐标（命中能力声明中哪一行），做 delta 设计和修改式开发。是两个 plugin 各自的增量能力。

### 算子类型

MC2 算子按设计模式分两类：
- **集合通信类**（collective-comm）：以集合通信原语（AllToAll/AllGather/AllReduce/ReduceScatter）+ 计算融合为核心。
- **MOE 类**（moe）：以 MOE 场景的 dispatch/combine 为核心，涉及 expert 路由、token 分发与重排。

### 流程性知识

决定"先做什么、后做什么、门禁条件是什么"的流程编排知识。归属 plugin 层（AGENTS.md / task-prompts / workflow），不归属 skill。
_Avoid_: 领域知识（领域知识是"怎么选 API、约束是什么"，归 skill）

### 领域知识

特定 MC2 路径下的技术参考——API 用法、工程结构、约束红线、设计模式、需求模板、需求拷问判据。归属 `ascendc-mc2-best-practice` skill。
