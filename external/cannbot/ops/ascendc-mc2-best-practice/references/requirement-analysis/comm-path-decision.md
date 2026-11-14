# MC2 通信路径决策（决策栈 L1 层）

通信路径 = 通信引擎 + 通信协议的组合选择（数据怎么跨卡搬）。由用户决策。仅当用户询问时，根据以下信息给出建议。

- **通信引擎**：通信的驱动/执行单元——AIV（向量核直驱）、AICPU（传统直驱方案）、CCU（专用通信引擎）
- **通信协议**：跨卡互连协议——URMA（UDMA 为其同义称呼）、UBMEM（UB 内存域，超节点内全互联）
- **通信库**（编程接口层，屏蔽引擎/协议差异，归属 L2 编程抽象）：HCOMM 通信基础库（最底层；HCCL 集合通信库与 APACE 通信基础 API 均基于它构建）、SHMEM（独立通信库，与 HCOMM 无关，驱动 UDMA/URMA）

> 通信路径选定后，可用的编程抽象底座（blaze-shmem / apace / ascendc-api / hccl-matmul）与已验证路线组合（chip × op_type × 调用形态）见 [`../capability-declaration.md`](../capability-declaration.md)。

## 通信路径选项

| 通信路径 | 引擎 + 协议 | 接口来源 | 说明 | 已验证架构 | 典型场景 |
|---------|------------|---------|------|-----------|---------|
| HCCL 高阶集合通信 | 服务端调度（引擎对 Kernel 不可见） | `asc-devkit/adv_api/hccl/`（如 `Hccl::AllReduce`） | HCCL 集合通信库（Ascend C 高阶 API 一部分，基于 HCOMM 构建），服务端调度，Kernel 不能干预通信时序；仅支持 acnn 单算子调用，**不支持 Kernel 直调** | 全系列 | 注册算子 / acnn 调用场景 |
| AIV+URMA | AIV + URMA | `aclshmem_*`（host）、`aclshmemx_udma_*`（device，经 SHMEM 独立通信库） | Kernel 直调，GM 内存域跨卡大块搬运，通算流水掩盖通信开销 | Ascend 950（dav-3510） | 计算密集型（AllToAll+Matmul 等集合通信类） |
| MTE通信 | AIV + UBMEM | host：`HcclAllocComResourceByTiling` 分配 window 资源；device：AIV 触发 MTE + `DataCopyPad` 按 window 地址搬运 | Kernel 直调（裸 Ascend C，不经任何通信库），HCCL window 资源寻址 + 状态位协议，适合 token 级路由通信 | A3（dav-2201）+ A5（Ascend 950/dav-3510）双平台 | 路由通信密集型（MoE Dispatch/Combine、专家并行 EP） |
| CCU | CCU + URMA | hcomm `writenbi` / `readnbi`（device） | Kernel 直调（编程接口为 HCOMM 通信基础库），通信与 AIC 计算流水深度重叠；注册场景已有实现（APACE 通信基础 API 基于 HCOMM 驱动 CCU） | Ascend 950（dav-3510） | 计算密集型（直调尚无参考工程） |

## 参考信息

### HCCL 高阶集合通信
- **接口文档**：https://hiascend.com/document/redirect/CannCommunityHcclCppApi
- **定位**：HCCL 集合通信库是 Ascend C 高阶 API 的一部分，基于 HCOMM 通信基础库构建
- **开发约束**：仅注册 / aclnn；通信走 `Hccl<HCCL_SERVER_TYPE_AICPU>` V2（`InitV2`+`SetCcTilingV2`），Matmul 走 `AscendC::Matmul`；禁止 SHMEM/Blaze。A2 仅 FullMesh，不支持 `AlltoAllV`/`Finalize<false>`。详见 `references/foundations/hccl-matmul/`
- **已验证芯片**：Ascend 910B（A2 / dav-2201）；A3（910_93）不在本期范围

### AIV+URMA
- **文档**：https://shmem-doc.pages.dev/
- **定位**：SHMEM 是独立通信库（cann/shmem），与 HCOMM 无关
- **开发约束**：通信走 SHMEM（禁止 HCCL 高阶 API），Matmul 走 Blaze 模板，详见 `ops/ascendc-mc2-best-practice/SKILL.md`
- **对齐要求**：512B

### MTE通信（AIV+UBMEM）
- **开发约束**：禁止用 HCCL 高阶通信原语替代 window 搬运；window 地址必须走 compat 层（禁止硬编码平台结构体偏移）；共享 GM/状态区走 `DataCopyPad`；状态协议先数据后状态、每核只写自己槽位、消费后清理。详见 SKILL.md ascendc-api 路线章节与 `references/foundations/ascendc-api/moe-dispatch-combine/api-rules/`
- **平台差异**：A3/A5 的 window 地址结构不同，由 compat 层统一封装访问方式

### CCU（CCU+URMA）
- **接口**：HCOMM 通信基础库的 `writenbi`（非阻塞写入）/ `readnbi`（非阻塞读取）
- **定位**：HCOMM 是最底层通信基础库（屏蔽引擎/协议差异）；HCCL 集合通信库与 APACE 通信基础 API 均基于 HCOMM 构建
- **现有实现**：APACE 的 CCU 变体（注册场景，APACE 通信基础 API 基于 HCOMM 驱动 CCU）；直调尚无参考工程
- **通算流水**：通信与 AIC 计算流水深度重叠，通过 CrossCoreSetFlag/CrossCoreWaitFlag 跨核同步
- **参考工程**：`ops/ascendc-mc2-best-practice/`（SHMEM+Blaze 架构可参考，通信层替换为 HCOMM 接口）

## 未知通信路径处理规则

当用户提出的通信路径**不在上述已知选项中**时：

1. **必须要求用户提供参考材料**——参考代码、API 使用说明、接口文档链接，至少提供其一
2. **不得仅凭用户口头描述就判定达标**——没有参考材料的路线无法让下游开发者"开始干活"
3. **用户无法提供参考材料时**——该通信路径所依赖的接口标记为"未实现"，列入 REQUIREMENTS.md 的未完成依赖清单，并判定不具备启动开发条件（详见拷问协议-开发就绪判断）
