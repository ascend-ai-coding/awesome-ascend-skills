# Brownfield 代码分析规则（Stub）

> **状态：Stub — Brownfield 模式扩展在后续阶段实施时填充本文档。**

## 用途

Brownfield 模式下，从已有算子代码推断其命中 [`capability-declaration.md`](capability-declaration.md) 中的哪条路径（chip × op_type × 调用形态 × 通信路径 × 编程抽象），做 delta 设计和修改式开发。

## 推断规则（待填充）

### 芯片型号推断

- CMakeLists.txt 中 `npu-arch` 值 → chip 坐标
- `#if defined(DAV_*)` 宏 → 辅助确认

### 通信路径与编程抽象推断

| 代码信号 | 推断通信路径 | 推断编程抽象（底座） |
|---------|------------|------------|
| `aclshmem*` / `aclshmemx_udma_*` + 独立 CMake 工程 | UDMA | blaze-shmem |
| `CollectiveComm<...>` + `block/` `tiling/` 共享层 | UDMA（直调）/ HCCL windows（注册） | apace |
| `winContext` / `mc2Context` / `HcclAllocComResourceByTiling` + window 地址搬运 | MTE通信（AIV+UBMEM） | ascendc-api |
| `Hccl::*` 高阶 API | HCCL 高阶（仅注册） | hccl-matmul |

### 算子类型推断

| 代码信号 | 推断 operator_type |
|---------|-------------------|
| AllToAll/AllGather/AllReduce/ReduceScatter 通信原语 + Matmul | collective-comm |
| expert routing / dispatch / combine / token 重排 | moe |

## Brownfield 流程（待填充）

> Brownfield 流程编排放 plugin（后续阶段），本文件仅提供代码分析的领域知识规则。
