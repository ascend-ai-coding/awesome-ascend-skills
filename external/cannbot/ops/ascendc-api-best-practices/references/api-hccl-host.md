# HCCL Host 侧 API 使用指南

> **适用场景**：跨卡算子 Host 侧 launcher 的通信域创建、channel 建链、engine context 管理、内存注册。本文档只覆盖 CANN HCCL 公共 host API 的签名与约束；具体框架的 builder 流程请参考对应框架文档。

---

## 目录

1. [概述](#1-概述)
2. [API 签名与参数](#2-api-签名与参数)
3. [最小通用示例](#3-最小通用示例)
4. [资源生命周期](#4-资源生命周期)
5. [常见错误](#5-常见错误)
6. [检查清单](#检查清单)

---

## 1. 概述

跨卡 kernel 直调需要 host 侧完成三件事：

1. **创建通信域**：rootInfo 交换 → `HcclCommInitRootInfoConfig`
2. **建链**：`HcclChannelAcquire` 获取 channel 句柄 + 获取各 rank 通信 buffer 地址
3. **下发上下文**：`HcclEngineCtxCreate/Copy` 把上下文结构写入 device GM，kernel 以 `__gm__` 指针接收

头文件：`hccl/hccl_comm.h`、`hccl/hccl_res.h`、`hccl/hccl_rank_graph.h`（CANN 安装目录 `include/hccl/`）。

---

## 2. API 签名与参数

### 2.1 通信域管理

| API | 签名 | 用途 |
|:---|:---|:---|
| `HcclGetRootInfo` | `HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo)` | rank0 生成 rootInfo |
| `HcclCommInitRootInfo` | `HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm)` | 创建通信域（默认配置） |
| `HcclCommInitRootInfoConfig` | `HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)` | 创建通信域（自定义配置，**推荐**） |
| `HcclCommDestroy` | `HcclResult HcclCommDestroy(HcclComm comm)` | 销毁通信域 |
| `HcclGetRankId` | `HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)` | 获取本 rank ID |
| `HcclGetRankSize` | `HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)` | 获取总 rank 数 |

> `HcclCommConfig` 用 `HcclCommConfigInit(&config)` 初始化；`hcclBufferSize` 字段控制 HCCL 内置 buffer 大小（MB），不设置时用默认值（CANN 默认 200MB）。

### 2.2 内存管理

| API | 签名 | 用途 |
|:---|:---|:---|
| `HcclGetHcclBuffer` | `HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)` | 取本 rank HCCL 内置 buffer（**库内管理，严禁释放**） |
| `HcclCommMemReg` | `HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)` | 注册通信内存 |

### 2.3 Channel 管理

| API | 签名 | 用途 |
|:---|:---|:---|
| `HcclChannelDescInit` | `HcclResult HcclChannelDescInit(HcclChannelDesc* desc, uint32_t num)` | 初始化 channel 描述符（**Acquire 前必须调用**，官方 @warning 强制） |
| `HcclChannelAcquire` | `HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc* desc, uint32_t num, ChannelHandle* channels)` | 创建 channel |
| `HcclChannelGetHcclBuffer` | `HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size)` | 取指定 channel 对端 rank 的 HCCL buffer 地址（URMA 路径填 Win 区地址用） |
| `HcclChannelGetRemoteMems` | `HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)` | 取对端注册内存列表（按 memTag 匹配，注册内存路径用） |

> **engine 一致性**：`HcclChannelAcquire` 与 `HcclEngineCtxCreate/Get/Copy` 必须使用同一 `CommEngine`，否则 engine ctx 复用失效。

### 2.4 Engine Context

| API | 签名 | 用途 |
|:---|:---|:---|
| `HcclEngineCtxCreate` | `HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)` | 创建引擎上下文 device 内存 |
| `HcclEngineCtxGet` | `HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)` | 查询已存在上下文（同 tag+engine 命中即复用） |
| `HcclEngineCtxCopy` | `HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char* ctxTag, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)` | host→device 拷贝上下文 |
| `HcclEngineCtxDestroy` | `HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)` | 显式销毁 engine ctx（不调用则随 `HcclCommDestroy` 释放） |

### 2.5 Rank Graph（链路发现）

| API | 签名 | 用途 |
|:---|:---|:---|
| `HcclRankGraphGetLayers` | `HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)` | 获取网络层 |
| `HcclRankGraphGetLinks` | `HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)` | 查询指定 netLayer 下 srcRank 与 dstRank 之间的链路信息（建链时按 `linkAttr.linkProtocol` 匹配取 endpoint） |

---

## 3. 最小通用示例

```cpp
// Step 1: rank0 生成 rootInfo，通过带外通道（如 TCP）分发给所有 rank
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);              // rank0 执行；其他 rank 接收

// Step 2: 创建通信域
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclWorldRankID = rankId;
HcclComm comm = nullptr;
HcclCommInitRootInfoConfig(rankNum, &rootInfo, rankId, &config, &comm);

// Step 3: 建链（每对 rank 一条 channel）
HcclChannelDesc desc;
HcclChannelDescInit(&desc, 1);           // 强制：Acquire 前必须 Init
desc.remoteRank = peerRank;
desc.channelProtocol = COMM_PROTOCOL_UBC_CTP;
ChannelHandle channel = 0;
HcclChannelAcquire(comm, engine, &desc, 1, &channel);

// Step 4: 取通信 buffer 地址（本 rank / 对端 rank）
void* localBuf = nullptr;  uint64_t bufSize = 0;
HcclGetHcclBuffer(comm, &localBuf, &bufSize);
void* remoteBuf = nullptr; uint64_t remoteSize = 0;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuf, &remoteSize);

// Step 5: 下发上下文到 device
void* devCtx = nullptr;
HcclEngineCtxCreate(comm, "my_ctx", engine, ctxBytes, &devCtx);
HcclEngineCtxCopy(comm, engine, "my_ctx", &hostCtx, ctxBytes, 0);

// Step 6: 所有 rank 完成建链握手后（带外 barrier），再 launch kernel
```

> 可用性验证：以上 API 均以 `hccl_res.h`/`hccl_comm.h` 头文件存在为准（CANN 9.1.0 安装头文件已包含全部上述声明）。

---

## 4. 资源生命周期

| 资源 | 管理方式 |
|:---|:---|
| HCCL 内置 buffer（`HcclGetHcclBuffer`） | **库内管理，严禁 `aclrtFree`** |
| engine ctx（`HcclEngineCtxCreate` 返回的 device 内存） | HCCL engine 管理：随 `HcclCommDestroy` 释放，或显式 `HcclEngineCtxDestroy`；**不要 `aclrtFree`** |
| `HcclComm` | `HcclCommDestroy` |
| ACL 资源 | `aclrtDestroyStream` → `aclrtResetDevice` → `aclFinalize` |

**释放顺序约束**：`HcclCommDestroy` 必须在 `aclrtResetDevice` 之前（comm 依赖 device）。

---

## 5. 常见错误

### 错误1：ctxTag 冲突

同 `ctxTag`+同 engine 命中 `HcclEngineCtxGet` 会**直接复用已有 ctx**——这是特性，但不同通信域/不同用途的上下文共用 tag 会导致互相覆盖。规则：不同通信域用不同 ctxTag。

### 错误2：跳过 HcclChannelDescInit

`HcclChannelAcquire` 前必须 `HcclChannelDescInit`（官方 @warning 强制），否则描述符字段为随机值，建链行为未定义。

### 错误3：释放 HCCL 内置 buffer

```cpp
// ❌ 错误：释放 HCCL 内置 buffer
void* buffer; uint64_t size;
HcclGetHcclBuffer(comm, &buffer, &size);
aclrtFree(buffer);  // 严禁！库内管理
```

### 错误4：资源释放顺序错误

```cpp
// ❌ 错误：先 ResetDevice 再 DestroyComm
aclrtResetDevice(deviceId);     // device 已重置
HcclCommDestroy(comm);          // comm 依赖 device，崩溃！
```

### 错误5：对 engine ctx 调 aclrtFree

engine ctx 的 device 内存由 HCCL engine 分配管理，用 `HcclEngineCtxDestroy` 或随 `HcclCommDestroy` 释放；`aclrtFree` 不是其配对接口。

### 错误6：建链未完成即 launch kernel

所有 rank 的 channel 握手完成前 launch kernel，device 侧轮询远端地址会无限挂死。必须在 launch 前做一次跨 rank 的带外 barrier（如 TCP barrier）。

---

## 检查清单

- [ ] 使用 `HcclCommInitRootInfoConfig`（非 `HcclCommInitRootInfo`）
- [ ] `HcclChannelAcquire` 前已 `HcclChannelDescInit`
- [ ] `HcclChannelAcquire` 与 `HcclEngineCtx*` 使用同一 engine
- [ ] 不同通信域使用不同 ctxTag
- [ ] HCCL 内置 buffer 未手动释放
- [ ] engine ctx 用 `HcclEngineCtxDestroy` 或随 `HcclCommDestroy` 释放（未用 `aclrtFree`）
- [ ] kernel launch 在跨 rank 带外 barrier 之后
- [ ] `HcclCommDestroy` 在 `aclrtResetDevice` 之前

---

## 相关文档

- [api-hcomm.md](api-hcomm.md) — Hcomm device 侧通信原语（channel 的消费端）
- [api-crosscore-sync.md](api-crosscore-sync.md) — 跨核同步 API
