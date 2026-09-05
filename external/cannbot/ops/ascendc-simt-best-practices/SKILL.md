---
name: external-cannbot-ops-ascendc-simt-best-practices
description: AscendC SIMT 最佳实践与 API 导航。提供 SIMT 算子开发的实践经验总结和专有 API 分类索引：VF函数声明与调用、线程排布模式、索引计算API、数据搬入搬出、计算指令映射、多核同步、核内共享内存、TensorList处理、调测接口、内置宏等。触发：开发SIMT算子kernel代码、设计SIMT编程模式、查询SIMT
  API分类或线程排布时。
original-name: ascendc-simt-best-practices
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# AscendC SIMT 最佳实践

> 本 skill 提供基于实际算子开发经验提炼的 SIMT 编程范式总结，以及 SIMT 专有 API 的分类索引和速查。SIMT 基础概念（核函数、线程架构、内存层级）请通过 `ascendc-docs-search` skill 查阅 `$ASC_DEVKIT_DIR/docs/guide/编程指南/编程模型/AI-Core-SIMT编程/`。

---

## API 分类索引表

| 类别 | $ASC_DEVKIT_DIR 目录 | 功能概述 | 头文件 |
|------|-----------------|----------|--------|
| 总览与核函数 | `SIMT-API/概述.md` | VF启动机制、头文件、混合编程辅助函数 | common_functions.h |
| 同步与内存栅栏 | `SIMT-API/同步与内存栅栏/` | asc_syncthreads/asc_threadfence/asc_threadfence_block | device_sync_functions.h |
| 数学函数 | `SIMT-API/数学函数/` | 三角/指数/对数/幂/误差/特殊/基础算术/整数工具 | math_functions.h |
| 原子操作 | `SIMT-API/原子操作/` | add/sub/exch/max/min/inc/dec/cas/and/or/xor | device_atomic_functions.h |
| Warp函数 | `SIMT-API/Warp函数/` | all/any/ballot/activemask/shfl/shfl_up/down/xor/reduce | device_warp_functions.h |
| 地址空间谓词 | `SIMT-API/地址空间谓词函数/` | __isGlobal/__isShared/__isConstant/__isLocal | asc_simt.h |
| 地址空间转换 | `SIMT-API/地址空间转换函数/` | __cast_to_xxx 地址空间转换 | asc_simt.h |
| 访存函数 | `SIMT-API/访存函数/` | asc_ldcg/asc_ldca/asc_stcg/asc_stwt | device_functions.h |
| 协作组 | `SIMT-API/协作组/` | 协作组编程模型 | asc_simt.h |
| SIMT编程简介 | `SIMT-API/SIMT编程简介/` | SIMT编程概念和入门 | asc_simt.h |
| 混合编程简介 | `SIMT-API/SIMD与SIMT混合编程简介/` | SIMD/SIMT混合编程指导 | asc_simt.h |

> **查阅完整 API 文档**：使用 `ascendc-docs-search` skill 查阅 SIMT API 官方文档（`find "$ASC_DEVKIT_DIR/docs/api/" -name "{APIName}*.md"`）。

---

## 编程阶段分类索引

| 编程阶段 | 核心主题 | 参考文档 | 关键要点 |
|---------|---------|---------|---------|
| **函数定义** | VF 函数声明 | [vf-declaration.md](references/vf-declaration.md) | `__simt_vf__` 标记、LAUNCH_BOUND、参数类型 |
| **函数调用** | VF_CALL 启动 | [vf-call.md](references/vf-call.md) | Dim3 线程数、GM_ADDR 直接转换 |
| **线程排布** | 统一 for 循环模式 | [thread-stride-pattern.md](references/thread-stride-pattern.md) | blockIdx*threadNum+threadIdx 初始、threadNum*blockNum 步进 |
| **索引计算** | API 与全局索引 | [index-calculation.md](references/index-calculation.md) | GetThreadIdx/GetThreadNum/GetBlockIdx/GetBlockNum |
| **数据搬运** | GM 直接 / UB 中转 | [data-transfer.md](references/data-transfer.md) | 简单场景直接 GM、共享数据用 UB 中转 |
| **计算指令** | C++ 运算符映射 | [compute-operators.md](references/compute-operators.md) | 优先运算符写法、显式 API 仅特殊需求 |
| **多核同步** | workspace + SyncAll | [multi-core-sync.md](references/multi-core-sync.md) | SetScheduleMode(1)、SyncAll 全核同步 |
| **核内共享** | UB buffer 共享 | [intra-core-shared.md](references/intra-core-shared.md) | TBuf VECCALC、__ubuf__ 指针、原子操作 |
| **动态输入** | TensorList 解析 | [tensorlist-handling.md](references/tensorlist-handling.md) | VF 内解析 / ListTensorDesc 外部解析 |

---

## 核心编程范式速查

```cpp
// 1. VF 声明
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count, __gm__ T* x, __gm__ T* y);

// 2. VF 调用
AscendC::Simt::VF_CALL<OpSimt<T>>(AscendC::Simt::Dim3(512), count, (__gm__ T*)x, (__gm__ T*)y);

// 3. 线程排布 for 循环
for (uint64_t i = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
     i < count; i += AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum()) { ... }
```

---

## SIMT vs SIMD 差异速查

| 维度 | SIMT | SIMD |
|------|------|------|
| **编程模型** | 多线程并行，每线程独立执行 | 单核向量指令，整块数据并行 |
| **函数标记** | `__simt_vf__` | 无特殊标记 |
| **启动方式** | `Simt::VF_CALL<VF>(Dim3(N), args)` | 直接在 kernel 内调用 |
| **线程索引** | GetThreadIdx / GetBlockIdx | 无线程概念，靠 tile 循环 |
| **数据访问** | GM 直接读写（`__gm__ T*`） | DataCopyPad 搬到 UB/L0C |
| **计算方式** | C++ 运算符（`y[i]=x[i]+1`） | 向量 API（`Add(xLocal, yLocal, zLocal)`） |
| **核内共享** | UB buffer + `__ubuf__` 指针 | TQue 流水线 queue |
| **多核同步** | `SyncAll()` + `SetScheduleMode(1)` | 流水线 set/wait flag |
| **适用场景** | 逐元素、条件分支、不规则访问 | 规则向量、高吞吐归约 |

---

## 参考资料索引

`references/` 按需加载：

**编程最佳实践：**

- **vf-declaration.md** -- VF 函数声明规范：`__simt_vf__` 标记、LAUNCH_BOUND 约束、参数类型支持表
- **vf-call.md** -- VF_CALL 启动语法与完整示例：Add 算子模板、GM_ADDR 直接转换 vs GlobalTensor 中转
- **thread-stride-pattern.md** -- 线程排布统一 for 循环模式：索引计算分解、Select 算子示例
- **index-calculation.md** -- 索引计算 API 完整列表：GetThreadIdx/GetThreadNum/GetBlockIdx/GetBlockNum 返回类型与用法、头文件引用
- **data-transfer.md** -- 数据搬入搬出方案选择：GM 直接计算 vs UB 中转、tiling 侧 SetLocalMemorySize
- **compute-operators.md** -- 计算指令映射：C++ 运算符优先写法、显式 API 适用场景
- **multi-core-sync.md** -- 多核同步机制：workspace 交换、SyncAll、SetScheduleMode(1)、VF_CALL 间同步限制
- **intra-core-shared.md** -- 核内共享内存：TBuf 申请、__ubuf__ 指针传递、原子操作注意事项
- **tensorlist-handling.md** -- TensorList 动态输入处理：ListTensorDesc 内存布局、Host/Device 侧解析、VF 内解析 vs 外部解析选择建议

**API 导航与速查：**

- **overview.md** -- SIMT API 总览：VF启动机制、头文件映射、核函数定义、混合编程辅助函数速查
- **debug-api.md** -- 调测接口速查：printf/assert/__trap 使用要点
- **macro-api.md** -- 内置宏速查：特殊值(INF/NAN/MAX/MIN)、数学常数(pi/e/ln2等)
- **编译与运行.md** -- 异构编译场景编译与运行速查：SIMT/AI CPU 算子的 bisheng 命令行与 CMake 编译命令、`<<<...>>>` 内核调用符

**完整 API 文档：**

- **$ASC_DEVKIT_DIR** -- 通过 `ascendc-docs-search` skill 查阅 SIMT API 文档