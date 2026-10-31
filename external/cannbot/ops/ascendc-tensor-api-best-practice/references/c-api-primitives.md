# C API 辅助能力速查（初始化 / 同步 / 锁 / 原子 / 缓存）

本参考在 Step 2/3（API 调查、Kernel Design）与实现节点按 implementation_route=tensor_api_custom 条件加载。补齐 `api-dataflow-layout.md` 只列入口名、`sync-and-lifecycle.md` 只讲约束的空隙：给出 C API 的**签名与调用形态**。

> 原则：Tensor API 做 Tensor/Layout/Copy/Mmad，C API 做初始化、同步、锁、原子、缓存控制。签名以 `~/asc-devkit` 当前头文件为准，本文只内化稳定入口与常见形态。

## 0. 头文件入口

```cpp
#include "c_api/asc_simd.h"    // 聚合入口（同步 / 部分 utils）
#include "c_api/sync/sync.h"   // 同步 + 锁
#include "c_api/utils/sys_init.h"   // asc_init
#include "c_api/utils/sys_var.h"    // 核 ID / 核数
#include "c_api/cache_ctrl/cache_ctrl.h"  // 缓存控制
#include "c_api/atomic/atomic.h"    // 原子操作
```

## 1. 初始化

```cpp
__aicore__ inline void asc_init();   // kernel 第一条，缺失即门禁失败
```

## 2. 核 ID / 核数（`utils/sys_var.h`）

```cpp
__aicore__ inline int64_t asc_get_block_idx();   // 核 ID（block_idx 的 deprecated 新名，POV 至 2027/09/07）
__aicore__ inline int64_t asc_get_block_num();   // 核数（block_num 同理）
__aicore__ inline int64_t asc_get_sub_block_num(); // Mix(1,2) 的 Vector 子核数
__aicore__ inline int64_t asc_get_sub_block_id(); // Mix(1,2) 的 Vector 子核 ID
__aicore__ inline int64_t asc_get_system_cycle(); // 系统周期计数（计时用）
__aicore__ inline int64_t asc_get_system_clock();
```

- 纯 Vector / 纯 Cube / Mix(1,1)：直接用 `block_idx`。
- Mix(1,2)：Cube 核用 `block_idx`；Vector 核用 `block_idx * asc_get_sub_block_num() + asc_get_sub_block_id()`。

## 3. 同步（`sync/sync.h`）

```cpp
__aicore__ inline void asc_sync_pipe(pipe_t pipe);                       // 排空指定流水在途指令
__aicore__ inline void asc_sync_notify(pipe_t pipe, pipe_t tpipe, event_t id); // 事件通知
__aicore__ inline void asc_sync_wait(pipe_t pipe, pipe_t tpipe, event_t id);   // 事件等待
__aicore__ inline void asc_sync_vec(int id);                             // Vector 流水浅同步
__aicore__ inline void asc_sync_mte2(int id);
__aicore__ inline void asc_sync_mte3(int id);
__aicore__ inline void asc_sync();                                       // 全流水 sync（deprecated 见下）
__aicore__ inline void asc_sync_data_barrier(mem_dsb_t arg);             // 数据屏障

// 跨核/块级同步（多核协作、AIC/AIV 交接）：
__aicore__ inline void asc_sync_block_arrive(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_block_wait(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_subblock_arrive(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_subblock_wait(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_inter_arrive(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_inter_wait(pipe_t pipe, int64_t flag_id);
__aicore__ inline void asc_sync_intra_arrive(pipe_t pipe, uint64_t sync_id);
__aicore__ inline void asc_sync_intra_wait(pipe_t pipe, uint64_t sync_id);
```

- `pipe_t` 为作用域枚举，成员含 `PIPE_MTE1/PIPE_MTE2/PIPE_MTE3/PIPE_V/PIPE_M/PIPE_FIX/PIPE_S/PIPE_ALL/PIPE_NONE`（以编译器内建枚举为准）；`event_t` 为事件 ID 类型。
- `asc_sync_notify(pipe, tpipe, id)` / `asc_sync_wait(pipe, tpipe, id)` 必须**成对**、同 ID；事件 ID 禁止用 6/7（见 `sync-and-lifecycle.md`）。

## 4. 锁（`sync/sync.h`）

```cpp
__aicore__ inline void asc_lock(pipe_t pipe, uint8_t mutex_id);
__aicore__ inline void asc_lock(pipe_t pipe, uint8_t mutex_id, const asc_mutex_execute_mode mode);
__aicore__ inline void asc_unlock(pipe_t pipe, uint8_t mutex_id);
__aicore__ inline void asc_unlock(pipe_t pipe, uint8_t mutex_id, const asc_mutex_execute_mode mode);
```

- `asc_lock/unlock` 同 pipe、同 mutex_id、同 mode 成对，不嵌套；用于保护同一流水内 buffer 访问时序（见 `kernel-skeleton.md` §4）。
- 单流水内顺序约束用对应 pipe 同步（如 `asc_sync_vec`），不用连续 lock/unlock 模拟。

## 5. 缓存控制（`cache_ctrl/cache_ctrl.h`）

```cpp
__aicore__ inline void asc_icache_preload(const void* addr, int64_t prefetch_len); // 指令缓存预取
__aicore__ inline void asc_dcci_single(__gm__ void* dst);       // 单地址 DCache 一致性
__aicore__ inline void asc_dcci_entire_ub();                    // 整 UB 一致性
__aicore__ inline void asc_dcci_entire_all();
__aicore__ inline void asc_datacache_preload(__gm__ uint64_t* address, int64_t offset); // 数据预取
__aicore__ inline void asc_dci();
__aicore__ inline void asc_set_scalar_cache_mode(asc_load_l2_cache_mode m);  // 置 L2 cache 读写模式
__aicore__ inline void asc_set_scalar_cache_mode(asc_store_l2_cache_mode m);
```

- L2 CacheMode 与 `copy`/`mmad` 的 cache hint 对应；搬运优化见 `performance-analysis.md` §4 内存访问分类。

## 6. 原子（`atomic/atomic.h`）

```cpp
// 标量原子（scalar_atomic.h）
__aicore__ inline int32_t asc_atomic_add(__gm__ int32_t* address, int32_t val); // 还有 float/uint64 等重载
__aicore__ inline int32_t asc_atomic_sub(__gm__ int32_t* address, int32_t val);
__aicore__ inline int32_t asc_atomic_exch(__gm__ int32_t* address, int32_t val);
// ...（另有 max/min/cas，type 覆盖 int32/uint32/float/int64/uint64）

// DMA 原子（datamove_atomic.h）：先配置类型再操作
__aicore__ inline void asc_set_atomic_add_float();
__aicore__ inline void asc_set_atomic_add_int32();
__aicore__ inline void asc_disable_dma_atomic();
// ...
```

## 7. 事件同步最小示例（跨块流水重叠）

```cpp
constexpr uint8_t event_id_0 = 0;   // 事件 ID 用 0~5（禁用 6/7）
// 预取下一块的搬入，再算当前块（Double Buffer 时序）：
asc_sync_notify(pipe_t::PIPE_MTE2, pipe_t::PIPE_V, event_id_0);   // 搬入完成 → 通知 Vector
asc_sync_wait(pipe_t::PIPE_V, pipe_t::PIPE_MTE2, event_id_0);     // Vector 等搬入完成
```

完整 owner 闭合规则见 `sync-and-lifecycle.md`；Double Buffer 实现见 `vector-performance-patterns.md` §6。

## 门禁

1. `asc_init()` 首、`asc_sync_pipe(PIPE_ALL)` 尾。
2. `notify/wait`、`lock/unlock` 严格成对、同 ID。
3. 事件 ID 禁用 6/7；具体 ID 范围、`pipe_t`/`event_t` 枚举、每个 C API 的模板约束以当前头文件为准，不凭记忆。

## 事实源

- `$ASC_DEVKIT_ROOT/include/c_api/{asc_simd.h, sync/sync.h, utils/sys_init.h, utils/sys_var.h, cache_ctrl/cache_ctrl.h, atomic/atomic.h}`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。