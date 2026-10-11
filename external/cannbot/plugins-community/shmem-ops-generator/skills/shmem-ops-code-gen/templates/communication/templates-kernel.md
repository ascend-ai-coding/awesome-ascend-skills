# Communication 算子 — Device Kernel 模板

> **阅读时机**：渐进式生成步骤 2–4（lifecycle/memory/transport/sync/compute）时读取本文件。
> 索引与约束见 [GUIDE.md](GUIDE.md)。

**统一实现（MUST）**：默认 **single path** 一套 kernel 逻辑 + UB 内分块（见 [code-patterns.md §6.4](../../references/code-patterns.md)）。**禁止**未证明 ≥5% 收益就生成 `*_small`/`*_big`、`*_small_data`/`*_big_data` 并行路径；外层 tile 仅因 GVA/symm 容量约束保留。

---

## src/<op_name>_kernel.h → `<op_name>/src/<op_name>_kernel.h`

```cpp
// ============================================================
// SHMEM Communication 算子 kernel 声明模板
// 使用: 写入 <op_name>/src/<op_name>_kernel.h
//       替换 <op_name>/<OpName> 占位符
// ============================================================

#ifndef <OP_NAME>_KERNEL_H
#define <OP_NAME>_KERNEL_H

#include <cstdint>

template <class T>
void <op_name>_kernel(uint32_t block_dim, void *stream, uint64_t fftsAddr,
                      uint8_t *input, uint8_t *output, uint8_t *symmetric,
                      int elements, int magic);

#endif // <OP_NAME>_KERNEL_H
```

---

## src/<op_name>_kernel.cpp → `<op_name>/src/<op_name>_kernel.cpp`

```cpp
// ============================================================
// SHMEM Communication 算子 Device kernel 模板
// 来源: shmem/examples/allgather/allgather_kernel.cpp 简化
// 使用: 复制到 <op_name>/src/<op_name>_kernel.cpp
//       替换 <op_name>/<OpName> 占位符，实现具体 phase 逻辑
// ============================================================

#include "<op_name>_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
// 性能调试时可选（默认关闭）：
// #include "utils/prof/shmemi_prof.h"   // 仅 SHMEMI_PROF_* 宏；勿调用 aclshmemi_get_state

#undef inline
#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#define inline inline attribute((always_inline))

using namespace AscendC;
using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;

// ==== 定制：按 design.md 的 schedule.phases 实现具体通信逻辑 ====
template <typename T>
ACLSHMEM_DEVICE void <op_name>_impl(uint64_t fftsAddr,
                                     __gm__ T *input,
                                     __gm__ T *output,
                                     __gm__ T *symmetric,
                                     int elements,
                                     int magic)
{
    util_set_ffts_config(fftsAddr);

    const int64_t aivNum = GetBlockNum();
    const int64_t aivIndex = GetBlockIdx();
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    // ---- 分核 ----
    // ==== 定制：按 schedule.core_partition 划分 sender / receiver / sync 核 ====
    // 示例：前半核为 sender（本地 → symmetric），后半核为 receiver（远端 symmetric → 本地）
    int core_group_num = aivNum / 2;
    bool is_sender = (aivIndex < core_group_num);

    // Symmetric buffer 布局
    __gm__ int32_t *sync_flags = (__gm__ int32_t *)symmetric;
    __gm__ T *data_buf = (__gm__ T *)(sync_flags + aivNum * SYNC_FLAG_INTERVAL);

    // UB 临时缓冲（ping-pong）
    __ubuf__ T *ping_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(1024 + 32));
    __ubuf__ T *pong_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(96 * 1024 + 32));

    uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t ub_elems = ub_size / sizeof(T);

    if (is_sender) {
        // ==== Phase: 本地 GM → symmetric (sender 核) ====
        // ==== 定制：按 design.md 的 phase 逻辑调整 put 目标和 chunk 策略 ====
        int64_t per_core = elements / core_group_num;
        int64_t my_offset = aivIndex * per_core;
        if (aivIndex == core_group_num - 1) {
            per_core = elements - my_offset;
        }

        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int64_t chunk_count = 0;

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size) ? ub_elems : (remaining / sizeof(T));

            SHMEMI_PROF_START(0);  // copy_in phase
            aclshmemx_mte_put_nbi(data_buf + my_offset + done_elems,
                                  input + my_offset + done_elems,
                                  ping_buf, ub_size, copy_elems,
                                  my_rank, EVENT_ID0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SHMEMI_PROF_END(0);

            chunk_count++;
            int64_t flag_val = chunk_count + magic;
            aclshmemx_signal_op(sync_flags + flag_offset, flag_val,
                                ACLSHMEM_SIGNAL_SET, my_rank);

            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
        }
    } else {
        // ==== Phase: 远端 symmetric → 本地 output (receiver 核) ====
        // ==== 定制：按 design.md 的 peer_model 和 addressing 调整 get 来源 ====
        int receiver_idx = aivIndex - core_group_num;
        // 示例：每个 receiver 核从一个远端 PE 拉取数据
        int64_t src_pe = receiver_idx;  // ==== 定制：映射逻辑 ====

        // 等待远端 sender 完成后 get
        // ==== 定制：按 schedule.phases 的 dependency 和 sync 调整等待逻辑 ====
        aclshmem_signal_wait_until(
            (__gm__ int32_t *)aclshmem_ptr(sync_flags, src_pe) + flag_offset,
            ACLSHMEM_CMP_GE, magic + 1);

        int64_t per_core = elements;
        int64_t dst_offset = src_pe * elements;
        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int pingpong = 0;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size / 2) ? (ub_size / 2 / sizeof(T))
                                                            : (remaining / sizeof(T));
            TEventID eid = (pingpong == 0) ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ T *buf = (pingpong == 0) ? ping_buf : pong_buf;

            WaitFlag<HardEvent::MTE3_MTE2>(eid);

            SHMEMI_PROF_START(1);  // remote_get phase
            aclshmemx_mte_get_nbi(output + dst_offset + done_elems,
                                  data_buf + done_elems,
                                  buf, ub_size / 2, copy_elems,
                                  src_pe, eid);
            SHMEMI_PROF_END(1);

            SetFlag<HardEvent::MTE3_MTE2>(eid);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
            pingpong = 1 - pingpong;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

// ---- Kernel 入口（按 dtype 展开）----
// ==== 定制：按 design.md 的 dtypes 调整支持的类型 ====
#define <OP_NAME>_FUNC_DEF(type)                                                    \
    extern "C" __global__ __aicore__ void Shmem<OpName>_##type(                     \
        uint64_t fftsAddr, GM_ADDR input, GM_ADDR output,                           \
        GM_ADDR symmetric, int elements, int magic)                                 \
    {                                                                               \
        <op_name>_impl<type>(fftsAddr, (__gm__ type *)input,                        \
                             (__gm__ type *)output, (__gm__ type *)symmetric,        \
                             elements, magic);                                      \
    }

<OP_NAME>_FUNC_DEF(int32_t);
<OP_NAME>_FUNC_DEF(float16_t);
<OP_NAME>_FUNC_DEF(bfloat16_t);

// ---- Host-callable wrapper ----
template <class T>
void <op_name>_kernel(uint32_t block_dim, void *stream, uint64_t fftsAddr,
                      uint8_t *input, uint8_t *output, uint8_t *symmetric,
                      int elements, int magic)
{
    if (std::is_same<T, int32_t>::value) {
        Shmem<OpName>_int32_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    } else if (std::is_same<T, fp16_t>::value) {
        Shmem<OpName>_float16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    } else if (std::is_same<T, bf16_t>::value) {
        Shmem<OpName>_bfloat16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    }
}

// ==== 定制：按实际使用的 dtype 显式实例化 ====
template void <op_name>_kernel<int32_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
template void <op_name>_kernel<fp16_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
template void <op_name>_kernel<bf16_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
```
