# 小 shape 时延优化：通用代码骨架（按需替换占位符）

> 配套方法论见 [index.md](index.md)。

> 目的：把 `SKILL.md` 里的优化杆落成可复用骨架。所有 `<...>` 为占位符，迁移时替换。
> 平台：Ascend 950 / DAV_3510（arch35）为例；A2/A3 需按 `ascendc-api-best-practices` 核对 API 可用性。

## 1. host：独立 fast path + 门控

```cpp
// tiling_key.h
constexpr uint64_t FAST_TILING_KEY = 5000UL;

// CanUseFast：只对能安全套用的 shape 返回 true
bool OpTiling::CanUseFast() const {
    constexpr uint64_t U32_MAX = 0xFFFFFFFFULL;
    if (isZeroShape_ || rank_ == 0 || rank_ > RANK_MAX) return false;
    if (!(elemBytes_ == 1 || elemBytes_ == 2 || elemBytes_ == 4 || elemBytes_ == 8)) return false;
    if (totalOut_ == 0 || totalOut_ > U32_MAX || dimProduct_ > U32_MAX) return false;
    return true;
}

// DoOpTiling：fast 优先
if (isFast_) return DoFastOpTiling();
// ...
if (isFast_) { /* 关闭会改变元素位宽/布局的既有优化，避免破坏 fast 语义 */ }

uint64_t OpTiling::GetTilingKey() const {
    if (isFast_) return FAST_TILING_KEY;
    // ... 原 key
}
```

## 2. host：紧凑 header + 64B 分组复制

```cpp
constexpr uint32_t HDR_BYTES  = 64;
constexpr uint32_t LINE_BYTES = 64;
constexpr uint32_t GROUP      = 5;   // 一组不超过 5 核

struct OpFastHeader {                 // 目标 <= 64B，全部按实际最大值选位宽
    uint32_t blockDim;
    uint32_t groupCount;
    uint32_t outElems;
    uint32_t innerSize;               // 每个索引对应的连续输出长度
    uint32_t elemBytes;
    uint32_t idxBytes;
    uint32_t divShift;                // 内层维为 2 的幂时使用
    uint32_t divPow2;
    uint32_t dims[RANK_MAX];          // 仅保留 kernel 需要的维
};
static_assert(sizeof(OpFastHeader) <= HDR_BYTES, "fit one cache line");

ge::graphStatus OpTiling::DoFastOpTiling() {
    constexpr uint64_t MIN_ELEMS_PER_CORE = 128;              // 实测较优，按平台调
    uint64_t need = (totalOut_ + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    need = std::min<uint64_t>(need, coreNum_ > 0 ? coreNum_ : 1);
    if (need == 0) need = 1;

    hdr_.blockDim = (uint32_t)need;
    // L6 tiling 分组复制默认关闭（gather_nd 实测 -7%）；确需时改为 (blockDim<=GROUP)?blockDim:ceil(blockDim/GROUP) 并 A/B
    hdr_.groupCount = 1U;
    hdr_.outElems   = (uint32_t)totalOut_;
    hdr_.innerSize  = (uint32_t)innerSize_;
    hdr_.elemBytes  = (uint32_t)elemBytes_;
    hdr_.idxBytes   = (uint32_t)idxBytes_;
    hdr_.divPow2 = 0;
    uint32_t g = hdr_.innerSize;
    if ((g & (g - 1)) == 0) {                                  // 2 的幂 → shift
        uint32_t s = 0; while (g > 1) { g >>= 1; s++; }
        hdr_.divShift = s; hdr_.divPow2 = 1;
    }
    for (uint32_t k = 0; k < RANK_MAX; k++) hdr_.dims[k] = (k < rank_) ? (uint32_t)dims_[k] : 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpTiling::PostTiling() {
    if (!isFast_) {
        /* ... 原逻辑 ... */
        return ge::GRAPH_SUCCESS;   // 必须 return，避免落入下方 fast tiling 覆盖原路径
    }
    size_t cap = context_->GetRawTilingData()->GetCapacity();
    if (cap < static_cast<size_t>(HDR_BYTES) + LINE_BYTES) {
        return ge::GRAPH_FAILED;    // tiling buffer 太小，避免越界写
    }
    uint32_t maxGroups = static_cast<uint32_t>((cap - HDR_BYTES) / LINE_BYTES);
    if (hdr_.groupCount > maxGroups) hdr_.groupCount = maxGroups;   // 容量裁剪
    if (hdr_.groupCount == 0) hdr_.groupCount = 1;

    uint8_t* dst = static_cast<uint8_t*>(context_->GetRawTilingData()->GetData());
    std::memcpy(dst, &hdr_, sizeof(hdr_));                          // 首 64B
    uint32_t dims[RANK_MAX];
    for (uint32_t k = 0; k < RANK_MAX; k++) dims[k] = hdr_.dims[k];
    for (uint32_t g = 0; g < hdr_.groupCount; g++)                  // 64B 对齐复制
        std::memcpy(dst + HDR_BYTES + g * LINE_BYTES, dims, sizeof(dims));

    context_->GetRawTilingData()->SetDataSize(HDR_BYTES + hdr_.groupCount * LINE_BYTES);
    context_->SetBlockDim(hdr_.blockDim);
    return ge::GRAPH_SUCCESS;
}
```

## 3. kernel：TPipe-free + C API + 标量读 tiling + 分组

```cpp
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/sys_var/sys_var.h"

constexpr uint32_t THREADS = 1024;   // 上限；实测 1024 常优于 2048
constexpr uint32_t MIN_THREADS = 64;
constexpr uint32_t HDR_BYTES = 64, LINE_BYTES = 64, GROUP = 5;

struct OpFastHeader {
    uint32_t blockDim, groupCount, outElems, innerSize, elemBytes, idxBytes, divShift, divPow2;
    uint32_t dims[RANK_MAX];
};

__aicore__ inline void OpFastDispatch(GM_ADDR in, GM_ADDR idx, GM_ADDR out, GM_ADDR tiling) {
    uint32_t bid = (uint32_t)asc_get_block_idx();
    const __gm__ OpFastHeader* hdr = reinterpret_cast<const __gm__ OpFastHeader*>(tiling);  // 标量读
    // 默认单份：直接读 tiling + HDR_BYTES。
    // 若开启 L6 分组复制，则改为：
    //   uint32_t grp = (hdr->blockDim <= GROUP) ? bid : (bid / GROUP);
    //   if (grp > hdr->groupCount - 1) grp = hdr->groupCount - 1;
    //   dims = tiling + HDR_BYTES + grp * LINE_BYTES;
    const __gm__ uint32_t* dims = reinterpret_cast<const __gm__ uint32_t*>(tiling + HDR_BYTES);
    // 按 elemBytes 分发到模板
    // OpFastRun<uint8_t/int32_t/...>(in, idx, out, hdr, dims, bid);
}
```

### per-output-element（合并访问）VF 模板

```cpp
template <typename T1, typename T2, uint32_t RANK>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void OpFastVf(
    __gm__ T1* __restrict outGm, const __gm__ T1* __restrict inGm, const __gm__ T2* __restrict idxGm,
    uint32_t begin, uint32_t end, uint32_t innerSize, uint32_t divShift, uint32_t divPow2,
    uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,   // 维
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)   // 由维现算的 stride
{
    for (uint32_t g = begin + threadIdx.x; g < end; g += blockDim.x) {
        uint32_t outer = (divPow2 != 0u) ? (g >> divShift) : (g / innerSize);
        uint32_t inner = g - outer * innerSize;
        // 读 idxGm[outer*RANK + k]，累加 srcBase（含边界检查/负索引）
        // outGm[g] = inGm[srcBase * innerSize + inner];
    }
}
```

### 自适应线程数

```cpp
uint32_t active  = end - begin;
uint32_t threads = (active + MIN_THREADS - 1) / MIN_THREADS * MIN_THREADS;
if (threads > THREADS) threads = THREADS;
asc_vf_call<OpFastVf<T1, T2, RANK>>(dim3(threads), ...);
```

## 4. 入口：fast 分支置于 TPipe 之前

```cpp
extern "C" __global__ __aicore__ void op(GM_ADDR in, GM_ADDR idx, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) return;
    SetSysWorkspace(workspace);
    if (GetUserWorkspace(workspace) == nullptr) return;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(FAST_TILING_KEY)) { OpFastDispatch(in, idx, out, tiling); return; }

    TPipe pipeIn;                 // 原路径
    // ... 原逻辑
}
```

## 5. 空核 floor 探针（仅诊断，务必回滚）

```cpp
    if (GetUserWorkspace(workspace) == nullptr) return;
    return;   // DIAG_FLOOR —— 测完立即删除
```

## 6. A/B 对比脚本

见 `SKILL.md` 的「A/B 测量协议」一节，可直接复用。
