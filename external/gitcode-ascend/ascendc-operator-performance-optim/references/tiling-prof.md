# 阶段 1：Tiling 优化 — 详细参考

## 1.1 多核切分

通过 `context->SetBlockDim(BLOCK_DIM)` 设置算子使用的核数。

| 架构 | 设置规则 |
|---|---|
| 耦合架构（Vector+Cube 一体） | `blockDim` = `GetCoreNumAiv()` 或 `GetCoreNumAic()` |
| 分离架构 — 纯 Vector 算子 | `blockDim` = AIV 核数（如 40） |
| 分离架构 — 纯 Cube 算子 | `blockDim` = AIC 核数（如 20） |
| 分离架构 — MIX（V+C）算子 | `blockDim` = 物理核组数（如 20 = 40 AIV / 2），**不可超过物理核数** |

`blockDim` 为逻辑核概念，取值范围 [1, 65535]。为充分利用硬件资源，一般设为
物理核数或其整数倍。AIC/AIV 核数分别通过 `GetCoreNumAic()` 和 `GetCoreNumAiv()`
获取。

## 1.2 L2Cache 切分

当 `输入数据量 + 输出数据量 > L2Cache 容量`（如 192 MB）时，将数据按 L2Cache
大小等分为多块，所有核协同处理同一块后再切换下一块。这样重复读取时可命中
L2Cache（~7 TB/s），避免频繁访问 HBM（~1.6 TB/s）。

**反例** — 未使能 L2Cache 切分，每个核的两个 tile 互相挤占 L2Cache：

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM;

class KernelSample {
public:
    __aicore__ inline void Init(GM_ADDR x) {
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process() {
        constexpr int32_t loopCount = 2;
        for (int32_t i = 0; i < loopCount; i++) {
            for (int32_t j = 0; j < TILE_NUM; j++) {
                CopyIn(j);    // 每个核读两个 tile，L2Cache 被反复淘汰
                Compute();
                CopyOut(j);
            }
        }
    }
};
```

**正例** — 使能 L2Cache 切分，外层循环按 L2Cache 分块，所有核协同处理：

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t TILE_NUM = 2;
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_LENGTH = TOTAL_LENGTH / TILE_NUM;
constexpr int32_t BLOCK_LENGTH = TILE_LENGTH / USE_CORE_NUM;

class KernelSample {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t index) {
        xGm.SetGlobalBuffer(
            (__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx() + index * TILE_LENGTH,
            BLOCK_LENGTH);
    }
    __aicore__ inline void Process() {
        constexpr int32_t loopCount = 2;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn();       // 每个核只读自己的切片，第二次读命中 L2Cache
            Compute();
            CopyOut();
        }
    }
};

extern "C" __global__ __aicore__ void simple_kernel(
    __gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    AscendC::KernelAdd op;
    for (int32_t i = 0; i < TILE_NUM; i++) {
        op.Init(srcGm, dstGm, i);
        op.Process();
    }
}
```

## 1.3 核间负载均衡

L2Cache 切分后，若每次计算所需块数不能被核数整除，则部分核会多分配尾块。

**问题**：核 1–5 每次 pass 多算一个块，始终最后完成，核 6–20 空等。

**解决方法**：在不同 pass 间交替分配尾块。例如 2 个 pass × 25 块 / 20 核，
pass 1 的尾块分配给核 1–5，pass 2 的尾块分配给核 6–10，全局来看核 1–10
各算 3 块，核 11–20 各算 2 块，达到全局负载均衡。
