# 阶段 3：API 使用优化 — 详细参考

## 3.1 TPipe 在 kernel 类外创建

`TPipe` 作为类成员时，初始化会设置全局 TPipe 指针，编译器认为类内存空间有被
外部污染的风险，因此**放弃对类内 Scalar 变量的常量折叠和常量传播优化**。

**反例** — TPipe 为类成员：

```cpp
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(...) {
        pipe.InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
    }
private:
    TPipe pipe;       // ← 在类内部，阻止 Scalar 优化
};

extern "C" __global__ __aicore__ void example_kernel(...) {
    KernelExample<float> op;
    op.Init(...);
}
```

**正例** — TPipe 在类外创建，以指针传入：

```cpp
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(..., TPipe* pipeIn) {
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
    }
private:
    TPipe* pipe;      // ← 仅存指针，类内存空间干净
};

extern "C" __global__ __aicore__ void example_kernel(...) {
    TPipe pipe;                     // ← 在类外创建
    KernelExample<float> op;
    op.Init(..., &pipe);
}
```

实测：平均 scalar_time 从 281 us 降至 236 us（**−17%**），scalar_time 占比从
21% 降至 17%。**任何场景**都建议使用此优化，scalar bound 场景收益尤为明显。

## 3.2 纯搬运算子使用 TQueBind

纯搬运算子不涉及 Vector 计算，标准 VECIN→VECOUT 模式会引入一次冗余的
LocalTensor→LocalTensor DataCopy。

**反例** — 冗余的 Vector 拷贝：

```cpp
TQue<QuePosition::VECIN, BUFFER_NUM> QueI;
TQue<QuePosition::VECOUT, BUFFER_NUM> QueO;

auto iLocal = QueI.AllocTensor<ComputeT>();
DataCopy(iLocal, inGm[i * 32], size);
QueI.EnQue(iLocal);
auto iLocal2 = QueI.DeQue<ComputeT>();
for (int j = 0; j < jLen; ++j) {
    auto oLocal = QueO.AllocTensor<ComputeT>();
    DataCopy(oLocal, iLocal2, size);   // LocalTensor → LocalTensor，浪费 Vector
    QueO.EnQue(oLocal);
    auto oLocal2 = QueO.DeQue<ComputeT>();
    DataCopyPad(outGm[j], oLocal2, ...);
    QueO.FreeTensor(oLocal2);
}
QueI.FreeTensor(iLocal2);
```

**正例** — TQueBind 消除冗余拷贝：

```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind;

auto bindLocal = queBind.AllocTensor<ComputeT>();
DataCopy(bindLocal, inGm[i * 32], size);
queBind.EnQue(bindLocal);
auto bindLocal2 = queBind.DeQue<ComputeT>();
for (int j = 0; j < len; ++j) {
    DataCopyPad(outGm[j], bindLocal2, ...);
}
queBind.FreeTensor(bindLocal2);
```

效果：`aiv_vec_time` 降至约 0。

## 3.3 Counter 模式（SetMaskCount）

Normal 模式需要手动计算主块/尾块的 mask 和迭代次数，涉及大量 Scalar 开销。
Counter 模式直接传入总元素数，硬件自动推断迭代次数。

**反例** — Normal 模式（half 类型，15000 个元素）：

```cpp
uint32_t ELE_SIZE = 15000;
AscendC::BinaryRepeatParams binaryParams;
uint32_t numPerRepeat    = 256 / sizeof(DTYPE_X);   // half → 128
uint32_t mainRepeatTimes = ELE_SIZE / numPerRepeat;  // 117
uint32_t tailEleNum      = ELE_SIZE % numPerRepeat;  // 24

AscendC::SetMaskNorm();
AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::NORMAL>(numPerRepeat);
AscendC::Add<DTYPE_X, false>(zLocal, xLocal, yLocal,
    AscendC::MASK_PLACEHOLDER, mainRepeatTimes, binaryParams);
if (tailEleNum > 0) {
    AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::NORMAL>(tailEleNum);
    AscendC::Add<DTYPE_X, false>(
        zLocal[mainRepeatTimes * numPerRepeat],
        xLocal[mainRepeatTimes * numPerRepeat],
        yLocal[mainRepeatTimes * numPerRepeat],
        AscendC::MASK_PLACEHOLDER, 1, binaryParams);
}
AscendC::ResetMask();
```

**正例** — Counter 模式，一次调用：

```cpp
uint32_t ELE_SIZE = 15000;
AscendC::BinaryRepeatParams binaryParams;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::COUNTER>(ELE_SIZE);
AscendC::Add<DTYPE_X, false>(zLocal, xLocal, yLocal,
    AscendC::MASK_PLACEHOLDER, 1, binaryParams);
AscendC::ResetMask();
```

当多个 Vector 指令处理相同元素数量时，Counter 模式优势更明显——无需反复
计算不同的主块/尾块 mask。

## 3.4 Matmul AtomicAdd

Matmul 结果 C(m,n) 需要与 GM 上矩阵 D(m,n) 相加时，可在搬出路径上融合。

**反例** — 手动搬运后在 UB 做 Add：

```cpp
mm.IterateAll(local_c);

DataCopy(local_d, gm_d, d_size);
event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
Add(local_d, local_d, local_c, d_size);
DataCopy(gm_d, local_d, d_size);
```

**正例** — AtomicAdd 融合到搬出：

```cpp
mm.IterateAll(gm_d, 1);          // enAtomic = 1
// 或在 Iterate 循环中：
// mm.GetTensorC(gm_d, 1);       // enAtomic = 1
```

M=64, N=256, K=256 实测：平均 cycle 从 154181 降至 135054（**−12.4%**）。

## 3.5 归约指令组合

将连续 buffer 全部累加为一个标量的场景：

| 方案 | 指令数 | 相对速度 |
|---|---|---|
| 2× WholeReduceSum | 2 | 最慢（WholeReduceSum 单条较慢） |
| 3× BlockReduceSum | 3 | 中等 |
| **1× BlockReduceSum + 1× WholeReduceSum** | **2** | **最快** |

推荐模式（float, shape=256）：

```cpp
static constexpr uint32_t BLK_LEN = 32;
TBuf<QuePosition::VECCALC> calcBuf;
pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
constexpr uint32_t c0Count = BLK_LEN / sizeof(float);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;

AscendC::SetMaskCount();
AscendC::SetVectorMask<float>(0, totalLength);
AscendC::BlockReduceSum<float, false>(tempTensor1, xLocal,
    AscendC::MASK_PLACEHOLDER, 1,
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();

AscendC::SetVectorMask<float>(0, blockNum0);
AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1,
    AscendC::MASK_PLACEHOLDER, 1,
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
```
