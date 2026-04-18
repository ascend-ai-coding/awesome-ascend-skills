# 阶段 4：内存优化 — 详细参考

## 存储层级一览

| Buffer | 用途 | 说明 |
|---|---|---|
| GM（HBM） | 全局内存 | 带宽约 1.6 TB/s |
| L2Cache | 共享缓存 | 约 192 MB，带宽约 7 TB/s |
| L1 Buffer | AI Core 本地存储 | Cube 数据中转 |
| L0A / L0B | Cube 输入 | 由 L1 加载 |
| L0C（CO1） | Cube 输出 | 支持原地累加 |
| UB（Unified Buffer） | Vector 输入/输出 | VECIN, VECOUT, VECCALC |
| BT Buffer（C2） | Bias 表 | 仅分离架构 |
| FP Buffer（C2PIPE2GM） | Fixpipe 参数 | 仅分离架构 |

## 4.1 UB Buffer 融合

连续多次 Vector 运算时，将中间结果保留在 UB 上，不经 GM 往返。
n 次连续运算的 GM 搬运次数从 `2n` 降为 `2`。

**反例** — 每次运算都经 GM 往返（Exp + Abs 需 4 次 GM 搬运）：

```cpp
class KernelSample {
    __aicore__ inline void Process() {
        CopyIn();       // GM → UB
        Compute();      // Exp
        CopyOut();      // UB → GM
        CopyIn1();      // GM → UB（重新读回 Exp 结果）
        Compute1();     // Abs
        CopyOut1();     // UB → GM
    }
};
```

**正例** — 在 UB 内链式计算（仅 2 次 GM 搬运）：

```cpp
class KernelSample {
    __aicore__ inline void Compute() {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> dstLocal  = outQueueDst.AllocTensor<float>();
        Exp(dstLocal, src0Local, 1024);
        Abs(dstLocal, dstLocal, 1024);   // 原地操作，留在 UB
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    __aicore__ inline void Process() {
        CopyIn();       // GM → UB（一次）
        Compute();      // Exp + Abs 融合
        CopyOut();      // UB → GM（一次）
    }
};
```

## 4.2 L0C 累加矩阵乘

`A1*B1 + A2*B2 + ...` 场景下，利用 Mmad 的内建累加功能将部分结果保留在
CO1（L0C）中。避免每次矩阵乘结果都 CO1→GM→UB 再做 Add。

**反例** — 逐次搬出后在 UB 求和：

```cpp
void Process() {
    Compute();       // Mmad → CO1
    CopyOut();       // CO1 → workspace（GM）
    CopyIn1();       // workspace → UB

    Compute1();      // Mmad → CO1
    CopyOut1();      // CO1 → workspace（GM）
    CopyIn2();       // workspace → UB

    Compute2();      // Add(result1, result2) in UB
    CopyOut2();      // UB → GM
}
```

**正例** — 在 L0C 中原地累加：

```cpp
void Compute() {
    MmadParams mmadParams;
    mmadParams.m = m;  mmadParams.n = n;  mmadParams.k = k;
    Mmad(c1Local, a2Local_1, b2Local_1, mmadParams);
    mmadParams.cmatrixInitVal = false;
    Mmad(c1Local, a2Local_2, b2Local_2, mmadParams);  // 在 CO1 原地累加
}
// 最后一次 CopyOut: CO1 → GM
```

## 4.3 小矩阵长驻 L1

当 L1 无法同时容纳左右矩阵（如左矩阵 992K、右矩阵 16K、L1 容量 512K）时，
将较小矩阵一次加载后常驻 L1，仅循环搬运较大矩阵。

**反例** — 每次迭代都重新加载两个矩阵：

```cpp
void Process() {
    for (uint32_t i = 0; i < 2; i++) {
        CopyInA1(i);      // 加载左矩阵切片
        SplitA();
        for (uint32_t j = 0; j < 2; j++) {
            CopyInB1(j);  // 每次都重新加载右矩阵
            SplitB();
            Compute(i, j);
        }
    }
    CopyOut();
}
```

**正例** — 右矩阵一次加载，仅循环搬运左矩阵：

```cpp
void Process() {
    CopyInB1();          // 右矩阵一次全载入 L1
    SplitB();            // L1 → L0B
    for (uint32_t i = 0; i < 2; i++) {
        CopyInA1(i);     // 循环加载左矩阵切片
        SplitA();
        for (uint32_t j = 0; j < 2; j++) {
            Compute(i, j);  // 右矩阵已在 L0B
        }
    }
    CopyOut();
}
```

2 个左矩阵切片时：搬运次数从 4+4=8 降为 1+2=3。

## 4.4 BT Buffer 存放 bias（分离架构）

将 bias 存入 BT Buffer（C2），在 Mmad 中一步融合 bias 加法，避免
CO1→GM→UB→Add→GM 的冗长路径。

**反例** — 在 UB 中单独做 bias Add：

```cpp
TQue<QuePosition::VECIN, 1> inQueueBias;
// Mmad 后：CO1 → workspace(GM) → UB
// bias: GM → UB
// Add(matmul_result, bias) in UB → GM
```

**正例** — 通过 BT Buffer 融合：

```cpp
TQue<QuePosition::C1, 1> inQueueC1;    // L1
TQue<QuePosition::C2, 1> outQueueC2;   // BT Buffer

void SplitBias() {
    LocalTensor<float> bias1Local = inQueueC1.DeQue<float>();
    LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();
    // L1 → BT
    DataCopy(bias2Local, bias1Local, {1, (uint16_t)(n * sizeof(float) / 64), 0, 0});
    outQueueC2.EnQue<float>(bias2Local);
    inQueueC1.FreeTensor(bias1Local);
}

void Compute() {
    LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
    MmadParams mmadParams;
    mmadParams.m = m;  mmadParams.n = n;  mmadParams.k = k;
    mmadParams.cmatrixInitVal = false;
    Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);  // 融合 bias
    outQueueC2.FreeTensor(bias2Local);
}
```

## 4.5 FP Buffer 存放量化参数（分离架构）

将量化参数存入 FP Buffer（C2PIPE2GM），通过 Fixpipe 在搬出路径上随路量化。
避免 CO1→GM→UB→量化计算→GM 的冗长路径。

**反例** — 在 UB 中单独做量化：

```cpp
TQue<QuePosition::VECIN, 1> inQueueDeq;    // 量化参数在 UB
// CO1 → workspace → UB
// 量化参数: GM → UB
// Cast + Mul + Cast in UB
// 结果 → GM
```

**正例** — 通过 FP Buffer 融合：

```cpp
TQue<QuePosition::C1, 1>        inQueueDeq1;   // L1
TQue<QuePosition::C2PIPE2GM, 1> inQueueDeq;    // FP Buffer

void SplitDeq() {
    LocalTensor<uint64_t> deq1Local = inQueueDeq1.DeQue<uint64_t>();
    LocalTensor<uint64_t> deqLocal  = inQueueDeq.AllocTensor<uint64_t>();
    // L1 → FP Buffer
    DataCopy(deqLocal, deq1Local, {1, (uint16_t)(cSize * sizeof(uint64_t) / 128), 0, 0});
    inQueueDeq.EnQue<uint64_t>(deqLocal);
    inQueueDeq1.FreeTensor(deq1Local);
}

void CopyOut() {
    LocalTensor<float>    c1Local  = outQueueCO1.DeQue<float>();
    LocalTensor<uint64_t> deqLocal = inQueueDeq.DeQue<uint64_t>();
    SetFixpipeNz2ndFlag(1, 0, 0);
    DataCopyCO12DstParams params;
    params.nSize    = n;
    params.mSize    = m;
    params.srcStride = m;
    params.dstStride = n;
    params.quantPre = QuantMode_t::VQF322B8_PRE;
    params.nz2ndEn  = true;
    DataCopy(cGM, c1Local, params);   // 搬出时随路量化
    outQueueCO1.FreeTensor(c1Local);
}
```
