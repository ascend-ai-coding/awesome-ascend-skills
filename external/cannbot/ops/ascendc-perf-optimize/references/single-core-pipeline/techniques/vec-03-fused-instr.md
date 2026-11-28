# 技术卡片 vec-03：融合指令（MulDstAdd / VMULA 替 Mul+Add）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC
- **一句话**：把成对相邻的乘 + 加合并成单条融合指令，减少 VEC 指令数与 high-level API 固定开销。

## 何时用（诊断特征）

- VECTOR 泳道上有**成对相邻、可合并的算子**（`Mul` 后紧接 `Add` 等）。
- hot loop 仍用 `FusedMulAdd`/`Mul`/`Add` 这类 high-level vector intrinsic，且在 A5 上反而比老平台慢（high-level API 自带 setVecMask / repeat header / PIPE_V 同步等固定开销）。
- `aiv_vec_ratio` 高，Compute 占总时间 >60%，指令数偏多。

## 原理

high-level vector API 每次调用带固定开销；改用 A5 MicroAPI VF，把「加载-乘加-写回」编译成紧凑指令片段、连续展开、吃满一拍 lane。乘加融合（`dst = dst*src1 + src2`）从两条指令降为一条。

## 怎么改（before / after）

```cpp
// before：Mul + Add 两条 high-level 指令（或 high-level FusedMulAdd）
Mul(x, x, w, len);
Add(x, x, b, len);

// after：A5 RegBase MicroAPI VF，单条 MulDstAdd 融合
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using namespace AscendC::MicroAPI;   // 实际 CANN 若为 Reg:  namespace MicroAPI = Reg;
__simd_vf__ inline void DoMulAddVf(__ubuf__ float* xBuf, __ubuf__ float* wBuf,
                                   __ubuf__ float* bBuf, uint32_t normDim) {
    RegTensor<float> xR, wR, bR;
    uint32_t lane = GetVecLen() / sizeof(float);              // A5 fp32=64，不要硬编码
    uint32_t repeatTimes = (normDim + lane - 1) / lane;
    for (uint32_t j = 0; j < repeatTimes; ++j) {
        MaskReg m = UpdateMask<float>(normDim);               // 尾段 mask
        LoadAlign(xR, xBuf + j * lane);
        LoadAlign(wR, wBuf + j * lane);
        LoadAlign(bR, bBuf + j * lane);
        MulDstAdd(xR, wR, bR, m);                             // x = x*w + b（in-place）
        StoreAlign(xBuf + j * lane, xR, m);
    }
}
#endif
```

顶层用 `__aicore__` 包装函数拿 `LocalTensor::GetPhyAddr()` 转裸指针再喂进 VF；调用侧按平台 `#if` 分流，老平台保留原 `Mul`+`Add`。

## 性能数据

msprof `aiv_time`：high-level `Mul`+`Add` → MicroAPI `MulDstAdd`，外部同类改造实测约 **-40%**（作为预期方向，非本库实测；须以本 case 上板 `aiv_time` 前后确认）。

## 红线（Stop rules）

- `MulDstAdd(dst, src1, src2, mask)` 语义 = `dst = dst*src1 + src2`（dst 既是源又是目标）。**替换前比对 ground truth 的参数顺序**：weight 走 src1、bias 走 src2；写反得到 `x*b+w`，静默编译通过但精度错。
- lane 数用 `GetVecLen()/sizeof(dtype)`，**不要借** `B32_DATA_NUM_PER_REPEAT(=64)` 或 `ONE_BLK_SIZE/sizeof(dtype)`——换 dtype/换代会全错。
- 入参用裸 `__ubuf__ T*`，**不要用 LocalTensor**（封装会阻止 RegTensor 寄存器分配）。
- VF 段后该有的 `PipeBarrier<PIPE_V>()` 要保留——VF 内部不替你做同步。
- 尾段单次 lane < 32 的短向量不值得改（VF 启动有固定开销）。

## 关联

- 源：`../vec.md` 策略 3（融合指令）+ 外部 VF 策略 2（`FusedMulAdd → MulDstAdd`）。
- 相邻卡片：vec-01（先融合 UB 链路）、vec-04/vec-05（配套消标量 round-trip 与中转 UB）。
