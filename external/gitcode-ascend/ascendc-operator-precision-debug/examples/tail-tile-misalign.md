# 案例：尾 tile 对齐 / 长度处理错误

## 现象

- 小 shape（如 1024 = tileLength 整数倍）通过
- 大 shape 或非整除 shape（如 1000、4097）失败
- 仅尾部若干元素错误

## 误差分析输出

```
shape=(4097,)  dtype=torch.float32
错误元素: 1/4097 (0.02%)
首错线性下标: 4096  多维坐标: [4096]
  NPU=0.000000  REF=1.234567  AbsErr=1.23e+00
```

## 根因

Compute 中尾 tile 使用了 `tileLength` 而非 `curTileLength` 作为向量 API 的长度参数，导致：
- 多计算的部分写入了无效数据
- 或尾 tile 实际只搬了 `curTileLength` 个元素，计算却读了 `tileLength` 个（UB 中残留脏数据）

```cpp
// ❌ 错误：统一用 tileLength
AscendC::Mul(tmp, xLocal, xLocal, tileLength);

// ✅ 正确：尾 tile 用 curTileLength
uint32_t len = (progress == loopCount - 1) ? curTileLength : tileLength;
AscendC::Mul(tmp, xLocal, xLocal, len);
```

## 定位关键

1. Phase 3 实验 D：shape=(1024,) 过, shape=(1025,) 挂 → 非整除时尾 tile 出错
2. Phase 2 检查 2.4：搜索 `tileLength` 在 Compute 中的用法，找到未区分尾 tile 的调用
