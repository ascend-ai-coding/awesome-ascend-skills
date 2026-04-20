# 案例：GM 偏移单位混淆（元素 vs 字节）

## 现象

- 全 1 输入通过，等差/随机输入失败
- 错误呈周期性，周期 = tileLength
- 首错下标恰好在第 2 个 tile 起始位置

## 误差分析输出

```
shape=(8192,)  dtype=torch.float16
错误元素: 7168/8192 (87.50%)
首错线性下标: 1024  多维坐标: [1024]
错误间隔: [1] → 从首错起连续错误
```

## 根因

CopyIn 的 GM 偏移多乘了 `sizeof(T)`，导致第 2 个 tile 起搬运地址偏移 2 倍。

```cpp
// ❌ 错误：偏移多乘了 sizeof(T)
AscendC::DataCopyPad(xLocal, xGm[progress * tileLength * sizeof(T)], ...);

// ✅ 正确：GlobalTensor 下标是元素偏移
AscendC::DataCopyPad(xLocal, xGm[progress * tileLength], ...);
```

## 定位关键

1. Phase 1：全 1 过但随机挂 → 实验 C 确认是地址类问题
2. Phase 1：首错在 1024（= tileLength）→ 第 2 个 tile 起偏移错误
3. Phase 4：0 核 tile=0 的 CopyIn dump 正确，tile=1 开始偏移 → 锁定 `progress * tileLength * sizeof(T)`
