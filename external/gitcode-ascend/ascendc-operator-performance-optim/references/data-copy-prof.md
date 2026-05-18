# 阶段 2：搬运优化 — 详细参考

## 2.1 单次搬运量 >= 16 KB

带宽利用率随单次搬运量增大而提升。实测经验：单次搬运 **>= 16 KB** 时，
UB↔HBM 两个方向均可达到接近峰值的带宽。低于此值时带宽利用率显著下降。

设计 Tiling 策略时应确保每次 `DataCopy` 搬运至少 16 KB。

## 2.2 GM 地址 512B 对齐

在 Atlas A2 训练系列 / Atlas 800I A2 推理产品上，GM 地址 512B 对齐可比 32B
对齐获得最高 **30%** 的带宽提升（最差场景下的差距）。

分配 GM Tensor 或计算偏移量时，应确保起始字节地址为 512 的整数倍。

## 2.3 使用 stride 参数代替 for 循环

使用 `DataCopyParams`（blockCount / blockLen / srcStride / dstStride）将间隔
搬运描述为一条 DMA 指令下发，而非用 for 循环逐行调用 `DataCopy`。

**反例** — for 循环，每次仅搬运 2 KB：

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth  = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
// 16 次独立的 2KB 搬运，带宽利用率极低
for (int i = 0; i < imgHeight; i++) {
    DataCopy(tensorIn[i * copyWidth], tensorGM[i * imgWidth], copyWidth);
}
```

**正例** — 单条 DMA 描述符，一次搬运 32 KB：

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth  = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
DataCopyParams copyParams;
copyParams.blockCount = imgHeight;                     // 16 行
copyParams.blockLen   = copyWidth / 8;                 // 单位: 32B DataBlock
copyParams.srcStride  = (imgWidth - copyWidth) / 8;    // src 行间间隔
copyParams.dstStride  = 0;                             // dst 连续写入
DataCopy(tensorGM, tensorIn, copyParams);
```

stride 方式下发一条 DMA 指令，硬件自主完成全部搬运，可充分利用带宽。
for 循环方式下发 16 条小 DMA 指令，每条之间还有 Scalar 开销。
