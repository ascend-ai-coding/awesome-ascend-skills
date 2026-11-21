# 5HD Layout Conversion

## 语义

逻辑输入 `[N, C, H, W]` 转为物理布局 `[N, C1, H, W, C0]`：

```text
C1 = ceil_div(C, C0)
out[n, c // C0, h, w, c % C0] = x[n, c, h, w]
```

`C` 不是 `C0` 整数倍时，最后一个 C1 block 的无效 channel 必须填零，反向转换裁剪到原始 C。准确的 PyTorch round-trip reference 见 `templates/dav3510/transpose_transdata_5hd.py`。

## PTO 数据流

Python tiling 先选择 N/C/H/W 的分核轴，再选择 UB 的 R/C split。连续 C0 block 优先 `T.copy`；跨 channel block 的重排使用目标仓库已经验证的 SIMD gather/scatter，无法表达的 dtype 使用 SIMT fallback。UB 输入、输出的两维 footprint 均按完整 SIMD 寄存器宽度 padding。

该语义不能通过简单的二维 `[H*W, C1*C0]` transpose 代替。实现必须显式保留 N、C padding、C0 内层和输出物理 stride。

## 精度与性能门禁

覆盖 `C<C0`、`C=C0`、`C=C0±1`、多 C1、H/W tile 边界、所有支持 dtype，以及 ND→5HD→ND round-trip。性能报告有效字节、padding 字节、GM burst、SIMD gather 成本、UB 占用与生成代码长度。
