# Row-Granularity Tile Design for Pooling

> **定位**：前向 reduce 专属（窗口 → 单值）。反向 scatter-add 见 [backward-patterns.md](backward-patterns.md)。

## 设计原理

Pooling 算子的核心计算是：对每个输出位置 (n,od,oh,ow)，在输入上取一个 KD×KH×KW 窗口，累加后除以元素数。

传统逐位置 tiling：每个 tile 处理一个 (ow, c) 切片，需要为每个 kw 位置重新加载数据 → **大量冗余 DataCopy**。

**行粒度 tiling**：每个 block 处理一个完整的输出行 (n,od,oh)，将 KD×KH 个输入行逐步加载、OW 个位置逐步累加。每行只加载一次，在 UB 中完成全部 OW 位置的累加。

## 数据流

```
For each (n, od, oh):                          # 一个 block
    acc[OW, C] = 0                             # UB 内累加器
    For each kd (valid):
        For each kh (valid):
            DataCopy: X[n,id,ih,:,:] → UB      # W*C 连续元素，一次搬运
            For each ow (0..OW):
                iw = ow*SW - PW + kw
                if iw in bounds:
                    Add(acc[ow*C:], in[iw*C:])  # C 宽度向量化
    acc /= divisor
    DataCopy: acc → Y[n,od,oh,:,:]              # OW*C 连续元素
```

## 关键参数

| 参数 | 含义 | avg_pool3d v2 典型值 |
|------|------|---------------------|
| W*C | 输入行大小（elements） | 32×16=512 ~ 64×64=4096 |
| OW*C | 输出行大小（elements） | 4×16=64 ~ 32×512=16384 |
| KD×KH | 窗口空间维度数 | 1~64 (scenario-dependent) |
| accSize | 累加器 UB 需求（fp32 bytes） | OW*C*4 |

## UB 预算估算

```
accBuf     = OW * C * sizeof(float)            # 累加器
inBuf      = W * C * sizeof(T)                 # 输入行（TQue 或 TBuf）
outBuf     = OW * C * sizeof(T)                # 输出行
inCastBuf  = W * C * sizeof(float)   (fp16 only)
outCastBuf = OW * C * sizeof(float)  (fp16 only)
reduceDRow = W * C * sizeof(T)       (KH==1 && KW==1 only)

total = accBuf + inBuf + outBuf + castBufs [+ reduceDBuf]
```

典型 fp32 场景（W=32, C=32, OW=8）: ≈ 8KB，远小于 UB 192KB。
典型 fp16 场景（W=64, C=512, OW=64）: ≈ 256KB，需检查 UB 限制。

## 设计检查清单

- [ ] OW*C*4 (acc pf32) + W*C*sizeof(T) (in) + OW*C*sizeof(T) (out) + castBufs < 192KB
- [ ] C ≥ 8（向量寄存器 256bit / 32bit = 8 fp32 elements）
- [ ] 窗口循环内无跨-ow 数据依赖（各 ow 独立累加）
- [ ] KW=1 时内层 ow 循环可简化（只有 kw=0 有效）

## TileLang 等价表达

```python
# TileLang 伪代码（语义蓝图，非可编译代码）
@T.prim_func
def avg_pool3d_row(X: T.Buffer((N,D,H,W,C), dtype), ...):
    for n, od, oh in T.grid(N, OD, OH):
        acc = T.alloc_buffer((OW, C), dtype="float32")
        # ... 窗口累加 ...
        for ow in T.serial(OW):
            Y[n, od, oh, ow, :] = acc[ow, :] / divisor
```

注意：TileLang 0.1.4 Ascend 后端无法高效执行此模式（转译映射见 tilelang2ascend-translator Pooling 类别 `references/pooling-patterns/references/tilelang-translation.md`），但行粒度语义蓝图可直接用于指导 AscendC 实现。
