# examples 占位符约定

本目录下每个 `<op>.md` 为 kernel 参考骨架，仅展示接口组合与轴切分模式（哪些轴 loop、哪些轴整块），不作为标准模板：loop 轴、`unroll_list`、tile shape、动态轴处理等需按实际 shape / dtype 与平台约束确定并调优；骨架未逐一经 NPU 编译验证。所有骨架共用以下占位符：

| 占位符 | 含义 |
|--------|------|
| `sl` | 输入 shape 列表，如 `[B, S, D]` |
| `ol` | 输出 shape 列表 |
| `pypto_dtype` | 元素 dtype，如 `pypto.DT_FP32` |
| `batch` | 被 loop 的外层轴长度（通常 `sl[0]`） |
| `inner` | 单次迭代处理的内层 shape，如 `sl[1:]` |
| `inner_out` | 单次迭代的输出内层 shape（末轴长度可能变化，如 glu/diff/mean） |
| `half` | last-dim 折半长度（rope/glu 等用） |
| `scale` | attention 缩放系数（attention） |
| `rep` | 复制次数（repeat / repeat_interleave） |
| `idx_inner` | 索引张量单迭代内层 shape（embedding） |
| `out_interleaved` | sort 交错输出（值+索引）shape（sort） |
| `num_tiles` / `tile_len` | 生成类沿输出轴切 tile 的分块数 / 每块长度（linspace 等） |

## 最小可运行 setup

```python
import pypto

B, D = 8, 128
sl, ol = [B, D], [B, 1]
pypto_dtype = pypto.DT_FP32
batch, inner, inner_out = B, [D], [1]
```

## Note 约定

每篇 Note 以一句话说明切分方式：loop 的轴、整块的轴及原因；无 batch-row loop 的骨架（cube、sort、生成/索引类）说明实际采用的切分方式，不得使用 "not applicable"。
