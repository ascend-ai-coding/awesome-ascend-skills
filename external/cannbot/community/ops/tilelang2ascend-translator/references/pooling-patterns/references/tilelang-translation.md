# TileLang → AscendC Translation Rules for Pooling

## 关键发现

**TileLang 0.1.4 Ascend 后端无法高效执行 Pooling 算子。** 设计阶段的 TileLang 代码应作为语义蓝图使用，实际落地必须转为手写 AscendC。

## TileLang 局限性（实测证伪）

### 1. AUTO_SYNC 每操作排空同步

每个 `T.copy` / `T.Parallel` 操作后，后端自动插入流水排空同步（~3.4us/op）。对于 pooling 这种 "大量小搬运 + 累加" 的模式，AUTO_SYNC 开销远超计算本身。

```
实测（TileLang）: geomean speedup = 0.007 → 0.010 (row-granularity 优化后)
需求达标线:      geomean speedup ≥ 0.6x
结论:            结构性延迟受限，TileLang 层不可达 0.6x
```

### 2. T.Pipelined 不支持滑动窗口归约体

```python
# ❌ 编译失败: "Can't handle the body of the loop because it is not a SeqStmt"
for kd, kh in T.grid(KD, KH):
    with T.Pipelined(current_stage=0):  # PipelinePlanner 报错
        T.copy(X[...], buf[...])
        for ow in T.serial(OW):
            acc[ow, :] = acc[ow, :] + buf[iw, :]
```

PipelinePlanner 无法分析包含内部 T.serial 循环 + 分支的归约体。

### 3. 3D T.Parallel 不支持

```python
# ❌ 编译失败: "Unsupported: 3D or higher dimensional parallel loops"
T.Parallel(W_out, C, ...)  # 3D parallel 不支持
```

### 4. Grid 并行无显著加速

Fixed Core 探针实验: block_num 从 1024 降至 24 核 → 性能变化在噪声内 (<2%)。瓶颈不在 launch 开销，而在单核内 AUTO_SYNC 累积延迟。

## 转译映射表

| TileLang 语义 | AscendC 对应 |
|--------------|------------|
| `T.copy(X[row, 0:W, 0:C], buf)` | `AscendC::DataCopy(inLocal_, xGM_[rowBase], W*C)` |
| `T.Parallel(C, lambda c: acc[ow,c] += buf[iw,c])` | `AscendC::Add(accLocal_[ow*C_], accLocal_[ow*C_], inFp32_[iw*C_], C_)` |
| `acc = T.alloc_buffer((OW, C), "float32")` | `AscendC::TBuf<VECCALC> accBuf_` + `accBuf_.Get<float>()` |
| `T.grid(N, OD, OH)` (block 并行) | `GetBlockIdx()` round-robin `for (outRow = workerId; ...; outRow += workerCount)` |
| 隐式同步 (AUTO_SYNC) | 显式 `PipeBarrier<PIPE_V>()` / `PipeBarrier<PIPE_MTE2>()` |
| `T.Pipelined` | 手动双 buffer（需要时） |

## 转译流程

```
TileLang 设计 (semantic blueprint only)
  │
  ├─ design/block_level/    → Block 粒度、tile size、UB 预算
  ├─ design/tile_level/     → 循环结构、数据搬运、累加模式
  └─ design/PERF_DESIGN.md  → 性能候选 (V1/V2/V3/...)
  │
  ▼
AscendC 手写实现
  ├─ op_kernel/*.cpp        → 按转译映射表逐语义翻译
  ├─ op_host/*.cpp          → NCDHW↔NDHWC permute + tiling params
  └─ CMakeLists.txt         → CANN 标准 CMake 模板
```

## 不应尝试的方向（已证伪）

| 方向 | 失效原因 | 证据 |
|------|---------|------|
| T.Pipelined 流水 | PipelinePlanner 不支持归约体 | p_retry=2 编译失败 |
| 多行合并 (R 行一同处理) | 3D T.Parallel 不支持 / 展平后正确性缺陷 | p_retry=3 失败 |
| Fixed Core pure TileLang | Grid 数非瓶颈，AUTO_SYNC 是主因 | V2 探针证伪 |
| 纯 TileLang 达标 (0.6x) | 每 op ~3.4us AUTO_SYNC | geomean 0.01x 上限 |

## 正确路径

1. TileLang 完成**语义设计和功能验证**（精度通过即 TileLang 交付）
2. 性能迭代在 TileLang 层做**尝试性探索**（3 轮 p_retry 上限）
3. 无论是否达标，产出 `PERF_DESIGN.md` 描述最优发现的 TileLang 设计结构
4. AscendC 层以 PERF_DESIGN.md 的设计结构为蓝图，用**显式流水控制**实现
5. AscendC 层可额外做 TileLang 无法表达的手动优化（如 reduce_d 快路径，实现见 `references/pooling-patterns/references/reduce-d-fastpath-implementation.md`）
