# Backward Patterns: Pooling 反向梯度算子设计模式（语义与策略）

> **定位**：反向 scatter-add 语义（与 [row-granularity.md](row-granularity.md) 的前向 reduce 语义相对）。
> 适用于 AvgPoolNGrad、MaxPoolNGrad、AdaptivePoolGrad 等「反向传播」算子。
> **实现篇**（AscendC 实现陷阱 + output-driven transpose-scatter 高性能落地，即本文件原 §11/§12）在
> tilelang2ascend-translator 的 Pooling 类别，见 `references/pooling-patterns/references/backward-implementation.md`（下称 backward-implementation.md）。

## 1. 语义：scatter-add（一对多）vs 前向 reduce（多对一）

| | 前向 pooling（reduce） | 反向 grad（scatter-add） |
|---|---|---|
| 数据流 | 读窗口 → 写单值（多对一） | 读单值 → 写窗口（一对多） |
| 计算 | `out[od] = Σ_{id∈win(od)} x[id] / div(od)` | `gx[id] += Σ_{od∋id} grad[od] / div(od)` |
| 写冲突 | 无（各输出独立） | **有**（多输出窗口覆盖同一输入位置） |
| 输入依赖 | 依赖 x 值 | AvgPool 只依赖 x 的 **shape**；MaxPool 依赖 **argmax**（x 值） |

**关键结论**：AvgPool 是线性算子，反向梯度只依赖输入 shape 和池化参数，**不依赖输入具体数值**。因此 `avg_pool3_d_grad` 的输入是 `orig_input_shape`（INT32 shape 值）+ `grads`（梯度），不需要原始数据 x。MaxPool 反向则需要 argmax（见 §6）。

## 2. 两种实现策略

> **✅ 最终决策（阅读指引）**：本节的早期表述（「策略 A 推荐」「gather 首选」）已被后续实测修订，**以下方框结论为准**（详见 backward-implementation.md §12 与 grad-v2-lessons.md §12.2，均为 tilelang2ascend-translator Pooling 类别）：
> - **默认实现：input-driven gather（division-free，见 grad-v2-lessons.md §12.1）**——小窗口零原子、代码简单，覆盖大多数场景；
> - **naive scatter（按输出位置 + 逐元素原子写）几乎全劣**——跨步读 + 原子写的代价在任意窗口体积下都高于 C 连续向量化累加（实测 15 case 中 14 个仅 0.04x~0.36x），**不要仅凭「窗口大」就切 scatter**；
> - **仅 output-driven transpose-scatter（backward-implementation.md §12：块读 + Transpose + UB 内向量化 Add + SetAtomicAdd）值得大 kernel 时做**——它把逐元素 GM 往返压成一次块读 + 一次块写；
> - 一句话：**性能上限由「GM 往返次数」决定，不由「原子操作」决定**。gather 是默认；窗口重叠（stride<kernel）或 kernel 大（≥8³）且追求极致性能时，用 output-driven transpose-scatter；naive scatter 不做。

### 策略 A：按输入位置 gather（division-free，默认实现）

对每个输入位置 `id`，用闭式公式找出覆盖它的所有输出窗口 `od`，累加 `grad[od] / div(od)`：

```
For each 输入位置 (id, ih, iw):        # 每个位置独立，无写冲突
    gx[id,ih,iw] = 0
    For od in reverse_window_d(id):    # 闭式公式，见 §3
        For oh in reverse_window_h(ih):
            For ow in reverse_window_w(iw):
                gx[id,ih,iw] += grad[od,oh,ow] / div(od,oh,ow)
```

**优点**：每个输入位置独立累加，**天然无原子冲突**，可并行、可向量化（C 维连续），且可用 ow-outer 循环实现 division-free（grad-v2-lessons.md §12.1）。这是反向算子的默认实现策略（早期文档曾标「首选」，已按 backward-implementation.md §12 / grad-v2-lessons.md §12.2 修订）。

### 策略 B：按输出位置 scatter（直观但需原子）

```
For each 输出位置 (od, oh, ow):
    For id in window(od):
        For ih in window(oh):
            For iw in window(ow):
                gx[id,ih,iw] += grad[od,oh,ow] / div(od,oh,ow)   # 原子累加！
```

**缺点**：多个输出窗口写同一输入位置，需要 `AtomicAdd` 或分块归约。

> ⚠️ **实测修正（对标 ops-nn avg_pool3_d_grad，推翻「gather 首选」）**：虽然 scatter 需要原子，但 AscendC 的 `SetAtomicAdd` 是硬件加速的；而 input-driven gather 落地时每个窗口元素要一次 GM 读 + 一次 `PipeBarrier`，kernel=8 时每输入位置 512 次串行 GM 往返 + 1024 次全流水屏障，性能只有 ops-nn 的 0.4~0.6x。**串行依赖链比原子操作更致命**。官方实现用的是 output-driven scatter + transpose 向量化（见 backward-implementation.md §12），而非 gather。结论：gather 仅在「窗口无重叠且 kernel 小」时简单可用；窗口重叠（stride<kernel）或 kernel 大时，output-driven transpose-scatter 才是首选。

## 3. 反向窗口闭式公式（核心）

输入位置 `id`（D 维）被输出窗口 `od` 覆盖，当且仅当：

```
od*stride - pad <= id < od*stride - pad + kernel
```

反解 `od` 得闭式区间：

```
od_min = ceil((id + pad - kernel + 1) / stride)
od_max = floor((id + pad) / stride)
od ∈ [max(0, od_min), min(Dout-1, od_max)]
```

H、W 维同构（把 `id/pad/kernel/stride/Dout` 换成对应维）。

> ⚠️ **实测踩坑（avg_pool3d_grad D 类根因）**：`od_max` 必须用 **raw start**（未 clamp 的 `id + pad`）计算，**不能**先用 clamp 后的 `od_min` 作为窗口起点再去推 `end`。错误写法 `od_max = od_min + kernel/stride` 会在 `pad > 0` 时把窗口右边界算偏，导致（1）gather 过散——把不属于该输入的输出梯度也累加进来，或（2）divisor 偏大——`count_include_pad=False` 时把 pad 区也算进有效元素数。两者都会造成 matched_ratio 掉到 0.6~0.9 的「隐性精度 bug」（不崩、shape 对、但数值错），容易被误判为 divisor 公式问题。定位方法：先用 Python 逐点复现 gather 算法与 `torch.autograd` 对齐（排除算法错），再用全 1 grads dump 单 case 观察梯度分布模式。

**为什么闭式公式重要**：它让反向从「对每个输出窗口扫描」变成「对每个输入位置直接算累加区间」，把 scatter 复杂度从 `Dout×kernel` 降到 `Dout×(kernel/stride)`（stride=1 时相同，stride>1 时显著降低），且无需原子操作。

## 4. divisor 与 forward 完全共享

反向的 `div(od,oh,ow)` 与前向的 divisor **是同一个量**——它是「输出位置」的属性，不是「输入位置」的属性。直接复用 tilelang2ascend-translator Pooling 类别的 `references/pooling-patterns/references/precision-patterns.md`（下称 precision-patterns.md）三级 divisor 策略：

1. `divisor_override > 0` → 固定 `Muls(1/override)`
2. `count_include_pad && !ceil_mode` → 恒 `KD*KH*KW` 单 Muls
3. `ceil_mode` 右截断 / `count_include_pad=False` → 逐输出位置 `ComputePaddedDivisor` / `ComputeDivisor`

**注意**：scatter 时每个输出位置用自己的 divisor；gather 时每个输入位置累加的多个输出位置可能 divisor 不同（ceil_mode / no-include-pad 边界），必须逐项除对应 divisor，**不能**先求和再统一除。

## 5. NDHWC 布局在反向

前向的 NDHWC 思想（[layout-strategy.md](layout-strategy.md)）反向同样适用：

- `grads` 若为 NDHWC `[N, Dout, Hout, Wout, C]`，则每个 `(n,od,oh)` 的整行 `Wout×C` 连续，C 维可向量化。
- 反向输出的 `gx` 为 `[N, Din, Hin, Win, C]`，每个 `(n,id,ih)` 的整行 `Win×C` 连续。
- gather 时，`gx[n,id,ih,:,:]` 的 C 维累加可用 `Add(C)` 向量化——与前向对称。

**但注意一个不对称**：前向是按「输出行 (od,oh)」做 block 划分（每个 block 读若干输入行、写一行输出）；反向按「输入行 (id,ih)」做 block 划分（每个 block 读若干输出行、写一行输入梯度）。两者 block 语义镜像，tile 划分逻辑需对调。

**data_format 约定（实测 avg_pool3d_grad 落地）**：`grads` 的布局由 `data_format` 属性决定——`NCDHW` 为 `[N,C,Dout,Hout,Wout]`、`NDHWC` 为 `[N,Dout,Hout,Wout,C]`；但输出 `grad_input` **恒为 NCDHW `[N,C,Din,Hin,Win]`**（ops-nn 约定，host 端做 permute 恢复）。因此 host 侧须在 `TORCH_LIBRARY`/`op_host` 里按 `data_format` 对 `grads` 做 NDHWC→NCDHW 的 permute，kernel 内统一按 NCDHW 计算；仅当 C 维连续（NDHWC 天然 C 连续、NCDHW 需保证 `C%8` 对齐）才可向量化。

## 6. AvgPoolGrad vs MaxPoolGrad（argmax 特殊性）

| | AvgPoolGrad | MaxPoolGrad |
|---|---|---|
| 反向输入 | `orig_input_shape` + `grads` | `argmax`（或原始 x）+ `grads` |
| 梯度分配 | 均匀：`grad/div` 给窗口内**所有**位置 | 选择性：`grad` 只给 **argmax 位置**，其余为 0 |
| 额外需求 | 无 | 前向须保存 argmax 索引，或反向重算 argmax |
| 原子/冲突 | gather 无冲突 | argmax 位置唯一，scatter 也无冲突（每窗口一个 argmax） |

MaxPoolGrad 若用 gather 视角：输入位置 `id` 的梯度 = 覆盖 `id` 的输出窗口中，argmax 恰好等于 `id` 的那些窗口的 `grad` 之和。需要 argmax 张量参与闭式判断，复杂度高于 AvgPoolGrad。

## 7. 多核划分

- **策略 A（gather）**：按输入位置（或输入行）round-robin 划分，各 block 独立 gather，**零原子冲突**，天然适合多核。
- **策略 B（scatter）**：按输出位置划分，跨 block 写同一输入位置时需 `AtomicAdd` 或二次归约，多核复杂。

> ⚠️ **修订（backward-implementation.md §12 / grad-v2-lessons.md §12.2）**：本结论是「简单且正确」的**默认**选择。追求大 kernel（≥8³）或窗口重叠（stride<kernel）场景的极致性能时，output-driven transpose-scatter（backward-implementation.md §12：块读 + Transpose + UB 内向量化 Add + SetAtomicAdd）才是首选；naive scatter（逐元素原子写）不做。

**结论**：反向默认按输入位置做 gather 划分（策略 A），把「写冲突」问题在算法层消除，而非在实现层用原子操作硬扛；仅在 backward-implementation.md §12 所述大 kernel / 重叠场景升级为 output-driven transpose-scatter。

## 8. TileLang 表达（语义蓝图）

```python
# AvgPool3DGrad gather 语义（TileLang 伪代码，非可编译）
@T.prim_func
def avg_pool3d_grad(Grad: T.Buffer((N,C,Dout,Hout,Wout), dtype),
                    Gx: T.Buffer((N,C,Din,Hin,Win), dtype)):
    for n, c, id, ih, iw in T.grid(N, C, Din, Hin, Win):
        acc = T.alloc_var("float32")
        acc = 0.0
        for od in T.serial(od_min(id), od_max(id)):   # 闭式区间
            for oh in T.serial(oh_min(ih), oh_max(ih)):
                for ow in T.serial(ow_min(iw), ow_max(iw)):
                    acc += Grad[n, c, od, oh, ow] / div(od, oh, ow)
        Gx[n, c, id, ih, iw] = acc
```

TileLang 0.1.4 Ascend 后端对动态闭式窗口区间支持有限（见 tilelang2ascend-translator Pooling 类别 `references/pooling-patterns/references/tilelang-translation.md`），此蓝图用于指导 AscendC 实现，实际落地仍以手写 AscendC 为主。

## 9. 与 forward 经验的可复用对照表

| 经验 | 前向 reduce | 反向 scatter | 复用？ |
|------|-----------|-------------|--------|
| NDHWC 布局 + host permute | ✅ | ✅ | **复用**（block 语义对调） |
| 对齐守卫 C%8 / W*C%16 | ✅ | ✅ | **复用** |
| fp32 累加 + Cast 辅助 | ✅ | ✅ | **复用** |
| divisor 三级策略 | ✅ | ✅ | **复用**（divisor 是输出位置属性） |
| row-granularity 窗口累加 | ✅ | ❌ | **不适用**（反向是 gather 区间累加） |
| reduce_d 快路径 (KH==KW==1) | ✅ | ❌ | **不适用**（可对偶设计「反向 gather 快路径」，但结构不同） |
| 多核 round-robin | ✅ | ✅ | **复用**（gather 视角零冲突） |
| 闭式反向窗口公式 | ❌ | ✅ | **反向新增** |

## 10. 设计检查清单（新反向 pooling 算子）

- [ ] 确认是 AvgPool（线性）还是 MaxPool（需 argmax）
- [ ] 反向窗口闭式公式的边界 clamp（`od_min` 可负、`od_max` 可超 `Dout`）
- [ ] divisor 逐输出位置除（不能先求和再统一除）
- [ ] 实现策略：默认 input-driven gather（division-free，零原子）；仅大 kernel / 窗口重叠时考虑 output-driven transpose-scatter（backward-implementation.md §12）；naive scatter 不做（grad-v2-lessons.md §12.2 修订）
- [ ] ceil_mode=True 末窗口跳过规则与 forward 一致（用 forward 输出 shape 推导）
- [ ] NDHWC 下 `gx` 行连续 + C 维向量化
- [ ] `orig_input_shape`（INT32）与 `grads` 的 shape 校验
- [ ] `data_format` 约定：grads 可 NCDHW/NDHWC，输出恒 NCDHW，host 侧 permute
- [ ] 反向窗口 `od_max` 用 raw start（未 clamp）计算，不用 clamp 后起点推 end（§3 陷阱）
- [ ] CopyOut `DataCopyPad`（MTE2 读 acc_）后尾随 `PipeBarrier<PIPE_ALL>`（backward-implementation.md §11.1 WAR 陷阱）
- [ ] 精度验证用 200-case 随机覆盖（NCDHW/NDHWC × fp32/fp16/bf16 × ceil/cip/divisor_override）
- [ ] 性能标杆选 ops-nn（`torch_npu` 的 F.avg_pool3d 反向），非 torch.autograd（backward-implementation.md §11.2 陷阱）
