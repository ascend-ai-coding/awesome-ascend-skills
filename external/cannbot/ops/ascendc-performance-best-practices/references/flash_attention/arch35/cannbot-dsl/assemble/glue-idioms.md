# FA 胶水：实施惯用法与同步校验器规则

本篇是最佳实践库第 5 层**胶水**的表达层规则：把[积木模块](index.md)粘成完整 kernel 时，DSL 表达层的合法写法与校验器约束。逐段机理解读见 [流水建模](../optimize/2-pipeline-modeling.md)。

> 定位：本文回答盲测暴露的落地缺口——知识文档给了结构与数值，但没给 DSL 表达层的合法写法。
> 与 [延迟线流水](../optimize/2-pipeline-modeling.md)、[Tiling 容量建模](../optimize/3-capacity-equations.md) 配套使用。
> 所有骨架代码来自已验证、逐位一致的参考实现，可直接迁移。
> **为什么需要本文**：盲测实证，知识文档给了结构与数值但没给 DSL 表达层的合法写法时，四大结构机制全部卡死在本文覆盖的缺口上（知识 v1 只能实施轻量机制，1.086×）。每条规则都对应一个真实翻车模式，不是风格建议。

## 1. L1/L0 视图声明规则（拆块拷贝的前提）

**规则：凡是要按行/列拆块拷入的 L0A/L0B buffer，声明时必须用 `TensorViewSpec` 显式钉死 NZ 物理布局；否则拆块 mem_copy 的布局会被编译器推导改写，数值错位。**（历史实测：不显式声明时一切编译通过但结果错，报错只有 `ValueError: Operation creation failed`，无定位信息。）

```python
from cannbotdsl import TensorViewSpec, ViewLayout   # 实际导入路径见 kernel 头部

# L0A：帧 (M=64, S2=128)，NZ 物理 (m1=8, n1=8? 按 (行块,列块,16,16))
self.l0a = Channel(MemLoc.L0A, (TILE_M, TILE_S2), dtype_16, depth=4,
    view=TensorViewSpec((TILE_M, TILE_S2), dtype_16,
        layout=ViewLayout("nz",
            physical_shape=(TILE_M // 16, TILE_S2 // 16, 16, 16),
            physical_stride=(TILE_S2 * 16, 256, 16, 1))))

# L0B：帧 (S2=128, CHUNK=64)，注意 physical_stride 以整帧为基准 —— 这使两个
# 64 行 L1 环槽可以分别拷进同一帧的上下半区（见 §2）
self.l0b = Channel(MemLoc.L0B, (TILE_S2, MM_FEATURE_CHUNK), dtype_16, depth=4,
    view=TensorViewSpec((TILE_S2, MM_FEATURE_CHUNK), dtype_16,
        layout=ViewLayout("nz",
            physical_shape=(MM_FEATURE_CHUNK // 16, TILE_S2 // 16, 16, 16),
            physical_stride=(TILE_S2 * 16, 256, 16, 1))))
```

**NZ reinterpret 的 offset 语义**：按物理 n-block 网格、以**字节**为单位的偏移（0.7.0 包实测口径；盲测中"元素偏移"理解导致 L0C 窗口错位、报 `Operation creation failed`）。参考实现的表达式自带 `BYTES_PER_16BIT_ELEMENT` 乘子，**整式照抄即可，不必自行换算单位**：

```python
half_frame_offset = (FRAME_ROWS // 2) * 16 * BYTES_PER_16BIT_ELEMENT
# 例：L0B 帧 128 行拆两个 64 行半区 → half*64*16*2
# 例：L0A 帧 64 行拆上下 32 行 → h*32*16*2
```

单环等价式（用单个 (128,512) L1 环替代双环时），通用 NZ 网格窗口偏移公式（盲测实测验证，逐位通过）：

```python
offset_bytes = blk_row * row_blk_stride_elems * 2 + n1blk * (m1_total_rows * 256) * 2
# 即：行块内偏移按 row_blk_stride（physical_stride[2]=16 的倍数）×2B；
#    跨行块按 (m1_total×256)×2B；feature 项系数随环行数变化（64 行环 ×feature，128 行环 ×2×feature...）
```

窗口超界/错位时报裸 `ValueError: Operation creation failed` 或**静默数值错**——先查 offset 网格换算。

## 2. KV 双环 → 128 行 L0B 帧装配（骨架）

L1 侧两个独立 CrossCore Channel（每环 64 行，depth=3），AIC 消费每个槽一次并把 handle 从 QK 带到 PV：

```python
self.right_lower = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                           depth=3, kind=ChannelKind.CrossCore)
self.right_upper = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                           depth=3, kind=ChannelKind.CrossCore)
```

QK 路径（`set_mmad_direction("m")`，B 帧 (S2,64)，收缩维在 K）——两个半环各拷进 B 帧上下半区：

```python
b = _bounded_local(self.l0b.produce(), (TILE_S2, width), (TILE_S2, width))
for half in range(2):                     # half=0 → lower 环, half=1 → upper 环
    source = right_lower if half == 0 else right_upper
    k = reinterpret(source, shape=(TILE_S2 // 2, width),
                     offset=(TILE_S2 // 2) * feature * BYTES_PER_16BIT_ELEMENT)
    dst = reinterpret(b, shape=(TILE_S2 // 2, width),
                      offset=half * (TILE_S2 // 2) * 16 * BYTES_PER_16BIT_ELEMENT)
    mem_copy(dst, k)
matmul(score, a, b, init=(feature == 0), unit_flag=... )   # 见 §6
```

PV 路径（`set_mmad_direction("n")`，B 帧转置装入 (CHUNK, S2)）：

```python
for half in range(2):
    source = right_lower if half == 0 else right_upper
    v = reinterpret(source, shape=(TILE_S2 // 2, MM_FEATURE_CHUNK),
                    offset=(TILE_S2 // 2) * feature * BYTES_PER_16BIT_ELEMENT)
    dst_b = reinterpret(b, shape=(MM_FEATURE_CHUNK, TILE_S2 // 2),
                        offset=half * MM_FEATURE_CHUNK * (TILE_S2 // 2) * BYTES_PER_16BIT_ELEMENT,
                        data_format="nz")
    mem_copy(dst_b, v.transpose(0, 1))    # V 以转置进 B，A 侧是 P
matmul(dst, a, reinterpret(b, shape=(MM_FEATURE_CHUNK, TILE_S2), data_format="nz"), init=True)
```

有界窗口惯用法（保 allocation pitch、缩运行期 extent）：

```python
def _bounded_local(buffer, shape, capacity, offset=0, stride=None):
    root = reinterpret(buffer, shape=capacity, offset=offset, stride=stride)
    return reinterpret(root, shape=make_tiler(shape, alignment=(1, 1)), stride=stride)
```

## 3. 同核双代 ping-pong 的合法形态（Q 驻留）

**规则：同核（AIC 内部）的双代缓冲用"普通 Buffer 数组 + 显式 parity 下标"，不引入任何 Channel/Ring/事件**。跨 tick 的写读靠程序序保证：旧代在消费完最后一个 chunk 后才被覆盖，程序序天然成立。

```python
# 声明：普通 Buffer 列表（不是 Channel！）
self.q_lower = [Buffer(MemLoc.L1, (32, w), dtype_16, data_format="nz") for w in (384, 384, 128)]
self.q_upper = [Buffer(MemLoc.L1, (32, w), dtype_16, data_format="nz") for w in (384, 384, 128)]

# 使用：parity 选代 —— 算 query i 时预取 query i+1 进另一代
load_q_wide(token, (i + 1) % 2)     # 宽段 384 双代
load_q_tail(token)                  # 尾段 128 单代（旧尾段消费完才载，时序上被宽段 MMAD 掩盖）
```

**会被同步校验器拒绝的形态**（实测 6 种全挂，报错均为
`retained cross-core version has an early release or an extra ready wait`
（`asccube.sync_intra_wait, pipe=4, wait_ready, banks=3`））：
- depth-2 L1 Ring + produce/consume 每 query 翻代
- 两个单槽 Channel + parity
- 普通 Buffer + 跨 tick 版本化同步原语

判别原则：**双代的两代数据如果只被一个核读写 → Buffer 数组 + parity；只要出现跨核交接 → Channel（depth 按握手数+裕量）**。把同核双代错误地表达成跨核同步是盲测第一大坑。

## 4. 动态 region 规则（if/循环体内）

| 规则 | 违反症状 | 替代写法 |
|---|---|---|
| 嵌套动态区域内赋值的 **Tensor 视图**不能在外层读取（标量可以 yield） | `UnboundLocalError` | 在分支外先建好全部候选视图，分支内只做 mem_copy/matmul；或整段复制 buffer 引用 |
| 动态区域内 `return` 早退 | `FE102_UNSAFE_EARLY_EXIT` | 所有动态分支汇聚到单一 return；用标志变量表达"提前结束" |
| 动态分支内创建 L0/L1 `_bounded_local`/reinterpret 窗口 op | `Operation creation failed`（无信息） | 窗口提升到分支外，无条件按 clamp 后的动态 extent 拷贝 |
| 动态长度窗口 | 同上 | `_bounded_local(buf, (rows, w), (CAP_ROWS, w))`：capacity 静态、extent 动态 |

## 5. 程序分叉与依赖杂项

- AIC/AIV 程序分叉：`if cannbotdsl.is_aic(): ... else: ...`（kernel 入口处按核型分流）。
- baseline 的参数校验模块（`sparse_flash_attention_validation.py`）是 kernel 文件的 import 依赖，新工作区必须一并携带，否则 `dsl_once.py` 直接 ImportError。
- 复制外部校验模块时注意它只含常量与参数检查，不含知识。

## 6. unit_flag 链式发射惯例

链式发射（同一流水内背靠背的 matmul/fixpipe，前一发完成即触发下一发）时：

```python
# matmul 链：非末尾 chunk 发 unit_flag=2，最后一个发 3
matmul(score, a, b, init=(feature == 0),
       unit_flag=3 if feature + MM_FEATURE_CHUNK == DIM_NOPE else 2)

# fixpipe 链：链尾用 3
mem_copy(dst, src, options=copy_options(unit_flag=3))
```

惯例：`2` = 链中继（完成信号传给下一个 unit），`3` = 链尾（完成信号交给等待者）。首块的 `init=True/False` 控制累加器是否清零，与 unit_flag 正交。QK 的最后 chunk 与 score fixpipe 各自是链尾 → 都发 3。

## 6.1 matmul 轴契约（盲测第 4 轮 A 段实证）

- `matmul(lhs, rhs, dst)` 的 **rhs 形状 = (N, K)，N=dim0**：输出 N 列、收缩 K。报错 `lhs K dimension does not match rhs K` 即 lhs 收缩维与 rhs dim1 不等。
- PV 的 V 必须**转置**进 B：B 帧 (CHUNK, TILE_S2) = (N=64, K=128)，即 F5 的 `v.transpose(0,1)` + `reinterpret(b, shape=(MM_FEATURE_CHUNK, TILE_S2))`；用 64 宽 B 不转置会报 `128 vs 64`。
- baseline 旧的 `offset=TILE_M*feature*4` 型 L0C 写公式只对 (64,512) 单 L0C 成立；per-chunk L0C buffer（`pv_l0c[feature//64]`）时 dst 就是整块 buffer，不要再叠 offset。

## 6.2 设备错误开放项

`RuntimeError: npuSynchronizeDevice … device error type 3, error code 507015`（同步期设备错误，无 dump/trace 指引）：目前无已验证的定位手段，遇到时记录触发前最后一步改动并回退；定位通道（msprof 参数/日志开关）是开放项。

## 7. 跨核 UB 通道的分区写（PV 分半 + O 原地累加的前提）

**合法骨架**（L0C 结果按行劈给两个 AIV、落到通道的 acc/pv 半区）：`partition(dim=0)` 直接施加在 **L0C buffer 本体**上，目的侧用 `tile_slice` 在**通道根张量**上选半区——不要先对 L0C 消费张量做 reinterpret 窗口再 partition：

```python
root = result_lo if feature < DIM_NOPE // 2 else result_hi   # pv_halves[0]/[1]
if first != 0:
    target = tile_slice(root, (m // 2, DIM_NOPE // 2), (0, 0))    # rows 0-31 = acc 区
else:
    target = tile_slice(root, (m // 2, DIM_NOPE // 2), (1, 0))    # rows 32-63 = pv 进料区
mem_copy(
    tile_slice(target, (m // 2, MM_FEATURE_CHUNK), (0, (feature // MM_FEATURE_CHUNK) % 4)),
    dst.partition(dim=0),          # ← partition 在 L0C buffer 上，硬件按行劈给两个 AIV
)
```

AIV 侧消费时：`acc = tile_slice(result, (32,256), (0,0))`、`pv = tile_slice(result, (32,256), (1,0))`；首 tile 的"确认所有权"只做 1-lane vload/vstore（写整 tile 会触发跨核版本校验）。

**非法/危险形态（实测）**：
- 先 `reinterpret` 出 L0C 窗口再 `partition(dim=0)` → **静默误译**（0.7.0 不报错）：每 AIV 行 0 正确、行 1-31 同量级小偏差。看到"首行对、其余偏"就是这个根因。
- AIV 侧对跨核通道槽做原地写（绕过 consume）→ 同步校验器通过但版本链断裂，后续 tick 数据错。

O 原地更新的合法循环结构（acc 区驻留、vmadd 原地、末 tile 归一化）见延迟线流水篇 §4.7；编译器所有权校验按"first 优先组织分支"通过。

## 8. 优化阶梯的前置条件（避免按错依赖顺序实施）

阶梯 µs 是**顺序依赖的消融值**，不是独立增益。每个大步的前置条件：

| 阶梯步 | 收益 | 前置条件 |
|---|---|---|
| 拆 KV 双环 | −0.7% | 需 §1 显式视图声明 + §2 装配骨架，否则数值错 |
| 索引提前一拍+成对地址 | →1482 | 假设二分/索引处理在 gather 关键路径上 |
| PV 分半交接 | →1359 | 依赖双环已就位（V 从环槽转置装 L0B） |
| Q 双代 ping-pong | →1184 | 需 §3 Buffer+parity 形态；Channel 形态必被拒 |
| 有效前缀快速路径 | →975 | **假设二分搜索仍在关键路径**；若索引预取已把二分挪出关键路径，收益≈0（盲测两轮均验证） |
| O 原地累加 | →934 | 需 §4 视图提升规则 + §7 分区写骨架；首 tile fixp 直写 acc 区，first 分支按"first 优先"组织 |
| Q 双代 ping-pong | →1184 | **收益前提是 PV 分半（阶梯前一步）已改变 AIC 关键路径**；单独实施≈中性（盲测第 2 轮验证） |

实施顺序建议：先做不依赖结构改造的轻量机制（索引预取、有效前缀、状态槽），再做结构机制（双环→PV 分半→Q 双代→O 原地），每步以 Task Duration 仲裁。

## 替换到同类算子：胶水怎么用

1. 本篇规则按"机制"组织（视图声明/双代/动态 region/unit_flag），与具体形状解耦——同构算子直接沿用规则，代入新形状重算 stride/槽位算术。
2. 每条规则附带报错对照（`ValueError: Operation creation failed`、`FE102`、`retained cross-core version` 等）：替换后遇到同族报错先查本文，再怀疑机制本身。
3. 新算子引入新表达层缺口时，按本文格式补条目：规则 + 翻车模式 + 最小复现，不写风格建议。
