# 小 shape（latency-bound）时延优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与实测范围以正文为准。

面向**单 case 只有几微秒、固定开销主导**的 AscendC 算子。核心不是"上一堆技巧"，而是：

> **先量化收益上限，再建隔离 fast path，然后一次只叠加一个优化杆做 A/B，不达标立即回滚。**

本指南把常见优化手段拆成**可独立叠加的杆**，每根杆都给出适用条件、预期、风险和验证方法。**没有哪根杆是必须的**——能否叠加取决于实测。

---

## 适用 / 不适用

- ✅ 单 case 时延 ~1–10us、输出元素数不大、launch/初始化/tiling 搬运等固定开销占比高。
- ✅ 已有算子（builtin 或自研）需要在**不改公开契约**前提下做时延优化。
- ✅ 想把一套已验证的提速模式迁移到同类算子（gather/scatter/index/小 elementwise/小 reduce）。
- ❌ 大 shape 计算/带宽主导（应先做 roofline 与 tiling 分核，不是本技能）。
- ❌ Cube/MatMul 主路径（走 matmul/Blaze 专题）。
- ❌ 精度未通过、功能未跑通的算子（先修正确性）。

---

## 阶段流程（顺序执行，禁止跳步）

```
S0 冻结用例 & 稳定测量
      │
S1 测空核 floor + 逐档 roofline  → 得到"可优化上限"，判断是否值得做
      │
S2 建独立 fast path + 严格门控（隔离，保证可回滚、不污染原路径）
      │
S3 逐杆叠加：每次只加一根杆 → 全量 A/B → 保留/回滚（记录台账）
      │
S4 收敛：确认无回归、补边界正确性、产出报告
```

### S0 冻结用例 & 稳定测量

1. 冻结性能用例集（覆盖 dtype × shape 档位 × 分支形态），**测量期间不改用例**。
2. 测量必须可复现：
   - **pin 空闲卡**（`--devices <空闲id>`），避开别人跑满的卡，否则会出现 100x 级假离群；
   - 每 case `--run 5`（取 median，跳首轮）；
   - 记录基线 CSV，后续所有 A/B 用同一协议。
3. 记录硬件与 CANN 版本、UB 大小、核数（后续判据都要用）。

### S1 测空核 floor + roofline（决定要不要做）

- **空核 floor**：临时在 kernel 入口加 `return;`（仅诊断），测一遍同一用例集 → 得到 launch+最小 setup 的不可压缩底噪 `floor_us`。
- **收益上限**：`max_speedup ≈ baseline_mean / floor_mean`。若目标要求 `> max_speedup`，直接判定不可达，不要浪费轮次。
- **逐档 roofline**：`ceiling_us ≈ 实际搬运字节 / 有效带宽 + floor`；对访存型算子按输出/输入字节估算，判断是 launch-bound 还是 memory-bound。
- **结论**：只有"非 floor 部分"才是可优化空间；把它按档位列出来，作为后续每根杆的目标。

> 诊断用的 `return;` 变体**必须回滚**，不得混入交付。

### S2 建独立 fast path + 严格门控

不要直接改原 kernel 算法。新增一条以**独立 tiling key** 命中的 fast path，并加 `CanUseFast()` 式门控：

- 只对满足条件的 shape/dtype/属性走 fast path，其余保持原路径；
- 先让 fast path 与基线**等价正确**（同精度），再谈性能；
- 门控条件要机器可判（rank、元素位宽、索引 dtype、各维/总量 ≤ uint32 等）。

好处：任何一根杆不达标可单独回滚，且不会影响未覆盖用例。

### S3 逐杆叠加（核心）

**每次只加一根杆**，全量跑同一用例集，计算：
- 总体 `baseline_mean / new_mean`
- 每档/每形态均值
- **最差单例比值**（防止平均变好但个别严重退化）

保留条件（建议）：总体变快 **且** 最差单例不显著退化（例如 ≥0.9x）。否则回滚该杆。把结果写入台账（见文末模板）。

### S4 收敛

- 连续 2 根杆无净收益 → 停止；
- 补边界/负数/非 2 次幂/空 tensor 等正确性用例；
- 产出报告：基线、floor、每杆 A/B、最终值、回滚项与原因。

---

## 优化杆清单（全部可选，按"通用性 × 性价比"排序）

> 编号不代表必须执行顺序；建议先 L1，再 L2，再按需 L3→L7。

### L1. 输出映射改为"最内层连续维合并 + 自适应核/线程数"（通用性最高）

- **目的**：让相邻线程的 GM 读/写连续（合并访问），避免一线程处理一个不连续块导致逐指令跨步。
- **改法**：以"输出元素"为线程索引（而非"索引元组/块"），线程序号 → 输出地址连续；同一输出的内层连续维用常量展开。
- **自适应规模**：核数按工作量 `ceil(outElems / MIN_ELEMS_PER_CORE)`（实测 128 较好）且不超物理核；线程数 `ceil(active, 64)` 上限按平台（实测 1024 优于 2048）。避免小 case 拉起大量空闲核/线程。
- **预期**：小中 shape 最常见的净收益来源；gatherND 单变量 A/B 显示 L1 边际 **+16%**（最终整体 1.17x，S 1.21x / M 1.23x）。
- **风险**：内层维不连续或需要跨步访问时无效；`MIN_ELEMS_PER_CORE` 需按平台调。
- **验证**：对比"按块映射"与"按元素映射"两版；看最差单例。

### L2. 独立 fast path + 门控（隔离前提，非性能杆）

- **目的**：安全迭代、可回滚、不污染原路径。
- **改法**：新 tiling key + `CanUseFast()` 门控 + 独立 kernel 头文件。
- **预期**：本身不改性能，但决定后续能否低风险叠加。
- **风险**：门控过宽会导致原路径退化或正确性问题；过窄则覆盖不到目标用例。

### L3. TPipe-free：kernel 自管 buffer/同步/初始化（不建 TQue/TBuf）

- **目的**：去掉 `TPipe`/`InitBuffer`/TQue 流水建立与同步开销。
- **改法**：fast 分支放在 `TPipe` 构造之前并 `return`；需要 UB 时用 `asc_get_phy_buf_addr(0)+offset` 直接算地址，用 C API 做拷贝/同步。
- **适用**：初始化开销占比高、或算法本身不需要复杂多缓冲流水。
- **预期**：单独收益通常较小（结构性），与 L1 叠加后更明显；**未单独 A/B 时不要宣称收益**。
- **风险**：手管 UB 与 SIMT/VF scratch、栈的地址可能冲突（见反模式）。

### L4. C API 化：去掉高阶 tensor/DMA API

- **目的**：去掉 `GlobalTensor/LocalTensor/DataCopyPad/Reg::*` 的封装与描述符构造开销。
- **改法**：用 `asc_get_block_idx`、`asc_vf_call`、裸 `__gm__` / `__ubuf__` 指针、`asc_copy_*`、`asc_sync`。
- **适用**：访存简单、访问模式规则（可直接裸指针）的小算子。
- **预期**：结构性；与 L1 叠加。
- **风险**：裸指针要自己保证对齐与越界；C API 头文件需在编译 include 路径内（ops-nn/atvoss 等已提供）。

### L5. 紧凑 tiling：header ≤64B + 标量直读 + 删除可现算字段

- **目的**：省掉把 tiling 搬进 UB 的步骤，减少字段读取与位宽。
- **改法**：
  - 只保留 kernel 必需字段，全部用能表示最大值的最小整型（核数/rank 用 `uint32` 甚至 `uint16`）；
  - **删除可由少量计算得到的字段**（如 stride、outputSize），在 kernel 里现算（rank 小时只是几次乘法）；
  - header 压到 ≤64B，kernel 用 `__gm__ const T*` 标量读，不 `DataCopy` 到 UB；
  - 数组维度按实际上限裁剪（如 xShape 只留 rank_max 个）。
- **适用**：tiling 字段多、每次 kernel 都整体搬运的场景。
- **预期**：结构性，通常几 % 级；多核同址读 tiling 的争用只在 payload 较大时值得处理；L6 默认关闭，须单变量 A/B 决定是否叠加（不要默认开启）。
- **风险**：现算字段要保证与 host 计算一致（溢出、位宽）。
- **代码骨架**：
  ```cpp
  struct OpFastHeader {           // 目标 ≤ 64B
      uint32_t blockDim, groupCount;
      uint32_t n, mainSize, ...;  // 仅必要标量
      uint32_t dims[4];           // 按 rank 上限裁剪
  };
  static_assert(sizeof(OpFastHeader) <= 64, "fit one cache line");
  // kernel:
  const __gm__ OpFastHeader* hdr = reinterpret_cast<const __gm__ OpFastHeader*>(tiling);
  ```

### L6. tiling 64B 分组复制（**默认关闭，必须 A/B**）

- **目的**：header 之外的 payload 被多核同时读同一 cache line 会掉速；复制成多份、按核分组各读一份。
- **改法**：
  - host：header 放首 64B；payload 复制到 `64 + g*64` 的 64B 对齐槽，`groupCount = (blockDim <= GROUP) ? blockDim : ceil(blockDim / GROUP)`（与 kernel 取槽公式一致：blockDim≤5 时每核一份）；容量不足时裁剪 groupCount；
  - kernel：`group = (blockDim <= GROUP) ? blockIdx : blockIdx / GROUP;`，读 `tiling + 64 + group*64`。
  - **`GROUP` 取 5**（同一 cache 共享组不超过 5 核）；核数 ≤5 时每核一段。
- **预期：不保证为正，可能明显为负。** gather_nd 实测 **-7%（1.17x → 1.09x）**，且 S/M/L/XL **四个档位一致变慢**——几百字节 tiling 的“复制 + 分组寻址”开销 > 同址争用收益。
- **何时可能为正**：tiling payload 较大（> 数 KB）、核数多、且 payload 被每核高频读取。
- **默认策略**：`groupCount = 1`（单份，不做复制）。只有存在独立证据表明“多核同址读 tiling 造成争用”时才开启，并按 S3 单变量 A/B 决定去留；不要因为“这是已知技巧”就默认开。
- **风险**：payload 必须每份完全一致；`SetDataSize` 与容量要对齐，避免越界；复制会增大 host 序列化与 tiling 体积。

### L7. 索引除法→移位 / 类型按位宽模板化 / 编译期展开

- **目的**：减少每元素算术与分支、减少模板实例数。
- **改法**：
  - host 判断除数是否 2 的幂，传 `divShift/divPow2`，kernel 用 `g >> shift`，否则回退除法；
  - 数据类型按**元素位宽**模板化（`uint8/16/32/64`），而不是按 dtype 各写一份；
  - rank/循环上限用模板常量 + `#pragma unroll`，避免动态数组与运行期分支。
- **预期**：结构性，几 %；在计算型小算子上更有效。
- **风险**：位宽模板要求算子对同宽类型是纯位拷贝语义；否则仍按 dtype 模板。

---

## 反模式（本项目实测失败，务必避免）

| 反模式 | 现象 | 原因 | 结论 |
|---|---|---|---|
| **单核标量循环**替代 SIMT/SIMD 处理小 case | 201 元素：2.6us → 13.9us，且精度异常 | 单核串行 GM 读**丢失内存级并行（MLP）**，延迟串行累积 | 即使几十个元素，SIMT 发射开销也远小于丢失 MLP 的代价 |
| **盲目全量入 UB** 再 gather | 精度失败；且仅极少数 case 能入 UB | UB 容量有限（如 950 约 248KB，扣 DCACHE 后 ~221KB）；手管 UB 与 VF/scratch 冲突 | 只在 `x_bytes ≤ 可用UB` 且索引复用高时才考虑；否则不适用 |
| 只看平均不看离群 | 平均变好但个别 case 2x 变慢 | 不同 shape 档位机制不同 | A/B 必须看**最差单例比值** |
| 不 pin 卡测性能 | 偶发 100x 离群 | 设备被其他任务占满 | 固定空闲卡 + `--run 5` |
| 不隔离直接改原 kernel | 无法回滚、影响全量用例 | 算法改动耦合原路径 | 先建独立 fast path + 门控 |
| **默认开启 tiling 分组复制（L6）** | 全部档位一致变慢 7% | 小 tiling 下复制/分组寻址开销 > 同址争用收益 | 默认单份；开 L6 前必须 A/B |

---

## 标准速记：5 项常见动作的定位

| 动作 | 本质 | 是否"必须" | 通用性 |
|---|---|---|---|
| 自管 buffer/同步（去 TQue） | 减固定开销 | 否 | 中（需要 UB 的算子才相关） |
| 全 C API 化 | 减封装/描述符开销 | 否 | 中 |
| 紧凑 tiling + 标量直读 | 减 tiling 搬运/读取 | 否 | 高（tiling 字段多的算子） |
| 删除可现算 tiling 字段 | 减 host/kernel 数据量 | 否 | 高 |
| tiling 64B 分组复制 | 减多核同址争用 | 否，**且可能负收益** | 视 tiling 大小/核数；gather_nd 实测 -7%，默认关 |

> 这 5 项**不是必选项**，而是可选杆；每加一项都要按 S3 单变量 A/B 决定去留。

**实测经验排序（gather_nd 200 条小 case，慎作普适结论，务必自测）**：
`L1 自适应核/线程（+16%）≫ L7 幂次移位（+1%）> L3/L4/L5 单独叠加（≈中性，1.00x）> L6 分组复制（−7%，回滚）`。
不要按编号顺序默认全开；**先加 L1，再逐项验证其余**。

---

## A/B 测量协议（可复制）

```bash
# 1) 基线（每杆 A/B 都用同一命令与同一设备集合）
python3 -m ttk kernel -i cases.csv --plugin <plugin.py> \
    --devices <空闲卡> --run 5 -o base.csv

# 2) 候选
python3 -m ttk kernel -i cases.csv --plugin <plugin.py> \
    --devices <空闲卡> --run 5 -o cand.csv

# 3) 对比：总体 + 分档 + 最差单例
python3 - <<'EOF'
import csv, statistics
b={r['testcase_name']:float(r['dyn_perf_us']) for r in csv.DictReader(open('base.csv'))}
c={r['testcase_name']:float(r['dyn_perf_us']) for r in csv.DictReader(open('cand.csv'))}
bs=statistics.mean(b.values()); cs=statistics.mean(c.values())
print(f"mean {bs:.3f} -> {cs:.3f}  speedup {bs/cs:.3f}x")
worst=sorted(c, key=lambda k: b[k]/c[k])[:5]
for k in worst: print(f"  {k:28s} {b[k]:.2f} -> {c[k]:.2f}  x{b[k]/c[k]:.2f}")
print("better/worse:", sum(c[k]<b[k] for k in c), "/", sum(c[k]>=b[k] for k in c))
EOF
```

**保留判据（建议）**：总体更快，且最差单例 ≥0.9x；否则回滚该杆并记录原因。

---

## 迁移到其他算子的适配清单

1. **输出映射**：找出输出的最内层连续维；把"最内层维 × 外层维"的线程映射改成元素连续映射（L1）。
2. **元素位宽**：确认算子是否等价于"按元素位宽做位运算/拷贝"；是则用位宽模板（L7），否则按 dtype 模板。
3. **rank 上限**：取实际支持的最大 rank（如 ≤4）作为模板上限，超过则走原路径。
4. **语义**：负索引、越界清零、广播、可选输入等全部保留；fast path 门控要覆盖不到时自动回退。
5. **tiling 字段**：列出 kernel 真正需要的量；能现算的（stride、outputSize、count）一律删除。
6. **UB 容量**：查平台 `ub_size`（如 `x86_64-linux/data/platform_config/*.ini`），DCACHE 预留后才是可用量；决定能否做 UB 方案。
7. **多核**：若启用 L6，host 分组与 kernel 取槽公式必须一致（`blockDim <= 5` 时每核一份），且 payload 每份一致；默认不启用。
8. **正确性回归**：原 ST 用例 + 边界用例 + 非 2 次幂/负数/空 tensor。

---

## 验收 Checklist

- [ ] 记录了基线均值、**空核 floor**、逐档 roofline；
- [ ] fast path 有严格门控，未覆盖用例走原路径；
- [ ] 每根杆都有单变量 A/B 记录（总体 + 分档 + 最差单例）；
- [ ] 最终最差单例无显著退化（≥0.9x）或已说明；
- [ ] 原 ST 用例 100% PASS；补充边界/负数/非 2 次幂用例 PASS；
- [ ] 诊断代码（空核 return、临时开关）已回滚；
- [ ] 报告写清"没达成的目标 + 证据（floor/roofline）"，不夸大。

---

## 性能台账模板（每杆一行）

真实台账（gather_nd，严格单变量 A/B，基线 mean=3.811us）：

| 轮次 | 变体 / 叠加的杆 | 改动摘要 | cand_mean | vs 基线 | 边际 | 最差单例 | 结论 |
|---|---|---|---|---|---|---|---|
| 0 | baseline | 原实现 | 3.811 | 1.00x | — | — | — |
| 1 | B1：fast 核心（L2+L3+L4+L5 + 按元素合并） | 全核、无移位、无复制 | 3.802 | 1.00x | 1.00x | 0.41x | 不单独保留 |
| 2 | B2：**+L1 自适应核/线程** | `ceil(out/128)` | 3.285 | **1.16x** | **1.16x** | 0.94x | ✅ 保留（主收益） |
| 3 | B3：**+L7 幂次移位** | 2 次幂用 `>>` | 3.262 | **1.17x** | 1.01x | 0.95x | ✅ 保留 |
| 4 | B4：**+L6 分组复制** | 64B × groupCount | 3.503 | 1.09x | **0.93x** | 0.89x | ❌ 回滚 |
| 5 | final = B3 | 移除分组寻址 | 3.256 | **1.17x** | — | 0.97x | ✅ 交付 |

失败尝试（另计，均已回滚）：
| 尝试 | 现象 | 结论 |
|---|---|---|
| UB 全量 staging | x≤UB 才适用；精度失败（UB/VF 冲突） | 回滚 |
| 单核标量替代 SIMT | 201 元素 2.6→13.9us，精度异常 | 回滚 |

> 台账要如实记录回滚项，避免后续重复踩坑。

---

## 参考实测（Ascend 950PR + CANN 9.2，仅用于校准预期）

- 空核 floor：**~1.09us**（min 0.86）；理论上限 ≈ 3.811/1.09 = **3.49x**。
- gather_nd 200 条小用例（严格单变量 A/B 后）：**3.811 → 3.256us，1.17x**，195/200 更快，最差单例 0.97x。
  - 分档：S 1.21 / M 1.23 / L 1.19 / XL 1.11；按形态 F1..F5 = 1.14/1.20/1.16/1.18/1.15。
  - 归因：**L1 自适应核/线程 +16%**，L7 +1%，L3/L4/L5 单独≈中性，**L6 −7%（回滚）**。
- 结论：当 floor 已占目标时延预算的大头（如 2x 目标 1.9us，floor 1.09us 占 57%），且大档位是稀疏随机访存时，**2x 通常不可达**；应尽早用 roofline 判定并停止。本算子最终 1.17x 即为该实现族上限附近。

---

## 代码骨架

- 可复制的通用骨架（门控 / 紧凑 tiling / TPipe-free 分发 / per-element VF / 自适应线程 / 空核探针）：[code-patterns.md](code-patterns.md)

---

## 相关技能

- 性能剖析与上板采集：`ops-profiling` / `ascendc-perf-optimize`
- API 约束与 C API 用法：`ascendc-api-best-practices` / `ascendc-docs-search`
- 精度调试：`ascendc-precision-debug`
