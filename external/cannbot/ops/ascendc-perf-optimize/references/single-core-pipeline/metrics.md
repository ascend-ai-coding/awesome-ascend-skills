# 采集与指标口径（msprof 统一）

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


> 本页是 single-core-pipeline 的**统一采集与读图口径**。诊断页（vec/scalar/memory/no-bound）与技术卡片库（`techniques/`）出现的所有指标、泳道、判定阈值，一律以本页定义为准。
> 采集工具统一为 **msprof**（上板 profiling）+ **CANN Simulator**（仿真 trace），不使用任何外部私有采集工具。

---

## 1. 采集

两条采集路径，按目标支持范围及有无 NPU 选择；本库 `ops-simulator` 路线仅用于其支持的 950 目标，A2/A3 缺板时记录动态证据缺口，不用 A5 仿真替代：

| 场景 | 工具 | 技能入口 | 产物 |
| --- | --- | --- | --- |
| 上板实测 | msprof（逐 case `aiv_time`） | `/ops-profiling` | `kernel_details.csv`（含 aiv_time/各 ratio）、`op_summary.csv` |
| 无硬件仿真 | CANN Simulator | `/ops-simulator` | 仿真流水 trace（chrome trace json）、指令时间线 |

- **逐 case 采集**：性能优化工作流对 `{cases_csv}` 中**每个 case** 单独采一次 `aiv_time`，禁止只挑代表性用例；执行协议取调用方与 `ops-profiling`；CANN-Bench A5 的采集流程仅适用于该目标/场景，不能替代 A2/A3 或普通任务协议。
- **仿真 trace** 用于在无板环境看流水形态（空泡、泳道占用）；**最终性能结论以上板 msprof `aiv_time` 前后对比为准**，仿真只作定性初判。

---

## 2. 指标词汇表

| 指标 | 定义 | 从哪读 |
| --- | --- | --- |
| `aiv_time` | AI Vector core 单 case 执行耗时（us） | msprof `kernel_details.csv`，优化前后比这个 |
| `aiv_vec_ratio` | Vector（计算）指令耗时占 aiv_time 比例 | msprof；判 VEC bound 主指标 |
| `aiv_mte2_ratio` | MTE2（GM→UB 载入）耗时占比 | msprof；判访存 bound |
| `aiv_scalar_ratio` | 标量指令耗时占比 | msprof；判 scalar bound |
| 各单元 util | SCALAR/MTE2/VECTOR/MTE3 泳道忙碌时长 / 总跨度 | 仿真 trace 或 msprof ratio |
| lane 利用率 | 一条向量指令实际点亮 lane 数 / 满 lane（VL） | 仿真 trace 逐指令 mask 宽度推算 |
| 指令发射率 (IPC) | 每周期发射指令数 | 仿真 trace 定性；无绝对达标线，只做前后对比 |
| `aiv_mte3_wait_ratio` | 写回泳道处于等待的时间占比 | msprof `ResourceConflictRatio.csv`；> 0.4 多为消费阶段紧跟异步计算（nobound-02） |
| 重叠因子 | 各泳道忙碌时长之和 / 实际耗时（见 §2.1） | msprof `PipeUtilization.csv`；理想上界 `Σ / max`，越接近说明流水重叠越充分 |

### 2.1 重叠因子：判"流水并行够不够"的入口指标

只用 `PipeUtilization.csv` 一张表即可，**上板随时能算，不需要仿真**：

```
重叠因子 = Σ(aiv_vec_time + aiv_scalar_time + aiv_mte2_time + aiv_mte3_time) / aiv_time
         = Σ(各 ratio)                                    # 两式等价，ratio 本就是 time/aiv_time
理想上界 = Σ / max(各 ratio)                               # 全并行时墙钟收敛到最长那条泳道
达成率   = 重叠因子 / 理想上界
```

实测算例（A5，56 核均值，某 RMSNorm+RoPE+KVCache 融合算子优化前）：

| 泳道 | 耗时 | ratio |
| --- | --- | --- |
| MTE2 | 29.52 us | 0.513 |
| VEC | 27.73 us | 0.483 |
| SCALAR | 27.57 us | 0.480 |
| MTE3 | 9.15 us | 0.159 |
| 合计 | **93.97 us** | **1.634** |

`aiv_time = 57.52 us`（Task Duration 60.08 us）。重叠因子 1.634、上界 `1.634/0.513 = 3.186`、**达成率 51%**——四条泳道忙了 93.97us 却只压缩到 57.52us，全并行本应收敛到最长的 29.52us。**没有单一泳道超过 0.52**，所以不是任何一种 bound，是编排问题 → 走 `no-bound.md`。

> 达成率没有硬阈值，但**接近 100% 说明已经排满、再调顺序无收益**；50% 量级说明一半时间在互相等，值得下钻。下钻手段见 `techniques/nobound-02`。

---

> **没有"绝对达标 IPC/util"**。判定永远是**优化前 vs 优化后同一指标对比**：`aiv_time` 降即正收益。指标占比阈值（如 Cast >20%、`aiv_vec_ratio` >80%）是经验判定线，阈值附近仍以 `aiv_time` 实测为准。

---

## 3. chrome trace 读图（5 条泳道）

A5（arch3510）RegBase 范式的仿真/msprof trace 按 5 条泳道观察。盯 **VECTOR（计算主力）** 泳道：越密（区间首尾相连、无空隙）说明计算单元越忙；空泡（bubble）越多说明被等待拖累，再看是被 MTE2/MTE3（访存）还是 SCALAR/SCALARLDST（标量准备）挡住。

| 泳道 | 含义 | 看什么 |
| --- | --- | --- |
| SCALAR | 标量计算 / 分支 / 地址准备 | 忙 → scalar 准备挡住计算（scalar bound 信号） |
| SCALARLDST | 标量 load/store、mask/setup | 与 SCALAR 一起忙 → 标量 round-trip、V_S/S_V 同步 |
| MTE2 | GM→UB 载入 | 忙且接近跨度 → 访存 bound；VF 路径几乎只搬 TilingData |
| VECTOR | 向量执行（计算主力） | 密 → 计算 bound；空泡 → 被依赖/寄存器/访存拖累 |
| MTE3 | UB→GM 写回 | 与 MTE2 成对频繁 → store→load 往返、未融合 |

**要点观察**：
- **DoubleBuffer 是否生效**：MTE2 与 VECTOR 交替出现 = 搬入与计算重叠（健康）；MTE2 全在前、VECTOR 全在后 = 串行（DB 未生效）。
- **依赖链空泡**：VECTOR 出现周期性空泡且此时 MTE2/MTE3 也不忙 → 依赖链长 / 寄存器压力（非访存）。
- **寄存器 spill**：VF 计算段内出现成对的 spill→reload 指令（preg 溢出，int64 等实际占用翻倍时常见），打断计算流。

> RegBase（RVEC）范式靠向量 load/store 在 UB↔向量寄存器间搬数，**MTE2/MTE3 几乎只搬 TilingData**。因此 VF 路径的瓶颈判断**不要拿 MTE2/MTE3 当 VF 瓶颈**——先看 VECTOR 空泡与 SCALAR 准备。

---

## 4. 概念对照表（从外部 VF 案例移植时翻译）

部分外部 VF 素材使用一套 RegBase 细粒度泳道/指标命名（RVEC 单元名 + 仿真 cycle/IPC）。移植其策略到本库卡片时，**一律翻译成下表右列的 msprof/trace 口径**，卡片正文只用右列词汇，不引任何外部工具命令、报告文件名或 golden 校验开关。

| RegBase 细粒度概念（仅作翻译参照） | 本库（msprof/trace）等价 | 说明 |
| --- | --- | --- |
| RVECEX（RVEC 执行单元）泳道 | VECTOR 泳道 / `aiv_vec_ratio` | 向量执行占比、计算主力 |
| RVECSU（RVEC setup 单元）忙 | SCALAR/SCALARLDST 泳道忙 + V_S/S_V 同步 | 标量地址/mask/准备开销 |
| RVECLD / RVECST（RVEC 载入/存储） | MTE2（载入）/ MTE3（写回）泳道 | 向量访存往返（VF 内为 UB↔寄存器） |
| preg spill/reload 指令成对出现 | trace 现 preg spill/reload | 寄存器溢出，int64 用双寄存器时实际 preg ×2 |
| 仿真 `total_cycle` / `IPC` | msprof `aiv_time`(us) / aicore cycle | 性能基准量，优化前后对比 |
| 外部工具的 golden 正确性校验 | 本工作流 `func_cases` 精度回归 | 正确性由算子级精度门（Step 0.5 / 阶段 2c）保证 |

> 说明：右列才是本库口径。凡卡片"何时用"引用泳道，用 VECTOR/MTE2/MTE3/SCALAR + `aiv_*_ratio`；凡"性能数据"给数字，用 `aiv_time`/aicore cycle；正确性一律指向 `func_cases` 精度回归，不引外部 golden 工具。
