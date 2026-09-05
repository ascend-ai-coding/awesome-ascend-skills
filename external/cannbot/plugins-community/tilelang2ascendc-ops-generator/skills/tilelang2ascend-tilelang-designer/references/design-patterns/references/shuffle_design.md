# 重排/搬运类算子设计要点（Design 视角）

> 本文件只含**设计阶段决策**（可用 TileLang DSL 表达）。AscendC 实现细节
> （GatherMask 指令参数、dstStride 单位、dtype 分路的 if constexpr 等）见
> translator references/ascendc_shuffle_patterns.md。

## 适用

交织重排、奇偶拆分、stride 切片重组（`chunk`/`split`/`cat`/`stack`）、gather/scatter、
广播消费（RoPE 交织、RotaryMul、permute 类）等数据搬运主导的矢量算子。

## 设计决策 1：规律重排用 pattern 结构，不建表

- 奇偶/隔点/半区对调等**规律重排**：设计时走"规律 pattern"结构——在 tile 数据流层面
  表达为固定步长取数（预留硬件 pattern 指令映射），**不生成逐项 offset 表**
- 每-launch 建表是固定开销，小 shape 下直接决定性能档位（实测仅建表 ~17.5µs/launch）

## 设计决策 1b：旋转/半区对调用分段搬移 + 常数乘，零表

- 旋转/半区对调类（RoPE/RotaryMul 的 `half`：`rotated = [-x[D/2:], x[:D/2]]`）设计成
  **两次"半区搬移" + 后半段对应系数乘 -1**——比 pattern 指令更轻，**连表都不用建**
- 负号**融合进对应分段**（如 r2 前半段乘 -1），不单独建 sign 表
- interleave（相邻列对调，`col^1`）官方算子多未覆盖，需单独设计（偶/奇列拆开重排），
  同样走搬移/重排思路，**不走 Gather**

## 设计决策 2：广播消费 = 多行共享源行

- 广播源（cos/sin S=1 等）设计成"一个 tile 内的行共享同一源行"的数据流：
  **行段游走（row-run）** + **源行去重**（一个 tile 只搬覆盖到的 1~2 个源行）
- 不物化中间张量（不做 expand 拷贝）——那是一次 S 倍设备拷贝

## 设计决策 3：按最终布局摆放

- 计算时就按输出**最终内存布局**摆放中间结果（利用 repeat-stride），让写回退化为
  单次连续搬移，避免"先算分段布局、再多次跨步拼写回"

## 设计决策 4：固定开销结构选择

- 小 shape（耗时与 N 无关）先算**固定成本账**：任何"每 launch 重建/重算"的结构都要质疑
  （offset 表、index 缓冲、多余 kernel 拆分）
- 先压固定开销，再谈数据通路

## 设计决策 5：host 核数分档（与归约类同规律）

- AIV block 分发存在 ~60-100ns/核的 ramp，小/中规模满核 launch 白付 1-3µs 分发倾斜
- 按规模带分档：小 → 每核至少若干行才多开核；中 → 夹在 [4,16] 区间；大 → 满核
- 档位边界在目标硬件实测确定（固定核数 × 规模阶梯 × 多次取均），不要拍脑袋

## 设计决策 6：按 dtype 分类设计（16bit 与 fp32 分路径）

重排/搬运类的结构选择**强依赖 dtype**，设计阶段就应分类讨论、单独实现：

| dtype | `GatherMask` 专用指令（硬件 pattern） | 搬移/重排方案（`data_move` 半区搬移、cat 式重排） | 建表 `Gather` |
|---|---|---|---|
| **fp16/bf16（16bit）** | ✅ stride-2 pattern（16bit 元素对） | ✅ 通用 | 不必要 |
| **fp32** | ⚠️ 指令层支持 float（CANN dav_c220 断言、asc-devkit 性能表均含 float），但 stride-2 pattern 实测行为不符预期 → 优先搬移方案 | ✅ **通用，且是无 pattern 时的首选** | 兜底 |

- `GatherMask` dtype 支持：stride-2 pattern 在 16bit 元素对上是主流路径；`float` 在指令层受支持
  （CANN 8.3.RC1 dav_c220 断言、asc-devkit 性能表均含 float），但 **fp32 上 stride-2 pattern 实测
  行为不符预期**——故 fp32 优先走 `data_move` 搬移/重排方案，而非硬凑 pattern
- **`data_move` 搬移等重排方案对所有 dtype 通用**（RoPE half 的搬移方案不挑 dtype）
- 按 dtype 分类单独实现，编译期 `if constexpr` 分派，不硬凑

> ⚠️ 术语区分：**`GatherMask` 专用指令（硬件 pattern）≠ `Gather` + 手工 mask 表（反模式）**。
> 后者在 half 上会崩溃（mask 符号扩展垃圾、offset 构造错误、旋转失效），切勿混用。

## 设计决策 7：克隆/全量搬运主导 = 独立 memcpy kernel（bulk copy）

- `output = clone(input)` + 少量更新（量化 scatter / index_put / 部分行覆写）形态，
  克隆:更新流量比 ≥100:1 时，**克隆段决定性能档位**——把它当独立 memcpy kernel 设计：
  input 扁平化 1D、按 chunk（8~64KB 级）均分全部核、双缓冲流水让搬入搬出重叠
- 更新段与克隆段存在跨核写依赖（互相覆盖）→ **拆双 kernel**（memcpy kernel + update
  kernel），顺序 launch 天然有序，不在 kernel 内做跨核同步
- TileLang 层若无法表达可靠的跨核写依赖时序（无跨核同步原语），不要硬撑单 kernel——
  设计期直接定双 kernel 结构并写入 PERF_DESIGN 待验证清单，Phase 4 按双 kernel 落地
- 微段化 + 全管线屏障是 memcpy 的头号反模式（实测 512B 段 × 段间全屏障把带宽钉死
  一个数量级）；段长只受 count 32B 对齐约束，与"安全"无关
