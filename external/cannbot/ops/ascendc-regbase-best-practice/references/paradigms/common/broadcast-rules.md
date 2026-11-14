# Broadcast 规则

## 0. 路线总纲

**术语**：**B 轴** = 广播轴（即 `inShape=1 && outShape>1` 的轴）；**A 轴** = 非广播轴（两端等长）；**尾轴 B** = B 轴位于最内维。

按需要 broadcast「数据所在位置」与「是否有可融合的后续计算」选路线：

| 场景 | 路线 |
|------|------|
| 数据在 GM（算子原始输入），需要广播形态 | **路线 1：NDDMA 搬运随路 broadcast** |
| 数据在 UB（中间计算结果），需要广播形态 | **路线 2：UB 内向量 broadcast**（若紧随的计算可与其合并在同一 VF，用 2.3 融合特例：广播结果不写入 UB，直接在寄存器内参与计算） |

**判定约束**：确定 broadcast 源的位置时，cast 视为透明——如 `broadcast(cast(y))`：源认定为 GM 上的 y，走路线 1（cast 在广播之后按 broadcast 后的 shape 做）。broadcast 前的操作是其他计算算子（add、mul、reduce、transpose、matmul、concat…）时，视为数据在 UB。

---

## 1. 路线 1：NDDMA 随路 broadcast（数据在 GM）

**结论**：broadcast 在 GM→UB 搬运阶段由 NDDMA 完成，零向量指令，不区分尾轴 / 非尾轴 B 轴，统一用「每轴 stride 描述」表达（B 轴 `srcStride=0` 原地重读）。搬入后 UB 内即广播展开形态，后续计算直接消费，无需额外 UB buffer。

**NDDMA 指令参考**：NDDMA 五字段定义与 dim[0]=最内维约定、`NdDmaDci()` Cache 刷新、维度上限与分块、随路 broadcast 填法与示例、硬件约束汇总，分别见 [nddma-rules.md](nddma-rules.md) §2.2 / §3.1 / §3.2-3.3 / §4.1 / §5。

---

## 2. 路线 2：UB 内向量 broadcast（数据在 UB）

**结论**：broadcast 只能在 UB 内完成时，全部用**寄存器级（Reg）向量指令手写**，不调用高阶 Broadcast API。按最内轴是否为 B 轴二选一：**尾轴 B——读一个、写一串**；**非尾轴 B——读一串、重复写**。

**通用原则**（任意轴型 AB / BA / ABA / ABAB…，由 shape 在 kernel 内直接推导）：

- 源与目标（下称 src / dst）同维数：**B 轴** src 长 1、dst 长 n；**A 轴**两端等长；
- 遍历：外层各轴每轴一层循环，地址 = 各轴下标 × 该轴 stride，逐项相加（src 的 B 轴长度为 1，下标恒为 0，即源不前进、被重复读取）；最内一或二维套 2.1 / 2.2 之一（最内轴是 B 轴 → 2.1，否则 → 2.2），代码只写最内二维，超过 4 维时最外面的轴套循环包住；dst 一整块连续铺开，顺序往后写；
- 标量源（`srcSize == 1`）：`Reg::Duplicate(reg, scalar)` 填寄存器后按 2.1 的写法重复落 UB。

### 2.1 尾轴 B（AB）：读一个、写一串

场景：src `[size0, 1]` → dst `[size0, size1]`（size1 = B 轴长）。

```cpp
constexpr uint32_t VL = GetVecLen() / sizeof(T);       // 向量寄存器可容纳的元素数
uint16_t repeatTime = (size1 + VL - 1) / VL - 1;       // 整轮数（不含尾轮）
uint32_t tailLen = size1 - repeatTime * VL;            // 尾块长度（整除时 = VL）
__ubuf__ T* dst = dstUb;
__VEC_SCOPE__
{
    Reg::RegTensor<T> brcReg;
    Reg::UnalignRegForStore ureg;
    for (uint32_t i = 0; i < size0; ++i) {
        // 广播分布 load：读 1 个元素 → 寄存器全 lane 同值
        // dtype 选型：1B→DIST_BRC_B8，2B→DIST_BRC_B16，4B/8B→DIST_BRC_B32
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B16>(brcReg, srcUb + i);
        for (uint16_t j = 0; j < repeatTime; ++j) {
            Reg::StoreUnAlign(dst, brcReg, ureg, VL);  // 整轮：count = VL，dst 随调用前移
        }
        Reg::StoreUnAlign(dst, brcReg, ureg, tailLen); // 尾轮：count = 尾块个数（整除时也是 VL）
    }
    Reg::StoreUnAlignPost(dst, ureg, 0);
}
```

要点：
- 若 size1 恒对齐，可换对齐路线：`UpdateMask(remaining)` + `StoreAlign(dstUb + i*size1 + j*VL, reg, mask)`，此时 mask 负责尾轮截断（remaining 引用传入、内部自动递减 VL），省去 ureg 与收尾。

### 2.2 非尾轴 B（BA）：读一串、重复写

场景：src `[1, size1]` → dst `[size0, size1]`（size0 = B 轴长即重复份数，size1 = 段长）。

```cpp
constexpr uint32_t VL = GetVecLen() / sizeof(T);        // 向量寄存器可容纳的元素数
uint16_t repeatTime = (size1 + VL - 1) / VL - 1;        // 整轮数（不含尾轮）
uint32_t tailLen = size1 - repeatTime * VL;             // 尾块长度（整除时 = VL）
__ubuf__ T* dst = dstUb;
__VEC_SCOPE__
{
    Reg::RegTensor<T> srcReg;
    Reg::RegTensor<T> tailReg;
    Reg::UnalignRegForStore ureg;
    Reg::LoadAlign(tailReg, srcUb + repeatTime * VL);   // 尾块源：各份相同，load 一次复用
    for (uint16_t i = 0; i < size0; ++i) {              // 外层按份
        for (uint16_t j = 0; j < repeatTime; ++j) {     // 整轮（不含尾轮）
            Reg::LoadAlign(srcReg, srcUb + j * VL);
            Reg::StoreUnAlign(dst, srcReg, ureg, VL);   // dst 随调用前移
        }
        Reg::StoreUnAlign(dst, tailReg, ureg, tailLen);
    }
    Reg::StoreUnAlignPost(dst, ureg, 0);
}
```

要点：
- `size1 <= VL` 时 repeatTime = 0，仅尾轮，即「读一次、写多次」的特例；

### 2.3 与计算融合的特例形态

**结论**：broadcast 的输出仅被紧随计算消费、且可合并在同一 VF 内时，广播结果不写入 UB，直接以寄存器内广播形态参与计算，体现为 **load 的分布模式** 或 **store 的重复次数**：`load → 计算 → store` 。

**通用流程**（以 `x - broadcast(y)` 类融合为例）：每轮按 dst 最内轴的 VL 粒度（整轮 + 尾轮，同 2.1 / 2.2）——load 广播源 y 构造寄存器内广播形态（尾轴 B 用 `DIST_BRC` 分布；非尾轴 B 读一段）→ load 另一操作数 x 同长一段 → 计算 → store。

**两种形态**（差异仅在 y 侧 load）：

1. **尾轴 B（AB）**：y 侧 load 随路广播（1 元素 → 全 lane），寄存器内即广播形态 → 计算 → store 一次写出 size1 个（超 VL 时 for 循环）。
2. **非尾轴 B（BA）**：y 侧 load 读连续 size1 个 → VF 内其他计算 → store 重复写 size0 次（同 2.2 写法）。

### 2.4 对齐选择原则

| 情形 | load | store |
|------|------|-------|
| 不确定是否对齐 | `LoadUnAlign`（连续多段配 `LoadUnAlignPre`） | `StoreUnAlign` + 结尾 `StoreUnAlignPost` |
| 确定对齐 | `LoadAlign` | `StoreAlign` + `UpdateMask()` |

- 2.1 / 2.2 示例均为「对齐 load + 非对齐流式 store」的通用写法（不要求 size1 对齐）；size1 恒对齐时 store 可换 `StoreAlign + UpdateMask`（见 2.1 要点）；

---

## 3. 指令速查

| 指令 / 手段 | 用途 | 所属路线 |
|-------------|------|----------|
| 多维 `DataCopy(NDDMA)`，B 轴 `srcStride=0` | 搬运随路广播 | 1 |
| `Reg::Duplicate(reg, scalar)` | 标量广播填充（V 通路） | 2 |
| `LoadAlign` + `DIST_BRC_B8/B16/B32` | 读 1 元素 → 全 lane（尾轴 B 广播形态构造） | 2 |
| `LoadAlign` / `LoadUnAlign(Pre)` | 常规读（对齐 / 非对齐） | 2 |
| `Reg::UpdateMask(n)` | 生成 mask 控制写出个数（n 引用传入、内部递减 VL） | 2 |
| `StoreAlign` + `UpdateMask()` | 对齐写 | 2 |
| `StoreUnAlign` + `StoreUnAlignPost` | 非对齐流式连续写（末尾收尾） | 2 |
