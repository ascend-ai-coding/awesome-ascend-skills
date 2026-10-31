# broadcast 范式 standard Tiling 预处理

> ub 切分前的预处理阶段

## 1 算子信息获取

### 1.1 平台信息获取

```cpp
coreNum      = ascendcPlatform.GetCoreNumAiv();
ubSize       = ascendcPlatform.GetCoreMemSize(UB, ubSize);
blockSize    = Ops::Base::GetUbBlockSize(context);      // 不要写 32
cacheLineSize = Ops::Base::GetCacheLineSize(context);   // 不要写 256/512
vectorSize   = Ops::Base::GetVRegSize(context);
```

**约束**：
- 每个返回值必须校验非 0，报错用 `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON`（param 取 `platform.<参数名>`，reason 注明查询接口名）
- 严禁将平台参数写死，必须通过接口获取

### 1.2 算子输入输出获取及校验

> 若本算子有输入需要在 Tiling 阶段读取张量**值**（而非仅 shape/dtype），属于值依赖，必须按 [value-depend-process.md](../../../common/value-depend-process.md) 在 op_def / INFERSHAPE / OPTILING 三处同时声明。Broadcast 类算子通常只依赖 shape，无值依赖；如有例外按该文档处理。

**约束**：
- 获取输入shape要使用 EnsureNotScalar
- 获取可选输入要使用 GetOptionalInputShape 和 GetOptionalInputDesc
- 可选输入为空时要特殊处理，不能直接返回失败
- 可选attr为空时要使用默认值，不能直接返回失败
- shape/dtype 要严格校验，不支持的 shape/dtype 必须报错。**错误日志分层**：参数值错误用 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON`、shape 维度错误用 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON`、dtype 错误用 `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON`（报错 param 用参数名如 `x1`，不用 `input(0)`；dtype 序列化用 `Ops::Base::ToString(dtype)`，不用 `ge::TypeUtils::DataTypeToSerialString`）
- **空 shape 防御**：任一输入/输出维度 `<= 0` 时必须 `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON` 报错返回（`HasNonPositiveDim`，见 §2）——0 维会以 maxDim=0 进入 maximumBroShape，使 totalTiles=0 → usedCoreNum=0，导致 CeilDiv 除零与 SetBlockDim(0) 退化 tiling
- 必须使用Info级别日志打印算子每个输入参数/属性，包括但不限于shape、dtype、attr

**Host 侧 attr 获取**：用 `GetBool`/`GetFloat`/`GetInt` 类型化 getter，返回指针，`nullptr` 表示未设→用默认值：

```cpp
auto attrs = ctx_->GetAttrs();
const bool *fp = attrs->GetBool(ATTR_INDEX);
fixed_min_ = (fp != nullptr && *fp) ? 1LL : 0LL;
const int64_t *np = attrs->GetInt(ATTR_INDEX);
int64_t num_bits = np ? *np : 8;
```

**TilingData 类型对齐**：TilingData 里**不用 bool**，统一用 `int64_t`（8 字节对齐）。host 侧 `bool`→`int64_t`（0/1），kernel 侧在 `Init()` 算好 multiplier，VF 里用算术消分支。

```cpp
// TilingData
int64_t fixed_min;  // 不用 bool

// Init
fixed_min_mul_ = td_->fixed_min ? 0.0f : 1.0f;

// VF — 无 if，Mul 替代 Select
AscendC::Reg::Min(regOCMin, regCmin, regZero, mask);
AscendC::Reg::Mul(regOCMin, regOCMin, fixedMinMul, mask);
```

**二进制配置**：属性不决定分发（同一 binary 处理所有 attr 组合）→ `"value": null`。属性决定分发（不同 attr 值走不同 binary）→ 不同 entry 不同 value。

**Broadcast 场景特有校验**：

输入预处理是每个 Broadcast 算子都要做的通用步骤。目标：将不同 shape 的入参和出参归一化，拉到同一个参照系 `maximumBroShape` 下。

预处理产出三类 shape：
- `maximumBroShape` — 坐标系，所有输入/输出的 broadcast 上界
- `normalInputShapes` — 归一化后各输入 shape，与 maximumBroShape 同 rank，输入 dim=1 标识 broadcast 轴
- `normalOutputShapes` — 归一化后各输出 shape，与 maximumBroShape 同 rank 且**必须稠密**（每维等于该维 broadcast 最大值）。广播输出（某维为 1）经 stride=0 映射后多核 tile 落到同一 GM 地址构成写-写数据竞争，Host 侧 `CheckBroadcastShape` 直接拒绝

## 2 合轴

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §5.1（输入预处理）

> 顺序：补 1 → 去 1 → 归一 → 合轴，**顺序不可换**

(1) 补 1: 低 rank 入参/输出在 shape 最前面补 1，拉齐到 maxRank（含入参+出参）。**只能在 shape 最前面补 1**，不允许在中间或末尾维度补 1
(2) 去 1: 对齐后，所有入参+出参在某轴都是 1 → squeeze 掉，消除废维度
(3) 归一: 去 1 后若 maximumBroShape 为空（全部均为标量），填 (1,)
(4) 合轴: 归一后从最后一维向前贪心合并相邻维。对候选组 [d..groupHighDim]（闭区间），每个入参/出参在组内的维度乘积必须为 **1**（整组广播）或等于**坐标系 maximumBroShape 在组内的乘积**（整组稠密）；组内部分维为 1、部分维稠密的混合模式不能合并。扩展失败时封闭当前组，从当前维重新开组

**校验**：补 1 后逐维检查 broadcast 兼容性——对每个维度，若某入参在该维的大小不为 1 且与其他入参同维不为 1 的大小不同，则报错；同时校验输出稠密性——该维存在非 1 值时输出不允许为 1（广播输出报错）。

**6 个示例**：

例 1（正常广播，补 1 个 1）：
```
入参 A: shape = (4, 8)
入参 B: shape = (1, 8)
入参 C: shape = (8,)       // 补 1 → (1,8)

补 1 → A=(4,8), B=(1,8), C=(1,8)
去 1 → dim0: A=4, B=1, C=1 → 不全为 1，保留
       dim1: A=8, B=8, C=8 → 全为 8≠1，保留
maximumBroShape = (4, 8)
校验 → dim0: 4,1,1 → ref=4 ✓
       dim1: 8,8,8 → ref=8 ✓

合轴（从最后一维 dim1 开始，向左尝试扩展）：
  尝试 [dim0,dim1]: A=4×8=32, B=1×8=8, C=1×8=8, 坐标系组乘积 = 32
                    → B/C 的 8 既非 1 也非 32 ✗ → 封闭组 (8)，从 dim0 开新组
  尝试 [dim0]: 单维成组

合轴后: maximumBroShape = (4, 8)   — 无可合并，保持不变
        normal_A = (4, 8)
        normal_B = (1, 8)
        normal_C = (1, 8)
```

例 2（正常广播，补多个 1）：
```
入参 A: shape = (2, 4, 8)
入参 B: shape = (8,)       // 补 1 → (1,1,8)

补 1 → A=(2,4,8), B=(1,1,8)
去 1 → dim0: A=2, B=1 → 不全为 1，保留
       dim1: A=4, B=1 → 不全为 1，保留
       dim2: A=8, B=8 → 全为 8≠1，保留
maximumBroShape = (2, 4, 8)
校验 → dim0: 2,1 → ref=2 ✓
       dim1: 4,1 → ref=4 ✓
       dim2: 8,8 → ref=8 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: A=4×8=32, B=1×8=8, 坐标系组乘积 = 32
                    → B 的 8 既非 1 也非 32 ✗ → 封闭组 (8)，从 dim1 开新组
  尝试 [dim0,dim1]: A=2×4=8, B=1×1=1, 坐标系组乘积 = 8
                    → A 稠密(8=maxProd)、B 整组广播(1) ✓ → 合并，当前组 = (8)

合轴后: maximumBroShape = (8, 8)
        normal_A = (8, 8)   — 原 dim0,dim2 合并为单维 8
        normal_B = (1, 8)   — 原前两维整组广播，合并后该维为 1
```

例 3（broadcast 不兼容报错）：
```
入参 A: shape = (4, 8)
入参 B: shape = (4, 3)

补 1 → A=(4,8), B=(4,3)     // 同 rank，无需补 1
去 1 → dim0: 4,4 → 不全为 1，保留
       dim1: 8,3 → 不全为 1，保留
maximumBroShape = (4, 8)
校验 → dim0: 4,4 → ref=4 ✓
       dim1: 8,3 → 两者均非 1 且 8≠3 → 报错！
→ OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "input size and ref size", "dim 1 input[1] size 3 and 8", "broadcast incompatible: input sizes must be equal or 1")
→ return false
（校验失败直接返回，不进入合轴 — 合轴必须在 CheckBroadcastShape 通过之后执行）
```

例 4（标量）：
```
入参 A: shape = ()
入参 B: shape = ()

补 1 → A=(), B=()     // rank=0，无轴可补
去 1 → maximumBroShape = ()
归一 → maximumBroShape = (1,)
normal_A = (1,)
normal_B = (1,)
校验 → dim0: 1,1 → 全为 1 ✓
合轴 → rank=1，单维无可合并，保持 (1,)
```

例 5（合轴，从最后一维向前贪心合并）：
```
入参 A: shape = (2, 3, 4, 5)
入参 B: shape = (2, 3, 1, 1)
输出 C: shape = (2, 3, 4, 5)

补 1 → rank 已对齐，无需补 1
去 1 → 每维均非全 1，无废维
坐标系 maximumBroShape = (2, 3, 4, 5)

合轴（从最后一维 dim3 开始，向左尝试扩展）：
  尝试 [dim2,dim3]: A=4×5=20, B=1×1=1, C=4×5=20
                    → 乘积均为 20 或 1 ✓ → 合并，当前组 = (20)
  尝试 [dim1,dim2,dim3]: A=3×20=60, B=3×1=3, C=3×20=60
                    → 3 既非 1 也非 60 ✗ → 封闭组 (20)，从 dim1 开新组
  尝试 [dim0,dim1]: A=2×3=6, B=2×3=6, C=2×3=6
                    → 乘积全等 ✓ → 合并，当前组 = (6)

合轴后: maximumBroShape = (6, 20)
        normal_A = (6, 20)
        normal_B = (6, 1)     // 整组广播 → 合并后该维为 1
        normal_C = (6, 20)
校验 → dim0: 6,6,6 → ref=6 ✓
        dim1: 20,1,20 → ref=20 ✓
```

例 6（广播输出报错）：
```
入参 A: shape = (4, 8)
输出 C: shape = (1, 8)     // 输出 dim0 为广播维

补 1 → A=(4,8), C=(1,8)
校验 → dim0: A=4（ref=4），C=1 → 该维存在非 1 值（ref != -1）而输出为 1
       → 输出稠密性违例，报错！
→ OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "output size and ref size",
   "dim 0 output[0] size 1 and 4",
   "broadcast output is not supported: output must be dense (equal to broadcast max dim)")
→ return false
（广播输出维 stride=0，多核 tile 映射到同一 GM 地址并发写构成写-写数据竞争，Host 侧直接拒绝）
```

**合轴的判定本质**：组内每个张量的乘积只有两种合法取值——`1`（该张量整组都是 1，合并后整维广播，stride=0）或 `坐标系组乘积`（该张量整组无 1，合并后整维稠密连续）。组内混有 1 和非 1（如上例 B 的 `3×1×1=3`）说明广播边界落在组内部，合并后会破坏元素映射，必须拒绝。

**合轴的收益**：rank 与循环维数下降、内层连续段变长——NDDMA 单次搬运量从"最内维 × 少数几维"提升为"整组连续元素"，且 rank>5 时 split 轴更容易进入 NDDMA 的 5 维窗口，避免退化成 kernel 侧逐次 DataCopy 的碎片化搬运。

**HasNonPositiveDim 完整参考实现**（空 shape 防御，在读取 raw shape 后、PadAndSqueeze 之前调用）：

```c++
// HasNonPositiveDim: 任一维度 <= 0（空 shape / 非法值）时返回 true。
// 0 维会以 maxDim=0 进入 maximumBroShape, 使 totalTiles=0 → usedCoreNum=0,
// 导致 CeilDiv 除零与 SetBlockDim(0) 退化 tiling, 须在 Host 侧拦截。
static bool HasNonPositiveDim(const std::vector<std::vector<int64_t>>& shapes)
{
    for (const auto &s : shapes) {
        for (int64_t d : s) {
            if (d <= 0) {
                return true;
            }
        }
    }
    return false;
}
```

**PadAndSqueeze 完整参考实现**：

```c++
// PadShape: 单个 shape 前补 1 对齐到 maxRank（只能在 shape 最前面补 1）。
// 依赖全部显式参数化（无捕获），替代原先的 lambda pad。
static std::vector<int64_t> PadShape(const std::vector<int64_t>& s, int64_t maxRank)
{
    std::vector<int64_t> p;
    p.assign(static_cast<size_t>(maxRank - static_cast<int64_t>(s.size())), 1);
    p.insert(p.end(), s.begin(), s.end());
    return p;
}

// PadAndSqueeze: 输入预处理——补 1（低 rank 在最前面补 1 对齐）→ 去 1（全 1 维 squeeze）
// → 归一（全标量时填 (1,)），产出 broadcast 坐标系 maximumBroShape 及与之间 rank 的
// normalInputShapes / normalOutputShapes。
void PadAndSqueeze(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<std::vector<int64_t>>& outputShapes,
    std::vector<int64_t>&                    maximumBroShape,
    std::vector<std::vector<int64_t>>&       normalInputShapes,
    std::vector<std::vector<int64_t>>&       normalOutputShapes)
{
    int64_t numInputs  = static_cast<int64_t>(inputShapes.size());
    int64_t numOutputs = static_cast<int64_t>(outputShapes.size());
    int64_t maxRank = 0;
    for (auto& s : inputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }
    for (auto& s : outputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }

    std::vector<std::vector<int64_t>> paddedIn(numInputs), paddedOut(numOutputs);
    for (int64_t i = 0; i < numInputs; i++) {
        paddedIn[i] = PadShape(inputShapes[i], maxRank);
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        paddedOut[i] = PadShape(outputShapes[i], maxRank);
    }

    maximumBroShape.clear();
    normalInputShapes.assign(numInputs, std::vector<int64_t>());
    normalOutputShapes.assign(numOutputs, std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            if (paddedIn[i][d] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedIn[i][d]);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            if (paddedOut[i][d] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedOut[i][d]);
        }
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs; i++) {
                normalInputShapes[i].push_back(paddedIn[i][d]);
            }
            for (int64_t i = 0; i < numOutputs; i++) {
                normalOutputShapes[i].push_back(paddedOut[i][d]);
            }
        }
    }
    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs; i++) {
            normalInputShapes[i].push_back(1);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            normalOutputShapes[i].push_back(1);
        }
    }
}
```

**BroadcastMergeAxis 完整参考实现**：

> ⚠ 必须在 `CheckBroadcastShape` 通过之后调用：合轴的乘积判等依赖"每维取值 ∈ {1, 该维最大值}"这一前提，未校验的 shape 可能出现乘积巧合相等但组内不连续的误合并。

```c++
// MergeShapeByGroups: 按分组闭区间列表把 shape 每组收缩为一维（组内乘积）。
// 依赖全部显式参数化（无捕获），替代原先的 lambda mergeShape。
static std::vector<int64_t> MergeShapeByGroups(
    const std::vector<int64_t>& s, const std::vector<std::vector<int64_t>>& groups)
{
    std::vector<int64_t> merged;
    merged.reserve(groups.size());
    for (auto& g : groups) {
        int64_t prod = 1;
        for (int64_t k = g[0]; k <= g[1]; k++) {
            prod *= s[k];
        }
        merged.push_back(prod);
    }
    return merged;
}

void BroadcastMergeAxis(
    std::vector<int64_t>&              maximumBroShape,
    std::vector<std::vector<int64_t>>& normalInputShapes,
    std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    int64_t numInputs  = static_cast<int64_t>(normalInputShapes.size());
    int64_t numOutputs = static_cast<int64_t>(normalOutputShapes.size());
    int64_t oldRank = static_cast<int64_t>(maximumBroShape.size());
    if (oldRank <= 1) {
        return;  // 单维无可合并
    }

    // 从最后一维向前贪心分组; groups[i] = {groupLowDim, groupHighDim} 闭区间，按从右到左的发现顺序收集
    std::vector<std::vector<int64_t>> groups;
    int64_t groupLowDim = oldRank - 1;
    int64_t groupHighDim = oldRank - 1;
    for (int64_t d = oldRank - 2; d >= 0; d--) {
        // 坐标系在 [d..groupHighDim] 上的组乘积（合并后该组的维度大小）
        int64_t maxProd = 1;
        for (int64_t k = d; k <= groupHighDim; k++) {
            maxProd *= maximumBroShape[k];
        }
        // 每个张量在 [d..groupHighDim] 上的乘积必须为 1（整组广播）或 maxProd（整组稠密）
        bool mergeable = true;
        for (int64_t i = 0; i < numInputs && mergeable; i++) {
            int64_t prod = 1;
            for (int64_t k = d; k <= groupHighDim; k++) {
                prod *= normalInputShapes[i][k];
            }
            if (prod != 1 && prod != maxProd) {
                mergeable = false;
            }
        }
        for (int64_t i = 0; i < numOutputs && mergeable; i++) {
            int64_t prod = 1;
            for (int64_t k = d; k <= groupHighDim; k++) {
                prod *= normalOutputShapes[i][k];
            }
            if (prod != 1 && prod != maxProd) {
                mergeable = false;
            }
        }
        if (mergeable) {
            groupLowDim = d;                          // dim d 并入当前组
        } else {
            groups.push_back({groupLowDim, groupHighDim});      // 封闭当前组，从 dim d 开新组
            groupLowDim = d;
            groupHighDim = d;
        }
    }
    groups.push_back({groupLowDim, groupHighDim});
    // groups 按从右到左发现，反转为从左到右
    std::reverse(groups.begin(), groups.end());

    // 每组收缩为一维：坐标系取组内乘积，张量取自身组内乘积（整组广播自然收缩为 1）
    maximumBroShape = MergeShapeByGroups(maximumBroShape, groups);
    for (int64_t i = 0; i < numInputs; i++) {
        normalInputShapes[i] = MergeShapeByGroups(normalInputShapes[i], groups);
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        normalOutputShapes[i] = MergeShapeByGroups(normalOutputShapes[i], groups);
    }
}
```

**CheckBroadcastShape 完整参考实现**：

> ⚠ 报错必须使用专用上报接口 `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON`，禁止裸 `OP_LOGE`。
> 逐维校验两条规则：(1) 输入同维非 1 大小必须一致；(2) 输出必须稠密——该维存在非 1 值（ref != -1）时输出不允许为 1。广播输出维 stride=0，多核 tile 映射到同一 GM 地址并发写构成写-写数据竞争，Host 侧直接拒绝。

```c++
bool CheckBroadcastShape(
    const std::vector<std::vector<int64_t>>& paddedIn,
    const std::vector<std::vector<int64_t>>& paddedOut,
    int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedIn[i][d];
                } else if (paddedIn[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "input size and ref size",
                        (std::string("dim ") + std::to_string(d) + " input[" + std::to_string(i) +
                         "] size " + std::to_string(paddedIn[i][d]) + " and " + std::to_string(ref)).c_str(),
                        "broadcast incompatible: input sizes must be equal or 1");
                    return false;
                }
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedOut[i][d];
                } else if (paddedOut[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) +
                         "] size " + std::to_string(paddedOut[i][d]) + " and " + std::to_string(ref)).c_str(),
                        "broadcast incompatible: output sizes must be equal or 1");
                    return false;
                }
            }
        }
        // 输出稠密性校验: 该维存在非 1 维值 (ref != -1) 时, 输出不允许为广播维 (1)。
        // 广播输出维 stride=0, 多核 tile 映射到同一 GM 地址并发写, 构成写-写数据竞争,
        // 广播输出无定义, Host 侧直接拒绝。
        if (ref != -1) {
            for (size_t i = 0; i < paddedOut.size(); i++) {
                if (paddedOut[i][d] == 1) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) +
                         "] size 1 and " + std::to_string(ref)).c_str(),
                        "broadcast output is not supported: output must be dense (equal to broadcast max dim)");
                    return false;
                }
            }
        }
    }
    return true;
}
```

**调用约定**：`HasNonPositiveDim` 在读取 raw shape 后、`PadAndSqueeze` 之前经 `OP_CHECK_IF` 拦截（任一维 `<= 0` 直接报错返回）。`PadAndSqueeze` 为 `void`（无失败路径——shape 合法性已由 `HasNonPositiveDim` 前置拦截），调用方无需检查返回值。broadcast 兼容校验由 `CheckBroadcastShape` 独立执行（输入同维非 1 一致 + 输出稠密），通过 `OP_CHECK_IF` 拦截：

```c++
// 空 shape 防御: 拒绝任一维 <= 0, 避免下游 CeilDiv 除零 / SetBlockDim(0)
OP_CHECK_IF(HasNonPositiveDim(rawInputShapes_) || HasNonPositiveDim(rawOutputShapes_),
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output",
        "dim size <= 0", "all dim sizes must be positive (>= 1)"),
    return ge::GRAPH_FAILED);

PadAndSqueeze(rawInputShapes_, rawOutputShapes_,
              maxBroShape_, normalInputShapes_, normalOutputShapes_);
rank_ = static_cast<int64_t>(maxBroShape_.size());

if (rank_ > RANK_8) {
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input",
        "rank exceeds limit", "rank must be <= 8");
    return ge::GRAPH_FAILED;  // 防止 delta = R - rank_ 为负数导致数组负索引
}

OP_CHECK_IF(
    !CheckBroadcastShape(normalInputShapes_, normalOutputShapes_, rank_),
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output",
        "incompatible", "check broadcast shape failed, shapes must be broadcast-compatible and outputs dense"),
    return ge::GRAPH_FAILED);

// 广播合轴：在 broadcast 校验通过后执行；后续 rank_/stride/FindSplitAxis/模板映射均基于合轴后 shape
BroadcastMergeAxis(maxBroShape_, normalInputShapes_, normalOutputShapes_);
rank_ = static_cast<int64_t>(maxBroShape_.size());
```

## 3 合轴产出校验

**对范式开发者的要求**：
- 1. 给一个复杂输入的样例，以合轴后的结果作为 golden，需要"人"保证正确性。
- 2. 删除 golden, agent读取本文档可以产出同样的合轴后的输出，如果产出不了，需要不断改进范式质量，直到成功
- 3. 样例随本章节上库

**golden 样例（Adam: 5 输入 3 输出, 输入 shape 不一致）**：

```
输入:
  IN0 (input0):           (256, 128, 64)
  IN1 (input1):           (256, 128, 64)
  IN2 (input2):           (256, 128, 64)
  IN3 (input3):           (256, 128, 64)
  IN4 (input4):           (256, 128, 64)
  mul0_x/mul1_x/mul2_x/mul3_x/add2_y: (256, 128, 64)
输出:
  OUT0 (out0):            (256, 128, 64)
  OUT1 (out1):            (256, 128, 64)
  OUT2 (out2):            (256, 128, 64)

补 1 → 所有 tensor rank=3，无需补 1
去 1 → dim0: 256（非 1，保留）; dim1: 128（非 1，保留）; dim2: 64（非 1，保留）
归一 → 坐标系 maximumBroShape = (256, 128, 64)
校验 → 所有维度一致 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: 各张量 = 128×64 = 8192，坐标系组乘积 = 8192
                    → 乘积均为 8192 = maxProd ✓ → 合并，当前组 = (8192)
  尝试 [dim0,dim1,dim2]: 各张量 = 256×8192 = 2097152，坐标系组乘积 = 2097152
                    → 乘积均为 2097152 = maxProd ✓ → 合并，当前组 = (2097152)

合轴后: maximumBroShape = (2097152,)
        normalInputShapes[0..9]  = (2097152,)  — 全部稠密连续
        normalOutputShapes[0..2] = (2097152,)
        inputStrides/outputStrides = (1,)
rank = 1 → mapped RANK = 4（rank ≤ 4）
```

shape 不一致的样例（假设 IN1 shape=(128,64)）：
```
补 1 → IN1=(1, 128, 64)，其他=(256, 128, 64)
去 1 → dim0: 256,1 → 不全为 1，保留
归一 → maximumBroShape = (256, 128, 64)
校验 → dim0: 256,1 → ref=256 ✓
       dim1/dim2: 全等 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: IN1 = 128×64 = 8192，其他 = 8192，坐标系组乘积 = 8192
                    → IN1 整组稠密（8192 = maxProd）✓ → 合并，当前组 = (8192)
  尝试 [dim0,dim1,dim2]: IN1 = 1×8192 = 8192，坐标系组乘积 = 256×8192 = 2097152
                    → 8192 既非 1 也非 2097152 ✗ → 封闭组 (8192)，从 dim0 开新组
  尝试 [dim0]: 单维成组

合轴后: maximumBroShape = (256, 8192)
        normalInputShapes[1] = (1, 8192)      — dim0 broadcast，dim1 整组稠密
        inputStrides[1] = (0, 1)              — dim0 stride=0（broadcast）
        其他输入/输出 = (256, 8192)，strides = (8192, 1)
rank = 2 → mapped RANK = 4（rank ≤ 4）
校验（合轴后坐标系） → dim0: 256,1 → ref=256 ✓
                        dim1: 8192,8192 → ref=8192 ✓
```
