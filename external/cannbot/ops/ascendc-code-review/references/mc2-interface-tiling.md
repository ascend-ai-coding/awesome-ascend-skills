# MC2 接口与 Tiling

<适用>
语言: 不限
侧别: All
领域: true
触发: hccl_, AllGather, AllReduce, ReduceScatter, AlltoAll, SyncAll, SetFlag, WaitFlag, PipeBarrier, SyncFunc, expert, expertIds, dispatch, combine, moeExpertNum, quant, BasicQuantMode, PERTENSOR, PERCHANNEL, AlltoAllV, sendCounts, InitV2, Finalize, SetCcTiling, CCU, AICPU, mc2, MC2, Mc2, Hccl, HCCL, AllToAll, Dispatch, Combine, all_gather_matmul, matmul_reduce_scatter, moe_distribute, groupEp, groupTp, GetHcclContext
关联适用: 任务输入或概要已用调用、注册或构建关系确认 MC2 归属，附关联文件、证据位置和来源
默认启用: true
排除场景: 与 MC2 无调用、注册或构建关系的独立计算或通信实现；仅出现同名变量或 API 而没有归属证据的代码。归属未确认时注明缺少的依据；用户说明保留来源。
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

领域说明和主题导航见 [MC2 总览](mc2-specific.md)。各条款的适用条件与检视方法可独立读取。

## 快速索引

按条款标题定位正文，读取到下一个同级或更高级标题之前，跳过代码块中的注释。每组容量见文件头，派发沿用 [负载均衡](../core/review-load-balance.md)。

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---|---|---|---|---|
| MC2-1.1 | 原型、属性读取和接口传参必须一致 | API使用 | 高 | [适用: Host] |
| MC2-1.2 | 参数校验和修改必须在实际调用入口生效 | API使用 | 高 | [适用: Host] |
| MC2-1.5 | tensor 的类型、布局和传入地址必须与实际存储一致 | 类型安全 | 高 | [适用: All] |
| MC2-1.6 | 输出 shape 和 dtype 的推导必须覆盖所有支持的输入 | API使用 | 高 | [适用: Host] |
| MC2-2.1 | 每种支持模式都必须能选中对应的 Kernel 实现 | Tiling设计 | 高 | [适用: All] |
| MC2-2.2 | Tiling 和任务参数必须完整传到执行端 | Tiling设计 | 高 | [适用: All] |
| MC2-2.3 | 分块和尾块处理不能遗漏数据或越界 | Tiling设计 | 高 | [适用: All] |
| MC2-2.5 | SIMT 边界检查不能让必需的同步参与线程提前退出 | 并发安全 | 高 | [适用: Kernel] |

## 详细规范

## MC2-1.1: 原型、属性读取和接口传参必须一致
<a id="mc2-1-1"></a>
<a id="mc2-1-3"></a>

MC2-1.1: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及原型、注册、属性读取或公开接口传参。

问题描述：属性读取和 aclnn 传参必须与当前算子的原型一致，包括字段顺序、类型、默认值，以及 EP、TP 各自表示的通信域。

### 错误示例

两个算子分别取得各自 context 的 attrs，属性顺序如下。

```cpp
// 两个算子的原型：Dispatch=[group,mode,epSize,tpSize]
//                  Combine =[group,mode,tpSize,epSize]
constexpr size_t EP_INDEX = 2;
auto dispatchEp = dispatchAttrs->GetAttrPointer<int64_t>(EP_INDEX);
auto combineEp = combineAttrs->GetAttrPointer<int64_t>(EP_INDEX); // 错：读到 TP
```

### 正确示例

```cpp
constexpr size_t DISPATCH_EP_INDEX = 2;
constexpr size_t COMBINE_EP_INDEX = 3;
auto dispatchEp = dispatchAttrs->GetAttrPointer<int64_t>(DISPATCH_EP_INDEX);
auto combineEp = combineAttrs->GetAttrPointer<int64_t>(COMBINE_EP_INDEX);
// 分别确认指针可用，再把值传给各自 EP 域；EP=4、TP=2 时均应读取4。
```

### 检视方法

以被检提交的算子原型和目标版本接口声明为准；已有 API 预研覆盖相同版本时可直接使用。

- 从算子注册和 Python、C++、aclnn 的接口签名，追踪到实际使用参数的位置。检查参数是否漏传、交换位置、被常量覆盖，或只用于打印日志；同时核对列表长度、输出顺序，以及 workspace、stream、通信域和设备参数。
- 用 EP≠TP、高位非零值和省略可选参数的输入检查上述问题。共享属性读取表或适配层重排参数时，按当前算子的原型确认对应关系。属性类型见 [TOPK 第 5 条](ascendc-topk.md)，Kernel 选择见 [MC2-2.1](#mc2-2-1)。

#### 条例结束

## MC2-1.2: 参数校验和修改必须在实际调用入口生效
<a id="mc2-1-2"></a>
<a id="mc2-1-4"></a>

MC2-1.2: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及入口注册、调用关系或参数校验。

问题描述：实际注册的入口，以及可独立调用的 aclnn、Python 和图入口，都必须执行修改后的逻辑，并在运算、访存或任务提交前完成必要的参数校验。

### 错误示例

本例展示目标 P 的实际注册结果，加载过程不会再替换该入口。Save 负责保存结果并返回状态，修改前后函数签名一致。

```cpp
Status BuildDispatchV2(int64_t count, int64_t epSize) {
    return Save(count / epSize);           // 当前实际入口，未校验
}
Status BuildDispatch(int64_t count, int64_t epSize) {
    if (epSize <= 0) return InvalidArgument; // 修改了旧函数
    return Save(count / epSize);
}
auto dispatchEntry = &BuildDispatchV2;     // 展开注册后实际绑定 V2
```

### 正确示例

```cpp
Status BuildDispatchV2(int64_t count, int64_t epSize) {
    if (epSize <= 0) return InvalidArgument; // 修复实际调用的函数
    return Save(count / epSize);
}
auto dispatchEntry = &BuildDispatchV2;
// epSize=0 在除法前返回；其他入口仍需核对各自的保护。
```

### 检视方法

- 从各个可独立调用的入口追踪到风险操作。按 [TOPK 第 7 条](ascendc-topk.md) 核对 shape、dtype 与模式的组合、专家数量的合法范围，以及零值的含义；用 epSize=0、越界索引和空指针检查是否及时返回错误。
- wrapper 或 InferShape 中的校验不能保护绕过它们的调用路径。更换注册入口或删除检查时，必须保留接口签名、合法默认值、失败码，以及公共 Tiling 对容量和拓扑的约束。

#### 条例结束

## MC2-1.5: tensor 的类型、布局和传入地址必须与实际存储一致
<a id="mc2-1-5"></a>
<a id="mc2-1-7"></a>
<a id="mc2-7-9"></a>

MC2-1.5: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及跨层绑定、dtype/stride 转换、DataCopy 搬运或缓冲传递。

问题描述：跨层传递 tensor 时，类型、布局和搬运范围必须与实际存储一致。转换 dtype 或将 tensor 连续化后，要传递新缓冲区的地址；HcclDataType 必须对应缓冲区中数据的实际编码。

### 错误示例

前提：专家索引原型为 int64，合法值包含 65537。

```cpp
// 输入是两个 int64_t 专家ID：65537、2。
int64_t ids[] = {65537, 2};
const int32_t *p = reinterpret_cast<const int32_t *>(ids);
// 小端示意读数为[65537,0,2,0]，第二个ID误读为0。
ConsumeTwoIds(p);
```

前提：MC2 接口明确允许逻辑 shape=[2,3] 且行 stride=4 的输入。

```text
shape=[2,3], stride=[4,1]
存储=[10,11,12,99,20,21,22,99]，99为padding
按 r*3+c 读取：[[10,11,12],[99,20,21]]
第二行首元素应为20，实际成为99
```

### 正确示例

```cpp
int64_t ids[] = {65537, 2};
ConsumeInt64Ids(ids, 2); // 匹配原型的步长和编码
// 若实际后端只支持int32：先逐元素证明范围，再转换到独立int32缓冲。
// 同时保活转换缓冲，不能只转换指针类型。
```

```text
按 r*4+c 读取：[[10,11,12],[20,21,22]]
或先连续化到新缓冲：[10,11,12,20,21,22]
后端接收新缓冲地址与连续stride=[3,1]
新缓冲owner覆盖异步计算和通信的最后一次读取
```

### 检视方法

涉及 DataCopy 等 API 时，按目标 CANN 版本、设备及实际重载核对参数单位和限制；优先使用匹配的预研结果，未覆盖时查官方文档和本地头文件。

- 按实际 stride、dtype 和 DataCopy 重载计算源、目的缓冲区的访问范围，检查尾块，以及换算后的长度或偏移是否被截断。对齐、repeat 限制和参数单位见 [KERNEL-1、KERNEL-3](ascendc-kernel-rules.md)。源缓冲区的 padding 必须可读，且不能混入有效输出。
- 按接口约定分别处理 None、空列表、空 tensor、表示动态维度的哨兵值和零 count。接口未承诺支持非连续输入时，可以拒绝这类输入。
- 跨层传参见 [MC2-1.1](#mc2-1-1)。转换后的缓冲区必须保持有效，直到异步操作不再使用它，见 [MC2-7.4](mc2-resources-errors.md#mc2-7-4)。

#### 条例结束

## MC2-1.6: 输出 shape 和 dtype 的推导必须覆盖所有支持的输入
<a id="mc2-1-6"></a>

MC2-1.6: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及 InferShape、InferDataType 或输出支持范围。

问题描述：每次推导都必须为所有必需输出设置本次的 shape 和 dtype，包括接口支持的不同模式、动态维度和空输入。

### 错误示例

Set* 表示推导 API。示例中的 payload 形状为 [tokens,H]，元数据需要 tokens×topK 个槽，dtype 根据当前模式确定。

```cpp
// Dispatch 有 payload、routeMetadata 两个必需输出。
SetPayloadShape(tokens, hidden);
return Success;                            // 错：routeMetadata 沿用上次描述
```

### 正确示例

```cpp
// 本例静态 tokens/topK 非负，乘积已通过位宽与容量检查。
SetPayloadShape(tokens, hidden);
SetRouteMetadataShape(tokens * topK);       // 本例每 token 对应 topK 个槽
SetPayloadDtype(payloadDtype);
SetRouteMetadataDtype(routeDtype);
return Success;                            // 两个输出均来自本次输入
```

### 检视方法

- 沿实际注册找到推导实现，逐个成功分支比较输出描述、分配容量与 Kernel 写入范围。检查模式切换、可选输入缺失、未知 rank 或维度、标量和零元素输入。
- 未知维度不能直接当作 0 或用作分配容量；接口不支持动态输入时，按约定拒绝。共享或自动生成的推导实现也要核实，强转 dtype 不能修复推导错误。注册要求见 [HOST-2](ascendc-host-rules.md)。

#### 条例结束

## MC2-2.1: 每种支持模式都必须能选中对应的 Kernel 实现
<a id="mc2-2-1"></a>
<a id="mc2-6-2"></a>

MC2-2.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 TilingKey、SEL、实例化或 quantMode 支持范围。

问题描述：入口接受的每种模式，包括 quantMode，都必须通过 TilingKey 和 SEL 选中实现该算法、且已编入产物的 Kernel。不支持的模式必须返回错误。

### 错误示例

本例只支持 mode0 和 mode1，入口已拒绝其他值。RunMode0 和 RunMode1 分别实现这两种模式。示例用普通 C++ 展示选择关系，实际 SEL 宏语法仍按目标版本核对。

```cpp
// Host 支持两种模式，本例给出宏展开后的普通 C++ 分派。
uint64_t key = mode == 0 ? 10 : 11;
if (key == 10) return RunMode0();
return Unsupported;                       // 错：合法 mode1 没有实现分支
```

SEL 参数位置也要核对。以下用展开后的 C++ 模板表示声明顺序 `[transposeA, quantMode]`；两参数均允许 0/1：

```cpp
// 模板声明：template<int transposeA, int quantMode> void Run();
constexpr int transposeA = 1, quantMode = 0;
Run<quantMode, transposeA>();              // 错：绑定成“不转置、量化”
```

### 正确示例

```cpp
uint64_t key = mode == 0 ? 10 : 11;
if (key == 10) return RunMode0();
if (key == 11) return RunMode1();           // 对应实例还须进入本次构建
return Unsupported;
// 若保留同一 key，须由另一个已传入的 mode 字段正确区分两种算法。
```

```cpp
Run<transposeA, quantMode>();              // 正确：“转置、不量化”
// 还须核对该组合生成的 key、编入实例和实际分支；0/0 测不出交换。
```

### 检视方法

SEL 和模板注册以目标 CANN 版本及本次构建使用的头文件为准；先复用对应预研，缺少的定义再查官方文档与本地实现。

- 对照枚举和 ops_info，逐模式检查入口校验、key 生成、SEL 参数的顺序与类型、取值范围、生成实例、链接和 Kernel 分支。检查是否存在空分支、提前返回，或被前面的分支先行匹配。
- 为兼容而保留的模式必须有依据。独立入口或互斥构建目标可以复用 key；合并执行路径前，要确认它们的编码、SEL 参数、生成实例和最终分支一致。
- 不能通过拒绝公开支持的模式来掩盖缺失的实现。模板语法见 [KERNEL-6、KERNEL-7](ascendc-kernel-rules.md)。

#### 条例结束

## MC2-2.2: Tiling 和任务参数必须完整传到执行端
<a id="mc2-2-2"></a>
<a id="mc2-8-5"></a>

MC2-2.2: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 Tiling、GenTask 注册、字段布局、序列化或启动实参。

问题描述：Tiling、launch 和 GenTask 必须传递本次调用的配置。注册必须指向可用的构造器，序列化布局必须与 Kernel 或任务执行端读取的布局一致。

### 错误示例

字段和 Serialize 用于演示数据布局。emptyKey、rounds 已按当前接口取得；空输入是否需要启动任务，由协议决定。

```cpp
// 上一请求 key11/count8；当前请求是合法空输入。
if (count == 0) return Success;             // 错：复用 data 中的旧值
data.count = count;
return Serialize(data);
```

### 正确示例

```cpp
TilingData data{};                         // 每次调用独立对象
data.count = count;
data.key = count == 0 ? emptyKey : modeKey;
data.rank = rank;
data.group = group;
data.rounds = roundsForThisRequest;         // 空输入仍保留协议所需轮次
return Serialize(data);                    // 失败不能改成 Success
```

### 检视方法

以被检提交的结构定义及目标版本的序列化、launch 接口为准，确认实际生效的构建分支。

- 从实际生效的注册追踪构造器、序列化和 launch，直到 GET_TILING_DATA 及字段使用处。核对 key、blockDim、workspace，以及实际读取字段的位宽、符号、偏移、单位、默认值和版本；确认序列化数据没有超过保存空间。公共 Fill 必须更新本次对象，失败后不能继续保存或提交。
- 用 epRank=3、epSize=4 和模式切换检查漏传、错传或覆盖。增加字段位宽后，要核对后续字段的位置和新旧布局的兼容性。
- 共享或自动生成的实现按实际能力判断，不能仅因命名不同就判定缺陷。保存指针时的生命周期要求见 [MC2-7.4](mc2-resources-errors.md#mc2-7-4)。

#### 条例结束

## MC2-2.3: 分块和尾块处理不能遗漏数据或越界
<a id="mc2-2-3"></a>
<a id="mc2-2-4"></a>

MC2-2.3: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及分块、尾块、核数、空输入或 RegBase mask。

问题描述：所有核、所有轮次处理的有效区间必须覆盖全部输出，不能遗漏数据、越界，或产生算法不允许的重复写入。

### 错误示例

以下 C++ 示例展示 RegBase 尾块访问的地址范围，valid=2，归约操作为 SUM。实际代码要使用目标平台支持的 load 和 store。

```cpp
// N=10、W=8、base=8；input 只分配10项。
float lane[8];
for (size_t j = 0; j < 8; ++j)
    lane[j] = input[base + j];             // 错：先读取了 input[10..15]
for (size_t j = 0; j < valid; ++j)
    output[base + j] = lane[j];            // valid=2，写回守卫补救不了读越界
```

### 正确示例

```cpp
float lane[8]{};                           // 本例后续为 SUM，0是中性值
for (size_t j = 0; j < valid; ++j)
    lane[j] = input[base + j];             // 仅读取 input[8..9]
for (size_t j = 0; j < valid; ++j)
    output[base + j] = lane[j];            // 仅写 output[8..9]
```

### 检视方法

- 列出各核、各轮的 [start,end) 区间，检查是否遗漏或重复写入，并核对 task ratio 和参与计算的核类型。检查空输入、长度不能整除分块大小、核数大于工作量的情况。数值计算见 [MC2-7.7](mc2-resources-errors.md#mc2-7-7)，尾块处理见 [KERNEL-1](ascendc-kernel-rules.md)。
- 分别核对 load、归约和 store 的有效 lane、最后访问地址、归约中性值及 mask 更新。API 允许读取的 padding 也不能影响有效结果；算法允许的原子操作或归约重叠可以保留。
- 空核和空 rank 仍必须参加通信协议要求的同步和通信。

#### 条例结束

## MC2-2.5: SIMT 边界检查不能让必需的同步参与线程提前退出
<a id="mc2-2-5"></a>

MC2-2.5: 条例正文

严重级别：高

适用范围：[适用: Kernel]；用于已确认属于 MC2 的代码，涉及 SIMT 索引、早退、屏障或启动维度。

问题描述：SIMT 边界检查必须阻止无效线程访存，同时保证后续同步所要求的线程仍参加对应轮次。

### 错误示例

BlockBarrier 代表要求整个线程块参加的原语，shared 对所有线程可见。

```cpp
// 启动8线程，i为本线程索引；input/output只有5项。
float x = input[i];                       // 错：i=5..7 已读越界
if (i >= 5) return;
BlockBarrier();                           // 错：要求8线程，已有3线程退出
output[i] = x;
```

### 正确示例

```cpp
bool active = i < 5;
if (active) shared[i] = input[i];
else        shared[i] = 0;                // shared已分配8项
BlockBarrier();                           // 8线程到达同一轮
if (active) output[i] = shared[i];
```

### 检视方法

先按目标 CANN 版本、设备和屏障重载确认参与线程及完成语义。已有预研未覆盖时查官方文档与本地头文件，不能用最新版本的约束代替目标版本。

- 展开多维索引计算，按 [SIMT 规范](ascendc-simt.md) 检查中间运算是否溢出，以及每次 load、store、atomic 前的边界检查。用 N=0、1、B-1、B、B+1 检查提前退出、循环轮次和屏障参与线程。
- 后续没有集合原语或共享数据依赖时，无效线程可以提前退出。只要求部分线程参加的屏障，按 API 规定的参与范围判断；增加超时时间不能修复参与线程不匹配的问题。

#### 条例结束
