# MC2 资源与错误处理

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
| MC2-7.1 | workspace 必须分配足够空间，资源用量不能超过上限 | 内存安全 | 高 | [适用: All] |
| MC2-7.3 | 拆分后的每次通信都必须满足后端的传输上限 | 资源管理 | 高 | [适用: Host] |
| MC2-7.4 | 异步资源必须由 owner 持有，直到最后一次使用完成 | 资源管理 | 高 | [适用: All] |
| MC2-7.6 | 每次请求的查询和执行必须使用同一份配置 | 并发安全 | 高 | [适用: All] |
| MC2-7.7 | 容量、地址和分块计算不能溢出或除零 | 数值安全 | 高 | [适用: All] |
| MC2-8.1 | 调用失败时必须返回错误，并准确记录原因 | API使用 | 高 | [适用: All] |
| MC2-8.2 | 替换错误处理宏不能改变判断逻辑，release 中的校验必须有效 | 输入验证 | 高 | [适用: All] |
| MC2-8.7 | 部分提交失败后，必须保留任务状态并正确处理重试和 fallback | 资源管理 | 高 | [适用: All] |

## 详细规范

## MC2-7.1: workspace 必须分配足够空间，资源用量不能超过上限
<a id="mc2-7-1"></a>
<a id="mc2-7-2"></a>

MC2-7.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 workspace、通信窗口、队列、双缓冲或资源上限。

问题描述：查询返回的 workspace 大小必须容纳执行时在其中访问的所有内存区间。计算各类资源的峰值用量时，要计入队列、双缓冲、临时区、对齐开销和系统预留。

### 错误示例

本例中，workspace 是字节指针。64 字节的通信窗口和 32 字节的元数据区同时使用，两段内存及总大小都满足目标平台的对齐要求。

```cpp
// 两段同时使用，均按32字节对齐。
size_t workspaceBytes = 64;                // 错：漏算元数据
size_t metaOffset = 64, metaBytes = 32;
auto *meta = workspace + metaOffset;       // workspace 只分配了64字节
Launch(workspace, meta, metaBytes);         // 使用 [64,96)，全部越界
```

### 正确示例

```cpp
size_t windowBytes = 64;
size_t metaOffset = 64, metaBytes = 32;
size_t workspaceBytes = metaOffset + metaBytes; // 本例为96字节
// 查询返回 workspaceBytes；调用者按该值分配，执行时提供 capacity。
if (metaOffset > capacity || metaBytes > capacity - metaOffset)
    return InvalidArgument;
auto *meta = workspace + metaOffset;       // 检查后再构造子区间指针
Launch(workspace, meta, metaBytes);
```

### 检视方法

先确认目标设备、CANN 版本及本次配置使用的 workspace 查询契约、对齐和资源上限；复用相同目标的预研，未覆盖时查官方资料与本地实现。

- 列出每段内存的字节偏移、长度、对齐要求和使用时间，逐一核对查询结果、实际分配大小与 launch 参数。检查单位换算是否溢出，区分 mSize 和 maxMSize，并验证零长度、区间末端恰好等于容量、最大输入等边界情况。
- 按同一时刻实际持有的存储计算资源峰值，并与对应资源的上限比较。静态双缓冲即使交替使用，也要计入两个槽的容量；多个 view 指向同一块存储时，只计一次。只有释放存储或确实共用存储，才能减少占用。
- 通信窗口大小、后端单次传输上限和实际分配容量要分别检查。拆分通信后，同时核对 counts 和轮次；查询失败时不能沿用旧结果。复用缓冲区前，确认别名关系允许复用，且此前的异步访问已经完成。

#### 条例结束

## MC2-7.3: 拆分后的每次通信都必须满足后端的传输上限
<a id="mc2-7-3"></a>

MC2-7.3: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及 CCU/AICPU 选择、通信量、拆分或 tail。

问题描述：每次通信的数据量必须符合目标后端和通信原语的限制。接口允许时可以拒绝超限请求；采用拆分或切换后端时，须确认接口支持。

### 错误示例

以下 Host 示例假设 elemBytes>0，通信原语允许拆分。SendRecvChunk 负责配对收发、处理状态并确认完成；L 只是示例使用的传输上限。

```cpp
// 本例已查得目标单次上限 L=268435456 bytes；count 单位为元素。
uint64_t count = 134217729, elemBytes = 2;
SendOneCcuChunk(count);                    // 错：268435458 bytes > L
```

### 正确示例

```cpp
uint64_t maxElements = limitBytes / elemBytes; // 本例134217728
if (maxElements == 0) return Unsupported;
for (uint64_t offset = 0; offset < count; ) {
    uint64_t chunk = std::min(maxElements, count - offset);
    SendRecvChunk(offset, chunk);          // 双方按相同轮次和偏移处理
    offset += chunk;                      // 本例两块：134217728、1
}
```

### 检视方法

1. 按目标 CANN 版本、设备、后端和通信原语查证单次传输上限及是否允许拆分。复用覆盖该目标的 API 预研，未覆盖时查询官方文档并核对本地头文件；256MB 只是历史参考值。
2. 按 dtype、count 单位和接口要求的 rank 倍数换算实际传输量，核对拆分后的每块、tail、接收 offset 和各 rank 的通信轮次。
3. 用上限附近的数据量和不对称负载验证已有拆分或选路逻辑。workspace 容量不能代替传输上限，AICPU 也要查证自身限制。接口允许拒绝超限请求时可以返回错误，无需新增拆分或切换后端功能。

#### 条例结束

## MC2-7.4: 异步资源必须由 owner 持有，直到最后一次使用完成
<a id="mc2-7-4"></a>
<a id="mc2-7-5"></a>

MC2-7.4: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及资源创建、借用、入队、销毁、移动、通信域字符串或 Python 临时存储。

问题描述：异步任务使用的实际 storage 必须保持有效，直到最后一次访问完成；本层只能释放自己拥有的资源。PTA 需要保存通信域名称时，必须持有字符串副本。

### 错误示例

Wrapper 是示例类型，ownsBuffer 在创建对象或转移所有权时设置。WaitLastUse 成功返回，表示最后一次异步访问已经完成。该 Destroy 不允许再次传入已删除的指针；如果接口支持重复 close，需要另行保存可重复检查的关闭状态。

```cpp
void Destroy(Wrapper *w) {
    Free(w->buffer);                       // 错：buffer 可能由调用者借入
    delete w;
}
```

```cpp
const char* group = MakeGroupName().c_str(); // 临时 string 已销毁
Enqueue([group] { UseGroup(group); });
```

### 正确示例

```cpp
Status Destroy(Wrapper *w) {
    Status s = WaitLastUse(w->ticket);     // 成功才证明后台不再访问
    if (s != Success) return s;            // 保留 w，交给原 owner 后续收尾
    if (w->ownsBuffer) Free(w->buffer);    // 借入路径 ownsBuffer=false
    delete w;                             // 包装对象由本层创建
    return Success;
}
```

```cpp
std::string group = MakeGroupName();
if (group.size() >= MAX_GROUP_NAME_LENGTH) return InvalidArgument;
Enqueue([group] { UseGroup(group.c_str()); }); // 持有至任务结束
```

### 检视方法

- 从通信提交到最后一次访问，核对 tensor、workspace、request、通信 handle 和字符串的 owner 与借用关系。只保存 data_ptr 不会延长底层对象寿命；实际接口是否保留地址、何时完成，按目标版本的预研或官方文档与本地实现确认。
- 异步接口保留通信域字符串时，检查保存方式及长度；同步调用不保存地址时可以直接借用。复制、终止符和容量的通用要求见 [安全编码第 2.3、5.3、7.1、7.2 条](cpp-secure.md)。
- 对代码已有的失败、回调、重配置、移动和 close 路径，检查所有权是否丢失、重复释放或过早释放。失败按公开接口约定返回状态或 Python 异常；无需为未承诺的使用方式新增生命周期机制。

#### 条例结束

## MC2-7.6: 每次请求的查询和执行必须使用同一份配置
<a id="mc2-7-6"></a>
<a id="mc2-7-10"></a>
<a id="mc2-8-6"></a>

MC2-7.6: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及成员初始化、查询/执行配对、executor、设备/通信域绑定、缓存、重配置或并发访问。

问题描述：首次调用、连续请求和并发执行都必须使用各自请求的配置。执行时的 executor、workspace、device/context、group/rank 和 stream/server 必须与查询准备阶段一致。

### 错误示例

前提：同一 Host 对象先处理 mode1 再处理 mode0，mode0 使用更少通信轮。

```cpp
// 前次mode1设置了4轮；本次 mode0 按接口定义只需 1 轮。
if (mode == 1) { roundNum_ = 4; }
Submit(roundNum_); // mode0仍提交4轮
```

Query/Execute 是两阶段接口的简化签名；实际接口是否在执行阶段接收 shape，以目标声明为准。

```cpp
Query(shapeA, &executorA, &needA);          // 成功，needA=96
Execute(shapeB, executorA, workspace, 64);  // 错：旧配置 + 容量不足
```

### 正确示例

```cpp
const uint32_t rounds = mode == 1 ? 4 : 1;
RequestState request{mode, EncodeKey(mode), rounds};
SubmitOwnedRequest(request); // 提交层按接口约定复制或持有本次状态
// 若必须写共享对象，则在整个读取使用期建立同步。
```

```cpp
Status s = Query(shapeB, &executorB, &needB);
if (s != Success) return s;                // 不能沿用 executorA/needA
if (capacity < needB) return InvalidArgument;
return Execute(shapeB, executorB, workspace, capacity);
// workspace 按目标要求对齐；executorB/内存保持到规定的完成点。
```

### 检视方法

- 找到各配置字段的赋值位置，检查从第一次读取到最后一次读取期间是否一直属于本次请求。交替使用不同 shape、dtype、模式、rank、context 和平台，核对缓存键、失效条件、双向模式切换，以及查询和执行的配置是否配套，包含 workspace 对齐、通信域成员和 stream/server。
- 合法的构造默认值或有效的上游赋值可以直接使用。group 名称相同或 handle 地址相同，不代表配置相同；增大 workspace 也不能修复配置不匹配，必须在提交前阻止执行。
- 跨 stream 或设备复用必须得到后端支持，否则重新创建。按 Graph/Eager 和仍支持的平台，检查允许的复用次数、依赖是否完成及最终销毁时机。并发请求还要检查整个读取期间是否会覆盖彼此的配置，并在报告中指出第一次错误读取及相关赋值位置。异步 request 的生命周期见 [MC2-7.4](#mc2-7-4)。

#### 条例结束

## MC2-7.7: 容量、地址和分块计算不能溢出或除零
<a id="mc2-7-7"></a>
<a id="mc2-7-8"></a>

MC2-7.7: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 count/offset/stride、类型转换、派生除数或 CeilDiv。

问题描述：通信 count、offset、stride、轮数和分块参数的中间运算及整数窄化，都不能超出类型的表示范围；实际使用的除数必须非零。

### 错误示例

前提：通信轮数 rounds 为 uint32，本例允许 rounds=0。

```cpp
uint32_t rounds=0; // 接口允许空轮
uint32_t last=rounds-1; // UINT32_MAX
ProcessRound(last);    // 巨大错误索引
```

### 正确示例

```cpp
if (rounds == 0) return HandleEmptyRounds();
uint32_t last=rounds-1; // rounds=1时为0
ProcessRound(last);
```

### 检视方法

1. 从通信 count、offset、stride 和分块参数的实际赋值处追踪到运算，逐步确认操作数类型、中间结果范围及写入 TilingData 或 API 参数时的窄化。先在窄类型中溢出，再赋给宽类型，仍然有误。
2. 检查已有上游保护是否覆盖当前运算及实际使用的值。单独限制两个乘数不一定保护乘积；coreNum 非零也不能保护 coreNum-1。有效校验后值未改变时，无需重复检查。
3. 按接口支持范围核对零轮次、最大合法输入、负哨兵转 unsigned，以及 CeilDiv 的 a+b-1 和后续乘法。合法的 0 或空输入按接口处理，不能为避免下溢而一律拒绝。

涉及设备核数、容量上限或 API 单位时，按目标 CANN 版本、设备及实际重载查阅对应预研、官方资料和本地定义。纯整数运算可直接按已确认的类型和值域判断。证据不足时列出缺少的值域或契约，不能仅因使用较窄类型报错。通用数值判据见 [红线第 1、3、4、9 条](ascendc-red-line.md)。

#### 条例结束

## MC2-8.1: 调用失败时必须返回错误，并准确记录原因
<a id="mc2-8-1"></a>
<a id="mc2-8-4"></a>

MC2-8.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及返回值、错误传播、日志或重试分类。

问题描述：准备、查询、提交或执行失败时，必须将错误传回公开接口的调用者，并停止依赖该结果的操作。错误码和日志必须反映实际原因。

### 错误示例

前提：MC2 准备接口返回 Status，Success=0，通信配置失败。

```cpp
using Status = int; // Success=0
Status Prepare() {
    if (!ConfigureComm()) { return false; } // 转为0
    return Success;
}
// 调用者看到Success，继续Submit。
```

### 正确示例

```cpp
Status Prepare() {
    if (!ConfigureComm()) { return ConfigFailure; } // 接口定义的非成功值
    return Success;
}
Status s = Prepare();
if (s != Success) { return s; }
Submit();
```

### 检视方法

沿 MC2 准备、查询、提交和执行链核对失败状态，检查非法 quantMode 或后端失败是否被转换成成功，以及失败后是否仍继续下发依赖任务。错误归因须来自实际返回契约，例如 GetRankSize 失败不能直接等同于 group 非法；需要解释 API 状态时，核对目标版本的预研或官方资料与本地实现。

通用返回值检查和诊断格式沿用 [TOPK](ascendc-topk.md)、[错误码规范](ascendc-error-code.md)及[日志规范](ascendc-log.md)。已有上游诊断和有效透传时，不要求每层重复上报。部分任务已提交的处理见 [MC2-8.7](#mc2-8-7)。

#### 条例结束

## MC2-8.2: 替换错误处理宏不能改变判断逻辑，release 中的校验必须有效
<a id="mc2-8-2"></a>
<a id="mc2-8-3"></a>

MC2-8.2: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及断言、错误/日志宏、编译条件或分支替换。

问题描述：替换宏时，必须保持条件判断、短路求值顺序、实参求值和失败返回的语义。不能只用 debug 断言保护 release 中的风险操作。

### 错误示例

前提：release 宏把 DEBUG_ASSERT 编译为空，epSize=0 可达。

```cpp
// release: #define DEBUG_ASSERT(x) ((void)0)
DEBUG_ASSERT(epSize > 0);
auto local = expertNum / epSize; // epSize=0仍可到达
```

### 正确示例

```cpp
if (epSize <= 0) { return InvalidArgument; }
auto local = expertNum / epSize;
// 调试断言可额外保留，但不承担公开输入的唯一守卫。
```

### 检视方法

按实际 include 和 release 构建条件展开本次替换的校验宏，确认 MC2 的 EP/TP、容量或模式校验仍在风险操作前生效。核对条件方向、实参求值及失败出口；已有有效上游保护时无需重复添加。仅修改日志文字且不改变控制流，不据此报行为缺陷。通用宏要求见 [TOPK](ascendc-topk.md)。

#### 条例结束

## MC2-8.7: 部分提交失败后，必须保留任务状态并正确处理重试和 fallback
<a id="mc2-8-7"></a>
<a id="mc2-8-8"></a>

MC2-8.7: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及部分提交、取消、超时、重试或 fallback。

问题描述：失败、取消或超时后，必须保留已被接受的任务的状态。重试不能重复执行这些任务；fallback 必须保持原输入、模式、输出 dtype、布局和精度。

### 错误示例

本例中，Submit 失败表示该任务尚未提交，Wait 成功表示最后一次使用已经结束。input 和任务票据都由生命周期足够长的请求对象持有。

```cpp
Status s = Submit(taskA, &ticketA);         // 已接受，仍会异步读取 input
if (s != Success) return s;
s = Submit(taskB, &ticketB);                // 失败
if (s != Success) {
    ReleaseOwnedInput();                   // 错：A 仍在使用
    return Success;                        // 错：把部分成功报成整体成功
}
```

### 正确示例

```cpp
Status s = Submit(taskA, &ticketA);
if (s != Success) return s;
s = Submit(taskB, &ticketB);
if (s != Success) {
    Status done = Wait(ticketA);
    if (done != Success) return done;      // 完成未确认，继续保留 input 和票据
    ReleaseOwnedInput();                   // A 已结束，本层拥有的资源才能释放
    return s;                             // 返回 B 的失败
}
```

### 检视方法

1. 记录本次请求中已被后端接受的任务及其完成状态。后续提交失败时，停止依赖失败结果的操作；释放输入或通信资源前确认已接受任务不再访问它们。接口的提交和完成语义按目标版本预研或官方文档与实际实现核对。
2. 如果代码实现了重试，检查通信域、epoch 和输出区间，确认不会重复产生远端副作用。清零本地状态不能撤销已经执行的通信。
3. 如果代码实现了 fallback，核对输入、通信域、量化模式、scale、输出类型及精度是否保持接口承诺。用任务 A 延迟完成、任务 B 提交失败的情况验证收尾。

是否提供重试或 fallback 由接口要求决定。允许直接失败时，正确收尾并返回错误即可；已有恢复能力不能在修复中丢失。

#### 条例结束
