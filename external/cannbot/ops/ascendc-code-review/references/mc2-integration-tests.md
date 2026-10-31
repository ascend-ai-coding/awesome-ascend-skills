# MC2 工程集成与测试

<适用>
语言: 不限
侧别: All
领域: true
触发: hccl_, AllGather, AllReduce, ReduceScatter, AlltoAll, SyncAll, SetFlag, WaitFlag, PipeBarrier, SyncFunc, expert, expertIds, dispatch, combine, moeExpertNum, quant, BasicQuantMode, PERTENSOR, PERCHANNEL, AlltoAllV, sendCounts, InitV2, Finalize, SetCcTiling, CCU, AICPU, mc2, MC2, Mc2, Hccl, HCCL, AllToAll, Dispatch, Combine, all_gather_matmul, matmul_reduce_scatter, moe_distribute, groupEp, groupTp, GetHcclContext
关联适用: 任务输入或概要已用调用、注册或构建关系确认 MC2 归属，附关联文件、证据位置和来源；已确认受本次公共组件修改影响的非 MC2 调用者，仅适用 MC2-9.8
默认启用: true
排除场景: 未确认 MC2 归属、也未确认受到本次公共组件修改影响的代码，即使命中同名变量或 API 也排除。非 MC2 调用者只保留 MC2-9.8；依据不足时注明缺口，用户说明保留来源。
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

领域说明和主题导航见 [MC2 总览](mc2-specific.md)。各条款的适用条件与检视方法可独立读取。

## 快速索引

按条款标题定位正文，读取到下一个同级或更高级标题之前，跳过代码块中的注释。每组容量见文件头，派发沿用 [负载均衡](../core/review-load-balance.md)。

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---|---|---|---|---|
| MC2-9.1 | 平台配置、注册和构建产物必须对应同一目标平台 | 编译配置 | 高 | [适用: All] |
| MC2-9.2 | 跨代际依赖必须符合当前项目的明确限制 | 编译配置 | 高 | [适用: All] |
| MC2-9.4 | 图融合必须满足实现的支持条件，并保留输出和外部依赖 | 精度保护 | 高 | [适用: Host] |
| MC2-9.6 | 图变换失败后必须恢复有效图，或禁止继续使用该图 | 资源管理 | 高 | [适用: Host] |
| MC2-9.7 | 构建产物和发布包必须包含所需实现、依赖和注册信息 | 编译配置 | 高 | [适用: All] |
| MC2-9.8 | 修改公共组件时必须检查受影响的调用者 | API使用 | 高 | [适用: All] |
| MC2-10.2 | 接口迁移必须实现当前需求，并保留调用者依赖的行为 | API使用 | 中 | [适用: All] |
| MC2-10.3 | 示例必须具备运行依赖，并启动全部通信 rank | API使用 | 中 | [适用: All] |
| MC2-10.4 | 测试 mock 不能替代被测逻辑，全局状态必须正确恢复 | 资源管理 | 高 | [适用: All] |
| MC2-10.5 | 测试必须能发现目标错误，并验证合法边界仍能通过 | 输入验证 | 高 | [适用: All] |
| MC2-10.7 | worker 的失败和异常必须传到调用者及启动器 | API使用 | 高 | [适用: All] |

## 详细规范

## MC2-9.1: 平台配置、注册和构建产物必须对应同一目标平台
<a id="mc2-9-1"></a>

MC2-9.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及平台查询、config、注册、模板或目标二进制。

问题描述：平台查询、算子原型、Tiling、config、GenTask 和最终二进制必须对应同一目标平台的实现。

### 错误示例

前提：本次构建目标为 A5，平台查询返回值按当前版本枚举解释。

```text
文件位于mc2/.../a5/
平台查询：PlatformKind::A3
活动分支：case A3 → BuildA3
错误结论：目录含a5，所以本次执行必定走A5
```

前提：本次迁移清单包含原型、tiling 注册、平台 config、gen_task 四项。

```text
迁移后：proto=A5，tiling=A5，config=A5
遗漏项：gen_task仍绑定BuildTaskA2
结果：A5入口可达，但任务交给A2专有构造器
```

```text
活动平台 P 的 config 指向 kernel_v2
固定目标只生成 kernel_v1，且无别名或生成适配
配置无法选到要求的实际产物
```

### 正确示例

```text
目标构建：A5；查询结果按当前头文件解释为PlatformKind::A5
枚举值 → case A5 → config[A5] → BuildA5
证据保存：平台查询调用、枚举定义版本、实际构建宏和分支
```

```text
迁移核对表：
原型 → A5合法组合；Tiling → A5字段布局
config → A5活动实现；gen_task → BuildTaskA5
旧A2分支仅在明确的A2目标启用，分别保存构建证据
```

```text
使活动 config 与实际产物一致，或补充满足接口要求的 v2 构建
同时核对其他平台仍指向其原有合法实现
```

### 检视方法

平台枚举和可用实现以目标 CANN 版本、设备及实际构建条件为准，优先使用对应预研或目标版本文档。

- 从平台查询结果、当前版本的枚举和嵌套宏，追踪到源文件列表及间接链接的依赖。核对模板、Graph/Eager、ND→NZ 转换、低精度和稀疏编码、server/stream，以及仍承诺支持的旧平台。
- 查询失败或返回未知值时，不能沿用旧配置。目录名或显式 -D 参数不能单独证明实际平台；socVersion→npuArch 按目标要求迁移，不能通过缩小支持范围掩盖缺失实现。跨代依赖见 [MC2-9.2](#mc2-9-2)。

#### 条例结束

## MC2-9.2: 跨代际依赖必须符合当前项目的明确限制
<a id="mc2-9-2"></a>

MC2-9.2: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及跨代头文件、调用、链接或注册依赖。

问题描述：只有同时满足以下条件，才判定跨代依赖违规：当前项目禁止该依赖方向，被依赖代码是代际专有实现，该依赖在本次构建中生效，且不符合项目允许的例外。

### 错误示例

```cmake
# 当前 target 为 A5；本例政策禁止依赖 A2 专有 Kernel。
# 已沿源列表确认 a2_private_kernel 编入了 A2 专有指令实现。
target_link_libraries(mc2_a5 PRIVATE a2_private_kernel)
```

### 正确示例

```cmake
# 使用 A5 目标可用的实现；共享算法通过不带专有依赖的公共库提供。
target_link_libraries(mc2_a5 PRIVATE a5_kernel mc2_common_math)
```

### 检视方法

沿 include、调用、链接和注册关系，核实实际依赖的专有实现，以及它是否参与本次构建。兼容例外要追踪到最终 Kernel，并核对 ABI 位宽、单位和布局转换。改名或移入 common 不改变依赖性质，没有显式 -D 也不代表宏未定义。公共实现可以合法复用；ABI 适配仍必须符合项目政策，且不能调用被禁止的 Kernel。

#### 条例结束

## MC2-9.4: 图融合必须满足实现的支持条件，并保留输出和外部依赖
<a id="mc2-9-4"></a>
<a id="mc2-9-5"></a>

MC2-9.4: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及图匹配、属性传递、节点替换或外部数据/控制边。

问题描述：图融合只适用于融合实现支持的输入和模式。融合后必须保持输出数值、布局，以及所有外部数据依赖和控制依赖，并区分归约前的值与归约后的值。

### 错误示例

图 API 仅用于示意。两个 rank 参与归约的值分别为 2 和 5 时，MAX 结果为 5，SUM 结果为 7。X=[1,2]、W=[[1,2],[3,4]] 时，X×Wᵀ=[5,11]；遗漏转置则得到 [7,10]。

```cpp
// 本融合实现仅支持 SUM，默认 transposeW=false。
if (IsMatmul(matmul) && IsAllReduce(reduce)) {
    return CreateFused(x, w);              // 错：漏查 reduceOp，漏传 transposeW
}
```

融合实现若提供等价的未归约输出，可分别重连 C 与归约消费者。

```cpp
// matmul.out 同时连到 reduce 和需要未归约值的 C。
ReplaceAllUses(matmul.out, fused.reducedOut); // 错：C 也收到归约后结果
Remove(matmul);
```

### 正确示例

```cpp
if (!IsMatmul(matmul) || !IsAllReduce(reduce)) return NotMatched;
if (reduceOp != SUM) return NotMatched;    // MAX 保留原图
return CreateFused(x, w, transposeW);      // 保留原 Matmul 属性
```

```cpp
// 本融合实现只有归约输出，不提供未归约旁路。
for (const auto &user : Users(matmul.out)) {
    if (user != reduce) return NotMatched; // 保留原图和 C 的输入
}
// 只有通过检查后才进入图重连；仍核对控制边和输出存储寿命。
```

### 检视方法

- 对照 pass 和融合实现支持的 shape、dtype、format、转置、reduceOp，以及可选输入、动态输入和空输入，检查输入索引、默认值、运算顺序和精度。如果实际执行了数据转置，还要检查新布局、缓冲区生命周期及转置成本。
- 删除节点前，逐个核对它的使用者，包括多输出、别名、数据边、控制边和通信依赖，并检查输出存储的生命周期。修改边时不能破坏遍历；无法提供原有值时保留原图，不能通过重复执行有通信副作用的计算来补充旁路输出。
- 如果依赖前序 pass 排除不支持的输入，要确认 pass 的执行顺序。隐式依赖边尚不明确时，不能认定删除节点是安全的。

#### 条例结束

## MC2-9.6: 图变换失败后必须恢复有效图，或禁止继续使用该图
<a id="mc2-9-6"></a>

MC2-9.6: 条例正文

严重级别：高

适用范围：[适用: Host]；用于已确认属于 MC2 的代码，涉及建节点、改边、删除、提交或回滚。

问题描述：图变换失败后，如果调用者还会使用该图，就必须恢复有效状态；否则必须终止处理，并禁止后续再使用该图。

### 错误示例

本例中，调用者在变换失败后仍会使用 graph。fused 的属性和输入已经准备好，只需重连示例中的输出。事务必须覆盖全部节点、数据边和控制边，并保证未 Commit 时能够回滚；Remove 也可能失败。

```cpp
Remove(oldMatmul);
Remove(oldReduce);
Status s = AddEdge(fused.out, consumer);
if (s != Success) return s;                // 错：原图已删，新图也不完整
```

### 正确示例

```cpp
GraphTransaction tx(graph);                // 本例：未 Commit 时恢复节点和边
Status s = tx.AddNode(fused);
if (s != Success) return s;
s = tx.AddEdge(fused.out, consumer);
if (s != Success) return s;
s = tx.Remove(oldMatmul);
if (s != Success) return s;
s = tx.Remove(oldReduce);
if (s != Success) return s;
return tx.Commit();                        // 完整重连后提交
```

### 检视方法

逐个检查创建节点、设置属性、修改边和删除节点的失败出口，确认节点、属性、输出映射、数据边及控制边都已恢复。清理不再使用的对象，回滚失败也要向上传递。没有事务 API 时，可以先构建待替换部分，或用明确的逆操作恢复，同时核对对象所有权。框架允许终止编译且禁止再用该图时，无需恢复执行。

#### 条例结束

## MC2-9.7: 构建产物和发布包必须包含所需实现、依赖和注册信息
<a id="mc2-9-7"></a>
<a id="mc2-10-1"></a>
<a id="mc2-9-3"></a>

MC2-9.7: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及构建依赖、头文件版本、后端配置、extension 注册、安装或加载。

问题描述：在文档声明的环境中，构建产物和发布包必须包含所需实现及公开接口的注册信息；头文件、后端、版本和数据布局必须兼容。

### 错误示例

本例要求发布包可以独立构建。comm_util.h 只有声明，源文件、生成物、对象库及间接链接的库都没有提供 ValidateComm 的定义。

```cmake
# sample.cpp 包含 comm_util.h，并调用 ValidateComm。
add_executable(mc2_sample sample.cpp)       # 错：没有提供 ValidateComm 定义
```

以下头文件示例以项目明确要求新版使用拆分头为前提。

```cpp
#include "kernel_operator.h" // 新版目标也无条件包含聚合头
```

### 正确示例

```cmake
# comm_util 是声明环境能取得的实现目标，版本和架构与头文件匹配。
add_executable(mc2_sample sample.cpp)
target_link_libraries(mc2_sample PRIVATE comm_util)
```

```cpp
#if ASC_DEVKIT_MAJOR >= PROJECT_SPLIT_API_MAJOR
#include "basic_api/kernel_operator_block_sync_intf.h"
#else
#include "kernel_operator.h"
#endif
// PROJECT_SPLIT_API_MAJOR 由目标文档确认后定义；示例仅使用同步 API。
```

### 检视方法

- 从 MC2 主库、extension 和示例目标追踪实际使用的源文件、生成物、链接依赖及安装清单，核对注册名、绑定和默认值。
- 按目标 CANN 版本及实际构建宏，核对 HCCL/HCOMM 头文件、库和 CCU/AICPU 后端是否配套。项目要求拆分头文件时，核实版本阈值，不能照搬 >=9；废弃接口检查沿用 [SUNSET-2](ascendc-sunset-api.md)。
- 验证发布问题时，在仅具备声明依赖的环境中执行构建、安装和最小调用，记录命令、结果及加载库路径。未实际运行的步骤标明未验证。

#### 条例结束

## MC2-9.8: 修改公共组件时必须检查受影响的调用者
<a id="mc2-9-8"></a>

MC2-9.8: 条例正文

严重级别：高

适用范围：[适用: All]；涉及 MC2 公共 helper、结构体、常量、缓存或 ABI；包含本次修改实际影响的非 MC2 调用者。

问题描述：修改公共 helper、结构体、常量或缓存时，必须保留实际受影响调用者所依赖的参数约定。

### 错误示例

本例采用均匀分片，参数都在 int 的表示范围内。非均匀分片不要求总量能够整除分片数。

```cpp
int LocalExperts(int total, int ep) {
    return 8 / ep;                         // 错：把 Dispatch 的8写成公共常量
}
// Combine(total=16, ep=2) 得到4，应为8；仅测试 Dispatch 看不出错误。
```

### 正确示例

```cpp
int LocalExperts(int total, int ep) {
    if (ep <= 0 || total < 0 || total % ep != 0) return Invalid;
    return total / ep;
}
// 分别验证 Dispatch(8,2)=4、Combine(16,2)=8，以及各自非法边界。
```

### 检视方法

在本次修改范围内追踪公共组件的实际调用者，分别核对 Dispatch、Combine 等入口依赖的值域、默认值、单位、布局和生命周期。均匀分片与非均匀分片分别判断，不能将某个调用者的整除条件或常量推广到所有入口。

记录已核对的调用者和证据缺口。非 MC2 使用方只检查本次公共修改造成的影响；通用 ABI 要求沿用 [安全编码第 7.4、7.5 条](cpp-secure.md)。

#### 条例结束

## MC2-10.2: 接口迁移必须实现当前需求，并保留调用者依赖的行为
<a id="mc2-10-2"></a>

MC2-10.2: 条例正文

严重级别：中

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及需求承诺、接口迁移、初始化或重配置。

问题描述：当前有效需求必须有对应实现。迁移接口时，要保留调用者依赖的返回数据，以及初始化和注册产生的副作用。

### 错误示例

接口要求 Configure(0) 将 capacity_ 从 8 更新为 0。

```cpp
Status Configure(size_t value) {
    if (value == 0) return Success;         // 错：capacity_ 仍保留旧值8
    capacity_ = value;
    return Success;
}
```

### 正确示例

```cpp
Status Configure(size_t value) {
    capacity_ = value;                     // 本例0合法，须更新为0
    return Success;
}
```

### 检视方法

对照目标 CANN 版本的接口定义与实际调用者，逐项核对迁移前后的返回数据、Tiling、初始化和回调注册。API 预研未覆盖时查询目标版本文档并检查本地实现，不能仅凭新旧函数名称判断等价。

检查合法的 0、空列表及重复配置是否仍按接口约定更新状态。改变 group 后，后续执行应使用相应 handle；是否阻塞或原子更新由接口契约决定。移除旧接口前确认已有可用替代路径，保留调用者依赖的合法默认值和副作用。

#### 条例结束

## MC2-10.3: 示例必须具备运行依赖，并启动全部通信 rank
<a id="mc2-10-3"></a>
<a id="mc2-10-6"></a>

MC2-10.3: 条例正文

严重级别：中

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及示例依赖、CLI、rank 启动或运行说明。

问题描述：示例必须提供文档声明的运行依赖，按当前 CLI 启动 worldSize 对应的全部 rank，并将失败状态传回启动器。

### 错误示例

前提：该 MC2 示例要求启动两个 rank，并为它们传入相同的 group 参数。

```text
文档声明worldSize=2
只启动：demo --rank 0 --world-size 2 --group demo
rank1没有启动，rank0等待必要参与者
```

### 正确示例

```text
参与者0：demo --rank 0 --world-size 2 --group demo
参与者1：demo --rank 1 --world-size 2 --group demo
按真实部署方式为两者分配设备及连接配置
预期：两进程都完成，结果按样例输入核对，失败状态传回启动器
```

### 检视方法

对照 CLI 和运行说明，核对参数、默认值、单位、版本、预期输出、初始化顺序及输入数据来源。逐 rank 检查 device、group、连接和启动方式，多机运行还要提供地址配置。仓内示例不要求一定能独立发布；进程号不能当作域内 rank，TODO 或“待测试”文字本身不构成缺陷。构建依赖见 [MC2-9.7](#mc2-9-7)，退出状态见 [MC2-10.7](#mc2-10-7)。

#### 条例结束

## MC2-10.4: 测试 mock 不能替代被测逻辑，全局状态必须正确恢复
<a id="mc2-10-4"></a>

MC2-10.4: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 mock、fixture、全局配置、用例并发或清理。

问题描述：UT 可以用 mock 替换外部依赖，但必须保留被测的判断和状态更新逻辑。正常结束、失败和并发执行后，全局状态都必须正确恢复。

### 错误示例

前提：单元测试目标是 Tiling 参数校验，应隔离真实 HCOM。

```text
测试目标：非法epSize在Tiling前被拒绝
fixture未装HCOM替身，实际调用读取系统通信配置
开发机碰巧已初始化而通过，隔离环境失败
```

### 正确示例

```text
fixture注入本UT控制的HCOM返回与调用计数器
输入epSize=0 → 参数错误，提交计数=0
输入合法EP → 只验证本UT要求的任务字段值和参数顺序
fixture退出时恢复原替身；真实通信另在声明环境的集成测试验证
```

### 检视方法

核对 HCOM mock 的符号、签名及生效范围，保证 Tiling 参数校验仍执行真实被测逻辑，隔离通信依赖的 UT 不会意外读取系统通信配置。检查用例改变的 EP/TP 配置、handle 和替身是否恢复，失败退出也应遵守 fixture 的清理约定。

只有用例确实并发访问这些状态时，才检查隔离或同步。使用真实设备通信的集成测试应声明依赖，不强制所有测试采用同一种 mock 或并发模型。

#### 条例结束

## MC2-10.5: 测试必须能发现目标错误，并验证合法边界仍能通过
<a id="mc2-10-5"></a>

MC2-10.5: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及参数、守卫、错误传播或对应测试。

问题描述：测试必须观察实际校验和状态更新的结果，既能发现目标错误，也能确认合法边界没有被错误拒绝。

### 错误示例

前提：capacity>0，回归目标是拒绝非法 expertId=capacity，并保留合法边界 capacity-1。

```python
result = run(expert_id=capacity)
assert result is not None  # Success也能通过，未检查任务是否下发
```

前提：本次修复涉及 EP 与 TP 大小不同的配置，但现有测试只有 EP=TP=1。

```text
测试仅覆盖EP=TP=1，epRank=tpRank=0
实现将EP/TP互换，测试仍通过
据此宣称多域映射已验证
```

### 正确示例

```python
before = backend.submit_count
result = run(expert_id=capacity)
assert result.status == INVALID_ARGUMENT
assert backend.submit_count == before
assert run(expert_id=capacity-1).status == SUCCESS
# 有效边界另验证结果，防止把所有输入一律拒绝也通过。
```

```text
EP大小4，TP大小2；两域名不同
EP成员全局设备[2,5,8,11] → 域内rank[0,1,2,3]
断言globalDevice5在该EP域的rank为1，不是5
核对EP/TP各自分片与收到的配置值
```

### 检视方法

按本次修改涉及的参数选择有区分度的输入：EP/TP 映射用 EP≠TP 和非连续设备号，专家边界用 capacity-1 与 capacity，错误传播检查状态码和任务提交次数。确认相关错误实现会使测试失败，合法边界仍能通过。

用例范围随改动确定，不要求每次运行全部组合。只记录实际运行的测试结果；用例存在或框架名称不能证明覆盖已通过。

#### 条例结束

## MC2-10.7: worker 的失败和异常必须传到调用者及启动器
<a id="mc2-10-7"></a>

MC2-10.7: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 future、worker、脚本、管道或退出码。

问题描述：所有决定任务成败的 worker 结果和异常，都必须由上层进程、脚本和启动器收集并反映在最终状态中。

### 错误示例

前提：两个 rank 的 worker 分别返回 Success 和 Failure，future 的返回值表示各自任务是否成功。

```cpp
for (auto &f : workers) f.wait();
return 0; // rank1业务返回Failure也被当成功
```

前提：shell 脚本运行 MC2 示例后再打印 done。

```bash
./mc2_demo   # 实际退出1
echo done    # 退出0
# 脚本最终退出0，CI误判成功。
```

### 正确示例

```cpp
bool ok = true;
for (auto &f : workers) {
    try {
        Status s = f.get();
        if (s != Success) ok = false;
    } catch (...) { ok = false; }
}
return ok ? 0 : 1;
```

```bash
rc=0
./mc2_demo || rc=$?
cleanup_rc=0
cleanup || cleanup_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$cleanup_rc"
```

### 检视方法

- 逐个核对 worker/future 的结果、异常和提前返回后的收尾处理。沿 wrapper、main 到 shell 检查状态传递，注意 bool 转换、数值截断、管道或 tee、后台任务，以及 cleanup 的返回值覆盖原错误；共享状态必须同步。
- 不能因前一个 worker 失败就跳过剩余的 future.get。join/wait 或 set -e 本身不能保证业务结果被汇总。发生挂起时另查超时和取消机制；已有可靠结果文件等报告方式时，不能只凭 return 0 判定错误被当成成功。

#### 条例结束
