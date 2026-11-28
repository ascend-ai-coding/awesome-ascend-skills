# CANN C++ 通用编码规范

```
<适用>
语言: C++
侧别: All
默认启用: true

领域: false
</适用>
```

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| GEN-1.1 | 对所有外部数据进行合法性检查 | 代码设计 | 高 | [适用: All] |
| GEN-1.3 | 清理无效冗余代码（不可达/重复调用链/只写不读/无效校验） | 代码设计 | 中 | [适用: All] |
| GEN-2.2 | 禁止头文件循环依赖 | 头文件 | 中 | [适用: All] |
| GEN-2.3 | 避免包含用不到的头文件 | 头文件 | 低 | [适用: All] |
| GEN-2.4 | 禁止通过 extern 声明引用外部接口 | 头文件 | 中 | [适用: All] |
| GEN-2.5 | 禁止在 extern "C" 中包含头文件 | 头文件 | 中 | [适用: All] |
| GEN-2.6 | 避免在头文件中使用 using 导入命名空间 | 头文件 | 中 | [适用: All] |
| GEN-3.1 | 避免滥用 typedef/#define 定义类型别名 | 数据类型 | 低 | [适用: All] |
| GEN-3.2 | 使用 using 而非 typedef 定义类型别名 | 数据类型 | 中 | [适用: Host] |
| GEN-4.2 | 禁止使用魔鬼数字/字符串 | 常量 | 中 | [适用: All] |
| GEN-4.3 | 每个常量保证单一职责 | 常量 | 低 | [适用: All] |
| GEN-5.2 | 避免全局变量，谨慎使用单例 | 变量 | 中 | [适用: All] |
| GEN-5.3 | 禁止在自增/自减表达式中再次引用该变量 | 变量 | 高 | [适用: All] |
| GEN-6.1 | 表达式比较遵循左变右不变 | 表达式 | 低 | [适用: All] |
| GEN-6.2 | 使用括号明确操作符优先级 | 表达式 | 中 | [适用: All] |
| GEN-7.1 | 使用 C++ 类型转换而非 C 风格转换 | 转换 | 中 | [适用: Host] |
| GEN-8.1 | switch 语句要有 default 分支 | 控制语句 | 中 | [适用: All] |
| GEN-9.1 | 禁止用 memcpy_s/memset_s 初始化非 POD 对象 | 声明初始化 | 高 | [适用: Host] |
| GEN-10.2 | 优先使用 unique_ptr 而非 shared_ptr | 指针数组 | 中 | [适用: Host] |
| GEN-10.3 | 使用 make_shared 而非 new 创建 shared_ptr | 指针数组 | 中 | [适用: Host] |
| GEN-10.4 | 使用智能指针管理对象 | 指针数组 | 中 | [适用: Host] |
| GEN-10.5 | 禁止使用 auto_ptr | 指针数组 | 中 | [适用: Host] |
| GEN-10.6 | 指针/引用形参不需要修改时使用 const | 指针数组 | 中 | [适用: All] |
| GEN-11.1 | 字符串存储操作确保有 '\0' 结束符 | 字符串 | 高 | [适用: Host] |
| GEN-12.1 | 断言不能用于运行期错误处理 | 断言 | 中 | [适用: All] |
| GEN-13.1 | delete 与 delete[] 配对使用 | 类与对象 | 高 | [适用: Host] |
| GEN-13.2 | 禁止使用 std::move 操作 const 对象 | 类与对象 | 中 | [适用: Host] |
| GEN-13.3 | 严格使用 virtual/override/final 修饰虚函数 | 类与对象 | 中 | [适用: Host] |
| GEN-14.1 | 使用 RAII 特性追踪动态分配 | 函数设计 | 中 | [适用: Host] |
| GEN-14.2 | 非局部 lambda 避免按引用捕获 | 函数设计 | 中 | [适用: Host] |
| GEN-14.3 | 禁止虚函数使用缺省参数值 | 函数设计 | 中 | [适用: Host] |
| GEN-14.4 | 使用强类型参数，避免 void* | 函数设计 | 低 | [适用: All] |
| GEN-15.1 | 函数传参顺序在同一文件内保持一致 | 函数使用 | 低 | [适用: All] |
| GEN-15.2 | 入参用 const T&，出参用 T& 或 T* | 函数使用 | 低 | [适用: All] |
| GEN-15.3 | 不涉及所有权时使用 T* 或 const T& 传参 | 函数使用 | 中 | [适用: Host] |
| GEN-15.4 | 传递所有权使用 shared_ptr + move 传参 | 函数使用 | 中 | [适用: Host] |
| GEN-15.5 | 单参数构造函数用 explicit 修饰 | 函数使用 | 中 | [适用: All] |
| GEN-15.6 | 拷贝构造和拷贝赋值操作符成对出现或禁止 | 函数使用 | 中 | [适用: Host] |
| GEN-15.7 | 禁止保存、delete 指针参数 | 函数使用 | 高 | [适用: Host] |

---

## GEN-1.1: 对所有外部数据进行合法性检查

**严重级别**：高

### 问题描述

对所有外部数据进行合法性检查，包括但不限于：函数入参、外部输入命令行、文件、环境变量、用户数据等。外部数据指本模块边界之外传入、不可假定其合法性的数据。

本条检查「使用外部数据前是否做了入口校验」；校验逻辑本身的安全语义（内存长度、越界、除零等）由 `cpp-secure.md` 安全规范覆盖，本条不重复检查。

### 错误示例

```cpp
// ❌ 直接解引用外部入参，未做合法性检查
ge::graphStatus InferShape(gert::Shape *outShape, const InferParams *params) {
    outShape->SetDim(0, params->batchSize);  // params 未判空即解引用
    ...
}
```

### 正确示例

```cpp
// ✅ 入口先校验再使用
ge::graphStatus InferShape(gert::Shape *outShape, const InferParams *params) {
    if (params == nullptr) {
        return GRAPH_FAILED;
    }
    outShape->SetDim(0, params->batchSize);
    ...
}
```

### 注意事项

1. 外部数据在「首次使用前」校验一次即可，后续传递已校验数据不必重复判（避免无效校验，见 GEN-1.3）
2. 安全语义校验（长度、越界、除零）以 `cpp-secure.md` 为准，本条只管入口校验习惯

---

## GEN-1.3: 清理无效冗余代码（不可达/重复调用链/只写不读/无效校验）

**严重级别**：中

### 问题描述

识别并清理四类无效、冗余或永不执行的代码：

1. **不可达/无外部调用者函数**：无任何调用路径的函数
2. **只写不读字段**：只被赋值、从未被读取的成员字段
3. **重复计算链**：同一计算链在兄弟分支中被完整重复调用
4. **无效校验**：校验条件比真实平台契约更宽松、不起保护作用的校验

本条检查代码的可达性与冗余；判定时须区分真正可删除代码与 ABI/代码生成兼容字段。数据来源与判定步骤见专属检视方法。

### 错误示例

```cpp
// ❌ 重复计算链：ComputeAllRoutes 被 ComputeXxxTiling 调一次，Branch 0/1/2/4/5 seam 又各调一次
__aicore__ inline void ComputeXxxTiling(...) {
    ComputeAllRoutes();                       // 调用 1 次
    if (branch == 0) { ComputeAllRoutes(); }  // 分支内重复整链路计算
    else if (branch == 1) { ComputeAllRoutes(); }
    ...
}
```

### 正确示例

```cpp
// ✅ 公共计算提取到分支外，只计算一次，结果供各分支复用
__aicore__ inline void ComputeXxxTiling(...) {
    auto allRoutes = ComputeAllRoutes();      // 仅调用 1 次
    if (branch == 0) { UseRoutes(allRoutes); }
    else if (branch == 1) { UseRoutes(allRoutes); }
    ...
}
```

### 注意事项

1. 保留项必须 retained-with-reason：ABI、代码生成兼容、真实运行时防御；「可能以后会用」不作为保留理由
2. 证据来源：Read 代码概要的「调用关系图」表；概要表不足以判定时用 Grep 工具反查兜底

### 专属检视方法

**1.3.1 不可达/无外部调用者函数** — 概要调用关系图标注「无外部调用者=是」且无白名单的函数 → SUSPICIOUS。白名单（不算死代码，不报）：
- 宏注册入口：`REG_OP`/`IMPL_OP`/`IMPL_OP_INFERSHAPE` 注册的函数
- Kernel 入口：`__global__ __aicore__` 入口函数
- `extern "C"` 导出函数
- UT 测试函数（tests/ 下接线，或被 `TEST`/`GTEST` 宏调用）

**1.3.2 只写不读字段** — 对 SUSPICIOUS 的无调用者函数内的成员字段，Grep 查赋值点（`field =`/`field_ =`）与读取点。同侧别内只赋值不读取 → SUSPICIOUS。豁免：跨侧别读写（Host 写 Kernel 读）、ABI 兼容字段、调试/日志用字段（需 retained-with-reason）。

**1.3.3 重复计算链** — 概要调用关系图标注「重复调用链=是」的函数（同函数被多个兄弟分支 if/else if/seam 重复完整调用）。确认重复调用执行相同计算链 → FAIL。

**1.3.4 无效校验** — 若概要/API 预研显示某参数有固定平台契约（如 Kernel 固定使用 256B），而代码校验条件更宽松（如 `vectorSize > 0` 未保护 256B 契约）→ SUSPICIOUS + 建议升级为真实契约校验。仅在概要/预研有契约证据时报，不凭空查。

**判定**：
- 1.3.3 重复计算链 → FAIL（明确冗余，应消除重复调用）
- 1.3.1/1.3.2/1.3.4 → SUSPICIOUS（需开发者确认是否可删，区分 ABI/兼容字段）

---

## GEN-2.2: 禁止头文件循环依赖

**严重级别**：中

### 问题描述

禁止头文件循环依赖：a.h 包含 b.h，b.h 包含 c.h，c.h 又包含 a.h 之类导致任何一个头文件修改，都导致所有包含了 a.h/b.h/c.h 的代码全部重新编译一遍。头文件循环依赖直接体现了架构设计上的不合理，可通过优化架构去避免。

本条检查头文件包含关系成环；单方向的多余包含归 GEN-2.3。

### 错误示例

```cpp
// a.h
#include "b.h"   // ❌ a → b
// b.h
#include "c.h"   // ❌ b → c
// c.h
#include "a.h"   // ❌ c → a，循环闭合
```

### 正确示例

```cpp
// c.h
class A;                     // ✅ 用前向声明打破循环，不再 #include "a.h"
void Func(const A &a);
```

### 注意事项

优先用前向声明、拆分公共类型头等方式打破循环，而非依赖 include guard 掩盖问题。

---

## GEN-2.3: 避免包含用不到的头文件

**严重级别**：低

### 问题描述

避免包含当前翻译单元用不到的头文件。未使用的头文件会增加编译依赖和编译时间。本条为建议级：某些头文件可能为未来功能扩展预留，或用于提供类型前向声明，检视时仅作提醒，不强制标记为 FAIL。

### 错误示例

```cpp
// ❌ 只用了 std::vector，却包含了大量无关头文件
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
```

### 正确示例

```cpp
// ✅ 按需包含
#include <vector>
```

### 注意事项

检视时仅作为提醒，不强制标记为 FAIL；未来扩展预留的包含可接受（需注释说明用途）。

---

## GEN-2.4: 禁止通过 extern 声明引用外部接口

**严重级别**：中

### 问题描述

禁止通过 extern 声明的方式引用外部函数接口、变量。extern 声明绕过头文件，缺少类型检查与一致性约束，接口签名变更时不会产生编译告警，易引入链接期与运行期错误，应包含提供接口的头文件。

本条检查 extern 形式的跨文件符号引用；extern "C" 修饰的 Kernel 入口声明除外（见注意事项）。

### 错误示例

```cpp
// ❌ 自行 extern 声明外部函数，绕开头文件
extern int32_t CalcTilingKey(const TilingParseContext *ctx);
```

### 正确示例

```cpp
// ✅ 包含提供该接口的头文件
#include "tiling_key_utils.h"
```

### 注意事项

Kernel 侧例外：允许 `extern "C"` 声明 Kernel 入口函数。

---

## GEN-2.5: 禁止在 extern "C" 中包含头文件

**严重级别**：中

### 问题描述

禁止在 extern "C" 块中包含（#include）头文件。extern "C" 块内的头文件会以 C 链接修饰编译，破坏其中 C++ 接口的名称修饰，引发链接错误或符号冲突。

### 错误示例

```cpp
// ❌ 头文件被包进 extern "C"，其中 C++ 接口的 mangling 被破坏
extern "C" {
#include "kernel_entry.h"
void KernelEntry();
}
```

### 正确示例

```cpp
// ✅ include 放在 extern "C" 之外
#include "kernel_entry.h"
extern "C" void KernelEntry();
```

### 注意事项

Kernel 侧说明：Kernel 入口必须使用 `extern "C"`，但不应在其中包含头文件。

---

## GEN-2.6: 避免在头文件中使用 using 导入命名空间

**严重级别**：中

### 问题描述

`using namespace` 在头文件中的传播范围取决于其作用域：
- **file-scope**（命名空间外部）：传播到所有包含该头文件的翻译单元
- **namespace-scoped**（`namespace X {}` 内部）：仅传播到重新打开同一命名空间的代码

本条检查头文件中的命名空间导入污染；FAIL/SUSPICIOUS 判定与危害评估步骤见专属检视方法，框架惯例豁免见注意事项。

### 错误示例

```cpp
// common/include/utils.h
using namespace std;   // ❌ 共享头文件 file-scope 导入大型命名空间，污染所有 includer

// op_host/xxx.h
using namespace ge;    // ❌ using namespace 位于后续 #include 之前，后续头文件在该命名空间上下文中编译
#include "foo.h"
```

### 正确示例

```cpp
// common/include/utils.h
// ✅ 头文件中使用全限定名
std::string Trim(const std::string &s);
```

### 注意事项

- **Kernel 侧豁免**：以下 `using namespace` 是 Ascend C 框架惯例，不标记：
  - `using namespace AscendC;`
  - 导入 AscendC 子命名空间：`matmul`、`regbaseutil`、`MicroAPI`、`AscendC::Impl::Detail`、`AscendC::MicroAPI`、`optiling`
  - 导入算子内部实现命名空间（如 `AttentionCommon`、`fa_base_matmul`、`NormCommon` 等）
- **Host 侧豁免**：导入**项目内部命名空间**（如 `Ops::Transformer::OpTiling`、`Ops::NN::Optiling`、`Ops::Base`、`mc2_matmul_v3_advanced` 等）不标记，这些命名空间受项目控制，冲突风险低

### 专属检视方法

**FAIL 条件**（满足任一即 FAIL，但须先做危害分析再定级）：

1. **file-scope `using namespace` 出现在同文件后续 `#include` 之前**——后续 include 在该命名空间上下文中编译
2. **共享头文件路径**（`common/include/` 等公共目录）在 file-scope 导入大型命名空间（`std`、`ge`、`gert` 等）

**定级前必须追溯传播链评估实际危害，结构违规 ≠ 实质危害：**

1. **确认 using 作用域**：在 `namespace X {}` 块内部则只对 X 可见，不泄漏到 includer 的 file-scope，**不是 file-scope 污染**
2. **追踪 include 链**：若被污染头文件的 include guard 在 using 之前已被更早的头文件激活，则**污染未实际发生**
3. **检查子头文件自主性**：若子头文件自带相同 using namespace，则外部污染是**冗余**，无新增风险
4. **定级**：有真实新增污染 → FAIL；结构违规但无实质危害 → SUSPICIOUS，注明原因

**SUSPICIOUS 条件**（标记提醒，不强制 FAIL）：

- 非共享头文件在 file-scope 导入 `std` / `ge` / `gert`
- API 头文件（op_api/）导入非项目命名空间（被外部调用者 include）
- 共享头文件在命名空间**内部**（非 file-scope）导入大型命名空间

---

## GEN-3.1: 避免滥用 typedef/#define 定义类型别名

**严重级别**：低

### 问题描述

避免滥用 typedef 或 #define 对基本类型起别名。无语义增益的别名（如 `typedef int MyInt`）只增加一层间接，降低可读性。本条为建议级；Host 侧 typedef 与 using 的选择见 GEN-3.2。

### 错误示例

```cpp
// ❌ 无语义的别名，纯增加间接层
typedef int MyInt;
#define UINT unsigned int
```

### 正确示例

```cpp
// ✅ 别名承载语义（单位、用途、约束）
using UbBlockSize = uint32_t;   // 明确表达 UB 块大小的语义
```

### 注意事项

#define 别名不做任何类型检查，优先用 typedef/using。

---

## GEN-3.2: 使用 using 而非 typedef 定义类型别名

**严重级别**：中

### 问题描述

使用 using 而非 typedef 定义类型的别名，避免类型变化带来的散弹式修改。using 支持模板别名，语义更清晰。本条适用于 Host 侧；Kernel 侧不适用（见注意事项）。

### 错误示例

```cpp
// ❌ typedef 定义别名
typedef std::shared_ptr<FooBar> FooBarPtr;
```

### 正确示例

```cpp
// ✅ using 定义别名
using FooBarPtr = std::shared_ptr<FooBar>;
```

### 注意事项

Kernel 侧不适用：Kernel 模板类无命名空间，using 定义别名不适用。

---

## GEN-4.2: 禁止使用魔鬼数字/字符串

**严重级别**：中

### 问题描述

代码中禁止直接使用未经命名的字面量（魔鬼数字/魔鬼字符串），应使用 `constexpr` 或 `const` 命名常量替代，使含义自解释。例外：`0`、`-1`、`1` 等基础值，数组索引 `0`，`true`/`false`/`nullptr`。

本条检查与硬件规格无关的通用裸数字；数值命中具体代际硬件规格的写死问题（如 block 大小、硬件参数）归 Kernel 侧对应规则与 tiling 接口取值约束，不在本条重复报。

### 错误示例

**【真实检视案例】**（来自 ops-math 历史检视 PR，均被人工采纳）

案例1 — Kernel 侧，`32` 作为对齐粒度反复出现（ops-math PR#377，`op_kernel/split_d.h:213`）：

> 评论：「避免魔鬼数字 32，建议用 static constexpr 变量替换，用变量名体现 32 的含义」

```cpp
 208 | template <typename T>
 209 | __aicore__ inline void DataCopyPadInAdaptive(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,const uint32_t calCount) {
 210 |     const uint64_t totalByte = calCount * sizeof(T);
 211 |     uint64_t st =  reinterpret_cast<uint64_t>(src.GetPhyAddr());
 212 |     uint64_t en = st + totalByte;
 213 |     uint32_t pre = (32 - st % 32) % 32;   // ❌ 魔鬼数字 32 = 对齐字节数，应定义为 constexpr uint32_t ALIGN_BYTES = 32;
 214 |     uint32_t aft = en % 32;               // ❌ 同一魔鬼数字多处重复
 215 |     uint32_t prenum = pre / sizeof(T);
 216 |     uint32_t midnum = (totalByte - pre - aft) / sizeof(T);
```

案例2 — tiling 代码，`32` 作为 blockSize（ops-math PR#1767，`op_host/div_mod_tiling.cpp:136`）：

> 评论：「建议通过接口获取 blockSize，消除魔鬼数字」

```cpp
 135 |     uint32_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
 136 |     uint32_t alignNum = 32 / typeSize;   // ❌ 魔鬼数字 32 = block 字节数，建议用接口获取 blockSize
 137 |     uint32_t totalLengthAligned = ((totalLength + alignNum - 1) / alignNum) * alignNum;
```

案例3 — tiling 代码，`4`/`10` 作为 buffer 数量（ops-math PR#387，`op_host/round_tiling.cpp:113`）：

> 评论：「魔鬼数字建议使用具有含义的常量表示」

```cpp
 110 |     ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
 111 |     if (dataType == ge::DT_INT32) {
 112 |         // x, y → 2 个，因为要做 doublebuffer 优化，所以 x(x2), y(x2) → 共 4 个
 113 |         ubDataNumber = 4;                 // ❌ 魔鬼数字 4 = buffer 数量，应定义为命名常量
 114 |     } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
 115 |         if (decimals) {
 116 |             // x(x2), y(x2), round_temp(x2), x_as_float32(x2), x_scaled(x2) → 共 10 个
 117 |     ubDataNumber = 10;            // ❌ 魔鬼数字 10
```

### 正确示例

```cpp
// ✅ 命名常量表达业务/硬件语义
static constexpr uint32_t ALIGN_BYTES = 32;
uint32_t pre = (ALIGN_BYTES - st % ALIGN_BYTES) % ALIGN_BYTES;

static constexpr uint32_t INT32_UB_BUFFER_NUM = 4;   // x(x2), y(x2) doublebuffer
static constexpr uint32_t BF16_DECIMALS_UB_BUFFER_NUM = 10;
```

### 注意事项

豁免（命中即 PASS，不计证据）：`0`/`1`/`-1` 基础值、数组下标 `[0]`、`i++`/`offset + 1` 类步进、已具名常量定义处的数字（命名即语义）、`sizeof(T)` 参与的换算。

### 专属检视方法

本方法负责候选定位与豁免判定；评分仍按 `core/methodology.md` 假设检验输出（GEN 为 clause 类，yaml 需带 evidence/confidence 字段）。

1. **候选定位**（Grep 变更代码，聚焦三类高危位置）：
   - 条件比较右侧：`grep -nE '(==|!=|>=|<=|>|<) ?[0-9]' {file}`
   - 赋值/初始化右侧：`grep -nE '= ?[0-9]+[;,)]' {file}`
   - API 实参裸数字：`grep -nE '\([0-9]+,|[0-9]+, ?[0-9]+' {file}`（重点盯 blockLen / repeatTime / tileNum 等参数）
2. **豁免排除**（命中任一即 PASS，不计证据）：`0`/`1`/`-1` 基础值、数组下标 `[0]`、`i++`/`offset + 1` 类步进；**已具名常量定义处**的数字（`constexpr int32_t ALIGN_BYTES = 32;` 命名即语义，合规）；`sizeof(T)` 参与的换算
3. **概要交叉核对**：Read 代码概要的「常量清单」表，已登记常量视为合规基线；同一数字 ≥2 处出现且未命名 → 规范违反 +40% + 上下文防御缺失 +30%（分值定义见 methodology 证据表）
4. **判定与建议**：非豁免位置的字面量 FAIL，修复建议必须给出建议常量名（表达业务/硬件语义，如 `MAX_SEQ_LEN`、`ALIGN_BYTES`、`UB_BUFFER_NUM`）

---

## GEN-4.3: 每个常量保证单一职责

**严重级别**：低

### 问题描述

建议每个常量保证单一职责：一个命名常量只表达一个语义，不因多处恰好同值而共用一个常量。同值不同义的位置各自定义常量，避免后续需求变化时牵连修改。

### 错误示例

```cpp
// ❌ 用一个常量承载两个语义：既是 buffer 数量又是循环展开因子
static constexpr uint32_t NUM_4 = 4;
uint32_t ubDataNumber = NUM_4;
#pragma unroll NUM_4
```

### 正确示例

```cpp
// ✅ 各自定义，语义单一
static constexpr uint32_t UB_BUFFER_NUM = 4;   // buffer 数量
static constexpr uint32_t UNROLL_FACTOR = 4;   // 循环展开因子
```

### 注意事项

本条为建议级，检视时作提醒，不强制 FAIL。

---

## GEN-5.2: 避免全局变量，谨慎使用单例

**严重级别**：中

### 问题描述

尽量避免使用全局变量，谨慎使用单例模式，避免滥用。全局变量与单例引入隐藏的共享状态与初始化顺序依赖，破坏可测试性与线程安全。本条检查新增全局可变状态；const 常量与编译期 constexpr 不受本条限制。

### 错误示例

```cpp
// ❌ 全局可变状态，初始化顺序与并发访问均不可控
static KernelConfig *g_config = nullptr;
```

### 正确示例

```cpp
// ✅ 状态收进类成员，随对象生命周期管理
class KernelRunner {
  private:
    KernelConfig config_;
};
```

### 注意事项

Kernel 侧说明：允许 `__aicore__` 模板类的成员变量（本质是全局存储）。

---

## GEN-5.3: 禁止在自增/自减表达式中再次引用该变量

**严重级别**：高

### 问题描述

禁止在变量自增或自减运算的表达式中再次引用该变量。此类表达式中修改与读取的求值顺序未定义，行为不确定，属正确性风险。本条只检查自增/自减复用模式；其它未定义行为由 `cpp-secure.md` 覆盖。

### 错误示例

```cpp
// ❌ 同一表达式中再次引用自增/自减变量
i = i++;
a[i] = i++;
ret = i++ + ++i;
```

### 正确示例

```cpp
// ✅ 拆分为独立语句，每条语句只做一件事
i = i + 1;
a[i] = i;
```

### 注意事项

本条属正确性风险，命中即 FAIL，不做 SUSPICIOUS 降级。

---

## GEN-6.1: 表达式比较遵循左变右不变

**严重级别**：低

### 问题描述

表达式的比较遵循左侧倾向于变化、右侧倾向于不变的原则：左值放变化的对象，右值放常量/期望值，符合自然语言阅读顺序。本条为建议级风格规则，只检查比较表达式两侧的摆放。

### 错误示例

```cpp
// ❌ 常量在左（Yoda 写法），与阅读直觉相反
if (SUCCESS != ret) {
  ...
}
```

### 正确示例

```cpp
// ✅ 变量在左、常量在右
if (ret != SUCCESS) {
  ...
}
```

### 注意事项

本条为建议级，检视时作提醒，不强制 FAIL。

---

## GEN-6.2: 使用括号明确操作符优先级

**严重级别**：中

### 问题描述

通过使用括号明确操作符的优先级，避免出现低级错误。混用 `&&`/`||`、位运算与比较等优先级易混淆的场景必须加括号，不依赖读者对优先级表的记忆。

### 错误示例

```cpp
// ❌ || 与 && 混用不加括号，意图依赖优先级表
if (cond1 || cond2 && cond3) {
  ...
}
```

### 正确示例

```cpp
// ✅ 括号明确分组
if (cond1 || (cond2 && cond3)) {
  ...
}
```

### 注意事项

无歧义场景（单目、乘除等）可不加括号，避免括号噪音。

---

## GEN-7.1: 使用 C++ 类型转换而非 C 风格转换

**严重级别**：中

### 问题描述

使用 C++ 提供的类型转换（static_cast/dynamic_cast 等），而不是 C 风格的强制转换，避免使用 const_cast 和 reinterpret_cast。C 风格转换依次尝试多种转换序列，意图不明且绕过类型检查。

### 错误示例

```cpp
// ❌ C 风格强转
uint32_t val = (uint32_t)offset;
auto *ctx = (ComputeContext *)userData;
```

### 正确示例

```cpp
// ✅ 意图明确的 C++ 转换
auto val = static_cast<uint32_t>(offset);
auto *ctx = static_cast<ComputeContext *>(userData);
```

### 注意事项

Kernel 侧不适用：Kernel 底层硬件操作必须用 `reinterpret_cast`（如 GM 地址转换）。

---

## GEN-8.1: switch 语句要有 default 分支

**严重级别**：中

### 问题描述

switch 语句要有 default 分支。枚举新增值或入参非法时，无 default 的 switch 会静默跳过，隐藏错误。default 用于兜底处理（报错或合理缺省），即使当前所有枚举值都已覆盖也应保留。

### 错误示例

```cpp
// ❌ 无 default，新增枚举值时静默漏处理
switch (dtype) {
    case ge::DT_FLOAT:
        return 4;
    case ge::DT_FLOAT16:
        return 2;
}
```

### 正确示例

```cpp
// ✅ default 兜底
switch (dtype) {
    case ge::DT_FLOAT:
        return 4;
    case ge::DT_FLOAT16:
        return 2;
    default:
        return 0;
}
```

### 注意事项

default 不应为空语句；至少打日志或返回错误码。

---

## GEN-9.1: 禁止用 memcpy_s/memset_s 初始化非 POD 对象

**严重级别**：高

### 问题描述

禁止用 `memcpy_s`、`memset_s` 初始化非 POD 对象。按位覆写会破坏虚表指针与成员对象的内部不变量，属未定义行为。非 POD 对象必须用构造函数初始化。

### 错误示例

```cpp
// ❌ 按位清零含 vtable/非平凡成员的对象
FooBar foo;
memset_s(&foo, sizeof(FooBar), 0, sizeof(FooBar));
```

### 正确示例

```cpp
// ✅ 用构造/值初始化
FooBar foo{};
```

### 注意事项

Kernel 侧不适用：Kernel 模板类都是 POD，且 `memset_s` 不可用。

---

## GEN-10.2: 优先使用 unique_ptr 而非 shared_ptr

**严重级别**：中

### 问题描述

优先使用 unique_ptr 而不是 shared_ptr。shared_ptr 引入引用计数开销与生命周期模糊；所有权单一时 unique_ptr 语义更准确、开销更低。

### 错误示例

```cpp
// ❌ 所有权独占却用 shared_ptr
std::shared_ptr<KernelContext> ctx = std::make_shared<KernelContext>();
```

### 正确示例

```cpp
// ✅ 单一所有权用 unique_ptr
std::unique_ptr<KernelContext> ctx = std::make_unique<KernelContext>();
```

### 注意事项

Kernel 侧不适用：Kernel 无智能指针。

---

## GEN-10.3: 使用 make_shared 而非 new 创建 shared_ptr

**严重级别**：中

### 问题描述

使用 std::make_shared 而不是 new 创建 shared_ptr。make_shared 一次分配同时承载对象与控制块，且避免裸 new 构造的泄漏窗口。

### 错误示例

```cpp
// ❌ 裸 new 构造 shared_ptr
std::shared_ptr<FooBar> foo(new FooBar());
```

### 正确示例

```cpp
// ✅ make_shared 构造
std::shared_ptr<FooBar> foo = std::make_shared<FooBar>();
```

### 注意事项

Kernel 侧不适用：Kernel 无智能指针。

---

## GEN-10.4: 使用智能指针管理对象

**严重级别**：中

### 问题描述

使用智能指针管理对象，避免使用 new/delete 裸管理动态内存。裸 new/delete 在异常与多返回路径下易泄漏，智能指针以 RAII 保证释放。

### 错误示例

```cpp
// ❌ 裸 new，多返回路径下泄漏
FooBar *foo = new FooBar();
if (!foo->Init()) {
    return GRAPH_FAILED;   // ❌ 泄漏
}
```

### 正确示例

```cpp
// ✅ 智能指针自动释放
auto foo = std::make_unique<FooBar>();
if (!foo->Init()) {
    return GRAPH_FAILED;   // ✅ 自动释放
}
```

### 注意事项

Kernel 侧不适用：Kernel 禁止动态内存分配，Buffer 由 `InitBuffer` 静态分配。

---

## GEN-10.5: 禁止使用 auto_ptr

**严重级别**：中

### 问题描述

禁止使用 auto_ptr。auto_ptr 的拷贝语义是转移所有权，与直觉相悖且已在 C++17 移除；需要独占所有权用 unique_ptr。

### 错误示例

```cpp
// ❌ auto_ptr 拷贝即转移，误用易产生悬垂
std::auto_ptr<FooBar> a(new FooBar());
std::auto_ptr<FooBar> b = a;   // a 变为空
```

### 正确示例

```cpp
// ✅ unique_ptr 语义明确
std::unique_ptr<FooBar> a = std::make_unique<FooBar>();
```

### 注意事项

Kernel 侧不适用：Kernel 无智能指针。

---

## GEN-10.6: 指针/引用形参不需要修改时使用 const

**严重级别**：中

### 问题描述

对于指针和引用类型的形参，如果是不需要修改的，要求使用 const。const 形参表达「只读」契约，防止函数内误写，并允许调用方传 const 对象。

### 错误示例

```cpp
// ❌ 只读入参未加 const，可被误改且限制调用方
uint32_t CalcSize(InputData *input);
bool IsEmpty(std::string &s);
```

### 正确示例

```cpp
// ✅ 只读形参加 const
uint32_t CalcSize(const InputData *input);
bool IsEmpty(const std::string &s);
```

### 注意事项

出参（需要写回调用方）不加 const；本条只约束只读形参。

---

## GEN-11.1: 字符串存储操作确保有 '\0' 结束符

**严重级别**：高

### 问题描述

对字符串进行存储操作，确保字符串有 '\0' 结束符。缺结束符的 C 风格字符串在后续 strlen/日志输出等操作中越界读，属内存安全风险。

### 错误示例

```cpp
// ❌ strncpy 拷满缓冲区不补 '\0'，后续输出越界读
char name[MAX_LEN];
strncpy_s(name, sizeof(name), inputName, sizeof(name));
OutputLog("name=%s", name);   // ❌ 无结束符
```

### 正确示例

```cpp
// ✅ 预留结束符空间并显式写入
char name[MAX_LEN] = {0};
strncpy_s(name, sizeof(name) - 1, inputName, sizeof(name) - 1);
name[sizeof(name) - 1] = '\0';
```

### 注意事项

Kernel 侧不适用：Kernel 无 C 风格字符串处理。Host 侧优先使用 std::string。

---

## GEN-12.1: 断言不能用于运行期错误处理

**严重级别**：中

### 问题描述

断言不能用于校验程序在运行期间可能导致的错误，可能发生的运行错误要用错误处理代码来处理。断言在 Release（NDEBUG）下被编译移除，依赖断言拦截运行期错误等于无校验。

### 错误示例

```cpp
// ❌ 用断言拦截运行期输入错误，Release 下失效
assert(context != nullptr);
context->GetInputDesc(0);
```

### 正确示例

```cpp
// ✅ 运行期错误用错误处理代码
if (context == nullptr) {
    return GRAPH_FAILED;
}
context->GetInputDesc(0);
```

### 注意事项

断言仅用于「违反即程序 bug」的内部逻辑不变式检查，不可用于外部输入校验。

---

## GEN-13.1: delete 与 delete[] 配对使用

**严重级别**：高

### 问题描述

单个对象释放使用 delete，数组对象释放使用 delete []。不配对使用属未定义行为：对数组用 delete 只析构首元素，对单对象用 delete[] 同样未定义。

### 错误示例

```cpp
// ❌ new[] 配 delete
const int kSize = 5;
int *number_array = new int[kSize];
delete number_array;   // ❌ 未定义行为
```

### 正确示例

```cpp
// ✅ 配对释放并置空
const int kSize = 5;
int *number_array = new int[kSize];
int *number = new int();
delete[] number_array;
number_array = nullptr;
delete number;
number = nullptr;
```

### 注意事项

Kernel 侧不适用：Kernel 禁止动态内存分配。Host 侧优先以 vector/智能指针替代裸 new[]。

---

## GEN-13.2: 禁止使用 std::move 操作 const 对象

**严重级别**：中

### 问题描述

禁止使用 std::move 操作 const 对象。对 const 对象 move 无法触发移动构造，静默退化为拷贝（或编译失败），语义与预期不符。

### 错误示例

```cpp
// ❌ move const 对象，退化为拷贝
const FooBar source;
auto sink = Sink(std::move(source));   // 实际执行拷贝
```

### 正确示例

```cpp
// ✅ 非 const 对象才 move；只读场景直接传 const 引用
FooBar source;
auto sink = Sink(std::move(source));
```

### 注意事项

Kernel 侧不适用：Kernel 无 `std::move`，模板类不支持移动语义。

---

## GEN-13.3: 严格使用 virtual/override/final 修饰虚函数

**严重级别**：中

### 问题描述

严格使用 virtual/override/final 修饰虚函数：基类虚函数声明用 virtual，派生类覆写用 override，禁止再覆写的终态用 final。override 让编译器校验签名一致性，防止「以为覆写实则新函数」。

### 错误示例

```cpp
// ❌ 派生类无 override，签名笔误时静默变成新函数
class Derived : public Base {
  public:
    void func();   // 若基类是 Func()，此处实为新函数
};
```

### 正确示例

```cpp
class Base {
  public:
    virtual void Func();
};

class Derived : public Base {
  public:
    void Func() override;
};

class FinalDerived : public Derived {
  public:
    void Func() final;
};
```

### 注意事项

Kernel 侧不适用：Kernel 无虚函数。

---

## GEN-14.1: 使用 RAII 特性追踪动态分配

**严重级别**：中

### 问题描述

使用 RAII 特性来帮助追踪动态分配：锁、句柄、内存等资源的获取与释放绑定对象生命周期，多返回路径与异常下不泄漏。

### 错误示例

```cpp
// ❌ 手工加解锁，中途 return 导致死锁
mutex_.lock();
if (!ready) {
    return;   // ❌ 未解锁
}
DoWork();
mutex_.unlock();
```

### 正确示例

```cpp
// ✅ lock_guard RAII 自动解锁
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ready) {
    return;   // ✅ 自动解锁
  }
  DoWork();
}
```

### 注意事项

Kernel 侧不适用：Kernel 无动态分配、无 `std::mutex`，无 RAII 场景。

---

## GEN-14.2: 非局部 lambda 避免按引用捕获

**严重级别**：中

### 问题描述

非局部范围使用 lambda 时，避免按引用捕获。lambda 逃逸出定义作用域（提交线程池、异步回调、存储后调用）后，引用捕获的局部变量已销毁，产生悬垂引用。

### 错误示例

```cpp
// ❌ 逃逸 lambda 按引用捕获局部变量，执行时局部变量已销毁
{
  int local_var = 1;
  auto func = [&]() { ...; std::cout << local_var << std::endl; };   // ❌ 悬垂引用
  thread_pool.commit(func);
}
```

### 正确示例

```cpp
// ✅ 按值捕获
{
  int local_var = 1;
  auto func = [local_var]() { ...; std::cout << local_var << std::endl; };   // ✅ 拷贝持有
  thread_pool.commit(func);
}
```

### 注意事项

Kernel 侧不适用：Kernel 不支持 lambda 表达式。同步调用（lambda 当帧执行完前已运行完毕）可按引用捕获。

---

## GEN-14.3: 禁止虚函数使用缺省参数值

**严重级别**：中

### 问题描述

禁止虚函数使用缺省参数值。缺省参数按静态类型绑定、虚函数按动态类型派发，两者组合产生「调用基类缺省值 + 执行派生类函数」的错配，行为反直觉。

### 错误示例

```cpp
// ❌ 虚函数带缺省参数
class Base {
  public:
    virtual void Func(int level = 1);
};
class Derived : public Base {
  public:
    void Func(int level = 2) override;
};
Base *p = new Derived();
p->Func();   // ❌ 执行 Derived::Func，但 level 取 Base 的 1
```

### 正确示例

```cpp
// ✅ 虚函数不带缺省参数，由调用方显式传参
class Base {
  public:
    virtual void Func(int level);
};
```

### 注意事项

Kernel 侧不适用：Kernel 无虚函数。

---

## GEN-14.4: 使用强类型参数，避免 void*

**严重级别**：低

### 问题描述

使用强类型参数、成员变量，避免使用 void*。void* 丢失类型信息，使用处强制转换无校验，类型错误只能运行期暴露。本条为建议级；框架回调等契约要求的 void* 入口除外。

### 错误示例

```cpp
// ❌ void* 传参，使用处强转无校验
void Process(void *data);
...
Process(&myTensor);
auto *tensor = static_cast<Tensor *>(data);   // ❌ 类型错误运行期才暴露
```

### 正确示例

```cpp
// ✅ 强类型接口
void Process(Tensor *data);
```

### 注意事项

框架契约（如回调签名）要求 void* 时，入口处立即转型并校验，不向内部传递 void*。

---

## GEN-15.1: 函数传参顺序在同一文件内保持一致

**严重级别**：低

### 问题描述

函数传参顺序在同一文件（或同一模块）内保持一致。不强制要求「入参在前、出参在后」，只要同一文件内的函数参数顺序风格统一即可（如统一采用入参在前，或统一采用出参在前）。本条为建议级，只检查同文件内风格一致性。

### 错误示例

```cpp
// ❌ 不一致：同文件内混用
bool FuncE(const std::string &in, FooBar *out);   // 入参在前
bool FuncF(Result *out, const int &val);           // 出参在前 → 风格不一致
```

### 正确示例

```cpp
// ✅ 风格统一：全部采用入参在前
bool FuncA(const std::string &in, FooBar *out1, FooBar *out2);
bool FuncB(const int &val, Result *out);

// ✅ 风格统一：全部采用出参在前
bool FuncC(FooBar *out1, FooBar *out2, const std::string &in);
bool FuncD(Result *out, const int &val);
```

### 注意事项

检视时以文件内多数函数的风格为基准，仅标记明显不一致的情况。

---

## GEN-15.2: 入参用 const T&，出参用 T& 或 T*

**严重级别**：低

### 问题描述

函数传参传递，入参用 `const T &`，出参用 `T &` 或 `T *`。出参使用引用或指针均可；本条为建议级，不限定出参必须为引用或指针，但同一文件内应保持一致。

### 错误示例

```cpp
// ❌ 同文件内混用（风格不一致）
void FuncA(const Input &in, Output &out);       // 出参用引用
void FuncB(const Input &in, Output *out);       // 出参用指针 → 同文件风格不一致
```

### 正确示例

```cpp
// ✅ 出参用引用（算子仓常见风格）
bool Func(const std::string &in, FooBar &out1, FooBar &out2);
void GetBasicShape(const gert::StorageShape *queryShape, uint32_t &b, uint32_t &s, uint32_t &h);

// ✅ 出参用指针（框架接口风格）
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
static ge::graphStatus InferShape(const gert::Shape *inShape, gert::Shape *outShape);
```

### 注意事项

算子仓实践中，`T &` 是更常见的出参方式（尤其标量出参和 TilingData 结构体），`T *` 多见于框架接口（如 `gert::Shape*`）或需要表达可选（nullable）语义的场景。

---

## GEN-15.3: 不涉及所有权时使用 T* 或 const T& 传参

**严重级别**：中

### 问题描述

函数传参传递，不涉及所有权的场景，使用 T * 或 const T & 作为参数，而不是智能指针。智能指针形参暗示所有权语义，不转移所有权却收智能指针会误导调用方并增加引用计数开销。

### 错误示例

```cpp
// ❌ 不转移所有权却收智能指针
bool Func(std::shared_ptr<FooBar> in);
```

### 正确示例

```cpp
// ✅ 只读场景用 const 引用
bool Func(const FooBar &in);
```

### 注意事项

Kernel 侧不适用：Kernel 无智能指针。

---

## GEN-15.4: 传递所有权使用 shared_ptr + move 传参

**严重级别**：中

### 问题描述

函数传参传递，如需传递所有权，建议使用 shared_ptr + move 传参。按值接收 + std::move 到成员，明确表达所有权移交，避免多余拷贝。

### 错误示例

```cpp
// ❌ 收 const 智能指针引用，无法表达所有权移交且产生拷贝
class Foo {
  public:
    explicit Foo(const std::shared_ptr<T> &x) : x_(x) {}
  private:
    std::shared_ptr<T> x_;
};
```

### 正确示例

```cpp
// ✅ 值传入 + move 到成员
class Foo {
  public:
    explicit Foo(std::shared_ptr<T> x): x_(std::move(x)) {}
  private:
    std::shared_ptr<T> x_;
};
```

### 注意事项

Kernel 侧不适用：Kernel 无智能指针。

---

## GEN-15.5: 单参数构造函数用 explicit 修饰

**严重级别**：中

### 问题描述

单参数构造函数必须用 explicit 修饰，多参数构造函数禁止使用 explicit 修饰。单参构造函数隐式充当类型转换运算符，意外转换产生临时对象；explicit 禁止隐式转换。

### 错误示例

```cpp
Foo(int x, int y=0);          // ❌ 可单参调用却无 explicit，允许隐式转换
explicit Foo(int x, int y);   // ❌ 恒两参调用，explicit 无意义
```

### 正确示例

```cpp
explicit Foo(int x);          // ✅ 单参构造加 explicit
explicit Foo(int x, int y=0); // ✅ 带缺省可单参调用，同样加 explicit
```

### 注意事项

判定标准是「能否以单参数调用」而非形参个数：带缺省值的多参构造若可单参调用，同样需要 explicit。

---

## GEN-15.6: 拷贝构造和拷贝赋值操作符成对出现或禁止

**严重级别**：中

### 问题描述

拷贝构造和拷贝赋值操作符应该是成对出现或者禁止。只定义其一会留下与预期不符的默认另一半（如浅拷贝）；无需拷贝语义时用 `= delete` 显式禁止。

### 错误示例

```cpp
// ❌ 只定义拷贝构造，拷贝赋值退化为默认浅拷贝
class Foo {
  private:
    Foo(const Foo&) = default;
};
```

### 正确示例

```cpp
class Foo {
  private:
    Foo(const Foo&) = default;
    Foo& operator=(const Foo&) = default;
    Foo(Foo&&) = delete;
    Foo& operator=(Foo&&) = delete;
};
```

### 注意事项

Kernel 侧不适用：Kernel 模板类通常只有默认实现，不显式定义拷贝/移动。

---

## GEN-15.7: 禁止保存、delete 指针参数

**严重级别**：高

### 问题描述

禁止保存、delete 指针参数。指针形参的所有权在调用方，被调方存储（超出调用生命周期使用）或释放（double free）都会产生悬垂指针与重复释放。

### 错误示例

```cpp
// ❌ 保存指针参数供后续使用 + 自行释放
void Register(const Config *cfg) {
    savedCfg = cfg;          // ❌ 生命周期不受控
}
void Cleanup(const Config *cfg) {
    delete cfg;              // ❌ 所有权在调用方
}
```

### 正确示例

```cpp
// ✅ 需要持有就拷贝数据；释放交还调用方
void Register(const Config &cfg) {
    savedConfig = cfg;       // ✅ 拷贝持有
}
```

### 注意事项

Kernel 侧不适用：Kernel 无 `delete`，指针参数来自 Buffer。确需移交所有权的接口用智能指针显式表达（见 GEN-15.4）。
