# CANN C++ 代码风格规范

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
| STYLE-1.1 | C++文件使用小写+下划线命名 | 命名 | 中 | [适用: All] |
| STYLE-1.2 | 函数命名使用大驼峰风格 | 命名 | 中 | [适用: All] |
| STYLE-1.3 | 类型命名采用大驼峰风格 | 命名 | 中 | [适用: All] |
| STYLE-1.4 | 变量命名采用小驼峰风格 | 命名 | 中 | [适用: All] |
| STYLE-1.5 | 宏、枚举值采用全大写下划线连接 | 命名 | 中 | [适用: All] |
| STYLE-1.6 | 命名含义明确，禁止模糊缩写，风格统一 | 命名 | 中 | [适用: All] |
| STYLE-1.7 | 算子 Kernel 符号遵循固定命名约定 | 命名 | 中 | [适用: All] |
| STYLE-2.1 | if 语句必须使用大括号 | 格式 | 中 | [适用: All] |
| STYLE-2.2 | for/while 循环必须使用大括号 | 格式 | 中 | [适用: All] |
| STYLE-2.3 | 多个变量定义不允许写在一行 | 格式 | 低 | [适用: All] |
| STYLE-2.4 | 合理安排空行，保持代码紧凑 | 格式 | 低 | [适用: All] |
| STYLE-3.1 | 注释使用 `//`，与代码间有空格 | 注释 | 低 | [适用: All] |
| STYLE-3.2 | 禁止使用 TODO/TBD/FIXME 及开发阶段注释 | 注释 | 中 | [适用: All] |
| STYLE-3.3 | 不要写空有格式的函数头注释 | 注释 | 低 | [适用: All] |
| STYLE-3.4 | 不用的代码直接删除，不要注释掉 | 注释 | 低 | [适用: All] |
| STYLE-3.5 | 注释语言与风格统一 | 注释 | 中 | [适用: All] |

---

## STYLE-1.1: C++文件使用小写+下划线命名

**严重级别**：中

### 问题描述

C++ 文件使用小写+下划线的方式命名，以 .cpp 结尾，头文件以 .h 结尾。

业界还有其它后缀表示方法（头文件 .hh/.hpp/.hxx，cpp 文件 .cc/.cxx/.c）。如果当前项目组使用了某种特定的后缀，可以继续使用，但必须保持风格统一；本规范默认使用 .h 和 .cpp。

本条只检查文件命名；代码内符号命名归 STYLE-1.2~1.7，Kernel 文件的 `_apt` 后缀禁令归 KERNEL-8。

### 错误示例

```
// ❌ 文件名大驼峰 / 混合大小写
AddOperator.cpp
KernelAdd.h
```

### 正确示例

```
// ✅ 小写 + 下划线
add_operator.cpp
kernel_add.h
```

### 注意事项

1. 本规范以 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) 为基础，参考 MindSpore 社区、华为通用编码规范，并结合业界共识整理而成
2. 对规则有异议可提 issue 并说明理由，经 CANN 运营团队评审后可接纳修改

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条逐文件检查文件名：后缀是否为 .h/.cpp（或项目统一后缀）、名字是否小写+下划线、同仓风格是否统一。

---

## STYLE-1.2: 函数命名使用大驼峰风格

**严重级别**：中

### 问题描述

函数（包括全局函数、作用域函数、成员函数）命名统一使用大驼峰风格，一般采用动词或动宾结构。

命名风格总表（驼峰规则全貌，类型/变量/常量细分见 STYLE-1.3~1.5）：

| 类型 | 命名风格 |
| ---------------------------------------- | --------- |
| 类类型，结构体类型，枚举类型，联合体类型等类型定义，作用域名称 | 大驼峰 |
| 函数(包括全局函数，作用域函数，成员函数) | 大驼峰 |
| 全局变量(包括全局和命名空间域下的变量，类静态变量)，局部变量，函数参数，类、结构体和联合体中的成员变量 | 小驼峰 |
| 宏，常量(const)，枚举值，goto 标签 | 全大写，下划线分割 |

表中**常量**是指全局作用域、namespace 域、类的静态成员域下，以 const 或 constexpr 修饰的基本数据类型、枚举、字符串类型的变量，不包括数组和其他类型变量；表中**变量**是指除常量定义以外的其他变量，均使用小驼峰风格。

### 错误示例

```cpp
// ❌ 函数名小写下划线 / 全小写，未用大驼峰
void add_element(const Element &element);
void deleteuser();
```

### 正确示例

```cpp
// ✅ 函数名大驼峰，动宾结构
class List {
public:
	void AddElement(const Element& element);
	Element GetElement(const unsigned int index) const;
	bool IsEmpty() const;
};

namespace Utils {
    void DeleteUser();
}
```

### 注意事项

大驼峰风格同时用于类名与函数名，靠动宾结构（函数）与名词（类）区分语义，不算冲突。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条 Grep 函数定义/声明处（`^\s*\w[\w:<>,\s*&]*\s+\w+\s*\(` 与类内方法），检查函数名是否大驼峰。

---

## STYLE-1.3: 类型命名采用大驼峰风格

**严重级别**：中

### 问题描述

所有类型命名——类、结构体、联合体、类型定义（typedef/using 别名）、枚举——使用相同的大驼峰约定；命名空间的命名建议使用大驼峰。

### 错误示例

```cpp
// ❌ 类型名小驼峰 / 全小写
class url_table { ...
struct urlTableProperties { ...
enum urlTableErrors { ...
```

### 正确示例

```cpp
// ✅ 类型名大驼峰
// classes, structs and unions
class UrlTable { ...
struct UrlTableProperties { ...
union Packet { ...
// typedefs
typedef std::map<std::string, UrlTableProperties*> PropertiesMap;
// enums
enum UrlTableErrors { ...

// namespace 命名建议使用大驼峰
namespace FileUtils {
}
```

### 注意事项

类型定义别名（typedef/using）同样使用大驼峰，与被别名类型的命名风格保持一致。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条 Grep `^(class|struct|union|enum)\s+\w+` 与 `typedef|using\s+\w+\s*=`，检查类型名是否大驼峰。

---

## STYLE-1.4: 变量命名采用小驼峰风格

**严重级别**：中

### 问题描述

通用变量命名采用小驼峰，包括全局变量、函数形参、局部变量、成员变量。

前缀/后缀约定：
- 全局变量应增加 `g_` 前缀，用于视觉突出，促使开发人员小心使用
- 类的成员变量命名以小驼峰加后下划线组成（如 `fileName_`）
- 全局静态变量命名与全局变量相同；函数内的静态变量与普通局部变量相同；类的静态成员变量和普通成员变量相同

### 错误示例

```cpp
std::string tablename;   // ❌ 多单词全小写，无大小写分隔，禁止此风格
std::string TABLE_NAME;  // ❌ 变量用全大写下划线（常量风格）
```

### 正确示例

```cpp
std::string tableName;  // ✅ 小驼峰
std::string path;       // ✅ 只有一个单词时，小驼峰为全小写

int g_activeConnectCount;   // ✅ 全局变量加 g_ 前缀

void Func()
{
    static int packetCount = 0;   // ✅ 函数内静态变量与普通局部变量相同
    ...
}

class Foo {
private:
    std::string fileName_;   // ✅ 成员变量后下划线，类似 K&R 风格
};
```

### 注意事项

全局变量是应当尽量少使用的，使用时应特别注意，加前缀促使开发人员对这些变量的使用更加小心。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条检查变量声明处命名：局部/形参小驼峰、全局 `g_` 前缀、类成员后下划线。

---

## STYLE-1.5: 宏、枚举值采用全大写下划线连接

**严重级别**：中

### 问题描述

宏、枚举值、goto 标签以及全局作用域内（含命名空间内、类静态成员）const 常量，采用全大写、下划线连接的格式。函数局部 const 常量和类的普通 const 成员变量使用小驼峰命名风格。枚举类型名用大驼峰，其取值用全大写下划线相连。

### 错误示例

```cpp
#define maxSize(a, b) ...          // ❌ 宏用小驼峰
enum TintColor { red, darkRed };   // ❌ 枚举值用小驼峰
```

### 正确示例

```cpp
#define MAX(a, b)   (((a) < (b)) ? (b) : (a)) // 仅对宏命名举例，并不推荐用宏实现此类功能

enum TintColor {    // 枚举类型名用大驼峰，取值全大写下划线相连
    RED,
    DARK_RED,
    GREEN,
    LIGHT_GREEN
};

int Func(...)
{
    const unsigned int bufferSize = 100;    // ✅ 函数局部常量用小驼峰
    char *p = new char[bufferSize];
    ...
}

namespace Utils {
	const unsigned int DEFAULT_FILE_SIZE_KB = 200;        // ✅ 全局常量全大写下划线
}
```

### 注意事项

本条与 STYLE-1.2 总表的「宏，常量(const)，枚举值，goto 标签 → 全大写下划线分割」行一致；局部 const 归入小驼峰。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条 Grep `#define\s+\w+`、枚举取值列表与全局 const 声明，核对大小写风格。

---

## STYLE-1.6: 命名含义明确，禁止模糊缩写，风格统一

**严重级别**：中

### 问题描述

命名风格正确（驼峰、大小写）只是底线，**命名必须能让读者不看上下文就理解其业务含义**。以下情形判 FAIL：

- **含义模糊的自造缩写**：如 `aI`、`aO`、`aITail`、`tmp1`、`flag2`、`res`——读者无法判断 `aI` 是 input、inner 还是 index
- **同一概念在 Host 侧与 Kernel 侧命名不一致**：如 tiling 结构体成员叫 `aI`，Kernel 侧使用处局部变量却叫 `a_i_seg`，两种风格（小驼峰 vs 下划线）混用，增加对照成本
- **用单个字母命名非循环变量**：单字母命名仅允许用于循环计数（`i`、`j`）及数学约定俗成场景（如坐标 `x`、`y`，张量输入用 `x`/`y`/`z` 时保持与算子定义一致）

本条检查命名的语义可读性与跨侧一致性；纯大小写风格归 STYLE-1.2~1.5。

### 错误示例

取自真实检视案例：tiling 切分结构，`aI` 实为「按轴切分的单次内层块长」，`aO` 为「外层循环次数」。

```cpp
// ❌ 含义模糊：aI / aO / aITail 无法自解释；Kernel 侧又用 a_i_seg（风格不一致）
struct SplitResult {
    int64_t axis;      // 切分轴
    int64_t aI;        // ？inner？input？
    int64_t aO;        // ？outer？output？
    int64_t aITail;    // ？
};

// Kernel 侧（同文件族）
int64_t a_i_seg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO,
                                  td_->split.aI, td_->split.aITail);
```

### 正确示例

```cpp
// ✅ 含义自解释，Host/Kernel 两侧风格统一（小驼峰）
struct SplitResult {
    int64_t splitAxis;       // 切分轴
    int64_t innerBlockSize;  // 单次内层处理的块长
    int64_t outerLoopCount;  // 外层循环次数
    int64_t innerBlockTail;  // 最后一次内层处理的尾块长
};

// Kernel 侧命名与 tiling 成员对应，风格一致
int64_t innerSegLen = GetUBSplitRange(flat % td_->split.outerLoopCount,
                                      td_->split.outerLoopCount,
                                      td_->split.innerBlockSize,
                                      td_->split.innerBlockTail);
```

### 注意事项

允许使用的缩写：领域内约定俗成的缩写（如 `gm`、`ub`、`tiling`、`num`、`idx`、`buf`、`params`）；其余自定义缩写应改为完整单词，或至少在声明处以注释说明含义。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

遇到 2~4 字母的短命名（尤其成组出现的 `xI`/`xO`/`xT` 型缩写），检查声明处是否有注释解释；无注释且无法从名字推断含义的，标记 FAIL 并在评论中给出建议命名。同一数据在 tiling 结构体、Kernel 成员、局部变量中的命名应保持同一单词与风格。

---

## STYLE-1.7: 算子 Kernel 符号遵循固定命名约定

**严重级别**：中

### 问题描述

AscendC 算子 Kernel 侧的类、入口函数、命名空间、TilingData 结构存在**社区固定约定**（源自官方样例与 CANN 算子仓既有实践），偏离约定或同一算子内混用两套风格的判 FAIL：

| 符号 | 固定约定 | 正例 |
|------|---------|------|
| Kernel 类名 | 大驼峰：`Kernel{OpName}`（官方样例风格）或与算子名一致的 `{OpName}`；同一算子内只用一种 | `KernelAdd`、`KernelSoftShrinkGrad`；`SoftShrinkGrad`、`AddExample` |
| 流程方法名 | 固定为 `Init` / `Process`；内部搬运/计算分离为 `CopyIn` / `Compute` / `CopyOut`（动宾结构、大驼峰） | `void Init(...)`、`void Process()`、`void CopyIn(...)` |
| Kernel 入口函数 | `extern "C" __global__ __aicore__ void {op_name}`，`{op_name}` 为全小写下划线，与算子注册名一致 | `extern "C" __global__ __aicore__ void soft_shrink_grad(GM_ADDR ...)` |
| 入口函数参数 | 与算子定义（spec）中的输入/输出名对齐，可用 snake_case（STYLE-1.4 的**例外**）；`workspace`、`tiling` 固定在最后 | `void add(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling)` |
| Kernel 命名空间 | `Ns{OpName}`（CANN 算子仓约定），大驼峰 | `namespace NsSoftShrinkGrad { ... }` |
| TilingData 结构 | `{OpName}TilingData`，大驼峰 | `SoftShrinkGradTilingData`、`AddExampleTilingData` |
| 张量/队列成员 | 标明存储位置与数据身份，同一算子内不混用两套前缀风格：官方样例风格 `{name}Gm` / `{name}Local` / `{name}Queue`，或仓内风格 `inputGM{name}` / `outputGM{name}` / `{name}Queue` | `xGm` + `xLocal` + `xQueue`；`inputGMX` + `xLocal` + `inputQueueX` |

### 错误示例

```cpp
// ❌ 类名小驼峰 addKernel（应为大驼峰）；入口与注册名 addExample 不一致
class addKernel { ... };
extern "C" __global__ __aicore__ void AddExample(GM_ADDR x, ...) { ... }

// ❌ 同一算子内混用两套张量命名风格：xGm 与 inputGMY 并存
GlobalTensor<T> xGm;
GlobalTensor<T> inputGMY;
```

### 正确示例

完整骨架，命名全部符合约定：

```cpp
namespace NsAddExample {
using namespace AscendC;

template <typename T>
class AddExample {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const AddExampleTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;    // Queue 后缀标明身份
    GlobalTensor<T> inputGMX;                            // GM 标明位于 Global Memory
    int64_t blockLength_ = 0;                            // 成员变量后下划线（STYLE-1.4）
};
} // namespace NsAddExample

// 入口：op_name 全小写下划线 = 算子注册名
extern "C" __global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    NsAddExample::AddExample<DTYPE_X> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
```

### 注意事项

文件命名（`{op_name}.cpp/.h`、禁止 `_apt` 后缀）由 KERNEL-8 与 STYLE-1.1 覆盖，本条只管代码内符号。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

类名/入口/命名空间/TilingData 四处对照上表，重点检查 PR 中新增算子是否与仓内既有算子的 `Ns{OpName}` 约定一致；张量成员只需确认「同一算子内不混用」，`xGm` 系与 `inputGM{x}` 系均认可。

---

## STYLE-2.1: if 语句必须使用大括号

**严重级别**：中

### 问题描述

if 语句都必须使用大括号，即便只有一条语句。理由：

- 代码逻辑直观，易读
- 在已有条件语句代码上增加新代码时不容易出错
- 在 if 语句中使用函数式宏时，有大括号保护不易出错（如果宏定义时遗漏了大括号）

### 错误示例

```cpp
// ❌ 单条语句省略大括号
if (cond)
    single line code;
```

### 正确示例

```cpp
// ✅ 即使 if 分支代码只有一行，也必须使用大括号
if (cond) {
  single line code;
}
```

### 注意事项

行宽、缩进、指针/引用对齐、大括号摆放、表达式换行等纯格式规则，以及文件头版权声明，已由仓内 pre-commit 钩子（clang-format、OAT）在提交阶段自动拦截，本组条款聚焦 formatter 无法判定或修复的风格项。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-2.2: for/while 循环必须使用大括号

**严重级别**：中

### 问题描述

和条件表达式类似，for/while 等循环语句必须加上大括号，即便循环体是空的，或循环语句只有一条。

### 错误示例

```cpp
// ❌ 循环体单条语句省略大括号 / 空语句
for (int i = 0; i < someRange; i++)
    DoSomething();
while (condition);
```

### 正确示例

```cpp
// ✅ 循环体使用大括号
for (int i = 0; i < someRange; i++) {
    DoSomething();
}

// ✅ 循环体是空，也使用大括号
while (condition) { }
```

### 注意事项

空循环体必须写显式空大括号，禁止用分号充当循环体。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-2.3: 多个变量定义不允许写在一行

**严重级别**：低

### 问题描述

多个变量定义和赋值语句不允许写在一行。每行只有一个变量初始化的语句，更容易阅读和理解。

### 错误示例

```cpp
int32_t base, offset, tail;   // ❌ 多个变量定义写在一行
int x = 1; int y = 2;         // ❌ 多条语句写在一行
```

### 正确示例

```cpp
int32_t base = 0;     // ✅ 每行一个变量
int32_t offset = 0;
int32_t tail = 0;
```

### 注意事项

本条为低严重级别建议项，检视时作提醒。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-2.4: 合理安排空行，保持代码紧凑

**严重级别**：低

### 问题描述

减少不必要的空行，可以显示更多的代码，方便代码阅读。规则：

- 根据上下内容的相关程度，合理安排空行
- 函数内部、类型定义内部、宏内部、初始化表达式内部，不使用连续空行
- 不使用连续 3 个及以上空行
- 大括号内的代码块行首之前和行尾之后不要加空行，但 namespace 的大括号内不作要求

### 错误示例

```cpp
int Foo()
{
    ...
}



int Bar()  // ❌ 连续 3 个空行（最多使用连续 2 个空行）
{
    ...
}


if (...) {
        // ❌ 大括号内的代码块行首不要加入空行
    ...
        // ❌ 大括号内的代码块行尾不要加入空行
}

int Foo(...)
{
        // ❌ 函数体内行首不要加空行
    ...
}
```

### 正确示例

```cpp
int Foo()
{
    ...
}

int Bar()
{
    ...
}

if (...) {
    ...
}
```

### 注意事项

本条为低严重级别建议项；namespace 大括号内的空行不作要求。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-3.1: 注释使用 `//`，与代码间有空格

**严重级别**：低

### 问题描述

代码注释置于对应代码的上方或右边；注释符 `//` 与注释内容之间要有 1 个空格，右置注释与前面代码至少 1 空格；使用 `//` 而不是 `/**/`。

注释总原则：尽量通过清晰的架构逻辑、好的符号命名来提高代码可读性，需要的时候才辅以注释说明；注释从读者的角度出发，**按需注释**；内容简洁、明了、无二义性，信息全面且不冗余；同一类型的注释保持统一的风格。

### 错误示例

```cpp
int foo;/*注释符与内容无空格，且使用 /**/ */
int bar;   //❌ 右置注释与代码间无空格
```

### 正确示例

```cpp
// this is multi-
// line comment
int foo; // this single-line comment
```

### 注意事项

连续多行注释统一使用 `//` 逐行表达，不混用 `/* */`。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-3.2: 禁止使用 TODO/TBD/FIXME 及开发阶段注释

**严重级别**：中

### 问题描述

交付代码中禁止残留开发过程的临时标记，包括：

- **TODO/TBD/FIXME** 等未完成事项标记（应提 issue 跟踪，不在代码中留标记）
- **AI 开发阶段注释**：开发过程中用于划分步骤、阶段的注释，如 `// Step 1`、`// 阶段一`、`// Phase 2`、`// 第一步：初始化`、`// === 计算核心 ===` 等过程性标记。这些注释在交付代码中应移除，或转为正式的函数/变量命名自解释

### 错误示例

```cpp
// ❌ 残留 AI 开发阶段注释
__aicore__ void Compute() {
    // Step 1: 数据搬运
    DataCopy(xLocal, xGm);
    // Phase 2: 计算
    Add(zLocal, xLocal, yLocal);
    // TODO: 后续优化流水线
    DataCopy(zGm, zLocal);
}
```

### 正确示例

```cpp
// ✅ 移除阶段注释，通过函数命名自解释
__aicore__ void Compute() {
    CopyIn();
    ComputeAdd();
    CopyOut();
}
```

### 注意事项

未完成事项建议提 issue 跟踪，不在代码中留 TODO 标记。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

本条 Grep `TODO|TBD|FIXME` 与 `Step \d|Phase \d|阶段|第.步|====` 型阶段标记（排除注释外字符串字面量）。

---

## STYLE-3.3: 不要写空有格式的函数头注释

**严重级别**：低

### 问题描述

不要写空有格式的函数头注释。并不是所有的函数都需要函数头注释，函数尽量通过函数名自注释，按需写函数头注释；函数原型无法表达的、却又希望读者知道的信息，才需要加函数头注释辅助说明。函数头注释内容可选，但不限于：功能说明、返回值、性能约束、用法、内存约定、算法实现、可重入的要求等。

### 错误示例

```cpp
// ❌ 空有格式没内容：参数、返回值栏空置；函数名信息冗余；关键的 buf 由谁释放没说清
/*
 * 函数名：WriteString
 * 功能：写入字符串
 * 参数：
 * 返回值：
 */
int WriteString(const char *buf, int len);
```

### 正确示例

```cpp
// ✅ 只写函数签名表达不了的关键信息
/*
 * 返回实际写入的字节数，-1表示写入失败
 * 注意，内存 buf 由调用者负责释放
 */
int WriteString(const char *buf, int len);
```

### 注意事项

本条为低严重级别建议项，检视时作提醒。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-3.4: 不用的代码直接删除，不要注释掉

**严重级别**：低

### 问题描述

不用的代码段直接删除，不要注释掉。被注释掉的代码无法被正常维护；当企图恢复使用这段代码时，极有可能引入易被忽略的缺陷。正确的做法是，不需要的代码直接删除掉；若再需要时，考虑移植或重写这段代码。

### 错误示例

```cpp
// ❌ 旧实现注释保留，无法维护且易误导
// int32_t ret = OldCalc(input, output);
int32_t ret = NewCalc(input, output);
```

### 正确示例

```cpp
// ✅ 直接删除旧实现，需要时从 git 历史找回
int32_t ret = NewCalc(input, output);
```

### 注意事项

本条为低严重级别建议项；大段注释掉的代码比单行更应删除。

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

---

## STYLE-3.5: 注释语言与风格统一

**严重级别**：中

### 问题描述

同一文件内注释语言与注释风格必须统一。中英文注释混杂、句式风格不一致会显著降低可读性。以下情形判 FAIL：

- **语言不统一**：同一文件内英文注释与中文注释混杂。CANN 开源仓代码注释推荐使用英文（面向国际社区）；若沿用中文，须整个算子（kernel + host 侧文件）统一，禁止同文件或同 PR 内中英夹杂
- **句式不统一**：英文注释句首大小写混乱（如一处 `// get block idx`、另一处 `// Get block idx`）；英文注释应统一句首大写、完整句子以句号结尾，短语式注释可省略句号但同一文件内保持一致
- **标点不统一**：英文注释使用半角标点，中文注释使用全角标点，禁止混用（如 `// 获取block idx。` 中英文标点夹杂）
- **对齐方式不统一**：右置注释与代码间至少 1 空格（与 STYLE-3.1 一致），连续多行注释统一使用行注释 `//` 逐行开头，禁止一部分用 `//` 一部分用 `/* */` 表达同一说明块

### 错误示例

```cpp
// ❌ 同一文件内中英文混杂、句式/标点不统一
int64_t blockTail = tilingData->dim0 - (blockNum - 1) * blockFormer;
// get block idx
int64_t blockIdx = AscendC::GetBlockIdx();
// 判断是否是最后一个block。
bool isLastBlock = (blockIdx == tilingData->blockNum - 1);
/* compute tail length */
int64_t tailLen = blockTail;
```

### 正确示例

```cpp
// ✅ 全文件统一英文注释，句首大写，句式一致
int64_t blockTail = tilingData->dim0 - (blockNum - 1) * blockFormer;
// Get the index of the current block.
int64_t blockIdx = AscendC::GetBlockIdx();
// Check whether the current block is the last one.
bool isLastBlock = (blockIdx == tilingData->blockNum - 1);
// Compute the tail length.
int64_t tailLen = blockTail;
```

### 注意事项

1. 文件级说明注释（说明入口、模块职责）可采用 Doxygen 风格 `/*! \file ... \brief ... */`，置于版权头之后；同一仓内要么都写要么都不写，保持统一
2. 注释中的算子名、API 名、变量名等代码符号保持原样（不翻译、不改写），便于 grep 检索
3. 中文注释里中英文/数字间的空格规范遵循仓内文档风格约定，不要自行发明规则

### 专属检视方法

确定性判定，不走假设检验流程（不建立 H0/H1、不收集证据分值、不计算自信值）：逐条对照代码，违反即 FAIL，无违反即 PASS。结果 yaml 使用 `category: style`，severity 取本条款快速索引值，禁止携带 confidence/evidence 字段，不加 `[STYLE]` 文本前缀。

对 PR 中每个新增/修改文件抽样统计注释语言比例：若同一文件中英注释均超过约 20% 且无明显分界（如整块中文区），判 FAIL；检查同一文件内句首大小写与标点是否一致。
