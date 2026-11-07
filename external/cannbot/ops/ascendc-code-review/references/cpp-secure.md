# CANN C++ 安全编码规范

<适用>
语言: C++
侧别: All
领域: true
触发: 必须触发
默认启用: true

适用场景: Host 侧和 Kernel 侧（Device 侧）的 C++ 安全编码规范
不适用场景: Python 代码（见 python-secure.md）、编译链接安全（见 compile-secure.md）
介绍: CANN C++ 安全编码规范，15 条条款覆盖 7 个安全类别
类别(All): 内存与指针安全(sizeof误用), 输入验证(内存操作长度校验), 类与对象安全(非trivially copyable对象位操作), 标准库安全(敏感信息清零)
类别(Host): 内存与指针安全(资源句柄释放后赋值/字符串空间), 资源管理(申请成功判断/new-delete配对/分配错误处理), 安全函数使用(安全函数库/destMax参数/返回值检查), 标准库安全(空指针string/c_str指针保存), LOG规范(格式化占位符)
> **说明**：安全编码红线规范，所有代码必须 100% 遵守。条款标注适用范围：`[适用: All]` / `[适用: Host]`
</适用>


## 快速索引

以下仅列出 All 条款；Host 专属条款直接见正文。条款按当前安全类别顺序连续编号，各类别内从 1 开始编号。

### 两者都适用 `[适用: All]`（4 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| SEC-1.2 | 禁止通过对指针变量进行sizeof操作来获取数组大小 | 内存安全 | 中 |
| SEC-2.1 | 内存操作长度校验 | 输入验证 | 高 |
| SEC-5.1 | 禁止通过整对象字节操作绕过非 trivially copyable 对象语义 | 类与对象 | 中 |
| SEC-6.3 | 敏感信息使用后清零 | 标准库 | 高 |

---

### 1. 内存与指针安全

#### SEC-1.1 指向资源句柄或描述符的变量，在资源释放后立即赋予新值 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无动态资源管理，Buffer 由 `InitBuffer` 静态分配，无需释放后置空。

**【描述】**
指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。

以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。

**【正确代码示例】**

```cpp
int foo(void)
{
    SomeStruct *msg = NULL;
    ... // 初始化msg->type，分配 msg->body 的内存空间

    if (msg->type == MESSAGE_A) {
        ...
        free(msg->body);
        msg->body = NULL;
    }

    ...
EXIT:
    ...
    free(msg->body);
    return ret;
}
```

#### SEC-1.2 禁止通过对指针变量进行sizeof操作来获取数组大小 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `LocalTensor<T>` 通过 API（如 `GetSize()`）获取大小，不能用 sizeof。

**【描述】**
将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。

**【错误代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

**【正确代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // 使用申请的缓冲区大小
```

#### SEC-1.3 确保字符串存储有足够的空间容纳字符数据和null结束符 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无 C 风格字符串处理。但 GM 数据搬运时需确保目标 Buffer 有足够空间。

**【描述】**
将数据复制到不足以容纳数据的缓冲区，会导致缓冲区溢出。

---

### 2. 输入验证

#### SEC-2.1 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `DataCopy` 的搬运长度需校验，确保不超过 UB 容量和 GM 数据范围。

**【描述】**
将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量足够大以容纳要复制的数据。

---

### 3. 资源管理

#### SEC-3.1 资源申请后必须判断是否成功 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无动态资源申请（malloc/new），Buffer 由 `InitBuffer` 静态分配，编译期确定。

**【描述】**
内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。

**【正确代码示例】**

```cpp
struct tm *make_tm(int year, int mon, int day, int hour, int min, int sec)
{
    struct tm *tmb = (struct tm *)malloc(sizeof(*tmb));
    if (tmb == NULL) {
        ... // 错误处理
    }
    tmb->year = year;
    ...
    return tmb;
}
```

#### SEC-3.2 new和delete配对使用，new[]和delete[]配对使用 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 禁止 new/delete。

##### 专属检视方法

除 Host 侧（`op_host/`）外，**测试代码**（`tests/`、`ut/`、`st/` 目录下的 `.cpp/.h`）同样适用 new/delete 配对检查——测试代码中 `new` 创建的对象若无对应 `delete`，会导致内存泄漏。

检视指引：
- Grep 测试文件中的 `new ` 创建对象语句 → 追踪同一作用域或对象生命周期内是否有对应 `delete`
- 重点关注：测试函数内的局部 `new`（函数结束前未释放）、SetUp/TearDown 中的成员对象创建与销毁配对、`new[]` 是否用 `delete[]`（而非 `delete`）释放
- 豁免：智能指针（`std::unique_ptr`/`std::shared_ptr`/`std::make_unique`/`std::make_shared`）管理的对象、RAII 模式封装的对象、测试框架自动管理的 fixture 成员

#### SEC-3.3 使用恰当的方式处理new操作符的内存分配错误 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 禁止 new。

---

### 4. 安全函数使用

#### SEC-4.1 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无 memcpy_s/memset_s，使用 Ascend C API（如 `Duplicate`、`DataCopyPad`）。

| 函数类别 | 危险函数 | 安全替代函数 |
|---------|---------|------------|
| 内存拷贝 | memcpy或bcopy | memcpy_s |
| 内存拷贝 | memmove | memmove_s |
| 字符串拷贝 | strcpy | strcpy_s |
| 字符串串接 | strcat | strcat_s |
| 格式化输出 | sprintf | sprintf_s |
| 格式化输出 | snprintf | snprintf_s |
| 格式化输入 | scanf | scanf_s |
| 内存初始化 | memset | memset_s |

#### SEC-4.2 正确设置安全函数中的destMax参数 `[适用: Host]`

> **Kernel 侧不适用**：本条针对 Host 使用的 Secure C 内存及字符串接口。

本条所称安全函数指项目实际引入的 Secure C 边界检查函数，不是所有名称带 `_s` 的接口都具有相同签名。常见的带 `destMax` 参数接口如下；完整声明及可选接口见 [OpenHarmony bounds_checking_function 的 securec.h](https://github.com/openharmony/third_party_bounds_checking_function/blob/master/include/securec.h)，使用时以项目实际版本为准。

| 类别 | 常见接口 | destMax 含义 |
|------|----------|-------------|
| 内存操作 | `memcpy_s`、`memmove_s`、`memset_s` | 从目标地址起可写区域的字节容量 |
| 窄字符串复制和追加 | `strcpy_s`、`strncpy_s`、`strcat_s`、`strncat_s` | 目标字符数组容量，包含字符串结束符的空间 |
| 窄字符串格式化输出 | `sprintf_s`、`vsprintf_s`、`snprintf_s`、`vsnprintf_s` | 目标字符数组容量，包含字符串结束符的空间 |

- `destMax` 不得超过目标地址起实际可写的容量；不能将源数据长度、指针的 `sizeof` 或期望写入长度直接当成目标容量。
- 目标为缓冲区内部地址时，应使用该位置起的剩余容量；计算前先确认偏移有效，防止容量减法回绕。
- `count` 等参数表示本次操作长度，不能替代 `destMax`。字符串追加还需计入目标已有内容及结束符；格式化输出需区分容量和输出长度限制。
- 宽字符版本及其他扩展接口按其声明核对容量单位，不能直接把字节数当成字符个数。`scanf_s` 一类接口没有统一的 `destMax` 参数，应按转换项要求提供各目标缓冲区大小，不能套用上述签名。
- 容量合法仍不代表操作一定成功，返回值按第 4.3 条处理。

#### SEC-4.3 必须检查安全函数返回值，并进行正确的处理 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无安全函数。

必须按具体接口的返回值约定处理失败，不能继续执行依赖成功结果的操作。返回 `errno_t` 的内存和字符串操作通常以 `EOK` 表示成功；格式化输出、输入接口返回字符数或成功转换项数等，不能统一用 `!= EOK` 判断。

```cpp
{
    ...
    err = memcpy_s(destBuff, destMax, src, srcLen);
    if (err != EOK) {
        MS_LOG("memcpy_s failed, err = %d\n", err);
        return FALSE;
    }
    ...
}
```

---

### 5. 类与对象安全

#### SEC-5.1 禁止通过整对象字节复制或填充绕过非 trivially copyable 对象的构造、赋值语义 `[适用: All]`

对非 trivially copyable 类型，禁止将 memcpy/memmove（含 _s 版本）、memset 或手工逐字节覆盖用于整对象复制、搬移或初始化，替代该类型的构造函数、赋值操作或公开初始化接口。典型对象包括 `std::string`、`std::vector`、拥有资源的智能指针以及含非平凡拷贝或析构操作的类；此类覆盖可能破坏所有权或内部状态。

不能仅凭类型是 struct、模板类或用于 Kernel 就认定其可按字节复制。应核实具体类型定义，或在编译环境支持时使用 std::is_trivially_copyable<T>::value 判断。AscendC 数据搬运、Duplicate 等 API 应按其操作的实际数据元素及接口契约判断，不能等同于对 C++ 包装对象执行整对象字节复制。

对 trivially copyable 类型，从有效的同类型对象复制字节，或保存并原样恢复同一对象的字节表示，还需满足标准对对象生命周期、存储大小和 potentially-overlapping subobject 的限制；memcpy 另有源目标不重叠要求。该性质不表示任意字节模式都是合法值，也不保证 memset(..., 0, ...) 等价于值初始化。对象初始化优先使用类型提供的初始化方式。

本规则不禁止在语言规则允许范围内通过 char、unsigned char 或 std::byte 观察对象表示，也不禁止对对象内部独立的字节缓冲区进行合法操作。不要仅因出现 reinterpret_cast 或字节访问就判定违规。

来源：

- [C++ 标准草案：Trivially copyable types](https://eel.is/c++draft/basic.types.trivial)：同类型字节复制与原样恢复保证及对象边界。
- [C++ 标准草案：Class properties](https://eel.is/c++draft/class.prop)：trivially copyable 的定义；standard-layout 与 trivially copyable 是不同性质。
- [C++ 标准草案：Type accessibility](https://eel.is/c++draft/basic.lval#11)：允许 char、unsigned char、std::byte 类型访问对象表示。


---

### 6. 标准库安全

#### SEC-6.1 禁止从空指针创建std::string `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### SEC-6.2 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针 `[适用: Host]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### SEC-6.3 内存中的敏感信息使用完毕后立即清0 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 UB 数据可通过 `Duplicate` 清零，GM 数据需在 Host 侧处理。

口令、密钥等敏感信息使用完毕后立即清零，避免被攻击者获取。

### 7. LOG 规范

> **适用范围**：本节检查 Host 侧的格式化 LOG 调用。

Host 侧使用 `OP_LOGE` / `OP_LOGD` / `OP_LOGW` 等格式化 LOG 宏，参数匹配要求见第 7.1 条。

LOG 宏签名（业务代码标准调用形式）：

```cpp
OP_LOGE(context->GetNodeName(), "format string %s %ld", arg1, arg2);
OP_LOGD(context->GetNodeName(), "format string %lu", arg1);
```

---

#### SEC-7.1 LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） `[适用: Host]`

**【问题说明】**

LOG 宏的格式化占位符与实际参数之间必须满足三个维度的一致性：

1. **数量一致**：参数少于占位符时，从栈上读取垃圾值，若被解释为 `%s` 将触发段错误
2. **类型匹配**：类型大小不匹配时（如 `uint64_t` 误用 `%d`），按说明符宽度截断，后续参数全部错位
3. **顺序对应**：参数顺序与格式符位置不对应时（如 `%s` 位置收到整数），整数被当作地址读字符串 → **段错误(SIGSEGV)**

> **⚠️ 禁止仅凭 grep 单行分析 LOG 调用。** 算子仓中大量 LOG 语句跨越多行（2-35 行），且常嵌套在 `OP_CHECK_IF` 等外层宏内。grep 命中后**必须 Read 前后至少 10 行**获取完整的格式字符串和全部参数，否则分析的是截断的不完整调用，结论无效。多行字符串拼接（`"a" "b"`）需先合并再解析。

**错误与正确示例**

```cpp
// ❌ 数量不一致：2 个占位符，1 个参数
OP_LOGD(ctx, "M: %ld, K: %ld", m);           // 缺少 k
// ✅
OP_LOGD(ctx, "M: %ld, K: %ld", m, k);

// ❌ 类型不匹配：uint64_t 用了 %d
OP_LOGE(ctx, "n = %d, ubSize = %d\n", n, ubSize); // n/ubSize 均为 uint64_t
// ✅
OP_LOGE(ctx, "n = %llu, ubSize = %llu\n", n, ubSize);

// ❌ 顺序错位：数量=5 格式符=5，但位置1和3的参数放反了
//   格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
//   参数:   inputName.c_str()(1) ... d0Size/NUM8(3) ...
//   → 位置1: %u 收到 const char*，位置3: %s 收到 uint → 段错误
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
// ✅ 参数顺序与格式符逐位对应
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

**类型与说明符速查**

| 类型 | 正确 | 常见错误 | 后果 |
|------|------|---------|------|
| `uint64_t` | `%llu` | `%u`, `%lu`, `%d` | 截断为 32 位，后续参数错位 |
| `int64_t` | `%lld` | `%d`, `%ld` | 同上 |
| `uint32_t` | `%u` | `%d` | 大值显示为负数 |
| `size_t` | `%zu` | `%d`, `%u` | 64 位系统上截断 |
| `bool` | `%d` 或 `? "true":"false"` + `%s` | `%s` 直传 | 未定义行为 |
| `void*` | `%p` | `%x` | 不可移植 |

**【检视方法】**

1. grep `OP_LOGE\|OP_LOGD\|OP_LOGW\|OP_LOGI` 找到所有 LOG 调用
2. Read 完整调用后，提取格式符序列和参数序列，逐位比对：数量是否一致 → 每个位置的参数类型是否兼容格式符
3. 高风险标记：`%s` 收到整数（段错误）、`uint64_t`/`int64_t` 配 `%d`（截断错位）
