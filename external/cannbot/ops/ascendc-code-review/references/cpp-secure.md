# CANN C++ 安全编码规范

```
<适用>
语言: C++
侧别: All
默认启用: true

领域: true
触发: 必须触发
</适用>
```


## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| SEC-1.1 | 指向资源句柄或描述符的变量，在资源释放后立即赋予新值 | 内存安全 | 高 | [适用: Host] |
| SEC-1.2 | 禁止通过对指针变量进行sizeof操作来获取数组大小 | 内存安全 | 高 | [适用: All] |
| SEC-1.3 | 确保字符串存储有足够的空间容纳字符数据和null结束符 | 内存安全 | 高 | [适用: Host] |
| SEC-2.1 | 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 | 输入验证 | 高 | [适用: All] |
| SEC-3.1 | 资源申请后必须判断是否成功 | 资源管理 | 高 | [适用: Host] |
| SEC-3.2 | new和delete配对使用，new[]和delete[]配对使用 | 资源管理 | 高 | [适用: Host] |
| SEC-3.3 | 使用恰当的方式处理new操作符的内存分配错误 | 资源管理 | 高 | [适用: Host] |
| SEC-4.1 | 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数 | 标准库 | 高 | [适用: Host] |
| SEC-4.2 | 正确设置安全函数中的destMax参数 | 标准库 | 高 | [适用: Host] |
| SEC-4.3 | 必须检查安全函数返回值，并进行正确的处理 | 标准库 | 高 | [适用: Host] |
| SEC-5.1 | 禁止通过整对象字节复制或填充绕过非 trivially copyable 对象的构造、赋值语义 | 类与对象 | 高 | [适用: All] |
| SEC-6.1 | 禁止从空指针创建std::string | 标准库 | 高 | [适用: Host] |
| SEC-6.2 | 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针 | 标准库 | 高 | [适用: Host] |
| SEC-6.3 | 内存中的敏感信息使用完毕后立即清0 | 标准库 | 高 | [适用: All] |
| SEC-7.1 | LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） | 日志规范 | 高 | [适用: Host] |

---

## SEC-1.1: 指向资源句柄或描述符的变量，在资源释放后立即赋予新值

**严重级别**：高

### 问题描述

指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。

以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。

Kernel 侧不适用：Kernel 无动态资源管理，Buffer 由 `InitBuffer` 静态分配，无需释放后置空。

### 错误示例

```cpp
// ❌ 释放后未置新值，EXIT 处重复释放
int foo(void)
{
    SomeStruct *msg = NULL;
    ... // 初始化msg->type，分配 msg->body 的内存空间

    if (msg->type == MESSAGE_A) {
        ...
        free(msg->body);  // 释放后未置新值
    }

    ...
EXIT:
    ...
    free(msg->body);  // 悬空指针重复释放
    return ret;
}
```

### 正确示例

```cpp
// ✅ 释放后立即赋予新值
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

### 注意事项

- 释放与置空必须就近成对完成；释放后到置空前不得存在任何可达的使用路径。
- 同一资源的其他别名（指针副本、句柄副本）须一并失效处理；悬空与判空要求的整体判定参见 ascendc-red-line.md 条例6。
- 文件描述符、socket、stream 等句柄类资源同理适用。

---

## SEC-1.2: 禁止通过对指针变量进行sizeof操作来获取数组大小

**严重级别**：高

### 问题描述

将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。

Kernel 侧说明：Kernel 中 `LocalTensor<T>` 通过 API（如 `GetSize()`）获取大小，不能用 sizeof。

### 错误示例

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// ❌ sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

### 正确示例

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // ✅ 使用申请的缓冲区大小
```

### 注意事项

- sizeof 只对数组类型返回全体大小；数组作为形参传递后退化为指针，sizeof 结果为指针宽度。
- 动态分配的内存大小须由申请时的长度变量或记录承载，不能来自 sizeof。

---

## SEC-1.3: 确保字符串存储有足够的空间容纳字符数据和null结束符

**严重级别**：高

### 问题描述

将数据复制到不足以容纳数据的缓冲区，会导致缓冲区溢出。

Kernel 侧不适用：Kernel 无 C 风格字符串处理。但 GM 数据搬运时需确保目标 Buffer 有足够空间。

### 错误示例

```cpp
char buf[8];
const char *src = "0123456789";  // 10 字符 + 结束符
// ❌ 目标空间不足，越界写入
(void)strcpy(buf, src);
```

### 正确示例

```cpp
char buf[16];
const char *src = "0123456789";
// ✅ 使用安全函数，容量不足时按返回值处理失败
errno_t ret = strcpy_s(buf, sizeof(buf), src);
if (ret != EOK) {
    return -1;
}
```

### 注意事项

- 计算目标容量时须计入字符串结束符的空间。
- 危险函数禁用与 destMax 容量语义分别按本文件 SEC-4.1、SEC-4.2 核对。

---

## SEC-2.1: 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性

**严重级别**：高

### 问题描述

将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量足够大以容纳要复制的数据。

Kernel 侧说明：Kernel 中 `DataCopy` 的搬运长度需校验，确保不超过 UB 容量和 GM 数据范围。

### 错误示例

```cpp
// ❌ 外部输入 len 未校验即作复制长度
const uint8_t *src = GetExternalInput(len);
(void)memcpy(dst, src, len);  // len 可能超过 dst 容量
```

### 正确示例

```cpp
// ✅ 先校验长度不超过目标容量，再复制
const uint8_t *src = GetExternalInput(len);
if (len > dstCapacity) {
    return -1;
}
(void)memcpy(dst, src, len);
```

### 注意事项

- 校验对象包括长度、偏移及两者的组合（容量减偏移），不能只校验单一数值。
- 由外部输入派生的长度（乘法结果、累加值）同样按外部输入对待。

---

## SEC-3.1: 资源申请后必须判断是否成功

**严重级别**：高

### 问题描述

内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。

Kernel 侧不适用：Kernel 无动态资源申请（malloc/new），Buffer 由 `InitBuffer` 静态分配，编译期确定。

### 错误示例

```cpp
// ❌ 申请失败未判断即使用
struct tm *tmb = (struct tm *)malloc(sizeof(*tmb));
tmb->year = year;  // tmb 可能为 NULL
```

### 正确示例

```cpp
// ✅ 判断成功后再使用
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

### 注意事项

- 判空之外还须保证生命周期、可访问范围和读写权限有效，整体策略参见 ascendc-red-line.md 条例6。
- 返回值按语义处理的要求参见 ascendc-topk.md 第 1 条。

---

## SEC-3.2: new和delete配对使用，new[]和delete[]配对使用

**严重级别**：高

### 问题描述

new 分配的单个对象必须由 delete 释放，new[] 分配的数组必须由 delete[] 释放；配对错误或遗漏释放会导致未定义行为或内存泄漏。

Kernel 侧不适用：Kernel 禁止 new/delete。

### 错误示例

```cpp
// ❌ new[] 用 delete 释放；new 的对象未释放
int32_t *arr = new int32_t[10];
Foo *obj = new Foo();
...
delete arr;   // 配对错误：未定义行为
// obj 未 delete，泄漏
```

### 正确示例

```cpp
// ✅ 严格配对
int32_t *arr = new int32_t[10];
Foo *obj = new Foo();
...
delete[] arr;  // new[] → delete[]
delete obj;    // new → delete
```

### 注意事项

- 除 Host 侧（`op_host/`）外，**测试代码**（`tests/`、`ut/`、`st/` 目录下的 `.cpp/.h`）同样适用 new/delete 配对检查——测试代码中 `new` 创建的对象若无对应 `delete`，会导致内存泄漏。
- 豁免：智能指针（`std::unique_ptr`/`std::shared_ptr`/`std::make_unique`/`std::make_shared`）管理的对象、RAII 模式封装的对象、测试框架自动管理的 fixture 成员。

### 专属检视方法

检视指引：

- Grep 测试文件中的 `new ` 创建对象语句 → 追踪同一作用域或对象生命周期内是否有对应 `delete`
- 重点关注：测试函数内的局部 `new`（函数结束前未释放）、SetUp/TearDown 中的成员对象创建与销毁配对、`new[]` 是否用 `delete[]`（而非 `delete`）释放

---

## SEC-3.3: 使用恰当的方式处理new操作符的内存分配错误

**严重级别**：高

### 问题描述

new 分配失败默认抛出 `std::bad_alloc`；若按项目约定使用 nothrow 形式或自定义失败处理，必须按所选形式检查并处理失败，不得假设分配必然成功。

Kernel 侧不适用：Kernel 禁止 new。

### 错误示例

```cpp
// ❌ 默认 new 失败抛异常，调用方未处理
Foo *obj = new Foo();  // badalloc 未捕获/未处理，流程异常终止
```

### 正确示例

```cpp
// ✅ 按所选形式处理失败
Foo *obj = new (std::nothrow) Foo();
if (obj == nullptr) {
    return -1;  // 失败路径明确处理
}
```

### 注意事项

- 默认 new 与 nothrow new 的失败语义不同（抛异常 vs 返回空指针），须按实际使用的形式检查。
- 未捕获 bad_alloc 的路径至少须保证异常安全：已申请资源可被正确释放。

---

## SEC-4.1: 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数

**严重级别**：高

### 问题描述

Host 侧涉及内存与字符串操作时，必须使用社区/项目提供的安全函数（`*_s` 系列），禁止使用无边界检查的危险函数。

Kernel 侧不适用：Kernel 无 memcpy_s/memset_s，使用 Ascend C API（如 `Duplicate`、`DataCopyPad`）。

### 错误示例

```cpp
// ❌ 使用无边界检查的危险函数
(void)strcpy(dst, src);
(void)sprintf(buf, "%s", name);
```

### 正确示例

```cpp
// ✅ 使用安全函数并处理返回值
errno_t ret = strcpy_s(dst, dstMax, src);
if (ret != EOK) {
    return -1;
}
```

### 注意事项

危险函数与安全替代对照：

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

---

## SEC-4.2: 正确设置安全函数中的destMax参数

**严重级别**：高

### 问题描述

本条所称安全函数指项目实际引入的 Secure C 边界检查函数，不是所有名称带 `_s` 的接口都具有相同签名。常见的带 `destMax` 参数接口如下；完整声明及可选接口见 [OpenHarmony bounds_checking_function 的 securec.h](https://github.com/openharmony/third_party_bounds_checking_function/blob/master/include/securec.h)，使用时以项目实际版本为准。

Kernel 侧不适用：本条针对 Host 使用的 Secure C 内存及字符串接口。

| 类别 | 常见接口 | destMax 含义 |
|------|----------|-------------|
| 内存操作 | `memcpy_s`、`memmove_s`、`memset_s` | 从目标地址起可写区域的字节容量 |
| 窄字符串复制和追加 | `strcpy_s`、`strncpy_s`、`strcat_s`、`strncat_s` | 目标字符数组容量，包含字符串结束符的空间 |
| 窄字符串格式化输出 | `sprintf_s`、`vsprintf_s`、`snprintf_s`、`vsnprintf_s` | 目标字符数组容量，包含字符串结束符的空间 |

### 错误示例

```cpp
// ❌ destMax 误用源长度 / 指针 sizeof
(void)memcpy_s(dst, sizeof(srcPtr), srcPtr, srcLen);  // sizeof(指针) 不是目标容量
(void)strcpy_s(dst, strlen(src), src);                // 源长度不是目标容量
```

### 正确示例

```cpp
// ✅ destMax = 目标地址起的实际可写容量（dst 为数组，容量已知）
errno_t ret = memcpy_s(dst, sizeof(dst), src, srcLen);
if (ret != EOK) {
    return -1;
}
```

### 注意事项

- `destMax` 不得超过目标地址起实际可写的容量；不能将源数据长度、指针的 `sizeof` 或期望写入长度直接当成目标容量。
- 目标为缓冲区内部地址时，应使用该位置起的剩余容量；计算前先确认偏移有效，防止容量减法回绕。
- `count` 等参数表示本次操作长度，不能替代 `destMax`。字符串追加还需计入目标已有内容及结束符；格式化输出需区分容量和输出长度限制。
- 宽字符版本及其他扩展接口按其声明核对容量单位，不能直接把字节数当成字符个数。`scanf_s` 一类接口没有统一的 `destMax` 参数，应按转换项要求提供各目标缓冲区大小，不能套用上述签名。
- 容量合法仍不代表操作一定成功，返回值按本文件 SEC-4.3 处理。

---

## SEC-4.3: 必须检查安全函数返回值，并进行正确的处理

**严重级别**：高

### 问题描述

必须按具体接口的返回值约定处理失败，不能继续执行依赖成功结果的操作。返回 `errno_t` 的内存和字符串操作通常以 `EOK` 表示成功；格式化输出、输入接口返回字符数或成功转换项数等，不能统一用 `!= EOK` 判断。

Kernel 侧不适用：Kernel 无安全函数。

### 错误示例

```cpp
// ❌ 返回值未检查，失败后继续使用目标缓冲区
err = memcpy_s(destBuff, destMax, src, srcLen);
Process(destBuff);  // 复制失败时 destBuff 内容无效
```

### 正确示例

```cpp
// ✅ 检查返回值，失败即处理
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

### 注意事项

- 返回值约定以实际引入的 Secure C 版本为准；errno_t、字符数、转换项数等不同接口家族的判断方式不同。
- 失败路径不得继续执行依赖目标缓冲区内容的操作。

---

## SEC-5.1: 禁止通过整对象字节复制或填充绕过非 trivially copyable 对象的构造、赋值语义

**严重级别**：高

### 问题描述

对非 trivially copyable 类型，禁止将 memcpy/memmove（含 _s 版本）、memset 或手工逐字节覆盖用于整对象复制、搬移或初始化，替代该类型的构造函数、赋值操作或公开初始化接口。典型对象包括 `std::string`、`std::vector`、拥有资源的智能指针以及含非平凡拷贝或析构操作的类；此类覆盖可能破坏所有权或内部状态。

对 trivially copyable 类型，从有效的同类型对象复制字节，或保存并原样恢复同一对象的字节表示，还需满足标准对对象生命周期、存储大小和 potentially-overlapping subobject 的限制；memcpy 另有源目标不重叠要求。该性质不表示任意字节模式都是合法值，也不保证 memset(..., 0, ...) 等价于值初始化。对象初始化优先使用类型提供的初始化方式。

### 错误示例

```cpp
// ❌ 对非 trivially copyable 对象整对象字节复制
std::string src = "...";
std::string dst;
(void)memcpy(&dst, &src, sizeof(dst));  // 破坏内部指针与所有权
```

### 正确示例

```cpp
// ✅ 使用类型自身的拷贝/赋值语义
std::string dst = src;
```

### 注意事项

- 不能仅凭类型是 struct、模板类或用于 Kernel 就认定其可按字节复制。应核实具体类型定义，或在编译环境支持时使用 `std::is_trivially_copyable<T>::value` 判断。AscendC 数据搬运、Duplicate 等 API 应按其操作的实际数据元素及接口契约判断，不能等同于对 C++ 包装对象执行整对象字节复制。
- 本规则不禁止在语言规则允许范围内通过 char、unsigned char 或 std::byte 观察对象表示，也不禁止对对象内部独立的字节缓冲区进行合法操作。不要仅因出现 reinterpret_cast 或字节访问就判定违规。
- 来源：
  - [C++ 标准草案：Trivially copyable types](https://eel.is/c++draft/basic.types.trivial)：同类型字节复制与原样恢复保证及对象边界。
  - [C++ 标准草案：Class properties](https://eel.is/c++draft/class.prop)：trivially copyable 的定义；standard-layout 与 trivially copyable 是不同性质。
  - [C++ 标准草案：Type accessibility](https://eel.is/c++draft/basic.lval#11)：允许 char、unsigned char、std::byte 类型访问对象表示。

---

## SEC-6.1: 禁止从空指针创建std::string

**严重级别**：高

### 问题描述

以空指针构造 `std::string` 是未定义行为；`const char*` 来源可能为空时，必须先判空再构造。

Kernel 侧不适用：Kernel 无 std::string。

### 错误示例

```cpp
// ❌ 未判空即构造
const char *name = GetName();  // 可能返回 nullptr
std::string nameStr(name);     // nullptr 构造，未定义行为
```

### 正确示例

```cpp
// ✅ 判空后再构造
const char *name = GetName();
std::string nameStr = (name != nullptr) ? name : "";
```

### 注意事项

- 空指针来源包括外部接口返回、可选属性、失败路径残留，均须判空。
- 判空与指针有效性的整体策略参见 ascendc-red-line.md 条例6。

---

## SEC-6.2: 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针

**严重级别**：高

### 问题描述

`c_str()`/`data()` 返回的指针仅在 string 对象下一次修改或销毁前有效；保存该指针跨越修改点或销毁点再使用，会读到悬空地址。

Kernel 侧不适用：Kernel 无 std::string。

### 错误示例

```cpp
// ❌ 保存 c_str 指针跨越修改点
std::string s = "abc";
const char *p = s.c_str();
s = "another value";  // s 修改后 p 可能失效
Use(p);                // 悬空读
```

### 正确示例

```cpp
// ✅ 就近取用不保存；确需长期持有内容时复制
std::string s = "abc";
Use(s.c_str());          // 就近使用
std::string copy = s;    // 需要长期持有时复制内容
```

### 注意事项

- 触发指针失效的操作包括赋值、append 等修改行为及对象销毁。
- 需要指针形态立即传参时，在调用处就近取 `c_str()`，不作为中间变量保存。

---

## SEC-6.3: 内存中的敏感信息使用完毕后立即清0

**严重级别**：高

### 问题描述

口令、密钥等敏感信息使用完毕后立即清零，避免被攻击者获取。

Kernel 侧说明：Kernel 中 UB 数据可通过 `Duplicate` 清零，GM 数据需在 Host 侧处理。

### 错误示例

```cpp
// ❌ 敏感缓冲区用后未清零即释放
char *key = (char *)malloc(KEY_LEN);
GetKey(key, KEY_LEN);
UseKey(key);
free(key);  // 内容仍残留于释放的内存中
```

### 正确示例

```cpp
// ✅ 用后立即清零再释放
char *key = (char *)malloc(KEY_LEN);
GetKey(key, KEY_LEN);
UseKey(key);
(void)memset_s(key, KEY_LEN, 0, KEY_LEN);
free(key);
```

### 注意事项

- 清零须在释放或复用前完成。
- 编译器可能消除对不再读取的缓冲区的普通写入；敏感清零应按项目安全要求选择可靠方式，并核对实际生效。

---

## SEC-7.1: LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序）

**严重级别**：高

### 问题描述

Host 侧使用 `OP_LOGE` / `OP_LOGD` / `OP_LOGW` 等格式化 LOG 宏，本条检查 Host 侧的格式化 LOG 调用。LOG 宏的格式化占位符与实际参数之间必须满足三个维度的一致性：

1. **数量一致**：参数少于占位符时，从栈上读取垃圾值，若被解释为 `%s` 将触发段错误
2. **类型匹配**：类型大小不匹配时（如 `uint64_t` 误用 `%d`），按说明符宽度截断，后续参数全部错位
3. **顺序对应**：参数顺序与格式符位置不对应时（如 `%s` 位置收到整数），整数被当作地址读字符串 → **段错误(SIGSEGV)**

业务代码标准调用形式：

```cpp
OP_LOGE(context->GetNodeName(), "format string %s %ld", arg1, arg2);
OP_LOGD(context->GetNodeName(), "format string %lu", arg1);
```

### 错误示例

```cpp
// ❌ 数量不一致：2 个占位符，1 个参数
OP_LOGD(ctx, "M: %ld, K: %ld", m);           // 缺少 k

// ❌ 类型不匹配：uint64_t 用了 %d
OP_LOGE(ctx, "n = %d, ubSize = %d\n", n, ubSize); // n/ubSize 均为 uint64_t

// ❌ 顺序错位：数量=5 格式符=5，但位置1和3的参数放反了
//   格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
//   参数:   inputName.c_str()(1) ... d0Size/NUM8(3) ...
//   → 位置1: %u 收到 const char*，位置3: %s 收到 uint → 段错误
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

### 正确示例

```cpp
// ✅ 数量、类型、顺序与占位符逐位一致
OP_LOGD(ctx, "M: %ld, K: %ld", m, k);
OP_LOGE(ctx, "n = %llu, ubSize = %llu\n", n, ubSize);
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

### 注意事项

类型与说明符速查：

| 类型 | 正确 | 常见错误 | 后果 |
|------|------|---------|------|
| `uint64_t` | `%llu` | `%u`, `%lu`, `%d` | 截断为 32 位，后续参数错位 |
| `int64_t` | `%lld` | `%d`, `%ld` | 同上 |
| `uint32_t` | `%u` | `%d` | 大值显示为负数 |
| `size_t` | `%zu` | `%d`, `%u` | 64 位系统上截断 |
| `bool` | `%d` 或 `? "true":"false"` + `%s` | `%s` 直传 | 未定义行为 |
| `void*` | `%p` | `%x` | 不可移植 |

### 专属检视方法

> **⚠️ 禁止仅凭 grep 单行分析 LOG 调用。** 算子仓中大量 LOG 语句跨越多行（2-35 行），且常嵌套在 `OP_CHECK_IF` 等外层宏内。grep 命中后**必须 Read 前后至少 10 行**获取完整的格式字符串和全部参数，否则分析的是截断的不完整调用，结论无效。多行字符串拼接（`"a" "b"`）需先合并再解析。

1. grep `OP_LOGE\|OP_LOGD\|OP_LOGW\|OP_LOGI` 找到所有 LOG 调用
2. Read 完整调用后，提取格式符序列和参数序列，逐位比对：数量是否一致 → 每个位置的参数类型是否兼容格式符
3. 高风险标记：`%s` 收到整数（段错误）、`uint64_t`/`int64_t` 配 `%d`（截断错位）
