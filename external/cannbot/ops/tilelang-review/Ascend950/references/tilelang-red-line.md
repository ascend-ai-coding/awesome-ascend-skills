# TileLang 算子红线问题清单

<适用>
语言: Python, TileLang DSL
侧别: All
领域: false
默认启用: true

适用场景: TileLang 算子开发中实际高频出现的编码红线问题
介绍: TileLang 红线类高频问题清单，8 条条款覆盖除零、索引、整数溢出、初始化、数据竞争、资源释放和 GM 地址范围
类别(All): 除零保护、索引边界、有符号整数不溢出、无符号整数不回绕、禁止读取未初始化数据、避免数据竞争
类别(Host): 资源释放
类别(Kernel): GM 内存偏移或大小使用足够宽的整数类型
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 3
</检视负载>

## 目的

检查 TileLang Host 与 Kernel 中的正确性、安全性和资源管理红线。

## 快速索引

### Host 和 Kernel 都适用 `[适用: All]`（6 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 1 | 除法/余数运算除零保护 | 数值安全 | 高 |
| 2 | GM/UB/fragment 索引校验 | 内存安全 | 高 |
| 3 | 有符号整数运算不溢出 | 数值安全 | 高 |
| 4 | 无符号整数运算不回绕 | 数值安全 | 高 |
| 5 | 禁止读取或写回未初始化数据 | 内存安全 | 高 |
| 6 | 避免线程、核和流水 stage 数据竞争 | 并发安全 | 高 |

### 仅 Host 侧适用 `[适用: Host]`（1 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 7 | 文件、子进程和临时资源可靠释放 | 资源管理 | 高 |

### 仅 Kernel 侧适用 `[适用: Kernel]`（1 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 8 | GM 内存偏移或大小使用足够宽的整数类型 | 内存安全 | 高 |

---

## 详细规范

### 1 确保除法和余数运算不会导致除以零 `[适用: All]`

**【检视策略】**

扫描 `/`、`//`、`%`、`T.ceildiv` 和归一化表达式，追踪除数来源。

| 除数来源 | 判定 |
|---------|------|
| 非零编译期常量 | PASS |
| 已由 factory/assert 限定为正的 specialization 参数 | PASS |
| 用户 Tensor shape、属性或动态中间值 | 必须有 wrapper 校验或 Kernel guard |
| `value - 1` 等派生表达式 | 重新证明结果非零 |

**错误示例**

```python
num_groups = T.ceildiv(hidden, group_size)
```

**正确示例**

```python
assert group_size in (16, 32, 64, 128)
num_groups = T.ceildiv(hidden, group_size)
```

---

### 2 外部数据作为索引时必须确保在范围内 `[适用: All]`

**【检视策略】**

追踪 gather index、expert id、stride offset 和动态 slice。保护条件必须支配 GM load/store；先越界读取再 mask 输出仍为错误。

**错误示例**

```python
value = x[index[row]]
if index[row] < num_rows:
    out[row] = value
```

**正确示例**

```python
if 0 <= index[row] and index[row] < num_rows:
    out[row] = x[index[row]]
```

---

### 3 确保有符号整数运算不溢出 `[适用: All]`

**【问题描述】**

`row * stride + col`、shape product、字节数、flatten offset 和 `T.view` 长度一旦在中间表达式中溢出，即使最终变量较宽也无法恢复正确值。Python factory 中的整数不溢出，不能证明 lowering 后的 IR 表达式仍使用足够宽的 dtype。

**【检视策略】**

1. 从公开最大 shape 推导每个乘法、加法和对齐表达式的最大绝对值。
2. 核对参与运算的每个 operand 在 IR 中的 dtype，确保扩宽发生在乘法之前。
3. 对 `T.cast`、函数参数、`T.alloc_var` 和 SIMD 窄整数逐个检查是否无依据缩窄。

**【排除规则】**：编译期常量表达式由 Python 求值且进入 IR 前已证明范围安全，或窄类型仅用于已证明范围内的 lane/index，可以 PASS。

**【判定方法】**：合法输入范围内中间值可溢出并影响地址、循环次数、分块或输出时判 FAIL；缺少公开最大 shape 时标记待确认并写明所需边界。

---

### 4 确保无符号整数运算不回绕 `[适用: All]`

**【问题描述】**

若 lowering 后的值为无符号整数，`value - 1` 在 `value == 0` 时会回绕为大正数，常进一步污染尾块长度、逆向循环和地址偏移。

**【检视策略】**

- 检查减法、ceil-div 展开、逆向循环、`n - offset` 和对齐公式。
- 追踪 operand 的 signedness，不以 Python 源码表面没有 `uint` 为依据。
- 下界 guard 必须支配减法本身；先回绕再比较无效。

**【判定方法】**：存在可达零值或较小左操作数，并且回绕结果参与访存、循环或输出时判 FAIL；完整上游下界证明存在时 PASS。

---

### 5 禁止读取或写回未初始化数据 `[适用: All]`

**【问题描述】**

`T.alloc_shared`、`T.alloc_fragment`、`T.alloc_var` 和 `torch.empty` 都不表示算法所需的清零。完整 tile 正常不代表尾 tile、条件分支和流水首次迭代已经覆盖所有有效输出。

**【检视策略】**

1. 从每个 buffer/标量的分配点开始，检查首次读取前所有可达路径是否已赋值。
2. 对完整寄存器访问核对 UB padding lane 是否已填充，不能只看有效 GM slice。
3. 对 int64/packed 数据经窄视图写入的路径，确认所有组成 word/byte 都已定义。
4. 对 reducer、partial output 和 atomic 目标同时检查 Host 初始化与 Kernel 初始化。

发现未初始化读取后继续追踪其值是否影响有效 GM 写回、地址、分支、同步或异常。只发生在无效尾行且可证明不可观察时，记录为待确认或健壮性建议，不能直接判功能 FAIL。

**正确示例**

```python
# int64 输出通过 int32 UB 写入时，高 32 位也必须初始化
with T.SimdVF():
    T.clear(acc_ub)
```

**【判定方法】**：未初始化值可到达有效输出、地址或控制流时判 FAIL；所有有效 lane 在每条路径均先写后读时 PASS。

---

### 6 需要避免数据竞争 `[适用: All]`

**【检视策略】**

1. `T.Parallel`/`T.SimtVF` 每线程写地址唯一，或使用正确 atomic。
2. `T.Persistent` 不让不同 core 写同一输出。
3. 多 stage buffer 的版本、producer/consumer 顺序和生命周期正确。
4. Atomic 操作的目标地址、dtype、初值及调用次数符合算法合约。

**【问题描述】**

仅仅让每个 core 获得不同 task id 不能自动证明写地址互斥；flatten/unflatten、shared expert、partial reduction 和 inplace alias 都可能让多个执行单元写入同一位置。非原子读改写也不能靠执行顺序碰巧正确。

**【排除规则】**：可从 task domain 和地址公式证明写集合两两不相交，或当前 TileLang/PTO lowering 明确支持所用 atomic dtype/操作且算法允许任意执行顺序时可以 PASS。

**【判定方法】**：存在两个可并发执行单元无同步地写同一位置，或 atomic 目标初值与累加语义不符时判 FAIL；缺少目标 atomic lowering 证据时标记待确认。

---

### 7 文件、子进程和临时资源必须可靠释放 `[适用: Host]`

**【问题描述】**

测试、构建和 profiling 失败时，未关闭的文件、子进程、临时目录或设备资源会污染后续用例，尤其会影响 pytest-xdist 和多设备任务。

**【检视策略】**：沿正常返回、异常、超时和用户中断路径检查资源生命周期；优先使用 context manager，必须跨作用域时使用覆盖所有出口的 `try/finally`。

**【判定方法】**：存在可达退出路径遗漏关闭、等待或清理动作时判 FAIL；由标准 context manager 完整托管时 PASS。

---

### 8 GM 内存偏移或大小必须使用足够宽的整数类型 `[适用: Kernel]`

**【问题描述】**

GM offset、元素总数和字节数必须覆盖公开最大 shape 下的完整范围。进入 Kernel 或 lowering 后无依据缩窄为 32 位，会使大 Tensor 的后半段访问错误地址。

**【检视策略】**

1. 推导最大 flatten offset、stride product 和 byte count。
2. 检查乘法发生前是否已经转为足够宽的整数类型。
3. 区分只在小范围内使用的 SIMD lane index 与最终 GM 地址；前者可以较窄，后者必须覆盖完整地址范围。

**【排除规则】**：公开接口和 factory specialization 已把元素总数严格限制在目标窄类型范围内，并且该约束覆盖全部入口时可以 PASS。

**【判定方法】**：合法最大输入可使 offset/size 截断并用于 GM 访问时判 FAIL；最大输入不明确时标记待确认。

---

## 检视检查清单

- [ ] 动态除数有非零保证
- [ ] GM/UB/fragment 索引和 slice 在界内
- [ ] 有符号整数运算不溢出
- [ ] 无符号整数运算不回绕
- [ ] 所有写回值在所有路径已初始化
- [ ] 线程、核和 stage 之间无数据竞争
- [ ] Host 资源在异常路径也能释放
- [ ] GM offset 和大小的整数类型覆盖完整范围
