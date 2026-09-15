# 调测接口 API 参考

> 快速参考导航 - 以下是本分类的 API 索引表，详细说明见下方各函数章节

# 调测接口

头文件: `#include "simt_api/asc_simt.h"`

SIMT VF函数内使用的调试接口。

| 函数 | 功能 | 原型 | 详情 |
|------|------|------|------|
| `printf` | 格式化输出(仅VF内) | `void printf(const __gm__ char* fmt, ...)` | `debug/printf-176.md` |
| `assert` | 运行时断言检查 | `void assert(bool condition)` | `debug/assert-177.md` |
| `__trap` | 立即终止当前线程执行 | `void __trap()` | `debug/__trap.md` |

**注意**: printf/assert仅用于调试，生产环境应去除。SIMT VF外使用 `AscendC::Simt::printf` 或 `PRINTF`。

---

## 函数详情


### assert-177

# assert<a name="ZH-CN_TOPIC_0000002523344786"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

本接口在SIMT VF调试场景下提供assert断言功能。在算子Kernel侧的SIMT VF实现代码中，如果assert的内部条件判断不为真，则会输出assert条件，并将输入的信息格式化打印在屏幕上，同时算子运行失败。

在算子Kernel侧代码的适当位置使用assert进行断言检查，并格式化输出一些调试信息。示例如下：

```
int assertFlag = 10;

assert(assertFlag == 10);
```

打印信息示例如下：

```
[ASSERT] /home/.../add_custom.cpp:44: : Assertion `assertFlag != 10' failed.
```

请注意，assert接口的打印功能对算子的实际运行性能有影响。

## 函数原型<a name="section2067518173415"></a>

```
assert(expr)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p541413413465"><a name="p541413413465"></a><a name="p541413413465"></a>expr</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p1441334144620"><a name="p1441334144620"></a><a name="p1441334144620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p84131146466"><a name="p84131146466"></a><a name="p84131146466"></a>assert断言是否终止程序的条件。条件为true则程序继续执行，条件为false则终止程序。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

该接口当前仅支持融合编译场景。

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"utils/debug/asc\_assert.h"头文件。

```
#include "utils/debug/asc_assert.h"
```

## 调用示例<a name="section82241477610"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void AddCustom(__gm__ bool* dst, __gm__ float* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    assert(!isnan(x[idx]));
    dst[idx] = x[idx];
}
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT] /home/.../add_custom.cpp:44: : Assertion `!isnan(x[idx])' failed.
```


---


### assert-179

# assert<a name="ZH-CN_TOPIC_0000002554424255"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

该接口实现AI CPU算子Kernel调试场景下的assert断言功能。

算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。

## 需要包含的头文件<a name="section78885814919"></a>

```
#include "aicpu_api.h"
```

## 函数原型<a name="section2067518173415"></a>

```
assert(expr)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p541413413465"><a name="p541413413465"></a><a name="p541413413465"></a>expr</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p1441334144620"><a name="p1441334144620"></a><a name="p1441334144620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p84131146466"><a name="p84131146466"></a><a name="p84131146466"></a>assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   该接口仅支持通过<<<...\>\>\>调用，并在[异构编译](编译与运行.md)场景使用。
-   kernel开发不要包含系统的assert.h，会导致宏定义冲突。
-   assert接口调用形式与C语言一致，不需要使用AscendC命名空间。

-   该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
-   使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu\_api为Device链接libaicpu\_api.a，通过--cce-aicpu-L指定libaicpu\_api.a的库路径。

    ```
    $bisheng -O2 foo.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o foo.aicpu.o
    ```

    $\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

## 调用示例<a name="section82241477610"></a>

在算子kernel侧实现代码中需要增加断言的地方使用assert检查代码示例如下：

```
int assertFlag = 10;
// 断言条件
assert(assertFlag == 12);
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT]` assertFlag == 12 ' at /home/.../test.cpp:36
```


---


### printf-176

# printf<a name="ZH-CN_TOPIC_0000002554424113"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

本接口提供SIMT VF调试场景下的格式化输出功能。在算子Kernel侧的SIMT VF实现代码中，需要输出日志信息时，调用printf接口打印相关内容。

## 函数原型<a name="section2067518173415"></a>

```
template <class... Args>
__attribute__((always_inline)) inline __simt_callee__ void printf(const __gm__ char* fmt, Args&&... args)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p45208478318"><a name="p45208478318"></a><a name="p45208478318"></a>fmt</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p135196472314"><a name="p135196472314"></a><a name="p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型的对象：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将原样不动地打印输出。</li><li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<div class="p" id="p158820115597"><a name="p158820115597"></a><a name="p158820115597"></a>支持的转换类型包括：<a name="ul541124915329"></a><a name="ul541124915329"></a><ul id="ul541124915329"><li>%d / %ld / %lld / %i / %li / %lli：输出十进制数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t</li><li>%f / %F：输出浮点数，支持打印的数据类型：float、half、bfloat16_t</li><li>%x / %lx / %llx：输出十六进制整数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t、uint8_t、uint16_t、uint32_t、uint64_t</li><li>%s：输出字符串</li><li>%u / %lu /%llu：输出unsigned类型数据，支持打印的数据类型：uint8_t、uint16_t、uint32_t、uint64_t</li><li>%p：输出指针地址</li></ul>
</div>
<p id="p1133101265017"><a name="p1133101265017"></a><a name="p1133101265017"></a><strong id="b164621818125011"><a name="b164621818125011"></a><a name="b164621818125011"></a>注意</strong>：上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。</p>
</li></ul>
</td>
</tr>
<tr id="row163919564263"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p1563916565265"><a name="p1563916565265"></a><a name="p1563916565265"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p59396564285"><a name="p59396564285"></a><a name="p59396564285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1257115311337"><a name="p1257115311337"></a><a name="p1257115311337"></a>附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   printf在SIMT VF中调用，会占用SIMT栈空间，请合理控制调用printf次数，防止SIMT栈溢出。
-   printf功能需要占用额外的Global Memory空间用于数据缓存，缓存空间大小默认为2MB。您可以通过acl.json中的"simt\_printf\_fifo\_size"字段进行配置，配置范围最小为1MB，最大为64MB。当打印数据量较大时，建议增加缓存空间。
-   在仿真环境下，使用printf接口会增加算子运行时间，通过在VF代码中判断线程ID，可以仅在部分线程中打印调试信息，减少重复内容的打印，更有利于调试。

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"utils/debug/asc\_printf.h"头文件。

```
#include "utils/debug/asc_printf.h"
```

## 调用示例<a name="section82241477610"></a>

```
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "utils/debug/asc_printf.h"

// asc_vf_call调用时dim3参数：dim3(8, 2, 8)
__simt_vf__ __launch_bounds__(128) inline void SimtCompute()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    printf("simt vf: d: (%d, %d, %d), f: %f, s: %s\n", x, y, z, 3.14f, "pass");
}
```

NPU模式下，程序运行时打印效果如下：

```
simt vf: d: (0, 0, 0), f: 3.140000, s: pass
simt vf: d: (0, 0, 1), f: 3.140000, s: pass
simt vf: d: (0, 0, 2), f: 3.140000, s: pass
simt vf: d: (0, 0, 3), f: 3.140000, s: pass
simt vf: d: (0, 0, 4), f: 3.140000, s: pass
simt vf: d: (0, 0, 5), f: 3.140000, s: pass
simt vf: d: (0, 0, 6), f: 3.140000, s: pass
simt vf: d: (0, 0, 7), f: 3.140000, s: pass
simt vf: d: (0, 1, 0), f: 3.140000, s: pass
simt vf: d: (0, 1, 1), f: 3.140000, s: pass
......
```


---


### printf-178

# printf<a name="ZH-CN_TOPIC_0000002554343839"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p27195352210"><a name="p27195352210"></a><a name="p27195352210"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。

## 需要包含的头文件<a name="section78885814919"></a>

```
#include "aicpu_api.h"
```

## 函数原型<a name="section2067518173415"></a>

```
void printf(const char* fmt, ...)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p45208478318"><a name="p45208478318"></a><a name="p45208478318"></a>fmt</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p135196472314"><a name="p135196472314"></a><a name="p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型的对象：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将原样不动地打印输出。</li><li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<div class="p" id="p158820115597"><a name="p158820115597"></a><a name="p158820115597"></a>支持的转换类型包括：<a name="ul541124915329"></a><a name="ul541124915329"></a><ul id="ul541124915329"><li>%d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t</li><li>%f：输出实数，支持打印的数据类型：float/half</li><li>%x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%s：输出字符串</li><li>%u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%p：输出指针地址</li></ul>
</div>
</li></ul>
</td>
</tr>
<tr id="row163919564263"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p1563916565265"><a name="p1563916565265"></a><a name="p1563916565265"></a>...</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p59396564285"><a name="p59396564285"></a><a name="p59396564285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1257115311337"><a name="p1257115311337"></a><a name="p1257115311337"></a>附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   该接口仅支持通过<<<...\>\>\>调用，并在[异构编译](编译与运行.md)场景使用。
-   该接口不支持打印除换行符之外的其他转义字符。
-   该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
-   使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu\_api为Device链接libaicpu\_api.a，通过--cce-aicpu-L指定libaicpu\_api.a的库路径。

    ```
    $bisheng -O2 foo.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o foo.aicpu.o
    ```

    $\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

## 调用示例<a name="section82241477610"></a>

在算子Kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```
#include "aicpu_api.h"

// 整型打印：
AscendC::printf("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
AscendC::printf("fmt string %f\n", a);

// 指针打印：
int b = 10;
int *c = &b;
AscendC::printf("TEST %p\n", c);
```

程序运行时打印效果如下：

```
fmt string 291
fmt string 3.140000
TEST 0xdfffd6fddd1c
```


---


### __trap

# \_\_trap<a name="ZH-CN_TOPIC_0000002523303642"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

在SIMT VF实现代码中调用此接口会中断算子的运行，适用于Kernel侧异常场景的调试。

## 函数原型<a name="section2067518173415"></a>

```
__simt_callee__ inline void __trap()
```

## 参数说明<a name="section158061867342"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"utils/debug/asc\_assert.h"头文件。

```
#include "utils/debug/asc_assert.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void SimtKernel(__gm__ bool* dst, __gm__ float* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (isnan(x[idx])) {
        __trap();
    }
    dst[idx] = x[idx];
}
```


---

