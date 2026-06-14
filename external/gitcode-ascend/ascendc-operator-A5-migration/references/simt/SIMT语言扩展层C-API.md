> **原始文档路径**: `asc-devkit/docs/guide/编程指南/语言扩展层/SIMT语言扩展层C-API.md`
# SIMT语言扩展层C API<a name="ZH-CN_TOPIC_0000002509743873"></a>

SIMT编程基于AI Core的硬件能力实现，可以使用[asc\_vf\_call](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/asc_vf_call.md)接口启动SIMT VF（Vector Function）子任务。当前，SIMT语言扩展层支持的C API类别如下：

-   [同步函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/同步函数.md)：提供内存管理与同步接口，解决不同核内的线程间可能存在的数据竞争以及线程的同步问题。
-   [数学函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/数学函数.md)：提供处理数学运算的函数接口集合。
-   [精度转换](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/精度转换.md)：提供不同精度类型间的转换功能的一系列API接口。
-   [比较函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/比较函数.md)：用于判断数据是否为有限数、无穷或nan。
-   [Atomic函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/Atmoic函数.md)：提供对Unified Buffer或Global Memory上的数据与指定数据执行原子操作的一系列API接口。
-   [Warp函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/Warp函数.md)：提供对单个Warp内32个线程的数据进行处理的相关操作的一系列API接口。
-   [类型转换](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/类型转换-141.md)：根据源操作数和目的操作数的数据类型进行精度转换。
-   [向量类型构造函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/向量类型构造函数.md)：向量类型构造相关接口。
-   专用函数：专用函数相关接口，包含[使能Cache Hints的Load/Store函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/使能Cache-Hints的Load-Store函数.md)相关接口。
-   [调测接口](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/printf-147.md)：SIMT VF调试场景下使用的相关接口。

