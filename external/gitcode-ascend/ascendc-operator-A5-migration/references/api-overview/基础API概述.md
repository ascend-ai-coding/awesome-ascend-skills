> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/基础API/概述.md`
# 概述<a name="ZH-CN_TOPIC_0000002518297265"></a>

[基础API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/基础API.md)实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。

**根据功能的不同，主要分为以下几类**：

-   [标量计算API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/标量计算.md)，实现调用Scalar计算单元执行计算的功能。
-   [Memory矢量计算API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/Memory矢量计算.md)，实现调用Vector计算单元执行计算的功能。
-   [矩阵计算API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/矩阵计算（ISASI）.md)，实现调用Cube计算单元执行计算的功能。
-   [数据搬运API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/Memory数据搬运.md)，计算API基于Local Memory数据进行计算，所以数据需要先从Global Memory搬运至Local Memory，再使用计算API完成计算，最后从Local Memory搬出至Global Memory。执行搬运过程的接口称之为数据搬运API，比如DataCopy接口。
-   [资源管理API](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/资源管理.md)，用于分配管理内存，比如AllocTensor、FreeTensor接口;
-   同步控制API，完成任务间的通信和同步，比如EnQue、DeQue接口。不同的API指令间有可能存在依赖关系，从[抽象硬件架构](../../编程模型/AI-Core-SIMD编程/抽象硬件架构.md)可知，不同的指令异步并行执行，为了保证不同指令队列间的指令按照正确的逻辑关系执行，需要向不同的组件发送同步指令。同步控制API内部即完成这个发送同步指令的过程，开发者无需关注内部实现逻辑，使用简单的API接口即可完成。

**根据对数据操作方法的不同，分为以下几类**：

-   连续计算API：支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维tensor的连续计算问题。

    ```
    Add(dst, src1, src2, n);
    ```

-   高维切分API：支持Repeat和Stride。功能灵活的计算API，提供与BuilitIn API完全对等编程能力，充分发挥硬件优势，支持对每个操作数的DataBlock Stride，Repeat Stride，Mask等参数的操作。

下图以矢量加法为例，展示了**连续计算API和高维切分API**的特点。

**图 1**  计算API几种计算方式的特点<a name="zh-cn_topic_0000001762058545_fig6847134062319"></a>  
![](../../../figures/计算API几种计算方式的特点.png "计算API几种计算方式的特点")

