> **原始文档路径**: `asc-devkit/docs/guide/编程指南/编程模型/AI-Core-SIMT编程/线程架构.md`
# 线程架构<a name="ZH-CN_TOPIC_0000002554358977"></a>

SIMT编程模型的线程层次结构分为两层：

-   线程块网格（Grid）：由多个线程块（Thread Block）组成，使用内置变量[gridDim](../../语言扩展层/SIMT-BuiltIn关键字.md#li20760123812911)来表示启用的线程块的个数，同一时刻一个AIV核只执行一个线程块任务。
-   线程块（Thread Block）：由若干线程（thread）组成，使用内置变量[blockDim](../../语言扩展层/SIMT-BuiltIn关键字.md#li076017381191)表示一个线程块启用的的线程个数，一个线程块最多可以启用2048个线程。

基于SIMT编程模型的程序，在AIV核上执行多个结构相同的线程块，执行的总线程数等于gridDim\*blockDim。

![](../../../figures/线程架构.png)

gridDim由三维结构[dim3](../../语言扩展层/SIMT-BuiltIn关键字.md#li1136665405)来表示，\{dimx，dimy，dimz\}用于指定3个不同维度的线程块的大小，三维乘积的总数不超过65535，各线程块可通过线程块索引[blockIdx](../../语言扩展层/SIMT-BuiltIn关键字.md#li1676053814914)进行标识。blockDim也由dim3三维结构表示，三维乘积的总数不超过2048，各线程可通过线程块内线程索引[threadIdx](../../语言扩展层/SIMT-BuiltIn关键字.md#li7760123814919)进行标识。线程索引的计算示例如下图所示：

![](../../../figures/线程架构-8.png)

底层调度过程中，同一时刻一个AIV只能执行一个线程块任务，每个线程块会被切分成多个Warp依次调度并完成执行。Warp是执行相同指令的线程的集合，每个Warp包含32个线程。每个AIV核包含4个Warp调度器（Warp Scheduler），调度器编号scheduler id为warp id % 4。

