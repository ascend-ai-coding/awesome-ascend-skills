> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/基础API/常用操作速查指导/如何使用归约计算API.md`
# 如何使用归约计算API<a name="ZH-CN_TOPIC_0000002271364328"></a>

归约指令将数据集合简化为单一值或者更小的集合。按照归约操作的数据范围的不同，归约指令分为以下几种，可参考[归约指令示意图](#zh-cn_topic_0000002267504648_fig16735034112710)：

-   ReduceMax/ReduceMin/ReduceSum：对所有的输入数据做归约操作，得到最大值和最大值索引/最小值和最小值索引/数据总和。
-   WholeReduceMax/WholeReduceMin/WholeReduceSum：对每个repeat内的输入数据做归约操作，得到每个repeat内的最大值和最大值索引/最小值和最小值索引/数据总和。返回索引时返回的是repeat内部索引。
-   BlockReduceMax/BlockReduceMin/BlockReduceSum：对每个datablock内的输入数据做归约操作，得到每个datablock内的最大值/最小值/数据总和。
-   PairReduce：相邻两个（奇偶）元素求和，例如（a1, a2, a3, a4, a5, a6...），归约后结果为（a1+a2,  a3+a4,  a5+a6, ......）。

**图 1**  归约指令示意图<a name="zh-cn_topic_0000002267504648_fig16735034112710"></a>  
![](../../../../figures/归约指令示意图.png "归约指令示意图")

针对归约指令，和其他的基础API一样也提供了**tensor高维切分计算**接口，可充分发挥硬件优势，支持开发者控制指令的**迭代执行**和操作数的**地址间隔**，功能更加灵活。但具体参数的单位和约束与[基础API](../接口分类说明/高维切分API.md)略有不同，下文将对这些差异点进行介绍。

-   repeatTime：迭代次数，开发者通过repeatTime来配置迭代次数，从而控制指令的多次迭代执行。
    -   ReduceMax/ReduceMin/ReduceSum对于repeatTime超过255的情况，在API内部进行了处理，所以repeatTime支持更大的取值范围，保证不超过int32\_t最大值的范围即可。
    -   WholeReduceMax/WholeReduceMin/WholeReduceSum/BlockReduceMax/BlockReduceMin/BlockReduceSum/PairReduce和其他基础API一样，repeatTime要求不超过255。

-   **mask**：用于控制每次迭代内参与计算的元素，mask参数的使用方法和基础API通用的使用方法一致。
-   **repeatStride**：表示相邻迭代间的地址步长。
    -   ReduceMax/ReduceMin/ReduceSum指令的目的操作数会归约成一个最大值/最小值/总和，所以其目的操作数不支持配置repeatStride。仅源操作数支持repeatStride，其含义、单位（datablock）和基础API通用说明一致。
    -   WholeReduceMax/WholeReduceMin/WholeReduceSum/BlockReduceMax/BlockReduceMin/BlockReduceSum/PairReduce源操作数和目的操作数都支持配置repeatStride，源操作数repeatStride的含义、单位（datablock）和基础API通用说明一致。目的操作数repeatStride的含义、单位和基础API通用说明有差异，因为归约后，目的操作数的长度会变短，比如WholeReduceSum归约后每个repeat会合并为一个值，所以迭代之间的间隔不能再使用一个datablock为单位，而以一个repeat归约后的长度为单位。

-   **dataBlockStride**：表示单次迭代内datablock的地址步长。
    -   ReduceMax/ReduceMin/ReduceSum指令的目的操作数会归约成一个最大值/最小值/总和，所以其目的操作数不支持配置dataBlockStride。源操作数也不支持dataBlockStride。
    -   WholeReduceMax/WholeReduceMin/WholeReduceSum/BlockReduceMax/BlockReduceMin/BlockReduceSum/PairReduce源操作数支持配置dataBlockStride，源操作数dataBlockStride的含义、单位（datablock）和基础API通用说明一致。目的操作数不支持dataBlockStride，因为归约后，目的操作数的长度会变短，比如WholeReduceSum归约后每个repeat会合并为一个值，不再有迭代内datablock和地址间隔的概念。

