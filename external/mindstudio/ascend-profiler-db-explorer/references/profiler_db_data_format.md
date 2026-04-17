## msprof导出db格式数据说明

msprof命令执行完成后，会生成一个汇总所有性能数据的msprof\_\{时间戳\}.db表结构文件，该文件推荐使用MindStudio Insight工具查看，也可以使用Navicat Premium等数据库开发工具直接打开。当前db文件汇总的性能数据如下：

>[!NOTE] 说明 
>db文件均以表格形式展示性能数据，且所有数据均以数字映射（例如opName字段下的算子名显示为194），数字与名称的映射表为[STRING\_IDS](#zh-cn_topic_0000002076410600_section116561584178)。

**单位相关**

1. 时间相关，统一使用纳秒（ns），且为本地Unix时间。
2. 内存相关，统一使用字节（Byte）。
3. 带宽相关，统一使用Byte/s。
4. 频率相关，统一使用MHz。

**ENUM\_API\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 1**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|API类型|

**表 2**  内容

|id|name|
|--|--|
|20000|acl|
|15000|model|
|10000|node|
|5500|communication|
|5000|runtime|
|50001|op|
|50002|queue|
|50003|trace|
|50004|mstx|

**ENUM\_MODULE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 3**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|组件名|

**表 4**  内容

|id|name|
|--|--|
|0|SLOG|
|1|IDEDD|
|2|SCC|
|3|HCCL|
|4|FMK|
|5|CCU|
|6|DVPP|
|7|RUNTIME|
|8|CCE|
|9|HDC|
|10|DRV|
|11|NET|
|22|DEVMM|
|23|KERNEL|
|24|LIBMEDIA|
|25|CCECPU|
|27|ROS|
|28|HCCP|
|29|ROCE|
|30|TEFUSION|
|31|PROFILING|
|32|DP|
|33|APP|
|34|TS|
|35|TSDUMP|
|36|AICPU|
|37|LP|
|38|TDT|
|39|FE|
|40|MD|
|41|MB|
|42|ME|
|43|IMU|
|44|IMP|
|45|GE|
|47|CAMERA|
|48|ASCENDCL|
|49|TEEOS|
|50|ISP|
|51|SIS|
|52|HSM|
|53|DSS|
|54|PROCMGR|
|55|BBOX|
|56|AIVECTOR|
|57|TBE|
|58|FV|
|59|MDCMAP|
|60|TUNE|
|61|HSS|
|62|FFTS|
|63|OP|
|64|UDF|
|65|HICAID|
|66|TSYNC|
|67|AUDIO|
|68|TPRT|
|69|ASCENDCKERNEL|
|70|ASYS|
|71|ATRACE|
|72|RTC|
|73|SYSMONITOR|
|74|AMP|
|75|ADETECT|
|76|MBUFF|
|77|CUSTOM|

**ENUM\_HCCL\_DATA\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 5**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|通信数据类型|

**表 6**  内容

|id|name|
|--|--|
|0|INT8|
|1|INT16|
|2|INT32|
|3|FP16|
|4|FP32|
|5|INT64|
|6|UINT64|
|7|UINT8|
|8|UINT16|
|9|UINT32|
|10|FP64|
|11|BFP16|
|12|INT128|
|255|RESERVED|
|65534|N/A|
|65535|INVALID_TYPE|

**ENUM\_HCCL\_LINK\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 7**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|通信链路类型|

**表 8**  内容

|id|name|
|--|--|
|0|ON_CHIP|
|1|HCCS|
|2|PCIE|
|3|ROCE|
|4|SIO|
|5|HCCS_SW|
|6|STANDARD_ROCE|
|255|RESERVED|
|65534|N/A|
|65535|INVALID_TYPE|

**ENUM\_HCCL\_TRANSPORT\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 9**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|通信传输类型|

**表 10**  内容

|id|name|
|--|--|
|0|SDMA|
|1|RDMA|
|2|LOCAL|
|255|RESERVED|
|65534|N/A|
|65535|INVALID_TYPE|

**ENUM\_HCCL\_RDMA\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 11**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，ID|
|name|TEXT|通信RDMA类型|

**表 12**  内容

|id|name|
|--|--|
|0|RDMA_SEND_NOTIFY|
|1|RDMA_SEND_PAYLOAD|
|255|RESERVED|
|65534|N/A|
|65535|INVALID_TYPE|

**ENUM\_MSTX\_EVENT\_TYPE**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 13**  格式

|字段名| 类型     |含义|
|--|--------|--|
|id| INTEGER |索引，Host侧tx打点数据event类型对应的ID|
|name| TEXT   |Host侧tx打点数据event类型|

**表 14**  内容

|id|name|
|--|--|
|0|marker|
|1|push/pop|
|2|start/end|
|3|marker_ex|

**ENUM\_MEMCPY\_OPERATION**

枚举表。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 15**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|主键，ID|
|name|TEXT|拷贝类型|

**表 16**  内容

|id|name|
|--|--|
|0|host to host|
|1|host to device|
|2|device to host|
|3|device to device|
|4|managed memory|
|5|addr device to device|
|6|host to device ex|
|7|device to host ex|
|65535|other|

**STRING\_IDS**

映射表，用于存储ID和字符串映射关系。

无对应开关。

**表 17**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|索引，string ID|
|value|TEXT|string value|

**SESSION\_TIME\_INFO**

时间表，用于存储性能数据中的开始结束时间。在采集未正常退出时，无结束时间。

无对应开关。

**表 18**  格式

|字段名|类型|含义|
|--|--|--|
|startTimeNs|INTEGER|任务开启时的Unix时间，单位ns|
|endTimeNs|INTEGER|任务结束时的Unix时间，单位ns|

**NPU\_INFO**

对应deviceId的芯片型号。

无对应开关。

**表 19**  格式

|字段名|类型|含义|
|--|--|--|
|id|INTEGER|设备ID，显示为-1时表示未采集到deviceId|
|name|TEXT|设备对应的芯片型号|

**HOST\_INFO**

hostUid及名称。

无对应开关。

**表 20**  格式

|字段名|类型|含义|
|--|--|--|
|hostUid|TEXT|标识Host的唯一ID|
|hostName|TEXT|Host主机名称，如localhost|

**TASK**

task数据，呈现所有硬件执行的算子信息。

由--task-time开关控制。

**表 21**  格式

|字段名|类型|含义|
|--|--|--|
|startNs|INTEGER|与globalTaskId联合索引，索引名称TaskIndex，算子任务开始时间，单位ns|
|endNs|INTEGER|算子任务结束时间，单位ns|
|deviceId|INTEGER|算子任务对应的设备ID|
|connectionId|INTEGER|生成host-device连线|
|globalTaskId|INTEGER|与startNs联合索引，索引名称TaskIndex，用于唯一标识全局算子任务|
|globalPid|INTEGER|算子任务执行时的PID|
|taskType|INTEGER|设备执行该算子的加速器类型|
|contextId|INTEGER|用于区分子图小算子，常见于MIX算子和FFTS+任务|
|streamId|INTEGER|算子任务对应的streamId|
|taskId|INTEGER|算子任务对应的taskId|
|modelId|INTEGER|算子任务对应的modelId|

**COMPUTE\_TASK\_INFO**

计算算子描述信息。

由--task-time开关控制。

**表 22**  格式

| 字段名             |类型|含义|
|-----------------|--|--|
| name            |INTEGER|算子名，STRING_IDS(name)|
| globalTaskId    |INTEGER|索引，全局算子任务ID，用于关联TASK表|
| blockDim        |INTEGER|算子运行切分数量，对应算子运行时核数|
| mixBlockDim     |INTEGER|mix算子从加速器的BlockNum值|
| taskType        |INTEGER|Host执行该算子的加速器类型，STRING_IDS(taskType)|
| opType          |INTEGER|算子类型，STRING_IDS(opType)|
| inputFormats    |INTEGER|算子输入数据格式，STRING_IDS(inputFormats)|
| inputDataTypes  |INTEGER|算子输入数据类型，STRING_IDS(inputDataTypes)|
| inputShapes     |INTEGER|算子的输入维度，STRING_IDS(inputShapes)|
| outputFormats   |INTEGER|算子输出数据格式，STRING_IDS(outputFormats)|
| outputDataTypes |INTEGER|算子输出数据类型，STRING_IDS(outputDataTypes)|
| outputShapes    |INTEGER|算子输出维度，STRING_IDS(outputShapes)|
| attrInfo        |INTEGER|算子的attr信息，用来映射算子shape，算子自定义的参数等，STRING_IDS(attrInfo)|
| opState         |INTEGER|算子的动静态信息，dynamic表示动态算子，static表示静态算子，N/A表示该场景或该算子不识别，STRING_IDS(opState)|
| hf32Eligible    |INTEGER|标识是否使用HF32精度标记，YES表示使用，NO表示未使用，N/A表示该场景或该算子不识别，STRING_IDS(hf32Eligible)|

**COMMUNICATION\_TASK\_INFO**

描述通信小算子信息。

由--task-time、--hccl、--ascendcl开关控制对应数据的采集。配置--task-time为非l0时数据有效。有通信数据的场景下默认生成该表。

**表 23**  格式

|字段名|类型|含义|
|--|--|--|
|name|INTEGER|算子名，STRING_IDS(name)|
|globalTaskId|INTEGER|索引，索引名称CommunicationTaskIndex，全局算子任务ID，用于关联TASK表|
|taskType|INTEGER|算子类型，STRING_IDS(taskType)|
|planeId|INTEGER|网络平面ID|
|groupName|INTEGER|通信域，STRING_IDS(groupName)|
|notifyId|INTEGER|notify唯一ID|
|rdmaType|INTEGER|RDMA类型，包含：RDMASendNotify、RDMASendPayload，ENUM_HCCL_RDMA_TYPE(rdmaType)|
|srcRank|INTEGER|源Rank|
|dstRank|INTEGER|目的Rank|
|transportType|INTEGER|传输类型，包含：LOCAL、SDMA、RDMA，ENUM_HCCL_TRANSPORT_TYPE(transportType)|
|size|INTEGER|数据量，单位Byte|
|dataType|INTEGER|数据格式，ENUM_HCCL_DATA_TYPE(dataType)|
|linkType|INTEGER|链路类型，包含：HCCS、PCIe、RoCE，ENUM_HCCL_LINK_TYPE(linkType)|
|opId|INTEGER|对应的大算子Id，用于关联COMMUNICATION_OP表|
|isMaster|INTEGER|标记主从流通信算子，分析时以主流算子为准，取值为：0：从流1：主流|
|bandwidth|NUMERIC|该通信小算子的带宽数据，单位Byte / s|

**COMMUNICATION\_OP**

描述通信大算子信息。

由--task-time、--hccl开关控制对应数据的采集。有通信数据的场景下默认生成该表。

**表 24**  格式

|字段名|类型|含义|
|--|--|--|
|opName|INTEGER|算子名，STRING_IDS(opName)，例：hcom_allReduce__428_0_1|
|startNs|INTEGER|通信大算子的开始时间，单位ns|
|endNs|INTEGER|通信大算子的结束时间，单位ns|
|connectionId|INTEGER|生成host-device连线|
|groupName|INTEGER|通信域，STRING_IDS(groupName)，例：10.170.22.98%enp67s0f5_60000_0_1708156014257149|
|opId|INTEGER|索引，通信大算子Id，用于关联COMMUNICATION_TASK_INFO表|
|relay|INTEGER|借轨通信标识|
|retry|INTEGER|重传标识|
|dataType|INTEGER|大算子传输的数据类型，如（INT8，FP32），ENUM_HCCL_DATA_TYPE(dataType)|
|algType|INTEGER|通信算子使用的算法，可分为多个阶段，STRING_IDS(algType)，如（HD-MESH）|
|count|NUMERIC|算子传输的dataType类型的数据量|
|opType|INTEGER|算子类型，STRING_IDS(opType)，例：hcom_broadcast_|
|deviceld|INTEGER|设备ID|

**CANN\_API**

CANN API数据。

由--ascendcl开关控制。

**表 25**  格式

|字段名|类型|含义|
|--|--|--|
|startNs|INTEGER|API的开始时间，单位ns|
|endNs|INTEGER|API的结束时间，单位ns|
|type|INTEGER|API类型，ENUM_API_TYPE(type)|
|globalTid|INTEGER|API所属的全局TID。高32位：PID，低32位：TID|
|connectionId|INTEGER|索引，用于关联TASK表和COMMUNICATION_OP表|
|name|INTEGER|API的名称，STRING_IDS(name)|

**QOS**

保存QoS的数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 26**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|eventName|NUMERIC|QoS事件名称，STRING_IDS(eventName)|
|bandwidth|NUMERIC|QoS对应时间的带宽，单位Byte / s|
|timestampNs|NUMERIC|本地时间，单位ns|

**AICORE\_FREQ**

AI Core频率信息。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 27**  格式

|字段名|类型|含义|
|--|--|--|
|deviceld|INTEGER|设备ID|
|timestampNs|NUMERIC|频率变化时的本地时间，单位ns|
|freq|INTEGER|AI Core频率值，单位MHz|

**ACC\_PMU**

ACC\_PMU数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 28**  格式

|字段名|类型|含义|
|--|--|--|
|accId|INTEGER|加速器ID|
|readBwLevel|INTEGER|DVPP和DSA加速器读带宽的等级|
|writeBwLevel|INTEGER|DVPP和DSA加速器写带宽的等级|
|readOstLevel|INTEGER|DVPP和DSA加速器读并发的等级|
|writeOstLevel|INTEGER|DVPP和DSA加速器写并发的等级|
|timestampNs|NUMERIC|本地时间，单位ns|
|deviceId|INTEGER|设备ID|

**SOC\_BANDWIDTH\_LEVEL**

SoC带宽等级信息。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 29**  格式

|字段名|类型|含义|
|--|--|--|
|l2BufferBwLevel|INTEGER|L2 Buffer带宽等级|
|mataBwLevel|INTEGER|Mata带宽等级|
|timestampNs|NUMERIC|本地时间，单位ns|
|deviceId|INTEGER|设备ID|

**NIC**

每个时间节点网络信息数据。

控制开关：

- msprof命令的--sys-io-profiling、--sys-io-sampling-freq
- Ascend PyTorch Profiler的sys\_io

**表 30**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|bandwidth|INTEGER|带宽，单位Byte/s|
|rxPacketRate|NUMERIC|收包速率，单位packet/s|
|rxByteRate|NUMERIC|接收字节速率，单位Byte/s|
|rxPackets|INTEGER|累计收包数量，单位packet|
|rxBytes|INTEGER|累计接收字节数量，单位Byte|
|rxErrors|INTEGER|累计接收错误包数量，单位packet|
|rxDropped|INTEGER|累计接收丢包数量，单位packet|
|txPacketRate|NUMERIC|发包速率，单位packet/s|
|txByteRate|NUMERIC|发送字节速率，单位Byte/s|
|txPackets|INTEGER|累计发包数量，单位packet|
|txBytes|INTEGER|累计发送字节数量，单位Byte|
|txErrors|INTEGER|累计发送错误包数量，单位packet|
|txDropped|INTEGER|累计发送丢包数量，单位packet|
|funcId|INTEGER|端口号|

**ROCE**

RoCE通信接口带宽数据。

控制开关：

- msprof命令的--sys-io-profiling、--sys-io-sampling-freq
- Ascend PyTorch Profiler的sys\_io

**表 31**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|bandwidth|INTEGER|带宽，单位Byte/s|
|rxPacketRate|NUMERIC|收包速率，单位packet/s|
|rxByteRate|NUMERIC|接收字节速率，单位Byte/s|
|rxPackets|INTEGER|累计收包数量，单位packet|
|rxBytes|INTEGER|累计接收字节数量，单位Byte|
|rxErrors|INTEGER|累计接收错误包数量，单位packet|
|rxDropped|INTEGER|累计接收丢包数量，单位packet|
|txPacketRate|NUMERIC|发包速率，单位packet/s|
|txByteRate|NUMERIC|发送字节速率，单位Byte/s|
|txPackets|INTEGER|累计发包数量，单位packet|
|txBytes|INTEGER|累计发送字节数量，单位Byte|
|txErrors|INTEGER|累计发送错误包数量，单位packet|
|txDropped|INTEGER|累计发送丢包数量，单位packet|
|funcId|INTEGER|端口号|

**LLC**

三级缓存带宽数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 32**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|llcId|INTEGER|三级缓存ID|
|timestampNs|INTEGER|本地时间，单位ns|
|hitRate|NUMERIC|三级缓存命中率(100%)|
|throughput|NUMERIC|三级缓存吞吐量，单位Byte/s|
|mode|INTEGER|模式，用于区分是读或写，STRING_IDS(mode)|

**TASK\_PMU\_INFO**

计算算子的PMU数据。

控制开关：

- msprof命令的--ai-core、--aic-mode=task-based开关控制该表生成，--aic-metrics开关控制具体数据采集
- Ascend PyTorch Profiler的aic\_metrics
- MindSpore Profiler的aic\_metrics

仅Atlas 200I/500 A2 推理产品和Atlas A2 训练系列产品/Atlas A2 推理系列产品支持采集该数据。

**表 33**  格式

|字段名|类型|含义|
|--|--|--|
|globalTaskId|INTEGER|全局算子任务ID，用于关联TASK表|
|name|INTEGER|PMU metric指标名，STRING_IDS(name)|
|value|NUMERIC|对应指标名的数值|

**SAMPLE\_PMU\_TIMELINE**

sample-based的PMU数据，用于timeline类的数据呈现。

控制开关：

- msprof命令的--ai-core、--aic-mode=sample-based开关控制该表生成，--aic-metrics开关控制具体数据采集
- Ascend PyTorch Profiler的aic\_metrics
- MindSpore Profiler的aic\_metrics

**表 34**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|totalCycle|INTEGER|对应core在时间片上的cycle数|
|usage|NUMERIC|对应core在时间片上的利用率（100%）|
|freq|NUMERIC|对应core在时间片上的频率，单位MHz|
|coreId|INTEGER|coreId|
|coreType|INTEGER|core类型(AIC或AIV)，STRING_IDS(coreType)|

**SAMPLE\_PMU\_SUMMARY**

sample-based的PMU数据，用于summary类的数据呈现。

控制开关：

- msprof命令的--ai-core、--aic-mode=sample-based开关控制该表生成，--aic-metrics开关控制具体数据采集
- Ascend PyTorch Profiler的aic\_metrics
- MindSpore Profiler的aic\_metrics

**表 35**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|metric|INTEGER|PMU metric指标名，STRING_IDS(metric)|
|value|NUMERIC|对应指标名的数值|
|coreId|INTEGER|coreId|
|coreType|INTEGER|core类型(AIC或AIV)，STRING_IDS(coreType)|

**NPU\_MEM**

NPU内存占用数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 36**  格式

|字段名|类型|含义|
|--|--|--|
|type|INTEGER|event类型，app或device，STRING_IDS(type)|
|ddr|NUMERIC|ddr占用大小，单位Byte|
|hbm|NUMERIC|hbm占用大小，单位Byte|
|timestampNs|INTEGER|本地时间，单位ns|
|deviceId|INTEGER|设备ID|

**NPU\_MODULE\_MEM**

NPU组件内存占用数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 37**  格式

|字段名|类型|含义|
|--|--|--|
|moduleId|INTEGER|组件类型，ENUM_MODULE(moduleId)|
|timestampNs|INTEGER|本地时间，单位ns|
|totalReserved|NUMERIC|内存占用大小，单位Byte|
|deviceId|INTEGER|设备ID|

**NPU\_OP\_MEM**

CANN算子内存占用数据，仅GE算子支持。

由--task-memory开关控制。

**表 38**  格式

|字段名|类型|含义|
|--|--|--|
|operatorName|INTEGER|算子名字，STRING_IDS(operatorName)|
|addr|INTEGER|内存申请释放首地址|
|type|INTEGER|用于区分申请或是释放，STRING_IDS(type)|
|size|INTEGER|申请的内存大小，单位Byte|
|timestampNs|INTEGER|本地时间，单位ns|
|globalTid|INTEGER|该条记录的全局TID。高32位：PID，低32位：TID|
|totalAllocate|NUMERIC|总体已分配的内存大小，单位Byte|
|totalReserve|NUMERIC|总体持有的内存大小，单位Byte|
|component|INTEGER|组件名，STRING_IDS(component)|
|deviceId|INTEGER|设备ID|

**HBM**

片上内存读写速率数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 39**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|bandwidth|NUMERIC|带宽，单位Byte/s|
|hbmId|INTEGER|内存访问单元ID|
|type|INTEGER|用于区分读或写，STRING_IDS(type)|

**DDR**

片上内存读写速率数据。

由--sys-hardware-mem、--sys-hardware-mem-freq开关控制。

**表 40**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|read|NUMERIC|内存读取带宽，单位Byte/s|
|write|NUMERIC|内存写入带宽，单位Byte/s|

**HCCS**

HCCS集合通信带宽数据。

控制开关：

- msprof命令的--sys-interconnection-profiling、--sys-interconnection-freq
- Ascend PyTorch Profiler的sys\_interconnection

**表 41**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|txThroughput|NUMERIC|发送带宽，单位Byte/s|
|rxThroughput|NUMERIC|接收带宽，单位Byte/s|

**PCIE**

PCIe带宽数据。

控制开关：

- msprof命令的--sys-interconnection-profiling、--sys-interconnection-freq
- Ascend PyTorch Profiler的sys\_interconnection

**表 42**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|本地时间，单位ns|
|txPostMin|NUMERIC|发送端PCIe Post数据传输带宽最小值，单位Byte/s|
|txPostMax|NUMERIC|发送端PCIe Post数据传输带宽最大值，单位Byte/s|
|txPostAvg|NUMERIC|发送端PCIe Post数据传输带宽平均值，单位Byte/s|
|txNonpostMin|NUMERIC|发送端PCIe Non-Post数据传输带宽最小值，单位Byte/s|
|txNonpostMax|NUMERIC|发送端PCIe Non-Post数据传输带宽最大值，单位Byte/s|
|txNonpostAvg|NUMERIC|发送端PCIe Non-Post数据传输带宽平均值，单位Byte/s|
|txCplMin|NUMERIC|发送端接收写请求的完成数据包最小值，单位Byte/s|
|txCplMax|NUMERIC|发送端接收写请求的完成数据包最大值，单位Byte/s|
|txCplAvg|NUMERIC|发送端接收写请求的完成数据包平均值，单位Byte/s|
|txNonpostLatencyMin|NUMERIC|发送端PCIe Non-Post模式下的传输时延最小值，单位ns|
|txNonpostLatencyMax|NUMERIC|发送端PCIe Non-Post模式下的传输时延最大值，单位ns|
|txNonpostLatencyAvg|NUMERIC|发送端PCIe Non-Post模式下的传输时延平均值，单位ns|
|rxPostMin|NUMERIC|接收端PCIe Post数据传输带宽最小值，单位Byte/s|
|rxPostMax|NUMERIC|接收端PCIe Post数据传输带宽最大值，单位Byte/s|
|rxPostAvg|NUMERIC|接收端PCIe Post数据传输带宽平均值，单位Byte/s。|
|rxNonpostMin|NUMERIC|接收端PCIe Non-Post数据传输带宽最小值，单位Byte/s|
|rxNonpostMax|NUMERIC|接收端PCIe Non-Post数据传输带宽最大值，单位Byte/s|
|rxNonpostAvg|NUMERIC|接收端PCIe Non-Post数据传输带宽平均值，单位Byte/s|
|rxCplMin|NUMERIC|接收端收到写请求的完成数据包最小值，单位Byte/s|
|rxCplMax|NUMERIC|接收端收到写请求的完成数据包最大值，单位Byte/s|
|rxCplAvg|NUMERIC|接收端收到写请求的完成数据包平均值，单位Byte/s|

**META\_DATA**

基础数据，当前仅保存版本号信息。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。

**表 43**  格式

|字段名|类型|含义|
|--|--|--|
|name|TEXT|字段名|
|value|TEXT|数值|

**表 44**  内容

|name|含义|
|--|--|
|SCHEMA_VERSION|总版本号，如1.0.2|
|SCHEMA_VERSION_MAJOR|大版本号，如1，仅当数据库格式存在重写或重构时更改|
|SCHEMA_VERSION_MINOR|中版本号，如0，当更改列或类型时更改，存在兼容性问题|
|SCHEMA_VERSION_MICRO|小版本号，如2，当更新表时都会更改，不具有兼容性问题|

**MSTX\_EVENTS**

mstx接口采集的Host侧数据，Device侧数据在TASK表中整合。

由--msproftx开关控制表格输出，mstx接口控制数据的采集。

**表 45**  格式

|字段名|类型|含义|
|--|--|--|
|startNs|INTEGER|Host侧tx打点数据开始时间，单位ns|
|endNs|INTEGER|Host侧tx打点数据结束时间，单位ns|
|eventType|INTEGER|Host侧tx打点数据类型，ENUM_MSTX_EVENT_TYPE(eventType)|
|rangeId|INTEGER|Host侧range类型tx数据对应的range ID|
|category|INTEGER|Host侧tx数据所属的分类ID|
|message|INTEGER|Host侧tx打点数据携带信息，STRING_IDS(message)|
|globalTID|INTEGER|Host侧tx打点数据开始线程的全局TID|
|endGlobalTid|INTEGER|Host侧tx打点数据结束线程的全局TID|
|domainId|INTEGER|Host侧tx打点数据所属域的域ID|
|connectionId|INTEGER|Host侧tx打点数据的关联ID，TASK(connectionId)|

**COMMUNICATION\_SCHEDULE\_TASK\_INFO**

通信调度描述信息，当前仅针对AI CPU通信算子的描述。

无对应开关，导出msprof\_\{时间戳\}.db文件时默认生成。需要采集环境中包含AI CPU通信算子。

**表 46**  格式

|字段名|类型|含义|
|--|--|--|
|name|INTEGER|算子名，STRING_IDS(name)|
|globalTaskId|INTEGER|主键，全局算子任务ID，用于关联TASK表|
|taskType|INTEGER|Host执行该算子的加速器类型，STRING_IDS(taskType)|
|opType|INTEGER|算子类型，STRING_IDS(opType)|

**MEMCPY\_INFO**

描述memcpy相关算子的拷贝数据量和拷贝方向。

由--runtime-api开关控制。

**表 47**  格式

|字段名|类型|含义|
|--|--|--|
|globalTaskId|NUMERIC|主键，全局算子任务ID，用于关联TASK|
|size|NUMERIC|拷贝的数据量|
|memcpyOperation|NUMERIC|拷贝类型，STRING_IDS(memcpyDirection)|

**CPU\_USAGE**

Host侧CPU利用率数据。

由--host-sys=cpu开关控制。

**表 48**  格式

|字段名|类型|含义|
|--|--|--|
|timestampNs|NUMERIC|采样时的本地时间，单位ns|
|cpuId|NUMERIC|cpu编号|
|usage|NUMERIC|利用率(%)|

**HOST\_MEM\_USAGE**

Host侧内存利用率数据。

由--host-sys=mem开关控制。

**表 49**  格式

|字段名|类型|含义|
|--|--|--|
|timestampNs|NUMERIC|采样时的本地时间，单位ns|
|usage|NUMERIC|利用率(%)|

**HOST\_DISK\_USAGE**

Host侧磁盘I/O利用率数据。

由--host-sys=disk开关控制。

**表 50**  格式

|字段名|类型|含义|
|--|--|--|
|timestampNs|NUMERIC|采样时的本地时间，单位ns|
|readRate|NUMERIC|磁盘读速率，单位B/s|
|writeRate|NUMERIC|磁盘写速率，单位B/s|
|usage|NUMERIC|利用率(%)|

**HOST\_NETWORK\_USAGE**

Host侧系统级别的网络I/O利用率数据。

由--host-sys=network开关控制。

**表 51**  格式

|字段名|类型|含义|
|--|--|--|
|timestampNs|NUMERIC|采样时的本地时间，单位ns|
|usage|NUMERIC|利用率(%)|
|speed|NUMERIC|网络使用速率，单位B/s|

**OSRT\_API**

Host侧syscall和pthreadcall数据。

由--host-sys=osrt开关控制。

**表 52**  格式

|字段名|类型|含义|
|--|--|--|
|name|INTEGER|OS Runtime API接口名|
|globalTid|NUMERIC|该API所在线程的全局TID。高32位：PID，低32位：TID|
|startNs|INTEGER|API的开始时间，单位ns|
|endNs|INTEGER|API的结束时间，单位ns|

**NETDEV\_STATS**

通过硬件采样带宽能力，可以部分识别通信问题，作为总览项，初步排查通信问题，如出现通信耗时异常，即可优先排查是否为网络拥塞导致。

控制开关：

- msprof命令的--sys-io-profiling、--sys-io-sampling-freq
- Ascend PyTorch Profiler的sys\_io
- MindSpore Profiler的sys\_io

**表 53**  格式

|字段名|类型|含义|
|--|--|--|
|deviceId|INTEGER|设备ID|
|timestampNs|INTEGER|采样时的本地时间，单位ns|
|macTxPfcPkt|INTEGER|MAC发送的PFC帧数|
|macRxPfcPkt|INTEGER|MAC接收的PFC帧数|
|macTxByte|INTEGER|MAC发送的字节数|
|macTxBandwidth|NUMERIC|MAC发送带宽，单位Byte / s|
|macRxByte|INTEGER|MAC接收的字节数|
|macRxBandwidth|NUMERIC|MAC接收带宽，单位Byte / s|
|macTxBadByte|INTEGER|MAC发送的坏包报文字节数|
|macRxBadByte|INTEGER|MAC接收的坏包报文字节数|
|roceTxPkt|INTEGER|RoCEE发送的报文数|
|roceRxPkt|INTEGER|RoCEE接收的报文数|
|roceTxErrPkt|INTEGER|RoCEE发送的坏包报文数|
|roceRxErrPkt|INTEGER|RoCEE接收的坏包报文数|
|roceTxCnpPkt|INTEGER|RoCEE发送的CNP类型报文数|
|roceRxCnpPkt|INTEGER|RoCEE接收的CNP类型报文数|
|roceNewPktRty|INTEGER|RoCEE发送的超时重传的数量|
|nicTxByte|INTEGER|NIC发送的字节数|
|nicTxBandwidth|NUMERIC|NIC发送带宽，单位Byte / s|
|nicRxByte|INTEGER|NIC接收的字节数|
|nicRxBandwidth|NUMERIC|NIC接收带宽，单位Byte / s|

**RANK\_DEVICE\_MAP**

rankId和deviceId的映射关系数据。

无对应开关，导出ascend\_pytorch\_profiler\_\{Rank\_ID\}.db文件时默认生成。

**表 54**  格式

|字段名|类型|含义|
|--|--|--|
|rankId|INTEGER|取值固定为-1。|
|deviceId|INTEGER|节点上的设备ID，显示为-1时表示未采集到deviceId。|


