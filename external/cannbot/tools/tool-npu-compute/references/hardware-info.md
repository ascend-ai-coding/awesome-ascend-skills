# HardwareInfo 字段说明

`HardwareInfo.jsonl` 记录采集时的主机和 NPU 环境快照。正常的 `npu-compute` 数据采集会生成该文件，无需把 HardwareInfo 作为 Section 单独指定。分析完整采集目录或比较多次采集结果时，应先读取该文件。

## 文件格式

- 文件采用 JSON Lines 格式，每行是一个独立 JSON 对象。
- 当前输出顺序固定为 `Host Info`、`Device Info`、`CPU Information`、`AI Core Information`、`Memory Information`。
- 每行的 `category` 字段标识该行类别，其他字段由类别决定。
- 容量字段是 JSON number，最多保留两位小数并去掉末尾的零；计数和频率字段是 JSON integer。
- 个别底层查询失败时，对应字段可能保留默认值 `0` 或空字符串。分析时应结合其他字段判断该值是否表示真实零值，不能统一当作设备确实没有该资源。

## Host Info

| 字段 | JSON 类型 | 单位 | 含义与来源 | 属性 | 分析用途和注意事项 |
|---|---|---|---|---|---|
| `category` | string | 无 | 固定为 `Host Info`，标识主机信息行 | 固定标识 | 用于按类别解析 JSONL，不是性能指标 |
| `cpu physical count` | integer | 个 | 读取在线 CPU 列表，并对各 CPU 的 `physical_package_id` 去重得到的物理 CPU 封装数量 | 主机配置，通常稳定 | 表示 Socket/Package 数量，不表示物理 Core 数量 |
| `cpu logical count` | integer | 个 | 通过 `_SC_NPROCESSORS_CONF` 获取主机配置的逻辑 CPU 数量 | 主机配置，通常稳定 | 可用于比较主机 CPU 规模，不等同于采集时实际在线或被目标程序使用的线程数 |
| `memory total size(MB)` | number | MB | `sysinfo` 返回的主机总内存字节数除以 `1024^2` | 主机配置，通常稳定 | 表示主机物理内存总量，不表示目标程序占用量或采集时空闲量 |
| `disk total size(GB)` | number | GB | 通过 `statvfs` 获取工具使用的文件系统容量，使用总块数乘块大小后除以 `1024^3` | 文件系统配置，通常稳定 | 表示对应文件系统的总容量，不表示所有磁盘容量，也不表示剩余空间 |

## Device Info

| 字段 | JSON 类型 | 单位 | 含义与来源 | 属性 | 分析用途和注意事项 |
|---|---|---|---|---|---|
| `category` | string | 无 | 固定为 `Device Info`，标识 NPU 设备信息行 | 固定标识 | 用于按类别解析 JSONL，不是性能指标 |
| `npu count` | integer | 个 | Runtime 当前可见的 NPU 设备数量 | 运行环境配置，通常稳定 | 用于确认采集环境可见的设备规模，不表示本次 Kernel 使用的设备数量 |
| `chip info` | string | 无 | SoC 名称与非空芯片版本以空格连接 | 设备型号，通常稳定 | 是判断两次采集是否来自同类 SoC 的主要字段；不要仅按字符串局部相似判断完全可比 |
| `arch info` | string | 无 | Runtime 返回的 NPU 架构属性转换为十进制字符串 | 设备架构，通常稳定 | 用于区分架构代际；示例值 `3510` 是字符串而不是 JSON integer |

## CPU Information

| 字段 | JSON 类型 | 单位 | 含义与来源 | 属性 | 分析用途和注意事项 |
|---|---|---|---|---|---|
| `category` | string | 无 | 固定为 `CPU Information`，标识设备侧 CPU 信息行 | 固定标识 | 用于按类别解析 JSONL，不是性能指标 |
| `control cpu count` | integer | 个 | 设备接口返回的 Control CPU Core 数量 | 设备配置，通常稳定 | 描述设备控制侧 CPU 资源，不表示主机 CPU 数量 |
| `ai cpu count` | integer | 个 | Runtime 设备属性返回的 AI CPU Core 数量 | 设备配置，通常稳定 | 描述设备 AI CPU 资源，不表示 AI Core、Cube Core 或 Vector Core 数量 |
| `ai cpu frequency(MHZ)` | integer | MHz | 设备管理接口返回的 AI CPU 当前频率 | 运行状态，可能变化 | 用于记录采集时 AI CPU 频率环境，不用于直接解释 AIC/AIV PMU 时长 |

## AI Core Information

| 字段 | JSON 类型 | 单位 | 含义与来源 | 属性 | 分析用途和注意事项 |
|---|---|---|---|---|---|
| `category` | string | 无 | 固定为 `AI Core Information`，标识设备计算 Core 信息行 | 固定标识 | 用于按类别解析 JSONL，不是性能指标 |
| `ai core count` | integer | 个 | Runtime 设备属性返回的 AI Core 数量 | 设备配置，通常稳定 | 按设备接口原值解释，不根据 Cube/Vector 数量自行推导 |
| `ai cube count` | integer | 个 | Runtime 设备属性返回的 Cube Core 数量 | 设备配置，通常稳定 | 用于比较 Cube 资源规模，也是读取频率信息时要求存在的计数字段 |
| `ai vector count` | integer | 个 | Runtime 设备属性返回的 Vector Core 数量 | 设备配置，通常稳定 | 用于比较 Vector 资源规模，也是读取频率信息时要求存在的计数字段 |
| `ai cube frequency(MHZ)` | integer | MHz | 优先读取 Cube Core 当前频率，失败时读取固定频率，再失败时使用 `1650` | 运行状态或回退值 | 可作为采集环境频率记录；仅凭该字段不能断言某个 CSV 公式实际采用了完全相同的配置值 |
| `ai vector frequency(MHZ)` | integer | MHz | 优先读取 Vector Core 当前频率，失败时读取固定频率，再失败时使用 `1650` | 运行状态或回退值 | 可作为采集环境频率记录；仅凭该字段不能断言某个 CSV 公式实际采用了完全相同的配置值 |

## Memory Information

| 字段 | JSON 类型 | 单位 | 含义与来源 | 属性 | 分析用途和注意事项 |
|---|---|---|---|---|---|
| `category` | string | 无 | 固定为 `Memory Information`，标识设备 HBM 信息行 | 固定标识 | 用于按类别解析 JSONL，不是性能指标 |
| `hbm total(MB)` | number | MB | 平台接口返回的 HBM 总字节数除以 `1024^2` | 设备配置，通常稳定 | 表示平台报告的 HBM 总容量，可用于判断设备内存规模 |
| `hbm used(MB)` | number | MB | Runtime 返回的 HBM 可分配总量减空闲量，再除以 `1024^2` | 采集时状态，可能变化 | 表示该 Runtime 接口可见内存池的已使用量，不应直接等同于整卡所有进程的物理 HBM 占用 |
| `hbm frequency(MHZ)` | integer | MHz | 设备管理接口返回的 HBM 频率 | 运行状态，可能变化 | 用于记录采集时内存频率环境，不直接代入当前五种 CSV 的公开计算公式 |

## 分析流程

1. 按行解析 JSON，不把整个文件当作一个 JSON 数组。
2. 根据 `category` 匹配类别和字段，不依赖行号代替类别判断。
3. 比较采集结果前，优先核对 `chip info`、`arch info`、Core 数量和频率环境。
4. 对 `hbm used(MB)`、AI CPU 频率、AI Core 频率和 HBM 频率等动态值，按采集时快照解释。
5. 遇到异常的零值或空字符串时，只说明字段值及其可能的数据不可用语义，不补造缺失硬件信息。
