# UB 检查：执行模式、布局与覆盖边界

遇到 UB 能力咨询或报告时读取本页，再按 checker 读取诊断参考。适用架构为 dav-3510。普通 SIMD 的 scalar 访问不等于 SIMT 线程访问。

## 按访问来源与检查类型区分

| 路径 | memcheck | initcheck | racecheck |
| --- | --- | --- | --- |
| 普通 SIMD 的受支持 vector、搬运及 scalar 访问 | 物理容量及所在分区 | 实际读写字节的初始化状态 | 已转换 UB 访问的潜在 RAW/WAR/WAW |
| SIMD VF | 按本次 launch 分区检查已还原的访问 | 实际读写字节的初始化状态 | VF 内外统一判序，潜在竞争使用 WARNING |
| SIMT VF | 逐线程 UB 写入，含记录可识别的原子/归约写 | 本次不扩展 | 本次不扩展 |

同一 launch 可混合多种路径；布局取自启动 kernel 的 `kernelVfType`，不能因没有采到 SIMT 记录就改判纯 SIMD。
布局及初始化状态按 launch 和相应存储归属隔离；不据此承诺跨 launch 竞争检查。

## 普通 SIMD 与 SIMD VF 的 memcheck 布局

以下公式限于默认预留配置：`U=256 KiB`、`R=8 KiB`（2 KiB ASC 预留和 6 KiB VF 栈）。
`C` 为包含 R 的编译期 UB 分配量，`D` 为本次启动的动态申请量，`S=C-R`。区间均为字节偏移、左闭右开。
关闭 VF 栈或 ASC 预留均超出支持范围；不能仅凭 C 的数值自动识别这些编译配置，也不能对已知非默认配置套用公式。

| kernel 类型 | 普通 SIMD/SIMD VF 使用的布局 |
| --- | --- |
| type 1：NO_VF | USER `[0,U-R)`、RESERVED `[U-R,U)`；不按 C/D 再细分 |
| type 2：SIMD_VF_ONLY，可包含普通 SIMD | STATIC `[0,S)`、DYNAMIC `[S,U-R)`、RESERVED `[U-R,U)`；无 SIMT Cache 或未分配动态区 |
| type 3：SIMT_VF_ONLY；type 4：SIMD_SIMT_MIX_VF，可包含普通 SIMD | `H=AlignUp(C+D,128B)`、`B=H-R`；STATIC `[0,S)`、DYNAMIC `[S,B)`、RESERVED `[B,H)`、SIMD_FORBIDDEN `[H,U)`；要求 H 不大于 `U-32 KiB` |

**type 2 的 D 不决定动态区访问上界。** 例如 C=64 KiB 时 S=56 KiB，D=0、32 或 64 KiB 的合法配置
都使用 56..248 KiB 动态区。不能把 `S+D` 之后的访问判为未分配。

type 3/4 的动态区包含 `[S+D,B)` 对齐填充。例如 C=65568B、D=32768B 时，S=57376、
S+D=90144、B=90240、H=98432，96B 填充属于动态区。SIMD 不需要推算 Cache 内部端点，整个 `[H,U)` 均为禁止区。

**同一操作数以其原始起点确定区域，所有实际片段必须留在该区域。** 从 STATIC 跨入 DYNAMIC，或 DYNAMIC 跨入 RESERVED，
即使两边各自允许访问，也属于跨区；完全位于 RESERVED 的访问可以通过 memcheck。源、目标操作数分别判断，
源在静态区、目标在动态区且各自合法时，不因一次搬运涉及两区报错。不同 VA 条目独立检查；stride 间隙不算实际访问。

缺少必要布局元数据或元数据矛盾时，不猜分区，相关 memcheck 分析不完整；**已确认超出 256 KiB 的物理访问仍可报告越界**。
物理容量内无诊断不能在布局未知时解释为分区检查通过。

## SIMD UB 的 initcheck 独立规则

**按实际访问字节检查，不按 STATIC/DYNAMIC/RESERVED/SIMD_FORBIDDEN 过滤。** type 1～4 及布局未知都使用同一流程：
检查实际读取的字节，按可见的实际写入更新目标状态。普通 SIMD 和 SIMD VF 在同一存储上下文共用 UB 字节状态；
不同 device、launch 和逻辑存储核隔离。stride 空洞未被写入，不自动初始化。

memcheck 报跨区、禁止区或布局未知，不抑制相应实际 UB 读写或合法 GM 写入的初始化跟踪。布局缺失本身不使
initcheck 不完整；真正的访问、转换或同步缺失仍限制结论。内部状态只能记录可表示且在物理容量内的字节，
超出该范围的异常记录不成为初始化证据，initcheck 也不承担越界诊断。

未初始化的 A 搬到 B 时可报告 A 的未初始化读取，但 B 已被实际写入；后续有可见性保证的读取 B 可不再报告未初始化。
这是写入历史跟踪，不做未初始化数据的污点传播，不能据此认定 B 的数值正确。

`UB address space: [0x0,0x40000)` 表示完整 UB 字节状态表，不是某个用户 buffer 或动态区边界。

## SIMT VF 的逐线程 UB 写入

写入必须完整落在本次执行的同一块已分配区域：STATIC、DYNAMIC（含对齐填充）、RESERVED 或 Data Cache。
静态/动态/预留边界沿用上面的含 SIMT 公式，但 **Cache 使用独立、可靠的实际 `[cacheBegin,cacheEnd)` 端点**。
SIMT 区内 Cache 写入可合法，不能套用 SIMD 的 `[H,U)` 禁止规则；四区外为未分配空间。

**检查使用本次 launch 的实际 Cache 区间。** 解读日志时直接核对实际端点，不用容量值、测试示例地址、同步成功或共享终点 H 推算端点。
若提供的日志缺少端点或写记录，应补充完整报告再判断四区检查结果；已有证据充分的非法写入诊断仍可解释。

非零写从 a 开始、长度 n，必须不超物理容量，且 `n <= end-a`，end 为起始区域终点；等于终点合法，
超过则跨区，即使后续进入另一个已分配区域。每个实际线程的每次非法写各报一条；线程 0 有效。
未执行线程、零字节写及 SIMT 读取不在本次写入检查范围。记录无法解出线程、地址、长度或存在回传丢失时说明分析不完整，
不把未知访问编造成越界。

## 指令覆盖与还原限制

- vector、scalar、搬运均以实际范围为准；scalar 不统一按 32B 放大，paired 访问需覆盖两个元素。缺失采集记录不能视为无访问。
- VTRANSPOSE.B16 的读写各为 512B；VBS32.F16/F32 输出每条 8B，F16 的保留字段也由指令写入。
- VMRGSORT（VMS4v2）支持已覆盖的 F32 非 suspend 组合；F16、suspend 及未支持参数组合保留覆盖缺口，不宣称完整通过。
- VNCHWCONV 等依赖 VA 状态；VA 缺失时只能保留可恢复范围。无地址记录中的占位 0 不代表真实访问 UB 偏移 0。
- LDVA 自身固定 32B 读取可检查；无有效 VA 快照时，后续依赖 VA 的访问仍可能不完整。不能编造开启快照的公开参数。

UB 竞争的同步模型、mask 警告和完整性要求见 [Racecheck 诊断](racecheck-diagnostics.md)；
越界字段见 [Memcheck 诊断](memcheck-diagnostics.md)，初始化字段见 [Initcheck 诊断](initcheck-diagnostics.md)。
