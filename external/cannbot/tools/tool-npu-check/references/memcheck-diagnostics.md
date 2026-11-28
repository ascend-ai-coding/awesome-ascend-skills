# Memcheck 重点诊断

在完成报告来源和可信度核对后，遇到内存访问报错或访问未完整分析时读取本页。
UB 报告先按 [UB 检查](ub-checks.md)区分 SIMD 与 SIMT 路径。

## 区分错误类型与判定边界

| 典型输出或场景 | 含义 | 需要核对与结论限制 |
| --- | --- | --- |
| `Invalid GM read/write of size N bytes`、`out of bounds` | 访问未完整落入本次设备上工具记录的有效分配范围 | 核对方向、访问区间、分配范围、device 与 launch。N 是访问长度，不能直接当作越界量，也不能仅凭源码行断定哪一个长度参数错误 |
| `used after free` | 访问命中工具记录为已释放的分配 | 核对分配、释放与实际使用的生命周期及原始命令。与普通越界区别解释；仅凭地址邻近某个 allocation，不能确认地址归属 |
| `Invalid L1/L0A/L0B/L0C read/write`、`capacity`、`valid offsets` | 对应片上空间的访问超出容量边界 | 按该空间的偏移和容量计算，不寻找 GM 分配记录。容量内访问不保证用户划分的逻辑 buffer 没有越界 |
| scalar 的短访问或一个 API 对应多条诊断 | 检查的是实际指令产生的访问 | 按报告的真实字节数判读，不把 scalar 统一扩为 32B。同一 API 可能展开成多条指令或多个访问片段，错误条数不等于源码缺陷数 |

例如，L0A 容量为 65536B，从偏移 65520 读 32B，实际超出容量 16B；报告的 `size 32 bytes`
仍是访问总长度。访问在合法分配或容量内却超出逻辑张量范围时，不能承诺工具一定检出。

## UB 分区与 SIMT 线程报告

| 典型输出或场景 | 含义与必看证据 |
| --- | --- |
| `Invalid UB read/write`、`UB layout (type N)` | 普通 SIMD/SIMD VF 的物理越界或分区违规。核对 launch、kernel 类型、各区间、`operand` 原始起点及实际 `fragment`，不能仅凭动态申请量 D 猜上界 |
| `outside DYNAMIC`、`first invalid address ... in RESERVED` | 同一操作数离开其起始动态区；不是预留区一律不能访问。完全落在预留区可合法，源、目标在不同区也应分别判定 |
| `UB layout: UNKNOWN; physical UB [0x0,0x40000)` | 布局未知仍可确认超出物理 UB 的访问；物理范围内则不能猜分区或宣布完整通过 |
| `Invalid UB dynamic memory write`、`SIMT thread`、`UB allocated regions` | SIMT VF 的线程写入没有完整落在同一已分配区域；核对线程号、实际写入区间、本次 launch 的四区边界。线程 0 有效，标题取起始区域，不表示整次访问都在该区 |

例如 DYNAMIC 为 `[0xe000,0x16000)`，一次 SIMD 写的片段为 `[0x15fe0,0x16020)`：
完整访问是 64B，后 32B 跨入预留区；`size 64 bytes` 不是越界 64B。stride 跳跃时首个非法地址可能跳过边界，
应按实际片段判断；负 stride 还需检查起始区域的下界，不能把首尾之间的空洞都算作访问。

SIMT 报告中的 Cache 区间必须有可靠来源；不能用测试中的 `[0x20000,0x40000)` 推断所有 launch。
SIMT 读取不在本次写入检查范围，每线程每次非法写可分别产生诊断，条数不等于源码缺陷数。

## 没有越界诊断，但访问未被完整检查

`inexact_device_operations` 表示有访问未被精确检查；`dropped_device_operations`、消息丢失或相关不完整原因
表示还存在遗漏。结合该版本报告的具体原因解释，不能仅凭 `errors=0` 或会话已正常结束声明目标访问通过。
这些字段与 `analysis_complete` 描述的层面可能不同，应一起核对，不要求它们始终联动。

典型场景是 NDDMA 缺少有效轴的 stride 寄存器状态：访问范围无法精确恢复时，memcheck 可以不生成确定的
GM 越界诊断，同时记录 inexact 访问。若同时启用 initcheck，还可能报告：

```text
Register nddma_loop0_stride was used before initialization
```

两者可以同时成立：前者说明范围检查存在缺口，后者说明消费指令的寄存器依赖未满足。
不能为了得到越界地址而把未知 stride 当作 0，也不能因 memcheck 没有错误而否定寄存器诊断。

## 资料与用例依据

- [使用手册](https://gitcode.com/cann/asc-tools/blob/master/docs/zh/02_npu_check.md)：通用命令与环境配置。UB 布局和跨区判定见 [UB 检查](ub-checks.md)。
- [scalar 用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/memcheck/scalar/README.md)、[片上容量用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/memcheck/cube/README.md)：真实访问字节数及各空间越界对照。
- [memory_access 用例断言](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/memcheck/memory_access/check_memory_access_end_to_end.sh)：`nddma_missing_active_stride`、API 多片段及 inexact 场景。

用例用于解释诊断字段和边界；分析具体问题时，使用本次 launch 的实际布局与访问记录。
