# Launcher 与证据合同

## 1. 职责边界

Launcher 只负责解析本次 case 的已冻结合同、分配 host/device/workspace、按物理数据合同执行 H2D/D2H、用 DESIGN 冻结的 entry/参数顺序/grid/workspace/stream 启动 Kernel，并保存 raw output 与运行记录。

Launcher 不生成随机输入、不计算 CPU Golden、不判定精度、不选择候选、不切换备选、不读取 golden 文件决定行为。

## 2. 逻辑到物理 buffer

每个 Tensor 分开冻结 logical shape/dtype、storage dtype/packed bytes、layout pattern、transpose/packing/padding、alignment/physical shape、stride/offset units 和 valid/tail range。

当需要 pack、transpose、alignment 或 padding 时，逻辑 ABI 仍是对外合同，物理 buffer 是设备实现细节。DESIGN 必须在 abi_crosswalk 中记录转换、物理 shape、byte span、offset/stride、padding 和首个设备消费者。

可预生成物理输入时，使用独立于 Launcher 的 data path 生成 physical witness。Launcher 从本 case 逻辑输入重新生成待 H2D buffer，并在 H2D 前执行 shape、size 和逐字节一致性检查；失败在 Host ABI/I/O 边界阻断，不能启动 Kernel。

没有逻辑到物理转换时，逻辑 bytes 本身就是事实源，不额外伪造 witness。

## 3. ACL 生命周期

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateContext(&context, deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    // H2D -> launch -> synchronize -> D2H
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtDestroyContext(context));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

每个成功分配的资源必须建立异常路径和释放顺序。deviceId、stream 数量和内存 flag 由项目合同决定。

## 4. Launch 与 verifier 证据隔离

一个 case 的完整证据必须同时包含 input files、physical H2D witness、launch command、launch return code、raw output、stdout/stderr、verifier command、verifier return code 和 timeout/termination state。

只启动成功、没有 verifier 结果或 verifier 未执行时，状态是 incomplete，不是 PASS。批处理循环不能吞掉 verifier 返回码；旧目录输出不能补当前 case 证据。

## 5. 零工作量输入

先完成输入文件、shape、dtype、group/batch metadata 和输出合同校验，再判断是否合法零工作量。

若合同允许逻辑行数为零，选择以下一种已证明路径：

1. 在调用 tiler、分配 device buffer 和启动 kernel 前生成合同规定的空输出；
2. 用源码和设备证据证明支持零维的 tiler/kernel 路径。

不得把零维输入试探性传入只支持正维度的 tiler 来判断行为。

## ABI Crosswalk 表结构

每个逻辑参数、输出和辅助 ABI 对象建立一行：

| 列 | 必填内容 |
|---|---|
| crosswalk_row_id | 稳定行 ID |
| 逻辑对象 | 参数/辅助对象 ID、角色、必选/可选语义 |
| 物理表示 | buffer、storage dtype/layout、byte span、offset/stride 单位、有效范围 |
| host 接线 | Launcher 参数、owner、分配/释放生命周期、H2D/D2H 方向 |
| device 接线 | 有序 Kernel GM 参数或入口绑定、可空条件 |
| 控制接线 | TilingData/Params 字段、workspace、grid、stream/dispatch |
| 消费者 | Kernel、Vector、最终输出消费者及 final/partial 时机 |
| 证据 | 手册/头文件/样例/设备 probe 的 source refs |

只替换地址而保留逻辑 stride、gate offset 或旧 Pattern 不算 ABI 闭合。

## 启动绑定字段

启动前逐项冻结：

    entry modifier and symbol
    template/specialization
    ordered GM arguments and physical byte rules
    TilingData/Params mapping
    workspace address/size/lifetime
    grid/core and stream
    source-backed signature witness

不得从函数名或相似工程猜测 `__global__`、`__cube__`、`__mix__`、模板参数、ratio 或参数顺序。

## SKIP_BUILD 产物绑定

跳过构建只能在以下条件全部满足时使用：

1. 默认构建目录存在。
2. 该目录由当前 source/design contract clean-build 产生。
3. binary 路径、构建目录和 source identity 可证明。

缺少 launcher、目录不匹配或产物 identity 无法证明时，首个失败边界是 Host/test setup，状态为 incomplete，不得启动设备或归因到 Kernel 精度。

## cwd 与项目根闭合

Runner 从脚本自身位置解析 PROJECT_ROOT，统一使用绝对 artifact path 或显式进入根目录。不得假设调用方 cwd 是项目根，也不得让 source 环境脚本改变业务路径语义。
