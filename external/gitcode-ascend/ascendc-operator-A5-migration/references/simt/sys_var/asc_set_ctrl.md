> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_ctrl.md

# asc_set_ctrl

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

对CTRL寄存器（控制寄存器）的特定比特位进行设置。

## 函数原型

```cpp
__aicore__ inline void asc_set_ctrl(uint64_t config)
```

## 参数说明

表1 参数说明

|参数名|输入/输出|描述|
| :------ | :---  | :------------ |
|config   |输入   |待设置的寄存器值。常用CTRL寄存器比特位说明参考表2。

表2 常用CTRL寄存器比特位说明

|CTRL比特位    |功能|
| :-------     | :---- |
|CTRL[48]      |用于控制整数计算指令的饱和模式 <br>- 1'b0：饱和模式，INF输出会被饱和为±MAX，NAN输出会被饱和为0；<br> - 1'b1：非饱和模式，INF/NAN保持原输出。<br> 该控制位仅支持如下数据类型：<br>- 浮点数计算时支持half数据类型；<br>- 浮点数精度转换时支持如下数据类型：hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、half、bfloat16_t。|
|CTRL[49]      |用于控制标量单元读取数据时是否通过datacache读取。datacache的读取速度快于ub，因此在对ub进行大量读取，且ub中数据不被vector单元写入时，可选择开启该模式。<br>- 1'b0：标量单元读取数据时，直接从ub中读取；<br> - 1'b1：标量单元读取数据时，从datacache中读取。|
|CTRL[53]      |用于控制整数计算指令的饱和模式 <br>- 1'b0：截断模式，溢出值按目标数据类型位数截断，保留低位，舍弃高位；<br>- 1'b1：饱和模式，溢出值饱和到±MAX。|
|CTRL[56]      |MASK模式控制位。MASK用于辅助SIMD指令执行，由于CTRL[56]指示两种模式：<br>- 1'b0：向量操作中的掩码寄存器。每位对应向量中一个元素，标记该元素是否参与计算。'1'表示对应元素将被计算；'0'表示对应元素将不被计算；<br>- 1'b1：MASK[31:0]用于指示SIMD指令实际操作的元素数量。实际重复次数由VECTOR自动推断，此时参数repeat time(Xt[63:56])被忽略。|
|CTRL[59]      |用于控制浮点数转整数或整数转整数时的精度转换饱和模式，仅在CTRL[60]使能时生效。<br>- 1'b0：饱和模式，溢出值饱和到±MAX；<br>- 1'b1：截断模式：溢出值按照目标数据类型位数截断，保留低位，舍弃高位。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 使用前需要先用[asc_get_ctrl()](asc_get_ctrl.md)获取当前值，仅对需要变更的bit位进行修改。
- 仅支持已设置CTRL[48]、CTRL[53]、CTRL[56]、CTRL[59]比特位。

## 调用示例

```cpp
// 将CTRL[48]设置为非饱和模式
uint64_t ori_config = asc_get_ctrl();
uint64_t mask = (static_cast<uint64_t>(1) << 48);
uint64_t config = ori_config | mask;
asc_set_ctrl(config);
```