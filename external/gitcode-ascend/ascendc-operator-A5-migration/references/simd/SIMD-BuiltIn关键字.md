> **原始文档路径**: `asc-devkit/docs/guide/编程指南/语言扩展层/SIMD-BuiltIn关键字.md`
# SIMD BuiltIn关键字<a name="ZH-CN_TOPIC_0000002509863893"></a>

## 预定义宏<a name="section2035104120360"></a>

如其他语言一样，会提供一些内置的宏方便用户编写程序。预定义宏一节着重介绍一些用户在做异构编程时会经常用到的宏，以及宏的解释。

-   **\_\_NPU\_ARCH\_\_**

    \_\_NPU\_ARCH\_\_是Device侧AI Core代码中的预处理宏，用于标识AI处理器的架构版本。该宏由四位数字组成，其中前三位数字用于标识AI Core的IP核\(Intellectual Property Core\)类型，第四位数字标识该AI Core同一个IP核的配置版本。通过该宏，开发者可以针对不同AI处理器，差异化进行代码适配和优化。AI处理器型号和\_\_NPU\_ARCH\_\_对应关系如下表所示：

    **表 1** AI处理器型号和\_\_NPU\_ARCH\_\_的对应关系

    <a name="table65291052154114"></a>
    <table><thead align="left"><tr id="row1852915527411"><th class="cellrowborder" valign="top" width="49.59%" id="mcps1.2.3.1.1"><p id="p12529952134112"><a name="p12529952134112"></a><a name="p12529952134112"></a><span id="ph5758231184918"><a name="ph5758231184918"></a><a name="ph5758231184918"></a>AI处理器</span>型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="50.41%" id="mcps1.2.3.1.2"><p id="p252955211411"><a name="p252955211411"></a><a name="p252955211411"></a>__NPU_ARCH__</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1986372151716"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p2864142161711"><a name="p2864142161711"></a><a name="p2864142161711"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p5880145213281"><a name="p5880145213281"></a><a name="p5880145213281"></a>3510</p>
    </td>
    </tr>
    <tr id="row75292521412"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p052935264116"><a name="p052935264116"></a><a name="p052935264116"></a><span id="ph85291852104120"><a name="ph85291852104120"></a><a name="ph85291852104120"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p11529165254118"><a name="p11529165254118"></a><a name="p11529165254118"></a>2201</p>
    </td>
    </tr>
    <tr id="row1652910526411"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p45291152194112"><a name="p45291152194112"></a><a name="p45291152194112"></a><span id="ph25291652164115"><a name="ph25291652164115"></a><a name="ph25291652164115"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p65291452164116"><a name="p65291452164116"></a><a name="p65291452164116"></a>2201</p>
    </td>
    </tr>
    <tr id="row155291352204114"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p115291852144111"><a name="p115291852144111"></a><a name="p115291852144111"></a><span id="ph11529115214412"><a name="ph11529115214412"></a><a name="ph11529115214412"></a><term id="zh-cn_topic_0000001312391781_term354143892110"><a name="zh-cn_topic_0000001312391781_term354143892110"></a><a name="zh-cn_topic_0000001312391781_term354143892110"></a>Atlas 200I/500 A2 推理产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p17530185224119"><a name="p17530185224119"></a><a name="p17530185224119"></a>3002</p>
    </td>
    </tr>
    <tr id="row1353025204112"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p153075220411"><a name="p153075220411"></a><a name="p153075220411"></a><span id="ph1653025204119"><a name="ph1653025204119"></a><a name="ph1653025204119"></a><term id="zh-cn_topic_0000001312391781_term4363218112215"><a name="zh-cn_topic_0000001312391781_term4363218112215"></a><a name="zh-cn_topic_0000001312391781_term4363218112215"></a>Atlas 推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p85307521410"><a name="p85307521410"></a><a name="p85307521410"></a>2002</p>
    </td>
    </tr>
    <tr id="row653005218416"><td class="cellrowborder" valign="top" width="49.59%" headers="mcps1.2.3.1.1 "><p id="p053055274116"><a name="p053055274116"></a><a name="p053055274116"></a><span id="ph7530652174110"><a name="ph7530652174110"></a><a name="ph7530652174110"></a><term id="zh-cn_topic_0000001312391781_term71949488213"><a name="zh-cn_topic_0000001312391781_term71949488213"></a><a name="zh-cn_topic_0000001312391781_term71949488213"></a>Atlas 训练系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.41%" headers="mcps1.2.3.1.2 "><p id="p155301052144110"><a name="p155301052144110"></a><a name="p155301052144110"></a>1001</p>
    </td>
    </tr>
    </tbody>
    </table>

    以下为通过\_\_NPU\_ARCH\_\_控制在不同AI处理器上算子输出值舍入模式的示例。

    ```
    __aicore__ static inline void CopyOut(uint64_t mulLen)
    {
    #if __NPU_ARCH__ == 2002
        Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, mulLen); // CAST_NONE表示舍入模式在转换有精度损失时使用CAST_RINT模式，在不涉及精度损失时不进行舍入
    #elif __NPU_ARCH__ == 2201
        Cast(dstLocal, srcLocal, RoundMode::CAST_RINT, mulLen); // CAST_RINT表示舍入模式为四舍六入五成双舍入
    #endif
        event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
        CommonCopyOut<float>(dstLocal, mulLen);  // 拷贝LocalTensor至GlobalTensor
    }
    ```

-   **ASCEND\_IS\_AIV、ASCEND\_IS\_AIC**

    ASCEND\_IS\_AIV和ASCEND\_IS\_AIC是通过C++宏实现的条件判断语句，用于在\_\_aicore\_\_修饰的函数中实现代码的条件编译。基于分离模式（AIC核和AIV核分离）开发融合算子时，算子逻辑中同时涉及AIV核和AIC核的处理逻辑，并需要进行核间同步，此时需要通过ASCEND\_IS\_AIV/ ASCEND\_IS\_AIC进行AIV和AIC核代码的隔离。

    >[!NOTE]说明 
    >当使用高阶API Matmul时，其内部已通过REGIST\_MATMUL\_OBJ宏方式实现了AIV与AIC核代码的隔离，用户无需再使用该宏进行处理。

    以MatmulNzCustom算子为例，该算子在分离模式下需要分别在AIV核和AIC核上实现不同的逻辑。具体而言，AIV核负责将矩阵数据搬入Unified Buffer，完成数据的重排（将矩阵数据转换为NZ格式），并将其写入Global Memory。而AIC核则直接从Global Memory读取已经重排好的NZ格式数据，并执行矩阵乘法（Matmul）计算。由于AIV核和AIC核的代码逻辑不同，需要通过ASCEND\_IS\_AIV和ASCEND\_IS\_AIC宏进行代码隔离，确保在编译时分别生成适用于AIV核和AIC核的代码。

    示例伪码如下：

    ```
    template <typename AType, typename BType, typename CType, typename BiasType>
    __aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::Process(AscendC::TPipe *pipe)
    {
        // 利用AIV核的Vector计算单元实现ND2NZ格式转换。如下代码中MatrixBtoNZ为将B矩阵进行ND2NZ格式转换的函数。
        if ASCEND_IS_AIV {
            pipe->InitBuffer(ubBuf, TOTAL_UB_SIZE);
            MatrixBtoNZ<typename B_TYPE::T>(tempGM,
                bGMNZ,
                tiling,
                isTransB,
                ubBuf,
                tiling.baseK,
                tiling.baseN);  // Vector侧实现的ND2NZ函数
            SyncAll();
            // AIC核和AIV核同步
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
            return;
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(0x4);   // 等待AIV核完成ND2NZ格式转换
        }
        ... ...
        // 设置左矩阵A、右矩阵B、Bias。
        matmulObj.SetTail(tailM, tailN);
        matmulObj.SetTensorA(aGlobal, false);
        matmulObj.SetTensorB(bGlobal, false);
        if (tiling.isBias) {
            matmulObj.SetBias(biasGlobal);
        }
        // 完成矩阵乘操作
        matmulObj.IterateAll(cGlobal);
        // 结束矩阵乘操作
        matmulObj.End();
    }
    ```

-   **ASCENDC\_CUBE\_ONLY**

    ASCENDC\_CUBE\_ONLY是通过C++宏实现的条件判断语句，用于在\_\_aicore\_\_修饰的函数中实现代码的条件编译。

    基于分离模式开发非融合算子时，在只有矩阵计算的算子场景下，可以通过设置ASCENDC\_CUBE\_ONLY，使能纯Cube模式完成Matmul计算，减少消息通信的性能开销，提升算子性能。

    >[!CAUTION]注意 
    >ASCENDC\_CUBE\_ONLY宏必须在\#include "lib/matmul\_intf.h"之前设置。

    以matmul\_custom算子为例，高阶API Matmul默认使用MIX模式，即用户从AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。这套消息处理机制会带来额外的Scalar性能开销。相较于MIX模式，纯Cube模式可以直接跳过消息通信框架，完成Matmul计算，提升算子性能。

    示例伪码如下：

    ```
    #define ASCENDC_CUBE_ONLY
    #include "lib/matmul_intf.h"
    
    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
    ```

## 函数执行空间限定符<a name="section1074418132518"></a>

函数执行空间限定符（Function Execution Space Qualifier）指示函数是在Host侧执行还是在Device侧执行，以及它是否可从Host侧或Device侧调用。

-   **\_\_global\_\_**

    \_\_global\_\_执行空间限定符声明一个Kernel函数。Kernel函数有如下性质：在Device上执行；只能被Host侧函数调用；\_\_global\_\_只是表示这是Device侧函数的入口，并不表示具体的设备类型，具体的设备类型由\_\_aicore\_\_标记。具有如下使用约束：

    -   一个\_\_global\_\_函数必须返回void类型，并且不能是class的成员函数。
    -   主机侧调用\_\_global\_\_函数必须使用<<<\>\>\>异构调用语法。
    -   \_\_global\_\_的调用是异步的，意味着函数返回，并不表示kernel函数在device侧已经执行完成，如果需要同步，需要使用Runtime同步接口显式同步，如**aclrtSynchronizeStream**接口。

-   **\_\_aicore\_\_**

    \_\_aicore\_\_执行空间限定符声明一个函数，它具有如下属性：

    -   在Device侧执行
    -   只能被\_\_global\_\_函数，或者其他\_\_aicore\_\_函数调用

    ```
    // Only callable from device functions with same kind
    // of execution space
    __aicore__ void bar() {}
    
    // Define a kernel function execute on AI Core device
    __global__ __aicore__ void foo() {
      bar(); // OK.
    }
    ```

-   **\_\_host\_\_**

    \_\_host\_\_执行空间限定符声明一个函数，它具有如下属性：

    -   只能在Host侧执行
    -   只能被Host侧函数调用
    -   \_\_global\_\_ 和\_\_host\_\_不能一起使用

    \_\_host\_\_限定符是可选项，无函数执行空间限定符定义的函数，默认是host函数。

    ```
    __aicore__ int f() {}
    
    // defines a host side function
    int foo() {}
    
    // defines a host side function
    __host__ int bar() {
      f();     // Error.
      foo();   // OK.
    }
    
    // Error.
    __global__ __host__ void kfunc() {}
    ```

-   **\_\_aicpu\_\_**

    AI CPU函数执行空间限定符\_\_aicpu\_\_用于指示函数是否为AI CPU Kernel函数，它具有如下属性：

    -   在Device侧执行且只能被Host侧函数调用，因此必须与\_\_global\_\_同时声明。
    -   一个\_\_global\_\_ \_\_aicpu\_\_函数不能是void返回类型，并且入参只能是一个指针。
    -   一个\_\_global\_\_ \_\_aicpu\_\_函数不能在.asc文件中进行定义，只能声明，且需要使用extern。
    -   Host侧调用\_\_global\_\_ \_\_aicpu\_\_函数时必须使用<<<\>\>\>异构调用语法，输入的函数入参在入参指针的基础上需要输入从指针中读取的数据大小。
    -   \_\_global\_\_的调用是异步的，意味着函数返回，并不表示kernel函数在Device侧已经执行完成，如果需要同步，需要使用Runtime同步接口显式同步，如**aclrtSynchronizeStream**接口

    ```
    // Define a AI CPU kernel function in AI CPU device file
    __aicpu__ void foo() {} // Error, single __aicpu__ identifier without _global__
    __global__ void foo() {} // Error, single __global__ identifier without __aicpu__ 
    __global__ __aicpu__  void foo() {} // Error, return type is void
    __global__ __aicpu__  int foo(void *a) {} // OK 
    __global__ __aicpu__  int foo(int a) {} // Error, input param is not pointer
    __global__ __aicpu__  int foo(void *a, void *b) {} // Error, input param num is not one
    ```

    ```
    // Declare a AI CPU kernel function in .asc file
    extern __global__ __aicpu__ uint32_t hello_world(void *args);// OK
    }
    ```

-   **\_\_inline\_\_**

    \_\_inline\_\_限定符声明一个函数，它具有如下属性：

    -   标识Device侧函数强制内联，可以减少函数频繁调用产生的指令压栈、出栈的开销，但可能会导致算子二进制增加。
    -   和C++函数修饰符inline的主要区别是Device侧\_\_inline\_\_是强制内联，C++的inline则是根据编译器优化选择性内联。
    -   AI Core对函数嵌套深度有限制，一般推荐嵌套深度不超过4层。使用强制内联可以减少调用层次。

-   **\_\_cube\_\_**

    标识该核函数仅在Cube核执行。针对耦合模式的硬件架构，该修饰符不生效。

    ```
    extern "C" __global__ __cube__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        KernelMmad op;
        op.Init(a, b, c);
        op.Process();
    }
    ```

-   **\_\_vector\_\_**

    标识该核函数仅在Vector核执行。针对耦合模式的硬件架构，该修饰符不生效。

    ```
     __vector__ __global__ __aicore__ void add_custom(){}
    ```

-   **\_\_mix\_\_\(cube, vec\)**

    标识该核函数同时在Cube核和Vector核上执行。\(cube, vec\)分别表示核函数启动的Cube核和Vector核的配比，支持的配比为\(1, 0\)，\(0, 1\)，\(1, 1\)， \(1, 2\)。针对耦合模式的硬件架构，该修饰符不生效。

-   **\_\_schedmode\_\_\(mode\)**

    标识该核函数的执行调度模式。如下图所示：

    -   mode = 0 : normal mode，尽可能选择空闲物理核下发执行核函数，若空闲物理核数无法满足当前核函数的需要，没有下发的部分等待核心空闲后执行。此时OP1和OP2算子会存在交叠执行（overlap）的情况。

    -   mode = 1 : batch mode，在下发核函数时先进行判断，若空闲物理核数无法满足当前核函数的需要，则等待至空闲物理核数满足该核函数所需要的所有物理核时，同时下发执行，OP1和OP2的执行被切分（split）开，不会出现交叠执行的情况。

    ![](../../figures/batchmode资料.png)

    在多流并发场景，多算子并行执行时，若执行总核数超过最大物理核数，且多个算子逻辑使用SyncALL等核间同步接口时，建议设置mode为1，防止多个算子之间互相等待空闲核调度，导致死锁。默认值mode为0。

    ```
     __schedmode__(1) __global__ __mix__(1, 2) void OP1() // OP1使用了SyncAll接口，且存在多流并发的可能，需要设置batch mode（mode 1）
    {
        AscendC::SyncAll();
        ....    
    }
     __schedmode__(1) __global__ __mix__(1, 2) void OP2() // OP2使用了SyncAll接口，且存在多流并发的可能，需要设置batch mode（mode 1）
    {
        AscendC::SyncAll();
        ....    
    } 
    __schedmode__(0) __global__ __vector__ void OP3() {...} // OP3没有使用SyncAll接口，可以设置为normal mode(mode 0)，按照正常规则执行算子。
    or
    __global__ __vector__ void OP3() {...} // 不设置__schedmode__，默认为normal mode。
    ```

## 函数标记宏<a name="section192521344610"></a>

-   \_\_simd\_vf\_\_

    函数标记宏，用于标记SIMD VF入口函数，函数无返回值。使用asc\_vf\_call调用SIMD VF入口函数，启动VF子任务。

    ```
    __simd_vf__ inline void KernelAdd(__ubuf__ float* x, __ubuf__ float* y, __ubuf__ float* z)
    ```

    \_\_simd\_vf\_\_标记的SIMD VF有以下入参约束：

    -   支持指针传参（Pass-by-Pointer），指针变量必须用\_\_ubuf\_\_地址空间限定符修饰。
    -   不支持引用传参（Pass-by-Reference）。
    -   不支持函数指针传参，函数对象。

        \_\_simd\_vf\_\_使用的示例如下：

        ```
        __simd_vf__ inline void simd_adds(__ubuf__ float *output, __ubuf__ float *input,
            uint32_t count, uint16_t one_repeat_size, uint16_t repeat_times)
        {
            AscendC::Reg::RegTensor<float> src_reg0;
            AscendC::Reg::RegTensor<float> dst_reg0;
            // asc_update_mask() will be supported later.
            // init MaskReg with the count of all numbers.
            AscendC::Reg::MaskReg mask_reg = AscendC::Reg::UpdateMask<float>(count);
            for (uint16_t i = 0; i < repeat_times; i++) {
                // asc_load, asc_adds and asc_store will be supported later.
                // load data from UB to RegTensor.
                AscendC::Reg::LoadAlign(src_reg0, input + i * one_repeat_size);
                AscendC::Reg::Adds(dst_reg0, src_reg0, 1.0f, mask_reg);
                // store data from RegTensor to UB.
                AscendC::Reg::StoreAlign(output + i * one_repeat_size, dst_reg0, mask_reg);
            }
        }
        ```

-   \_\_simd\_callee\_\_

    函数标记宏，函数可以有返回值，允许被SIMD VF入口函数或其他非入口函数调用。

    ```
    __simd_callee__ inline float add(float x, float y)
    ```

## 地址空间限定符<a name="section1624210295308"></a>

AI Core具备多级独立片上存储，各个地址空间独立编址，具备各自的访存指令，根据架构差异，有些存储空间具备统一地址空间（Generic Address Space），有些则没有。设备侧编程基于语法扩展允许地址空间作为合法的类型限定符，以提供针对不同地址空间的访问能力和地址空间合法性检查。

**表 2**  地址空间映射关系

<a name="table425403455214"></a>
<table><thead align="left"><tr id="row027120348524"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1227193465211"><a name="p1227193465211"></a><a name="p1227193465211"></a>地址空间限定符</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p327111349525"><a name="p327111349525"></a><a name="p327111349525"></a>AI Core物理存储空间</p>
</th>
</tr>
</thead>
<tbody><tr id="row162721534115215"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p92721034185210"><a name="p92721034185210"></a><a name="p92721034185210"></a>__gm__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10272143417521"><a name="p10272143417521"></a><a name="p10272143417521"></a>设备侧内存GM</p>
</td>
</tr>
<tr id="row72726341527"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4272634135214"><a name="p4272634135214"></a><a name="p4272634135214"></a>__ubuf__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1427214347521"><a name="p1427214347521"></a><a name="p1427214347521"></a>Vector Unified Buffer</p>
</td>
</tr>
<tr id="row62721334195214"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p182722346528"><a name="p182722346528"></a><a name="p182722346528"></a>__ca__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p102721034145219"><a name="p102721034145219"></a><a name="p102721034145219"></a>Cube L0A Buffer</p>
</td>
</tr>
<tr id="row182721834175211"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14272434145216"><a name="p14272434145216"></a><a name="p14272434145216"></a>__cb__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p327293455219"><a name="p327293455219"></a><a name="p327293455219"></a>Cube L0B Buffer</p>
</td>
</tr>
<tr id="row1027253413523"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12272434195211"><a name="p12272434195211"></a><a name="p12272434195211"></a>__cc__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1927223411527"><a name="p1927223411527"></a><a name="p1927223411527"></a>Cube L0C Buffer</p>
</td>
</tr>
<tr id="row2272034135217"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p92721347529"><a name="p92721347529"></a><a name="p92721347529"></a>__cbuf__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p727233495213"><a name="p727233495213"></a><a name="p727233495213"></a>Cube L1 Buffer</p>
</td>
</tr>
<tr id="row14137103317189"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4798642111813"><a name="p4798642111813"></a><a name="p4798642111813"></a>__fbuf__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p117981342201811"><a name="p117981342201811"></a><a name="p117981342201811"></a>Fixpipe Buffer</p>
</td>
</tr>
<tr id="row127473841817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14798942101818"><a name="p14798942101818"></a><a name="p14798942101818"></a>__ssbuf__</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p07981642171819"><a name="p07981642171819"></a><a name="p07981642171819"></a>SSBuffer</p>
</td>
</tr>
</tbody>
</table>

地址空间限定符可以在变量声明中使用，用于指定对象分配的区域。如果对象的类型被地址空间名称限定，那么该对象将被分配在指定的地址空间中。同样地，对于指针，指向的类型可以通过地址空间进行限定，以指示所指向的对象所在的地址空间。

```
// declares a pointer p in the __gm__ address space that
// points to an object(has int type) in the __gm__ address space
__gm__ int *p;

__global__ __aicore__ void foo(...)
{
  // declares an array of 4 floats in the private address space.
  float x[4];
}
```

地址空间限定符不能用于非指针返回类型，非指针函数参数，函数类型，同一个类型上不允许使用多个地址空间限定符。

```
// OK.
__aicore__ int f() {...}

// Error. Address space qualifier cannot be used with a non-pointer return type.
__ubuf__ int f() { ... }

// OK. Address space qualifier can be used with a pointer return type.
__ubuf__ int *f() { ... }

// Error. Multiple address spaces specified for a type.
__ubuf__ __gm__ int i;

// OK. The first address space qualifies the object pointed to and the second
// qualifies the pointer.
__ubuf__ int * __gm__ ptr;
```

>[!NOTE]说明 
>**重要**：不同地址空间指针的大小可能不同。例如，不能认为 sizeof\(\_\_gm\_\_ int \*\)总是等于sizeof\(\_\_ubuf\_\_ int \*\)，譬如编译器或许可能在某些系统上以32bit存储\_\_ubuf\_\_指针。

-   **private地址空间**

    private地址空间是大多数变量的默认地址空间，特别是局部变量。

    ```
    // m is in a specific kernel parameter address space, 
    // it's physical location is implementation determined.
    __global__ __aicore__ void foo(int m) {
      // OK. i is an int variable allocated in private address space
      int i;
    }
    
    __aicore__ void bar(int k) { //OK. k is in private address space
      // OK. i is an int variable allocated in private address space
      int i; 
    }
    ```

-   **\_\_gm\_\_地址空间**

    \_\_gm\_\_地址空间限定符用来表示分配于设备侧全局内存的对象，全局内存对象可以声明为标量、用户自定义结构体的指针。

    ```
    __gm__ int *var; // var point to an array of int elements
    
    typedef struct {
        float a[3];
        int b[2];
    } foo_t;
    
    __gm__ foo_t *info; // info point to an array of foo_t elements
    ```

-   **\_\_ubuf\_\_地址空间**

    \_\_ubuf\_\_地址空间用来描述存储于AI Core核内UB存储空间的变量。

    ```
    __global__ __aicore__ void foo() {
      // ptr is in private address space, point to __ubuf__
      __ubuf__ int *ptr;
    }
    ```

-   **\_\_ca\_\_, \_\_cb\_\_, \_\_cc\_\_, \_\_cbuf\_\_地址空间**

    上述几个地址空间主要用于特定的DMA指令访问，不具备标量直接访问能力。

    ```
    class ObjTy{
      ObjTy(){...}
      void print(){...}
    
    private:
      int a;
      int b;
    };
    
    __global__ __aicore__ 
    void foo(__ca__ int * ptr) { // Error. Cannot have __ca__
                                 // qualifier in kernel arguments
      // OK
      __ca__ int *ptr; 
    }
    ```

## 内置常量<a name="section784531219338"></a>

<a name="table878553753515"></a>
<table><thead align="left"><tr id="row578553718354"><th class="cellrowborder" valign="top" width="16.14838516148385%" id="mcps1.1.4.1.1"><p id="p1378573713352"><a name="p1378573713352"></a><a name="p1378573713352"></a>常量名</p>
</th>
<th class="cellrowborder" valign="top" width="25.847415258474154%" id="mcps1.1.4.1.2"><p id="p1378553763512"><a name="p1378553763512"></a><a name="p1378553763512"></a>取值</p>
</th>
<th class="cellrowborder" valign="top" width="58.00419958004199%" id="mcps1.1.4.1.3"><p id="p127858377356"><a name="p127858377356"></a><a name="p127858377356"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row15785103718358"><td class="cellrowborder" valign="top" width="16.14838516148385%" headers="mcps1.1.4.1.1 "><p id="p1678553710357"><a name="p1678553710357"></a><a name="p1678553710357"></a>constexpr int32_t g_coreType</p>
</td>
<td class="cellrowborder" valign="top" width="25.847415258474154%" headers="mcps1.1.4.1.2 "><a name="ul107856379351"></a><a name="ul107856379351"></a><ul id="ul107856379351"><li>AscendC::AIC</li><li>AscendC::AIV</li></ul>
</td>
<td class="cellrowborder" valign="top" width="58.00419958004199%" headers="mcps1.1.4.1.3 "><p id="p778516378353"><a name="p778516378353"></a><a name="p778516378353"></a>常量值由框架自动设置，AIC核下，配置为AscendC::AIC，AIV核下，配置为AscendC::AIV。</p>
<p id="p278616379351"><a name="p278616379351"></a><a name="p278616379351"></a>可以通过对该常量值的判断，来实现了AIV与AIC核代码的区分和隔离。功能等同于直接使用ASCEND_IS_AIV、ASCEND_IS_AIC。</p>
</td>
</tr>
</tbody>
</table>

## 内置变量<a name="section199434523343"></a>

<a name="table169229153513"></a>
<table><thead align="left"><tr id="row769289183515"><th class="cellrowborder" valign="top" width="15.988401159884011%" id="mcps1.1.4.1.1"><p id="p1669269163510"><a name="p1669269163510"></a><a name="p1669269163510"></a>变量名</p>
</th>
<th class="cellrowborder" valign="top" width="25.977402259774017%" id="mcps1.1.4.1.2"><p id="p116921597358"><a name="p116921597358"></a><a name="p116921597358"></a>对应API</p>
</th>
<th class="cellrowborder" valign="top" width="58.03419658034197%" id="mcps1.1.4.1.3"><p id="p14692294358"><a name="p14692294358"></a><a name="p14692294358"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row06921195350"><td class="cellrowborder" valign="top" width="15.988401159884011%" headers="mcps1.1.4.1.1 "><p id="p156921796359"><a name="p156921796359"></a><a name="p156921796359"></a>block_num</p>
</td>
<td class="cellrowborder" valign="top" width="25.977402259774017%" headers="mcps1.1.4.1.2 "><p id="p369213993514"><a name="p369213993514"></a><a name="p369213993514"></a>GetBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="58.03419658034197%" headers="mcps1.1.4.1.3 "><p id="p969217912357"><a name="p969217912357"></a><a name="p969217912357"></a>当前任务配置的核数，用于代码内部的多核逻辑控制等。</p>
</td>
</tr>
<tr id="row1669214993515"><td class="cellrowborder" valign="top" width="15.988401159884011%" headers="mcps1.1.4.1.1 "><p id="p1769219913351"><a name="p1769219913351"></a><a name="p1769219913351"></a>block_idx</p>
</td>
<td class="cellrowborder" valign="top" width="25.977402259774017%" headers="mcps1.1.4.1.2 "><p id="p116921990355"><a name="p116921990355"></a><a name="p116921990355"></a>GetBlockIdx</p>
</td>
<td class="cellrowborder" valign="top" width="58.03419658034197%" headers="mcps1.1.4.1.3 "><p id="p169211933512"><a name="p169211933512"></a><a name="p169211933512"></a>当前核的索引，用于代码内部的多核逻辑控制及多核偏移量计算等。</p>
</td>
</tr>
</tbody>
</table>

通常，建议用户使用内置变量对应的API获取所需值，不建议用户直接使用内置变量。因为内置变量反映的是单个硬件资源的配置信息，对于软件栈整合硬件资源、扩展硬件的功能，内置变量的值与实际语义可能不符。

例如，在Atlas 推理系列产品中，当启用KERNEL\_TYPE\_MIX\_VECTOR\_CORE时，算子会同时运行在AI Core和Vector Core上。此时，block\_idx在这两种核心上都是从0开始计数，用户无法直接通过block\_idx来切分数据和控制多核逻辑。而GetBlockIdx在Vector Core上对block\_idx增加偏移量（AI Core的block\_num），从而保证返回的值能够正确反映多核环境下的实际逻辑。

## SIMD与SIMT混合编程场景<a name="section952111671314"></a>

SIMD与SIMT混合编程场景中，SIMT VF的入口函数使用\_\_simt\_vf\_\_进行标识，通过在SIMD的\_\_aicore\_\_函数中使用[asc\_vf\_call](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/asc_vf_call.md)调用SIMT入口函数。被SIMT VF入口函数调用的函数使用\_\_simt\_callee\_\_进行标识。

-   \_\_simt\_vf\_\_ <a name="li611618392141"></a>

    函数标记宏，用于标记SIMT VF入口函数，函数无返回值。使用asc\_vf\_call接口调用SIMT VF入口函数，启动VF子任务。

    ```
    __simt_vf__ inline void KernelAdd(__gm__ float* x, __gm__ float* y, __gm__ float* z)
    ```

    \_\_simt\_vf\_\_标记的SIMT VF函数支持的参数类型如下。\_\_simt\_vf\_\_的使用示例请参考[编程示例](../编程模型/AI-Core-SIMD编程/SIMD与SIMT混合编程.md#section776244992018)。

    -   指针类型：\_\_ubuf\_\_ \*、\_\_gm\_\_ \*；
    -   标量类型：bool、int8\_t、uint8\_t、int16\_t、uint16\_t、half、bfloat16、int32\_t、uint32\_t、float、int64\_t、uint64\_t。

-   \_\_simt\_callee\_\_

    函数标记宏，用于标记SIMT VF非入口函数，函数可以有返回值，允许被SIMT VF入口函数或其他非入口函数调用。

    ```
    __simt_callee__ inline float add(float x, float y)
    ```

Ascend C为SIMT编程、SIMD与SIMT混合编程提供了布尔、整形、浮点型的标量数据类型和短向量数据类型，提供了用于表达线程块、线程网格三维信息的内置变量。 关于内置数据格式的详细说明请参见[内置数据类型](SIMT-BuiltIn关键字.md#section1835494915576)，内置变量请参见[内置变量](SIMT-BuiltIn关键字.md#section13165113520576)。

