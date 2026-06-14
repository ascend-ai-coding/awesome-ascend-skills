> **原始文档路径**: `asc-devkit/docs/guide/编程指南/编程模型/AI-Core-SIMD编程/SIMD与SIMT混合编程.md`
# SIMD与SIMT混合编程<a name="ZH-CN_TOPIC_0000002477823926"></a>

## 抽象硬件架构<a name="section3163536195018"></a>

AI Core上SIMD（Single Instruction Multiple Data，单指令多数据）与SIMT（Single Instruction Multiple Thread，单指令多线程）混合编程，结合了SIMD多数据并行计算能力与SIMT离散访存的优势，实现向量级并行与线程级并行的高效协同。当前仅支持Ascend 950PR/Ascend 950DT。

整个执行过程以Vector Function（VF）为基本调度单位，VF为一个基本函数块。SIMD与SIMT混合编程支持在同一算子中灵活切换SIMD与SIMT执行方式，两种不同类型的VF可以快速切换，每个VF代表一个独立的计算任务片段，通常对应算子中的一段可并行处理的逻辑，从而在性能、能效与开发效率之间取得更优平衡。在SIMD与SIMT混合编程中：

-   一个核函数中可包含多个VF。
-   每个VF可以选择使用SIMD或SIMT方式进行编程。
-   不同类型VF之间可以快速切换，切换粒度为单个VF。
-   在同一时刻，一个AIV核只能执行SIMT或SIMD任务。

在SIMD与SIMT混合编程中，SIMT能够简化复杂算子与不规则控制流的开发；而SIMD基于向量寄存器与指令，实现高效的数据并行处理，即单指令处理多数据，提升每周期的吞吐量。SIMD与SIMT混合编程支持开发者根据算子特征进行精细化映射：规则的逐元素elementwise操作通过SIMD获得高带宽和高算力利用率，不规则或包含分支的计算通过SIMT来缓解发散和控制复杂度。在系统层面，这有利于提高硬件利用率和能效；同时，也更便于进行算子融合和数据复用等优化。同一个算子中既包含SIMD擅长的连续规整计算，也包含SIMT擅长的离散访问等任务，从而在同一算子中同时利用SIMD和SIMT的优势。

如[图1](SIMD与SIMT混合编程.md#fig17744131364413)所示，SIMD和SIMT的内部执行流程为：

-   Scalar计算单元将VF发射到Vector Function Queue中。
-   SIMD与SIMT混合编程的工作模式以VF为粒度进行切换，SIMT工作模式下的Data Cache（DCache）数据在VF切换时会被保留。
-   SIMD和SIMT之间的VF串序执行，同一时刻，一个AIV核仅能执行SIMD或SIMT任务。
-   VF执行完成后，结果数据被写回Unified Buffer或Global Memory。

**图 1** SIMD与SIMT混合编程硬件架构图<a name="fig17744131364413"></a>  
![](../../../figures/SIMD与SIMT混合编程硬件架构图.png "SIMD与SIMT混合编程硬件架构图")

SIMD与SIMT编程存在以下差异：

**表 1**  SIMD与SIMT核心差异点

<a name="table1651312912417"></a>
<table><thead align="left"><tr id="row151314292249"><th class="cellrowborder" valign="top" width="15.981598159815983%" id="mcps1.2.4.1.1"><p id="p17513182916244"><a name="p17513182916244"></a><a name="p17513182916244"></a>维度</p>
</th>
<th class="cellrowborder" valign="top" width="42.22422242224223%" id="mcps1.2.4.1.2"><p id="p1851392911242"><a name="p1851392911242"></a><a name="p1851392911242"></a>SIMD</p>
</th>
<th class="cellrowborder" valign="top" width="41.7941794179418%" id="mcps1.2.4.1.3"><p id="p451382915241"><a name="p451382915241"></a><a name="p451382915241"></a>SIMT</p>
</th>
</tr>
</thead>
<tbody><tr id="row5513182914244"><td class="cellrowborder" valign="top" width="15.981598159815983%" headers="mcps1.2.4.1.1 "><p id="p14513029132412"><a name="p14513029132412"></a><a name="p14513029132412"></a>编程模型</p>
</td>
<td class="cellrowborder" valign="top" width="42.22422242224223%" headers="mcps1.2.4.1.2 "><p id="p35132294246"><a name="p35132294246"></a><a name="p35132294246"></a>单指令多数据（SIMD），基于向量寄存器与向量指令。</p>
</td>
<td class="cellrowborder" valign="top" width="41.7941794179418%" headers="mcps1.2.4.1.3 "><p id="p175147295243"><a name="p175147295243"></a><a name="p175147295243"></a>单指令多线程（SIMT），以线程为单位并行执行。</p>
</td>
</tr>
<tr id="row16514172912412"><td class="cellrowborder" valign="top" width="15.981598159815983%" headers="mcps1.2.4.1.1 "><p id="p15514102917245"><a name="p15514102917245"></a><a name="p15514102917245"></a>数据搬运方式</p>
</td>
<td class="cellrowborder" valign="top" width="42.22422242224223%" headers="mcps1.2.4.1.2 "><p id="p194616581337"><a name="p194616581337"></a><a name="p194616581337"></a>通过显式Load/Store将数据从<span id="ph43501137162513"><a name="ph43501137162513"></a><a name="ph43501137162513"></a>Unified Buffer</span>搬运到向量寄存器。</p>
<p id="p1998118518347"><a name="p1998118518347"></a><a name="p1998118518347"></a>不支持直接从<span id="ph721809124812"><a name="ph721809124812"></a><a name="ph721809124812"></a>Global Memory</span>搬运数据到SIMD的向量寄存器。</p>
</td>
<td class="cellrowborder" valign="top" width="41.7941794179418%" headers="mcps1.2.4.1.3 "><p id="p55141929162410"><a name="p55141929162410"></a><a name="p55141929162410"></a>支持直接读写<span id="ph1410121110487"><a name="ph1410121110487"></a><a name="ph1410121110487"></a>Global Memory</span>或<span id="ph587461554814"><a name="ph587461554814"></a><a name="ph587461554814"></a>Unified Buffer</span>中的数据。</p>
</td>
</tr>
<tr id="row1216192522716"><td class="cellrowborder" valign="top" width="15.981598159815983%" headers="mcps1.2.4.1.1 "><p id="p141612253271"><a name="p141612253271"></a><a name="p141612253271"></a>适用场景</p>
</td>
<td class="cellrowborder" valign="top" width="42.22422242224223%" headers="mcps1.2.4.1.2 "><p id="p101611725162712"><a name="p101611725162712"></a><a name="p101611725162712"></a>规则、连续的逐元素操作（elementwise），如卷积、矩阵乘法、向量操作等。</p>
</td>
<td class="cellrowborder" valign="top" width="41.7941794179418%" headers="mcps1.2.4.1.3 "><p id="p111612025142713"><a name="p111612025142713"></a><a name="p111612025142713"></a>不规则、含分支、动态访问等复杂逻辑，如注意力机制、稀疏操作等。</p>
</td>
</tr>
</tbody>
</table>

尽管SIMD与SIMT在编程模型和执行机制上有显著差异，但在硬件层面上共享以下关键资源：

-   SIMT VF与SIMD VF共享ICache（Instruction Cache），提升指令预取效率。
-   SIMT与SIMD共享Vector ALU单元，基于该单元执行的功能和性能基本相同。
-   Unified Buffer内存空间中一部分为SIMT与SIMD共享空间，另一部分作为SIMT的Data Cache。

## 内存层级<a name="section1676401713532"></a>

在SIMD与SIMT混合编程场景，可以访问多种内存空间，下表汇总了常见内存类型的作用域及其生命周期。
<a name="table193141614125513"></a>
<table><thead align="left"><tr id="row431461485515"><th class="cellrowborder" valign="top" width="14.018598140185981%" id="mcps1.1.5.1.1"><p id="p12314161495516"><a name="p12314161495516"></a><a name="p12314161495516"></a>内存类型</p>
</th>
<th class="cellrowborder" valign="top" width="30.23888376162383762%" id="mcps1.1.5.1.2"><p id="p16314171435516"><a name="p16314171435516"></a><a name="p16314171435516"></a>线程作用域</p>
</th>
<th class="cellrowborder" valign="top" width="17.928207179282072%" id="mcps1.1.5.1.3"><p id="p203141214165520"><a name="p203141214165520"></a><a name="p203141214165520"></a>生命周期</p>
</th>
<th class="cellrowborder" valign="top" width="17.514818518148186%" id="mcps1.1.5.1.4"><p id="p17401742185918"><a name="p17401742185918"></a><a name="p17401742185918"></a>物理位置</p>
</th>
<th class="cellrowborder" valign="top" width="51.814818518148186%" id="mcps1.1.5.1.4"><p id="p17401742185918"><a name="p17401742185918"></a><a name="p17401742185918"></a>特点</p>
</th>
</tr>
</thead>
<tbody><tr id="row831491414559"><td class="cellrowborder" valign="top" width="14.018598140185981%" headers="mcps1.1.5.1.1 "><p id="p1780259331"><a name="p1780259331"></a><a name="p1780259331"></a>全局内存</p>
</td>
<td class="cellrowborder" valign="top" width="30.23888376162383762%" headers="mcps1.1.5.1.2 "><p id="p123149149558"><a name="p123149149558"></a><a name="p123149149558"></a>所有核的核函数</p>
</td>
<td class="cellrowborder" valign="top" width="17.928207179282072%" headers="mcps1.1.5.1.3 "><p id="p2314614185513"><a name="p2314614185513"></a><a name="p2314614185513"></a>应用程序</p>
</td>
<td class="cellrowborder" valign="top" width="17.514818518148186%" headers="mcps1.1.5.1.4 "><p id="p999075216140"><a name="p999075216140"></a><a name="p999075216140"></a>Device</p>
</td>
<td class="cellrowborder" valign="top" width="51.814818518148186%" headers="mcps1.1.5.1.4 "><p id="p999075216141"><a name="p999075216140"></a><a name="p999075216140"></a>大容量，低带宽</p>
</td>
</tr>
<tr id="row6314181413550"><td class="cellrowborder" valign="top" width="14.018598140185981%" headers="mcps1.1.5.1.1 "><p id="p1437313147519"><a name="p1437313147519"></a><a name="p1437313147519"></a>共享内存</p>
</td>
<td class="cellrowborder" valign="top" width="30.23888376162383762%" headers="mcps1.1.5.1.2 "><p id="p731481435515"><a name="p731481435515"></a><a name="p731481435515"></a>单核核函数</p>
</td>
<td class="cellrowborder" valign="top" width="17.928207179282072%" headers="mcps1.1.5.1.3 "><p id="p5314151445515"><a name="p5314151445515"></a><a name="p5314151445515"></a>核函数</p>
</td>
<td class="cellrowborder" valign="top" width="17.514818518148186%" headers="mcps1.1.5.1.4 "><p id="p7740204211595"><a name="p7740204211595"></a><a name="p7740204211595"></a>Vector Core</p>
</td>
<td class="cellrowborder" valign="top" width="51.814818518148186%" headers="mcps1.1.5.1.4 "><p id="p999075216141"><a name="p999075216140"></a><a name="p999075216140"></a>小容量，高带宽</p>
</td>
</tr>
<tr id="row1242419814517"><td  class="cellrowborder" valign="top" width="14.018598140185981%" headers="mcps1.1.5.1.1 "><p id="p12690146153"><a name="p12690146153"></a><a name="p12690146153"></a>SIMT 寄存器</p>
</td>
<td class="cellrowborder" valign="top" width="30.23888376162383762%" headers="mcps1.1.5.1.2 "><p id="p7690561553"><a name="p7690561553"></a><a name="p7690561553"></a>Thread</p>
</td>
<td class="cellrowborder" valign="top" width="17.928207179282072%" headers="mcps1.1.5.1.3 "><p id="p16690961515"><a name="p16690961515"></a><a name="p16690961515"></a>SIMT VF函数</p>
</td>
<td class="cellrowborder" valign="top" width="17.514818518148186%" headers="mcps1.1.5.1.4 "><p id="p36901661456"><a name="p36901661456"></a><a name="p36901661456"></a>Device</p>
</td>
<td class="cellrowborder" valign="top" width="51.814818518148186%" headers="mcps1.1.5.1.4 "><p id="p999075216141"><a name="p999075216140"></a><a name="p999075216140"></a>极小容量，极高带宽</p>
</td>
</tr>
<tr id="row311525417103"><td  class="cellrowborder" valign="top" width="14.018598140185981%" headers="mcps1.1.5.1.1 "><p id="p12690146153"><a name="p12690146153"></a><a name="p12690146153"></a>SIMD 寄存器</p>
</td>
<td  class="cellrowborder" valign="top" width="30.23888376162383762%" headers="mcps1.1.5.1.2 "><p id="p19664352141016"><a name="p19664352141016"></a><a name="p19664352141016"></a>Register File</p>
</td>
<td class="cellrowborder" valign="top" width="17.928207179282072%" headers="mcps1.1.5.1.3 "><p id="p166643525102"><a name="p166643525102"></a><a name="p166643525102"></a>SIMD VF函数</p>
</td>
<td class="cellrowborder" valign="top" width="17.514818518148186%" headers="mcps1.1.5.1.4 "><p id="p1666415522108"><a name="p1666415522108"></a><a name="p1666415522108"></a>Vector Core</p>
</td>
<td class="cellrowborder" valign="top" width="51.814818518148186%" headers="mcps1.1.5.1.4 "><p id="p999075216141"><a name="p999075216140"></a><a name="p999075216140"></a>极小容量，极高带宽</p>
</td>
</tr>
</tbody>
</table>

- 全局内存即Global Memory，存储空间大，带宽低，生命周期与整个应用程序一致，通常用于存储输入输出数据；
- 共享内存是每个AIV核拥有的独立Unified Buffer（UB），生命周期和AIV核函数一致，常作为全局内存的缓存；
- 最靠近Vector计算单元的是寄存器，SIMT和SIMD模式各自拥有一块私有寄存器内存。

整体内存架构如下图所示：

**图 2** SIMD与SIMT混合编程内存模型示意图<a name="fig237411381089"></a>  
![](../../../figures/SIMD与SIMT混合编程内存模型示意图.png "SIMD与SIMT混合编程内存模型示意图")

在SIMT工作模式下，各个内存的工作流程如下：
-   每个线程独立的寄存器和栈，用于存储局部变量。可用寄存器数量与线程块中线程数有关，具体支持情况请见[表5](../../语言扩展层/SIMT-BuiltIn关键字.md#table1715318510594)。
-   线程块内所有线程共享本地内存Unified Buffer（UB），该内存区域由线程块内所有线程共同访问，且其生命周期和线程块一致。
-   所有线程均可通过Data Cache（DCache）访问全局内存（GM），Data Cache是从UB上单独划分出来的一个缓存空间，内存大小可配置范围为32KB到128KB，具体配置方法详见下文[UB内存分配](#section3725125414229)。

在SIMD工作模式下，各个内存的工作流程如下：
-   SIMD的Register File（简称RF）包含多种类型的Reg矢量计算寄存器，用于SIMD VF函数内部存储计算数据，Reg的类型请见[Reg数据类型定义](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/RegTensor.md)。
-   单核内所有VF Reg寄存器共享内存资源UB。
-   在SIMD模式下，不支持直接从全局内存加载数据到Reg矢量计算寄存器，需先将数据从全局内存GM搬运至UB，再通过显式的Load/Store指令，从UB加载到Reg矢量计算寄存器中执行计算操作。

## UB内存分配<a name="section3725125414229"></a>

UB（即Unified Buffer）内存空间总大小为256KB，参考[图3](#fig184031623113510)，按功能划分为四个主要区域，从低地址向高地址依次为静态内存、动态内存、 预留空间 、Data Cache，具体结构如下：

1.  静态内存：从内存的起始地址分配一段指定大小的内存空间，其大小在编译时确定，不可动态修改。

    ```
    // 静态内存通过数组分配，例如：
    __ubuf__ char staticBuf[1024];
    ```

2.  <a name="li688923143920"></a>动态内存：位于静态内存之后，通过[<<<...\>\>\>](#li8828116155217)中参数dynUBufSize指定的动态内存大小空间，可通过以下方式申请使用：

    -   通过TPipe的相关接口申请。
    -   通过LocalMemAllocator的Alloc接口申请。
    -   使用动态数组分配。

        ```
        // 动态内存通过动态数组分配，例如：
        extern __ubuf__ char dynamicBuf[];
        ```

    由于上述三种方法申请动态内存时均从静态内存结束位置之后开始分配，如果同时使用可能会导致地址空间重叠，从而引发未定义行为，因此只能选择其中一种方法进行申请。

3.  预留空间：编译器和Ascend C预留空间，大小固定为8KB。
4.  Data Cache：SIMT专有的Data Cache空间，内存大小必须大于或等于32KB。

>[!NOTE]说明 
>-   DataCache =  UB总大小（256KB） –  静态内存 – 动态内存 – 预留空间\(8KB）
>-   若DataCache小于32KB，会出现校验报错。
>-   在SIMD与SIMT混合编程的场景下，算子内部不能使用全部的Unified Buffer空间，除了预留8KB空间外，还需至少为SIMT预留32KB的Data Cache空间。

**图 3**  UB内存分配图<a name="fig184031623113510"></a>  
![](../../../figures/UB内存分配图.png "UB内存分配图")

## 核函数的定义<a name="section167235919512"></a>

-   核函数定义方式
    -   SIMT VF函数定义：

        定义SIMT VF核函数时，[\_\_launch\_bounds\_\_](../../语言扩展层/SIMT-BuiltIn关键字.md#li23861114618)\(thread\_num\)是可选配置，用于在编译期指定核函数启动的最大线程数，如果不配置thread\_num，thread\_num默认为1024。

        SIMD与SIMT混合编程中SIMT VF核函数定义的[\_\_simt\_vf\_\_](../../语言扩展层/SIMD-BuiltIn关键字.md#li611618392141)、\_\_gm\_\_修饰符需要单独进行标识。关于SIMT VF函数编程的相关约束请参考[附录](../../附录/CPP标准支持/语法限制/函数.md)。

        ```
        __simt_vf__ __launch_bounds__(thread_num) inline void simt_vector_function(__ubuf__ float* input, …)
        ```

    -   SIMD VF函数定义：

        SIMD VF核函数使用\_\_simd\_vf\_\_修饰符进行标识。

        ```
        __simd_vf__ inline void my_kernel(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z);
        ```

        >[!NOTE]注意 
        >**SIMD\_VF和SIMT\_VF的入参只支持PoD（Plain Old Data）数据类型。**
        >-   PoD数据类型：包括基础数据类型（int32\_t、float等）以及这些基本数据类型组成的数组和结构体；不包括构造函数、析构函数、复制构造函数、复制赋值操作符、非静态成员函数或虚函数的类或结构体。

    -   SIMD与SIMT混合编程核函数的定义：

        1.  核函数使用\_\_global\_\_、\_\_aicore\_\_修饰符进行标识。
        2.  核函数的入参和SIMD函数的用法一致。
        3.  在SIMD与SIMT混合编程核函数中调用SIMT VF函数和SIMD VF函数。

        ```
        __global__ __aicore__ void my_kernel(__gm__ float*,…)
        ```

-   <a name="li8828116155217"></a>SIMD与SIMT混合核函数调用方式：

    1.  核函数的调用请参见[核函数](核函数.md)。执行配置由3个参数决定：
        -   numBlocks：设置核函数启用的核数，通过<<<...\>\>\>的方式传入。
        -   dynUBufSize：用于指定动态内存大小。动态内存的申请方式请参见UB内存分配中的[动态内存](#li688923143920)。
        -   stream：类型为aclrtStream，用于维护异步操作的执行顺序，确保在device上按照程序中的代码调用顺序执行。

    2.  开发者需要保证核函数内使用的动态内存大小不超过dynUBufSize，超出会越界访问预留空间或者Data Cache，引发未定义行为。
    3.  可配置的最大动态内存大小 = 256KB - 保留空间（8KB）- 32KB（最小DCache）- 静态内存。

    ```
    kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...)
    ```

## 调用层级<a name="section13964105005215"></a>

-    核函数：使用\_\_global\_\_ \_\_aicore\_\_标识，是Device侧的入口函数，在Host侧可以通过<<<...\>\>\>语法进行调用。
-   \_\_aicore\_\_函数：使用\_\_aicore\_\_标识该函数在Device侧执行。核函数内可以调用\_\_aicore\_\_函数。
-   simd vf函数：使用\_\_simd\_vf\_\_标记，能被核函数通过asc\_vf\_call接口调用。simd vf函数内只能调用\_\_simd\_callee\_\_函数和constexpr函数。
-   simt vf函数：使用\_\_simt\_vf\_\_标记，能被核函数通过asc\_vf\_call接口调用。simt vf函数内只能调用\_\_simt\_callee\_\_函数和constexpr函数。
-   \_\_simd\_callee\_\_子函数：被simd vf函数调用的子函数，子函数可能有返回值或者通过引用传参，这类子函数通过\_\_simd\_callee\_\_标识。\_\_simd\_callee\_\_函数内只能调用\_\_simd\_callee\_\_函数和constexpr函数。
-   \_\_simt\_callee\_\_子函数：被simt vf函数调用的子函数，子函数可能有返回值或者通过引用传参，这类子函数通过\_\_simt\_callee\_\_标识。\_\_simt\_callee\_\_函数内只能调用\_\_simt\_callee\_\_函数和constexpr函数。

具体支持的调用关系图如下所示。

**图 4**  函数调用关系图<a name="fig2064152983610"></a>  
![](../../../figures/函数调用关系图.png "函数调用关系图")

## 编程示例<a name="section776244992018"></a>

样例中介绍的算子完整代码请参见[SIMD与SIMT混合编程实现gather&adds算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/00_introduction/03_fusion_operation/simt_gather_and_simd_adds)。

```
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void simt_gather(
    __gm__ float* input,
    __gm__ uint32_t* index,
    __ubuf__ float* gather_output,
    uint32_t input_total_length,
    uint32_t index_total_length,
    uint32_t output_total_length)
{
    if (threadIdx.x >= output_total_length) {
        return;
    }
    // blockIdx will be supported later.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= index_total_length) {
        return;
    }

    uint32_t gather_idx = index[idx];
    if (gather_idx >= input_total_length) {
        return;
    }

    gather_output[threadIdx.x] = input[gather_idx];
}

__simd_vf__ inline void simd_adds(__ubuf__ float* output, __ubuf__ float* input,
    uint32_t count, uint32_t one_repeat_size, uint16_t repeat_times)
{
    AscendC::Reg::RegTensor<float> src_reg0;
    AscendC::Reg::RegTensor<float> dst_reg0;
    AscendC::Reg::MaskReg mask_reg; 

    for (uint16_t i = 0; i < repeat_times; i++) {
        mask_reg = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::LoadAlign(src_reg0, input + i * one_repeat_size);
        AscendC::Reg::Adds(dst_reg0, src_reg0, ADDS_ADDEND, mask_reg);
        AscendC::Reg::StoreAlign(output + i * one_repeat_size, dst_reg0, mask_reg);
    }
}

__global__ __aicore__ void gather_and_adds_kernel(__gm__ float* input, __gm__ uint32_t* index, __gm__ float* output,
    uint32_t input_total_length, uint32_t index_total_length)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub_allocator;
    // 1. gather numbers from input.
    uint32_t index_total_length_per_block = index_total_length / AscendC::GetBlockNum();
    AscendC::LocalTensor<float> gather_output = ub_allocator.Alloc<float>(index_total_length_per_block);
    asc_vf_call<simt_gather>(dim3(THREAD_COUNT),input, index,
                             (__ubuf__ float *)gather_output.GetPhyAddr(),
                             input_total_length,
                             index_total_length,
                             index_total_length_per_block);

    // 2. use reg compute api to do addition.
    AscendC::LocalTensor<float> adds_output = ub_allocator.Alloc<float>(index_total_length_per_block);
    constexpr uint32_t one_repeat_size = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeat_times = (index_total_length_per_block + one_repeat_size - 1) / one_repeat_size;
    asc_vf_call<simd_adds>((__ubuf__ float *)adds_output.GetPhyAddr(),
        (__ubuf__ float *)gather_output.GetPhyAddr(), index_total_length_per_block, one_repeat_size, repeat_times);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    // 3. copy data to global memory.
    AscendC::GlobalTensor<float> output_global_tensor;
    output_global_tensor.SetGlobalBuffer(output + index_total_length_per_block * AscendC::GetBlockIdx());
    AscendC::DataCopy(output_global_tensor, adds_output, index_total_length_per_block);
}

int main(int argc, char *argv[])
{
     …
     //numBlocks only supports one dimension currently.
     gather_and_adds_kernel<<<numBlocks, dynUBufSize, stream>>>(input_device, index_device, output_device, input_total_length, index_total_length);
     …
}
```
