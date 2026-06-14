> **原始文档路径**: `asc-devkit/docs/guide/编程指南/语言扩展层/SIMT-BuiltIn关键字.md`
# SIMT BuiltIn关键字<a name="ZH-CN_TOPIC_0000002477663934"></a>

## 函数执行空间限定符<a name="section204112391232"></a>

函数执行空间限定符（Function Execution Space Qualifier）指示函数是在Host侧执行还是在Device侧执行，以及能被调用的空间范围。

**表 1**  函数执行空间限定符概览

<a name="table121121062614"></a>
<table><thead align="left"><tr id="row4291032613"><th class="cellrowborder" rowspan="2" align="center" valign="top" id="mcps1.2.6.1.1"><p id="p1023104264"><a name="p1023104264"></a><a name="p1023104264"></a>函数执行空间限定符</p>
<p id="p894291414343"><a name="p894291414343"></a><a name="p894291414343"></a></p>
</th>
<th class="cellrowborder" colspan="2" align="center" valign="top" id="mcps1.2.6.1.2"><p id="p19680135592815"><a name="p19680135592815"></a><a name="p19680135592815"></a>执行空间</p>
</th>
<th class="cellrowborder" colspan="2" align="center" valign="top" id="mcps1.2.6.1.3"><p id="p184801804290"><a name="p184801804290"></a><a name="p184801804290"></a>允许调用函数空间</p>
</th>
</tr>
<tr id="row189411914153411"><th class="cellrowborder" align="center" valign="top" id="mcps1.2.6.2.1"><p id="p994218146344"><a name="p994218146344"></a><a name="p994218146344"></a>Host</p>
</th>
<th class="cellrowborder" align="center" valign="top" id="mcps1.2.6.2.2"><p id="p394211443414"><a name="p394211443414"></a><a name="p394211443414"></a>Device</p>
</th>
<th class="cellrowborder" align="center" valign="top" id="mcps1.2.6.2.3"><p id="p10942214163417"><a name="p10942214163417"></a><a name="p10942214163417"></a>Host</p>
</th>
<th class="cellrowborder" align="center" valign="top" id="mcps1.2.6.2.4"><p id="p1994218142341"><a name="p1994218142341"></a><a name="p1994218142341"></a>Device</p>
</th>
</tr>
</thead>
<tbody><tr id="row62110122613"><td class="cellrowborder" valign="top" width="28.51%" headers="mcps1.2.6.1.1 mcps1.2.6.2.1 "><p id="p17221010263"><a name="p17221010263"></a><a name="p17221010263"></a>__host__， 无限定符</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="12.839999999999998%" headers="mcps1.2.6.1.2 mcps1.2.6.2.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="19.42%" headers="mcps1.2.6.1.2 mcps1.2.6.2.3 "><p id="p7670131016189"><a name="p7670131016189"></a><a name="p7670131016189"></a>x</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="13.209999999999999%" headers="mcps1.2.6.1.3 mcps1.2.6.2.4 "><p id="p381084520358"><a name="p381084520358"></a><a name="p381084520358"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="26.02%" headers="mcps1.2.6.1.3 "><p id="p6382184111353"><a name="p6382184111353"></a><a name="p6382184111353"></a>x</p>
</td>
</tr>
<tr id="row9231082616"><td class="cellrowborder" valign="top" width="28.51%" headers="mcps1.2.6.1.1 mcps1.2.6.2.1 "><p id="p142510112618"><a name="p142510112618"></a><a name="p142510112618"></a>__aicore__</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="12.839999999999998%" headers="mcps1.2.6.1.2 mcps1.2.6.2.2 "><p id="p153318813366"><a name="p153318813366"></a><a name="p153318813366"></a>x</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="19.42%" headers="mcps1.2.6.1.2 mcps1.2.6.2.3 "><p id="p17983564355"><a name="p17983564355"></a><a name="p17983564355"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="13.209999999999999%" headers="mcps1.2.6.1.3 mcps1.2.6.2.4 "><p id="p6166191083620"><a name="p6166191083620"></a><a name="p6166191083620"></a>x</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="26.02%" headers="mcps1.2.6.1.3 "><p id="p4290115883510"><a name="p4290115883510"></a><a name="p4290115883510"></a>√</p>
</td>
</tr>
<tr id="row16229203212367"><td class="cellrowborder" valign="top" width="28.51%" headers="mcps1.2.6.1.1 mcps1.2.6.2.1 "><p id="p1722914323365"><a name="p1722914323365"></a><a name="p1722914323365"></a>__global__</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="12.839999999999998%" headers="mcps1.2.6.1.2 mcps1.2.6.2.2 "><p id="p622916323366"><a name="p622916323366"></a><a name="p622916323366"></a>x</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="19.42%" headers="mcps1.2.6.1.2 mcps1.2.6.2.3 "><p id="p184411197378"><a name="p184411197378"></a><a name="p184411197378"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="13.209999999999999%" headers="mcps1.2.6.1.3 mcps1.2.6.2.4 "><p id="p6991810183716"><a name="p6991810183716"></a><a name="p6991810183716"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="26.02%" headers="mcps1.2.6.1.3 "><p id="p10451171219379"><a name="p10451171219379"></a><a name="p10451171219379"></a>x</p>
</td>
</tr>
</tbody>
</table>

\_\_global\_\_修饰的函数是核函数入口，有以下使用约束：

-   函数返回类型必须为void，不能是class、struct或者union的成员函数。
-   不支持递归调用。
-   对\_\_global\_\_函数的调用是异步的，调用后即返回Host侧的主机线程。
-   只能被Host侧函数调用，在Device上执行。

\_\_aicore\_\_修饰的函数只能在Device侧执行，只能被\_\_global\_\_函数，或者其他\_\_aicore\_\_函数调用。

\_\_host\_\_修饰的函数只能在Host侧被调用和执行。

## 内存空间限定符<a name="section20787845172517"></a>

使用内存空间限定符\_\_ubuf\_\_来表示动、静态内存，静态内存的大小在编译期是确定的，动态内存的大小在核函数执行时确定。

-   静态内存通过数组分配：

    ```
    __ubuf__ half staticBuf[1024];
    ```

-   动态内存通过以下方式申请使用：

    ```
    extern __ubuf__ half dynamicBuf[]; 
    ```

    动态内存的实际内存大小需要在核函数启动时配置，具体内容请参考[核函数配置](#section97005415463)。

## 内置变量<a name="section13165113520576"></a>

-   dim3<a name="li1136665405"></a>

    用于指定和获取线程网格（grid）、线程块（block）在x、y、z维度上的内置结构体。

    dim3由3个无符号整数组成，结构体定义为\{dimx，dimy，dimz\}，用于指定3个不同维度的大小，三维总数为dimx \* dimy \* dimz。开发者可以通过如下方式创建dim3结构。

    ```
    dim3(x); // 创建一维结构，dimy和dimz为默认值1
    dim3(x, y); // 创建二维结构，dimz为默认值1
    dim3(x, y, z); // 创建三维结构
    ```

当前提供了以下仅在Device上可用的dim3结构的内置变量：

-   blockDim<a name="li076017381191"></a>

    内置全局变量，在核函数中可以直接使用，用于获取线程块中配置的线程的三维层次结构，即启动VF时配置的dim3结构体实例值。blockDim.x，blockDim.y，blockDim.z分别表示线程块中三个维度的线程数。

-   gridDim<a name="li20760123812911"></a>

    内置全局变量，只能在核函数中使用，表示整个计算任务在各个维度上分别由多少个线程块构成。

    -   gridDim.x是x维度上的线程块数量。
    -   gridDim.y是y维度上的线程块数量，目前只能返回1。
    -   gridDim.z是z维度上的线程块数量，目前只能返回1。

-   blockIdx<a name="li1676053814914"></a>

    内置全局变量，只能在核函数中使用，用于获取块索引。表示当前线程所在的线程块在整个网格中的位置坐标。

    -   blockIdx.x的范围是0到gridDim.x - 1。
    -   blockIdx.y的范围是0到gridDim.y - 1，目前只能返回0。
    -   blockIdx.z的范围是0到gridDim.z - 1，目前只能返回0。

-   threadIdx<a name="li7760123814919"></a>

    内置全局变量，在核函数中可以直接使用，用于获取当前线程在线程块内部的索引。threadIdx.x，threadIdx.y，threadIdx.z分别表示当前线程在3个维度的索引，threadIdx.x的范围为\[0, blockDim.x\)，threadIdx.y的范围为\[0, blockDim.y\)，threadIdx.z的范围为\[0, blockDim.z\)。线程块内线程的索引与线程ID对应关系如下：

    -   对于一维线程块，其线程ID为threadIdx.x。
    -   对于二维线程块，其线程ID为（threadIdx.x + threadIdx.y \* blockDim.x）。
    -   对于三维线程块，其线程ID为（threadIdx.x + threadIdx.y \* blockDim.x + threadIdx.z \* blockDim.x \* blockDim.y）。

当前提供了以下仅在Device上可用的int类型的内置变量：

-   warpSize

    运行时变量，表示一个线程束（warp）中的线程数量，当前为固定值32。

## 内置数据类型<a name="section1835494915576"></a>

目前提供了一系列适用于Device侧的数据类型，包括标量和短向量。短向量是由多个元素组成的简单向量。

**表 2**  标量数据类型

<a name="table114921244133211"></a>
<table><tbody><tr id="row11523124473213"><td class="cellrowborder" valign="top" width="7.5200000000000005%"><p id="p356595195817"><a name="p356595195817"></a><a name="p356595195817"></a>类型</p>
</td>
<td class="cellrowborder" valign="top" width="28.51%"><p id="p1252334493212"><a name="p1252334493212"></a><a name="p1252334493212"></a><strong id="b252317441327"><a name="b252317441327"></a><a name="b252317441327"></a>数据类型</strong></p>
</td>
<td class="cellrowborder" valign="top" width="25.4%"><p id="p115231244103214"><a name="p115231244103214"></a><a name="p115231244103214"></a><strong id="b9523144193211"><a name="b9523144193211"></a><a name="b9523144193211"></a>描述</strong></p>
</td>
<td class="cellrowborder" valign="top" width="12.2%"><p id="p15230443324"><a name="p15230443324"></a><a name="p15230443324"></a>Size（bit）</p>
</td>
<td class="cellrowborder" valign="top" width="26.369999999999997%"><p id="p769365816503"><a name="p769365816503"></a><a name="p769365816503"></a>取值范围</p>
</td>
</tr>
<tr id="row16523144173216"><td class="cellrowborder" valign="top" width="7.5200000000000005%"><p id="p125651859582"><a name="p125651859582"></a><a name="p125651859582"></a>布尔型</p>
</td>
<td class="cellrowborder" valign="top" width="28.51%"><p id="p1752384403213"><a name="p1752384403213"></a><a name="p1752384403213"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="25.4%"><p id="p15523104414328"><a name="p15523104414328"></a><a name="p15523104414328"></a>全0代表false，否则代表true。</p>
</td>
<td class="cellrowborder" valign="top" width="12.2%"><p id="p1252464414329"><a name="p1252464414329"></a><a name="p1252464414329"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="26.369999999999997%"><p id="p569365818504"><a name="p569365818504"></a><a name="p569365818504"></a>true, flase</p>
</td>
</tr>
<tr id="row752484423212"><td class="cellrowborder" rowspan="8" valign="top" width="7.5200000000000005%"><p id="p155656525811"><a name="p155656525811"></a><a name="p155656525811"></a>整形</p>
<p id="p75655513587"><a name="p75655513587"></a><a name="p75655513587"></a></p>
<p id="p7565857582"><a name="p7565857582"></a><a name="p7565857582"></a></p>
<p id="p256515518584"><a name="p256515518584"></a><a name="p256515518584"></a></p>
<p id="p856513575815"><a name="p856513575815"></a><a name="p856513575815"></a></p>
<p id="p256555105817"><a name="p256555105817"></a><a name="p256555105817"></a></p>
<p id="p15651952584"><a name="p15651952584"></a><a name="p15651952584"></a></p>
<p id="p1956519517584"><a name="p1956519517584"></a><a name="p1956519517584"></a></p>
</td>
<td class="cellrowborder" valign="top" width="28.51%"><p id="p185244442322"><a name="p185244442322"></a><a name="p185244442322"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="25.4%"><p id="p11524164410326"><a name="p11524164410326"></a><a name="p11524164410326"></a>unsigned char</p>
</td>
<td class="cellrowborder" valign="top" width="12.2%"><p id="p6524184417325"><a name="p6524184417325"></a><a name="p6524184417325"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="26.369999999999997%"><p id="p116931358105018"><a name="p116931358105018"></a><a name="p116931358105018"></a>[0, 255]</p>
</td>
</tr>
<tr id="row9524104414329"><td class="cellrowborder" valign="top"><p id="p55245445324"><a name="p55245445324"></a><a name="p55245445324"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p175249445326"><a name="p175249445326"></a><a name="p175249445326"></a>signed char</p>
</td>
<td class="cellrowborder" valign="top"><p id="p15244442322"><a name="p15244442322"></a><a name="p15244442322"></a>8</p>
</td>
<td class="cellrowborder" valign="top"><p id="p869335805010"><a name="p869335805010"></a><a name="p869335805010"></a>[-128, 127]</p>
</td>
</tr>
<tr id="row13524124415322"><td class="cellrowborder" valign="top"><p id="p652444413320"><a name="p652444413320"></a><a name="p652444413320"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1952420443327"><a name="p1952420443327"></a><a name="p1952420443327"></a>unsigned short</p>
</td>
<td class="cellrowborder" valign="top"><p id="p3524174410328"><a name="p3524174410328"></a><a name="p3524174410328"></a>16</p>
</td>
<td class="cellrowborder" valign="top"><p id="p18693115812504"><a name="p18693115812504"></a><a name="p18693115812504"></a>[0, 65535]</p>
</td>
</tr>
<tr id="row205248445323"><td class="cellrowborder" valign="top"><p id="p1252474415325"><a name="p1252474415325"></a><a name="p1252474415325"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p652414445323"><a name="p652414445323"></a><a name="p652414445323"></a>signed short</p>
</td>
<td class="cellrowborder" valign="top"><p id="p185241044203211"><a name="p185241044203211"></a><a name="p185241044203211"></a>16</p>
</td>
<td class="cellrowborder" valign="top"><p id="p9693458175016"><a name="p9693458175016"></a><a name="p9693458175016"></a>[-32768, 32767]</p>
</td>
</tr>
<tr id="row1252464413322"><td class="cellrowborder" valign="top"><p id="p11524134417324"><a name="p11524134417324"></a><a name="p11524134417324"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p15524114453219"><a name="p15524114453219"></a><a name="p15524114453219"></a>unsigned int</p>
</td>
<td class="cellrowborder" valign="top"><p id="p852415445326"><a name="p852415445326"></a><a name="p852415445326"></a>32</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1469311586502"><a name="p1469311586502"></a><a name="p1469311586502"></a>[0, 4294967295]</p>
</td>
</tr>
<tr id="row15524244123211"><td class="cellrowborder" valign="top"><p id="p19524244153211"><a name="p19524244153211"></a><a name="p19524244153211"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p2052414446324"><a name="p2052414446324"></a><a name="p2052414446324"></a>signed int</p>
</td>
<td class="cellrowborder" valign="top"><p id="p3524644153216"><a name="p3524644153216"></a><a name="p3524644153216"></a>32</p>
</td>
<td class="cellrowborder" valign="top"><p id="p469325845018"><a name="p469325845018"></a><a name="p469325845018"></a>[-2147483648, 2147483647]</p>
</td>
</tr>
<tr id="row35241144113212"><td class="cellrowborder" valign="top"><p id="p145241244153216"><a name="p145241244153216"></a><a name="p145241244153216"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p752418446324"><a name="p752418446324"></a><a name="p752418446324"></a>unsigned long</p>
</td>
<td class="cellrowborder" valign="top"><p id="p5524174493219"><a name="p5524174493219"></a><a name="p5524174493219"></a>64</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1968436163610"><a name="p1968436163610"></a><a name="p1968436163610"></a>[0,18446744073709551615]</p>
</td>
</tr>
<tr id="row552424418327"><td class="cellrowborder" valign="top"><p id="p4524174418327"><a name="p4524174418327"></a><a name="p4524174418327"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p5524134420322"><a name="p5524134420322"></a><a name="p5524134420322"></a>signed long</p>
</td>
<td class="cellrowborder" valign="top"><p id="p125241344143217"><a name="p125241344143217"></a><a name="p125241344143217"></a>64</p>
</td>
<td class="cellrowborder" valign="top"><p id="p8813204316281"><a name="p8813204316281"></a><a name="p8813204316281"></a>[-9223372036854775808, 9223372036854775807]</p>
</td>
</tr>
<tr id="row12524194413212"><td class="cellrowborder" rowspan="6" valign="top" width="7.5200000000000005%"><p id="p451813217594"><a name="p451813217594"></a><a name="p451813217594"></a>浮点型</p>
</td>
<td class="cellrowborder" valign="top" width="28.51%"><p id="p1652414420324"><a name="p1652414420324"></a><a name="p1652414420324"></a>float8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" width="25.4%"><p id="p15524194412324"><a name="p15524194412324"></a><a name="p15524194412324"></a>符号位宽1，指数位宽4，尾数位宽3</p>
</td>
<td class="cellrowborder" valign="top" width="12.2%"><p id="p9524154418326"><a name="p9524154418326"></a><a name="p9524154418326"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="26.369999999999997%"><p id="p1969311582506"><a name="p1969311582506"></a><a name="p1969311582506"></a>[2<sup id="sup134721452234"><a name="sup134721452234"></a><a name="sup134721452234"></a>6</sup> - 2<sup id="sup947215518239"><a name="sup947215518239"></a><a name="sup947215518239"></a>9</sup>, 2<sup id="sup34721351236"><a name="sup34721351236"></a><a name="sup34721351236"></a>9</sup> - 2<sup id="sup1347285142319"><a name="sup1347285142319"></a><a name="sup1347285142319"></a>6</sup>]</p>
</td>
</tr>
<tr id="row9524144163216"><td class="cellrowborder" valign="top"><p id="p135241844113216"><a name="p135241844113216"></a><a name="p135241844113216"></a>float8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p85241344173217"><a name="p85241344173217"></a><a name="p85241344173217"></a>符号位宽1，指数位宽5，尾数位宽2</p>
</td>
<td class="cellrowborder" valign="top"><p id="p14524244123213"><a name="p14524244123213"></a><a name="p14524244123213"></a>8</p>
</td>
<td class="cellrowborder" valign="top"><p id="p469310585508"><a name="p469310585508"></a><a name="p469310585508"></a>[2<sup id="sup647115572316"><a name="sup647115572316"></a><a name="sup647115572316"></a>13</sup> - 2<sup id="sup04713516232"><a name="sup04713516232"></a><a name="sup04713516232"></a>16</sup>, 2<sup id="sup124711755231"><a name="sup124711755231"></a><a name="sup124711755231"></a>16</sup> - 2<sup id="sup1547111522311"><a name="sup1547111522311"></a><a name="sup1547111522311"></a>13</sup>]</p>
</td>
</tr>
<tr id="row9524644153216"><td class="cellrowborder" valign="top"><p id="p552454412325"><a name="p552454412325"></a><a name="p552454412325"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p6959161523517"><a name="p6959161523517"></a><a name="p6959161523517"></a>符号位宽1，点域位宽2，指数与尾数位宽由点域编码决定</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1352464419323"><a name="p1352464419323"></a><a name="p1352464419323"></a>8</p>
</td>
<td class="cellrowborder" valign="top"><p id="p9693185865018"><a name="p9693185865018"></a><a name="p9693185865018"></a>点域编码决定数据精度与取值范围</p>
</td>
</tr>
<tr id="row19524544113220"><td class="cellrowborder" valign="top"><p id="p352544453213"><a name="p352544453213"></a><a name="p352544453213"></a>half</p>
</td>
<td class="cellrowborder" valign="top"><p id="p5525174493216"><a name="p5525174493216"></a><a name="p5525174493216"></a>符号位宽1，指数位宽5，尾数位宽10</p>
</td>
<td class="cellrowborder" valign="top"><p id="p952544473213"><a name="p952544473213"></a><a name="p952544473213"></a>16</p>
</td>
<td class="cellrowborder" valign="top"><p id="p7693105855012"><a name="p7693105855012"></a><a name="p7693105855012"></a>[2<sup id="sup1347211532312"><a name="sup1347211532312"></a><a name="sup1347211532312"></a>5</sup> - 2<sup id="sup147210517238"><a name="sup147210517238"></a><a name="sup147210517238"></a>16</sup>, 2<sup id="sup1147235192313"><a name="sup1147235192313"></a><a name="sup1147235192313"></a>16</sup> - 2<sup id="sup147214582314"><a name="sup147214582314"></a><a name="sup147214582314"></a>5</sup>]</p>
</td>
</tr>
<tr id="row5525154443219"><td class="cellrowborder" valign="top"><p id="p13525164463215"><a name="p13525164463215"></a><a name="p13525164463215"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top"><p id="p45253449322"><a name="p45253449322"></a><a name="p45253449322"></a>符号位宽1，指数位宽8，尾数位宽7</p>
</td>
<td class="cellrowborder" valign="top"><p id="p175251844103212"><a name="p175251844103212"></a><a name="p175251844103212"></a>16</p>
</td>
<td class="cellrowborder" valign="top"><p id="p769312588502"><a name="p769312588502"></a><a name="p769312588502"></a>[2<sup id="sup84721956237"><a name="sup84721956237"></a><a name="sup84721956237"></a>120</sup> - 2<sup id="sup1447220519237"><a name="sup1447220519237"></a><a name="sup1447220519237"></a>128</sup>, 2<sup id="sup124721953238"><a name="sup124721953238"></a><a name="sup124721953238"></a>128</sup> - 2<sup id="sup947212522316"><a name="sup947212522316"></a><a name="sup947212522316"></a>120</sup>]</p>
</td>
</tr>
<tr id="row1052544410322"><td class="cellrowborder" valign="top"><p id="p252554410321"><a name="p252554410321"></a><a name="p252554410321"></a>float</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1252534413212"><a name="p1252534413212"></a><a name="p1252534413212"></a>符号位宽1，指数位宽8，尾数位宽23</p>
</td>
<td class="cellrowborder" valign="top"><p id="p852516444327"><a name="p852516444327"></a><a name="p852516444327"></a>32</p>
</td>
<td class="cellrowborder" valign="top"><p id="p1169365815016"><a name="p1169365815016"></a><a name="p1169365815016"></a>[2<sup id="sup1047210542315"><a name="sup1047210542315"></a><a name="sup1047210542315"></a>104</sup> - 2<sup id="sup247212517234"><a name="sup247212517234"></a><a name="sup247212517234"></a>128</sup>, 2<sup id="sup1947211562311"><a name="sup1947211562311"></a><a name="sup1947211562311"></a>128</sup> - 2<sup id="sup74721855236"><a name="sup74721855236"></a><a name="sup74721855236"></a>104</sup>]</p>
</td>
</tr>
</tbody>
</table>

短向量数据类型分为Vector X2、Vector X3、Vector X4，表示一个短向量变量有2、3、4个元素，当前支持的类型分布如下：

<a name="table49986503175"></a>
<table><tbody><tr id="row12995135017175"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p108797465237"><a name="p108797465237"></a><a name="p108797465237"></a>元素数据类型</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p6995650191711"><a name="p6995650191711"></a><a name="p6995650191711"></a>Vector X2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p916583931816"><a name="p916583931816"></a><a name="p916583931816"></a>Vector X3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p216573951814"><a name="p216573951814"></a><a name="p216573951814"></a>Vector X4</p>
</td>
</tr>
<tr id="row179955509177"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p138795466233"><a name="p138795466233"></a><a name="p138795466233"></a><span>unsigned char</span></p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p19951050111714"><a name="p19951050111714"></a><a name="p19951050111714"></a>uchar2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p79951450161718"><a name="p79951450161718"></a><a name="p79951450161718"></a>uchar3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p1599515506173"><a name="p1599515506173"></a><a name="p1599515506173"></a>uchar4</p>
</td>
</tr>
<tr id="row6995150121717"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p520813615266"><a name="p520813615266"></a><a name="p520813615266"></a><span>signed char</span></p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p1399595019178"><a name="p1399595019178"></a><a name="p1399595019178"></a>char2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p10995175031711"><a name="p10995175031711"></a><a name="p10995175031711"></a>char3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p129951450181712"><a name="p129951450181712"></a><a name="p129951450181712"></a>char4</p>
</td>
</tr>
<tr id="row6996105019176"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p138791246152313"><a name="p138791246152313"></a><a name="p138791246152313"></a>unsigned short (16bit)</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p17995650141719"><a name="p17995650141719"></a><a name="p17995650141719"></a>ushort2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p2099585051712"><a name="p2099585051712"></a><a name="p2099585051712"></a>ushort3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p99965500173"><a name="p99965500173"></a><a name="p99965500173"></a>ushort4</p>
</td>
</tr>
<tr id="row8996185012174"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p187913467238"><a name="p187913467238"></a><a name="p187913467238"></a><span>signed short (16bit)</span></p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p209968502179"><a name="p209968502179"></a><a name="p209968502179"></a>short2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p79962050161711"><a name="p79962050161711"></a><a name="p79962050161711"></a>short3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p399611500176"><a name="p399611500176"></a><a name="p399611500176"></a>short4</p>
</td>
</tr>
<tr id="row0996250181714"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1387914611236"><a name="p1387914611236"></a><a name="p1387914611236"></a><span>unsigned int</span></p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p1899655081717"><a name="p1899655081717"></a><a name="p1899655081717"></a>uint2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p89968504179"><a name="p89968504179"></a><a name="p89968504179"></a>uint3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p19996105017176"><a name="p19996105017176"></a><a name="p19996105017176"></a>uint4</p>
</td>
</tr>
<tr id="row189961750201712"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p18879184616231"><a name="p18879184616231"></a><a name="p18879184616231"></a><span>signed int</span></p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p799613501179"><a name="p799613501179"></a><a name="p799613501179"></a>int2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p1699665012173"><a name="p1699665012173"></a><a name="p1699665012173"></a>int3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p109961750131712"><a name="p109961750131712"></a><a name="p109961750131712"></a>int4</p>
</td>
</tr>
<tr id="row10996850111715"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p4879124614237"><a name="p4879124614237"></a><a name="p4879124614237"></a>无符号的长整型 (64bit)</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p13996115016176"><a name="p13996115016176"></a><a name="p13996115016176"></a>ulonglong2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p1099611503174"><a name="p1099611503174"></a><a name="p1099611503174"></a>ulonglong3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p179961450121719"><a name="p179961450121719"></a><a name="p179961450121719"></a>ulonglong4</p>
</td>
</tr>
<tr id="row399685017170"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p4879164612319"><a name="p4879164612319"></a><a name="p4879164612319"></a>有符号的长整型 (64bit)</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p499645021715"><a name="p499645021715"></a><a name="p499645021715"></a>longlong2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p189969503173"><a name="p189969503173"></a><a name="p189969503173"></a>longlong3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p1996165041719"><a name="p1996165041719"></a><a name="p1996165041719"></a>longlong4</p>
</td>
</tr>
<tr id="row10996750131719"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1387994611236"><a name="p1387994611236"></a><a name="p1387994611236"></a>无符号的长整型 (32bit)</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p799675061716"><a name="p799675061716"></a><a name="p799675061716"></a>ulong2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p1599695041716"><a name="p1599695041716"></a><a name="p1599695041716"></a>ulong3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p159961350201718"><a name="p159961350201718"></a><a name="p159961350201718"></a>ulong4</p>
</td>
</tr>
<tr id="row179961650101720"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p18879946162314"><a name="p18879946162314"></a><a name="p18879946162314"></a>有符号的长整型 (32bit)</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p11996145013170"><a name="p11996145013170"></a><a name="p11996145013170"></a>long2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p599619508171"><a name="p599619508171"></a><a name="p599619508171"></a>long3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p2996125016171"><a name="p2996125016171"></a><a name="p2996125016171"></a>long4</p>
</td>
</tr>
<tr id="row189971750141710"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p4879124615236"><a name="p4879124615236"></a><a name="p4879124615236"></a>浮点型，1符号位，2指数位，1尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p1899619506175"><a name="p1899619506175"></a><a name="p1899619506175"></a>float4_e2m1x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p13997165017174"><a name="p13997165017174"></a><a name="p13997165017174"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p299715011716"><a name="p299715011716"></a><a name="p299715011716"></a>-</p>
</td>
</tr>
<tr id="row4997135021712"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p8879446142315"><a name="p8879446142315"></a><a name="p8879446142315"></a>浮点型，1符号位，1指数位，2尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p1499795013177"><a name="p1499795013177"></a><a name="p1499795013177"></a>float4_e1m2x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p10997105017177"><a name="p10997105017177"></a><a name="p10997105017177"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p1699715508179"><a name="p1699715508179"></a><a name="p1699715508179"></a>-</p>
</td>
</tr>
<tr id="row12997115001717"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p9879194642317"><a name="p9879194642317"></a><a name="p9879194642317"></a>浮点型，1符号位，4指数位，3尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p19997185071717"><a name="p19997185071717"></a><a name="p19997185071717"></a>float8_e4m3x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p399795001718"><a name="p399795001718"></a><a name="p399795001718"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p89971350171718"><a name="p89971350171718"></a><a name="p89971350171718"></a>-</p>
</td>
</tr>
<tr id="row1899710509175"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1087914618237"><a name="p1087914618237"></a><a name="p1087914618237"></a>浮点型，1符号位，5指数位，2尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p199710500179"><a name="p199710500179"></a><a name="p199710500179"></a>float8_e5m2x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p149976503172"><a name="p149976503172"></a><a name="p149976503172"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p499755013177"><a name="p499755013177"></a><a name="p499755013177"></a>-</p>
</td>
</tr>
<tr id="row18997650181714"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1787974652320"><a name="p1787974652320"></a><a name="p1787974652320"></a>浮点型 hif8</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p599755041713"><a name="p599755041713"></a><a name="p599755041713"></a>hifloat8x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p12997135017175"><a name="p12997135017175"></a><a name="p12997135017175"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p12997205015174"><a name="p12997205015174"></a><a name="p12997205015174"></a>-</p>
</td>
</tr>
<tr id="row99975505178"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1187994610232"><a name="p1187994610232"></a><a name="p1187994610232"></a>浮点型，1符号位，5指数位，10尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p2997115021715"><a name="p2997115021715"></a><a name="p2997115021715"></a>half2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p3997165010177"><a name="p3997165010177"></a><a name="p3997165010177"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p16997135051713"><a name="p16997135051713"></a><a name="p16997135051713"></a>-</p>
</td>
</tr>
<tr id="row699865011170"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p1687944642312"><a name="p1687944642312"></a><a name="p1687944642312"></a>浮点型，1符号位，8指数位，7尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p179971350151716"><a name="p179971350151716"></a><a name="p179971350151716"></a>bfloat16x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p199971150151719"><a name="p199971150151719"></a><a name="p199971150151719"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p3997185061710"><a name="p3997185061710"></a><a name="p3997185061710"></a>-</p>
</td>
</tr>
<tr id="row999818508170"><td class="cellrowborder" valign="top" width="41.005899410059%"><p id="p128791246192315"><a name="p128791246192315"></a><a name="p128791246192315"></a>浮点型，1符号位，8指数位，23尾数位</p>
</td>
<td class="cellrowborder" valign="top" width="21.11788821117888%"><p id="p1499815509170"><a name="p1499815509170"></a><a name="p1499815509170"></a>float2</p>
</td>
<td class="cellrowborder" valign="top" width="18.838116188381164%"><p id="p6998650191715"><a name="p6998650191715"></a><a name="p6998650191715"></a>float3</p>
</td>
<td class="cellrowborder" valign="top" width="19.038096190380962%"><p id="p69984505172"><a name="p69984505172"></a><a name="p69984505172"></a>float4</p>
</td>
</tr>
</tbody>
</table>

**表 3**  短向量数据类型

<a name="table13856144241"></a>
<table><tbody><tr id="row585151414242"><td class="cellrowborder" valign="top" width="60.39%"><p id="p98514141249"><a name="p98514141249"></a><a name="p98514141249"></a>数据类型</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p78551418240"><a name="p78551418240"></a><a name="p78551418240"></a>内存大小（字节）</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p1085201418248"><a name="p1085201418248"></a><a name="p1085201418248"></a>地址对齐（字节）</p>
</td>
</tr>
<tr id="row1285161422412"><td class="cellrowborder" valign="top" width="60.39%"><p id="p198651472418"><a name="p198651472418"></a><a name="p198651472418"></a>char2, uchar2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p178641420249"><a name="p178641420249"></a><a name="p178641420249"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p11861714172410"><a name="p11861714172410"></a><a name="p11861714172410"></a>2</p>
</td>
</tr>
<tr id="row486414142412"><td class="cellrowborder" valign="top" width="60.39%"><p id="p1861214172413"><a name="p1861214172413"></a><a name="p1861214172413"></a>char3, uchar3， char4, uchar4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p186141413240"><a name="p186141413240"></a><a name="p186141413240"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p886914182410"><a name="p886914182410"></a><a name="p886914182410"></a>4</p>
</td>
</tr>
<tr id="row586171442413"><td class="cellrowborder" valign="top" width="60.39%"><p id="p208612148244"><a name="p208612148244"></a><a name="p208612148244"></a>short2, ushort2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p586161416241"><a name="p586161416241"></a><a name="p586161416241"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p486131492417"><a name="p486131492417"></a><a name="p486131492417"></a>4</p>
</td>
</tr>
<tr id="row1486314192410"><td class="cellrowborder" valign="top" width="60.39%"><p id="p2861914152411"><a name="p2861914152411"></a><a name="p2861914152411"></a>short3, ushort3，short4, ushort4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p208681472412"><a name="p208681472412"></a><a name="p208681472412"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p16861514172417"><a name="p16861514172417"></a><a name="p16861514172417"></a>8</p>
</td>
</tr>
<tr id="row1086114182419"><td class="cellrowborder" valign="top" width="60.39%"><p id="p148671419247"><a name="p148671419247"></a><a name="p148671419247"></a>int2, uint2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p27064244135"><a name="p27064244135"></a><a name="p27064244135"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p686171402415"><a name="p686171402415"></a><a name="p686171402415"></a>8</p>
</td>
</tr>
<tr id="row68613143247"><td class="cellrowborder" valign="top" width="60.39%"><p id="p88719147244"><a name="p88719147244"></a><a name="p88719147244"></a>int3, uint3， int4, uint4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p148714143245"><a name="p148714143245"></a><a name="p148714143245"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p08712148242"><a name="p08712148242"></a><a name="p08712148242"></a>16</p>
</td>
</tr>
<tr id="row58781462412"><td class="cellrowborder" valign="top" width="60.39%"><p id="p108718147247"><a name="p108718147247"></a><a name="p108718147247"></a>long2，ulong2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p98714144244"><a name="p98714144244"></a><a name="p98714144244"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p1987151482412"><a name="p1987151482412"></a><a name="p1987151482412"></a>8</p>
</td>
</tr>
<tr id="row205761327101413"><td class="cellrowborder" valign="top" width="60.39%"><p id="p16576627191411"><a name="p16576627191411"></a><a name="p16576627191411"></a>long3，ulong3，long4，ulong4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p65761527131413"><a name="p65761527131413"></a><a name="p65761527131413"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p115762275146"><a name="p115762275146"></a><a name="p115762275146"></a>16</p>
</td>
</tr>
<tr id="row310145571416"><td class="cellrowborder" valign="top" width="60.39%"><p id="p41075571416"><a name="p41075571416"></a><a name="p41075571416"></a>longlong2，ulonglong2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p1410455181416"><a name="p1410455181416"></a><a name="p1410455181416"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p310125591414"><a name="p310125591414"></a><a name="p310125591414"></a>16</p>
</td>
</tr>
<tr id="row161011553147"><td class="cellrowborder" valign="top" width="60.39%"><p id="p98731422420"><a name="p98731422420"></a><a name="p98731422420"></a>longlong3，ulonglong3，longlong4，ulonglong4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p410105581413"><a name="p410105581413"></a><a name="p410105581413"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p0101255191412"><a name="p0101255191412"></a><a name="p0101255191412"></a>32</p>
</td>
</tr>
<tr id="row787191412417"><td class="cellrowborder" valign="top" width="60.39%"><p id="p489914122418"><a name="p489914122418"></a><a name="p489914122418"></a>float2</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p146121260161"><a name="p146121260161"></a><a name="p146121260161"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p787714102416"><a name="p787714102416"></a><a name="p787714102416"></a>8</p>
</td>
</tr>
<tr id="row138710146247"><td class="cellrowborder" valign="top" width="60.39%"><p id="p1687714172410"><a name="p1687714172410"></a><a name="p1687714172410"></a>float3，float4</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p5879143243"><a name="p5879143243"></a><a name="p5879143243"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p1887141442410"><a name="p1887141442410"></a><a name="p1887141442410"></a>16</p>
</td>
</tr>
<tr id="row148761462413"><td class="cellrowborder" valign="top" width="60.39%"><p id="p118713146248"><a name="p118713146248"></a><a name="p118713146248"></a>float4_e2m1x2_t， float4_e1m2x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p208715145248"><a name="p208715145248"></a><a name="p208715145248"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p108711142246"><a name="p108711142246"></a><a name="p108711142246"></a>1</p>
</td>
</tr>
<tr id="row488141452411"><td class="cellrowborder" valign="top" width="60.39%"><p id="p15882142245"><a name="p15882142245"></a><a name="p15882142245"></a>float8_e4m3x2_t，float8_e5m2x2_t，hifloat8x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p19881214142414"><a name="p19881214142414"></a><a name="p19881214142414"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p888514142418"><a name="p888514142418"></a><a name="p888514142418"></a>2</p>
</td>
</tr>
<tr id="row288314162415"><td class="cellrowborder" valign="top" width="60.39%"><p id="p58801442416"><a name="p58801442416"></a><a name="p58801442416"></a>half2，bfloat16x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="19.950000000000003%"><p id="p78851482412"><a name="p78851482412"></a><a name="p78851482412"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="19.66%"><p id="p888114102411"><a name="p888114102411"></a><a name="p888114102411"></a>4</p>
</td>
</tr>
</tbody>
</table>

## 运算符<a name="section186787252339"></a>

SIMT编程提供了一系列运算符，用于执行数学运算。以下是支持的运算符列表。

**表 4**  SIMT编程支持的运算符列表

<a name="table11745172843710"></a>
<table><thead align="left"><tr id="row874617286370"><th class="cellrowborder" valign="top" width="11.75%" id="mcps1.2.8.1.1"><p id="p774672811375"><a name="p774672811375"></a><a name="p774672811375"></a>类别</p>
</th>
<th class="cellrowborder" valign="top" width="8.89%" id="mcps1.2.8.1.2"><p id="p47469283375"><a name="p47469283375"></a><a name="p47469283375"></a>运算符</p>
</th>
<th class="cellrowborder" valign="top" width="8.48%" id="mcps1.2.8.1.3"><p id="p16746828203711"><a name="p16746828203711"></a><a name="p16746828203711"></a>bool</p>
</th>
<th class="cellrowborder" valign="top" width="31.03%" id="mcps1.2.8.1.4"><p id="p1774652813378"><a name="p1774652813378"></a><a name="p1774652813378"></a>int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t</p>
</th>
<th class="cellrowborder" valign="top" width="14.149999999999999%" id="mcps1.2.8.1.5"><p id="p17746152823720"><a name="p17746152823720"></a><a name="p17746152823720"></a>half/bfloat16_t/float</p>
</th>
<th class="cellrowborder" valign="top" width="16.13%" id="mcps1.2.8.1.6"><p id="p77469281379"><a name="p77469281379"></a><a name="p77469281379"></a>half2/bfloat16x2_t</p>
</th>
<th class="cellrowborder" valign="top" width="9.569999999999999%" id="mcps1.2.8.1.7"><p id="p7746102843711"><a name="p7746102843711"></a><a name="p7746102843711"></a>hifloat8_t</p>
</th>
</tr>
</thead>
<tbody><tr id="row2746162863719"><td class="cellrowborder" rowspan="8" valign="top" width="11.75%" headers="mcps1.2.8.1.1 "><p id="p9746728143715"><a name="p9746728143715"></a><a name="p9746728143715"></a>算术运算符</p>
<p id="p2074662843716"><a name="p2074662843716"></a><a name="p2074662843716"></a></p>
<p id="p19746142873718"><a name="p19746142873718"></a><a name="p19746142873718"></a></p>
</td>
<td class="cellrowborder" valign="top" width="8.89%" headers="mcps1.2.8.1.2 "><p id="p187461428163718"><a name="p187461428163718"></a><a name="p187461428163718"></a>+</p>
</td>
<td class="cellrowborder" valign="top" width="8.48%" headers="mcps1.2.8.1.3 "><p id="p15746328113715"><a name="p15746328113715"></a><a name="p15746328113715"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="31.03%" headers="mcps1.2.8.1.4 "><p id="p174632815371"><a name="p174632815371"></a><a name="p174632815371"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="14.149999999999999%" headers="mcps1.2.8.1.5 "><p id="p1474622813375"><a name="p1474622813375"></a><a name="p1474622813375"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="16.13%" headers="mcps1.2.8.1.6 "><p id="p10746728153713"><a name="p10746728153713"></a><a name="p10746728153713"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="9.569999999999999%" headers="mcps1.2.8.1.7 "><p id="p87469283372"><a name="p87469283372"></a><a name="p87469283372"></a>x</p>
</td>
</tr>
<tr id="row2746182816377"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p2746142820377"><a name="p2746142820377"></a><a name="p2746142820377"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p11746628103715"><a name="p11746628103715"></a><a name="p11746628103715"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p7746192843716"><a name="p7746192843716"></a><a name="p7746192843716"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p1746152812377"><a name="p1746152812377"></a><a name="p1746152812377"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p12746028153714"><a name="p12746028153714"></a><a name="p12746028153714"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p107463280371"><a name="p107463280371"></a><a name="p107463280371"></a>x</p>
</td>
</tr>
<tr id="row1574672853712"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p67464286375"><a name="p67464286375"></a><a name="p67464286375"></a>*</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p1774672853711"><a name="p1774672853711"></a><a name="p1774672853711"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p174622812371"><a name="p174622812371"></a><a name="p174622812371"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p157461228123711"><a name="p157461228123711"></a><a name="p157461228123711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p16746122863713"><a name="p16746122863713"></a><a name="p16746122863713"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p167465287377"><a name="p167465287377"></a><a name="p167465287377"></a>x</p>
</td>
</tr>
<tr id="row47461828143714"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p7746182816378"><a name="p7746182816378"></a><a name="p7746182816378"></a>/</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p1474672817374"><a name="p1474672817374"></a><a name="p1474672817374"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p8746152853718"><a name="p8746152853718"></a><a name="p8746152853718"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p474617285375"><a name="p474617285375"></a><a name="p474617285375"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p17746102833712"><a name="p17746102833712"></a><a name="p17746102833712"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p6746142811379"><a name="p6746142811379"></a><a name="p6746142811379"></a>x</p>
</td>
</tr>
<tr id="row1074662813372"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p17746132863714"><a name="p17746132863714"></a><a name="p17746132863714"></a>%</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p17746628113713"><a name="p17746628113713"></a><a name="p17746628113713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p974617282376"><a name="p974617282376"></a><a name="p974617282376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p274792815371"><a name="p274792815371"></a><a name="p274792815371"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p674712863718"><a name="p674712863718"></a><a name="p674712863718"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p18747128183716"><a name="p18747128183716"></a><a name="p18747128183716"></a>x</p>
</td>
</tr>
<tr id="row57476286372"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p474713283376"><a name="p474713283376"></a><a name="p474713283376"></a>++</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p12747728183715"><a name="p12747728183715"></a><a name="p12747728183715"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p15747028153711"><a name="p15747028153711"></a><a name="p15747028153711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p17471628193714"><a name="p17471628193714"></a><a name="p17471628193714"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p07471528133711"><a name="p07471528133711"></a><a name="p07471528133711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p15747192817372"><a name="p15747192817372"></a><a name="p15747192817372"></a>x</p>
</td>
</tr>
<tr id="row18747162873719"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p13747142873710"><a name="p13747142873710"></a><a name="p13747142873710"></a>--</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p16747112816375"><a name="p16747112816375"></a><a name="p16747112816375"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p1474772812374"><a name="p1474772812374"></a><a name="p1474772812374"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p11747162811374"><a name="p11747162811374"></a><a name="p11747162811374"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p774722819376"><a name="p774722819376"></a><a name="p774722819376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p4747728123717"><a name="p4747728123717"></a><a name="p4747728123717"></a>x</p>
</td>
</tr>
<tr id="row874712286377"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p974719284373"><a name="p974719284373"></a><a name="p974719284373"></a>- (取反)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p177476281370"><a name="p177476281370"></a><a name="p177476281370"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p10747102803717"><a name="p10747102803717"></a><a name="p10747102803717"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p15747162823711"><a name="p15747162823711"></a><a name="p15747162823711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p17473289372"><a name="p17473289372"></a><a name="p17473289372"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p10747132817378"><a name="p10747132817378"></a><a name="p10747132817378"></a>x</p>
</td>
</tr>
<tr id="row374712843713"><td class="cellrowborder" rowspan="6" valign="top" width="11.75%" headers="mcps1.2.8.1.1 "><p id="p074732823714"><a name="p074732823714"></a><a name="p074732823714"></a>比较运算符</p>
</td>
<td class="cellrowborder" valign="top" width="8.89%" headers="mcps1.2.8.1.2 "><p id="p874742883718"><a name="p874742883718"></a><a name="p874742883718"></a>&lt;</p>
</td>
<td class="cellrowborder" valign="top" width="8.48%" headers="mcps1.2.8.1.3 "><p id="p19747152893712"><a name="p19747152893712"></a><a name="p19747152893712"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="31.03%" headers="mcps1.2.8.1.4 "><p id="p1674719280375"><a name="p1674719280375"></a><a name="p1674719280375"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="14.149999999999999%" headers="mcps1.2.8.1.5 "><p id="p774772843715"><a name="p774772843715"></a><a name="p774772843715"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="16.13%" headers="mcps1.2.8.1.6 "><p id="p16747162811376"><a name="p16747162811376"></a><a name="p16747162811376"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="9.569999999999999%" headers="mcps1.2.8.1.7 "><p id="p1874752843717"><a name="p1874752843717"></a><a name="p1874752843717"></a>x</p>
</td>
</tr>
<tr id="row17748828103711"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p774852853714"><a name="p774852853714"></a><a name="p774852853714"></a>&lt;=</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p5748228183719"><a name="p5748228183719"></a><a name="p5748228183719"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p117486284376"><a name="p117486284376"></a><a name="p117486284376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p574842818373"><a name="p574842818373"></a><a name="p574842818373"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p1374852853713"><a name="p1374852853713"></a><a name="p1374852853713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p167485285371"><a name="p167485285371"></a><a name="p167485285371"></a>x</p>
</td>
</tr>
<tr id="row574892833718"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p1974815284379"><a name="p1974815284379"></a><a name="p1974815284379"></a>&gt;</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p1748112819377"><a name="p1748112819377"></a><a name="p1748112819377"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p15748192893718"><a name="p15748192893718"></a><a name="p15748192893718"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p1774815287377"><a name="p1774815287377"></a><a name="p1774815287377"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p274852813713"><a name="p274852813713"></a><a name="p274852813713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p1074832833715"><a name="p1074832833715"></a><a name="p1074832833715"></a>x</p>
</td>
</tr>
<tr id="row2748928193713"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p37482283373"><a name="p37482283373"></a><a name="p37482283373"></a>&gt;=</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p774811288377"><a name="p774811288377"></a><a name="p774811288377"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p974822818377"><a name="p974822818377"></a><a name="p974822818377"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p4748172803711"><a name="p4748172803711"></a><a name="p4748172803711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p1374822833712"><a name="p1374822833712"></a><a name="p1374822833712"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p0748112816378"><a name="p0748112816378"></a><a name="p0748112816378"></a>x</p>
</td>
</tr>
<tr id="row1374822815373"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p8748132818373"><a name="p8748132818373"></a><a name="p8748132818373"></a>==</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p18748028153712"><a name="p18748028153712"></a><a name="p18748028153712"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p37484282373"><a name="p37484282373"></a><a name="p37484282373"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p15748142816377"><a name="p15748142816377"></a><a name="p15748142816377"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p12748142823713"><a name="p12748142823713"></a><a name="p12748142823713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p18748102883718"><a name="p18748102883718"></a><a name="p18748102883718"></a>x</p>
</td>
</tr>
<tr id="row147481428143716"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p57489283370"><a name="p57489283370"></a><a name="p57489283370"></a>!=</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p1574892820379"><a name="p1574892820379"></a><a name="p1574892820379"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p12748162893719"><a name="p12748162893719"></a><a name="p12748162893719"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p974852815371"><a name="p974852815371"></a><a name="p974852815371"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p2748172816376"><a name="p2748172816376"></a><a name="p2748172816376"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p074872843711"><a name="p074872843711"></a><a name="p074872843711"></a>x</p>
</td>
</tr>
<tr id="row97481228183718"><td class="cellrowborder" rowspan="6" valign="top" width="11.75%" headers="mcps1.2.8.1.1 "><p id="p174818285372"><a name="p174818285372"></a><a name="p174818285372"></a>位运算符</p>
</td>
<td class="cellrowborder" valign="top" width="8.89%" headers="mcps1.2.8.1.2 "><p id="p974813287375"><a name="p974813287375"></a><a name="p974813287375"></a>&amp;</p>
</td>
<td class="cellrowborder" valign="top" width="8.48%" headers="mcps1.2.8.1.3 "><p id="p67481428193713"><a name="p67481428193713"></a><a name="p67481428193713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="31.03%" headers="mcps1.2.8.1.4 "><p id="p274815280376"><a name="p274815280376"></a><a name="p274815280376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="14.149999999999999%" headers="mcps1.2.8.1.5 "><p id="p2748428193716"><a name="p2748428193716"></a><a name="p2748428193716"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="16.13%" headers="mcps1.2.8.1.6 "><p id="p20748628113713"><a name="p20748628113713"></a><a name="p20748628113713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="9.569999999999999%" headers="mcps1.2.8.1.7 "><p id="p374815286375"><a name="p374815286375"></a><a name="p374815286375"></a>x</p>
</td>
</tr>
<tr id="row57481328203718"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p137482281371"><a name="p137482281371"></a><a name="p137482281371"></a>|</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p374832873713"><a name="p374832873713"></a><a name="p374832873713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p6748328183713"><a name="p6748328183713"></a><a name="p6748328183713"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p1474818289377"><a name="p1474818289377"></a><a name="p1474818289377"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p1074914288375"><a name="p1074914288375"></a><a name="p1074914288375"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p17749192803719"><a name="p17749192803719"></a><a name="p17749192803719"></a>x</p>
</td>
</tr>
<tr id="row10749102818374"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p674942853711"><a name="p674942853711"></a><a name="p674942853711"></a>^</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p207492028183713"><a name="p207492028183713"></a><a name="p207492028183713"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p1374992803714"><a name="p1374992803714"></a><a name="p1374992803714"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p1174932815377"><a name="p1174932815377"></a><a name="p1174932815377"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p1749152833719"><a name="p1749152833719"></a><a name="p1749152833719"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p97491628113716"><a name="p97491628113716"></a><a name="p97491628113716"></a>x</p>
</td>
</tr>
<tr id="row127490289377"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p13749152815370"><a name="p13749152815370"></a><a name="p13749152815370"></a>~</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p674952853719"><a name="p674952853719"></a><a name="p674952853719"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p27493282376"><a name="p27493282376"></a><a name="p27493282376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p177498283371"><a name="p177498283371"></a><a name="p177498283371"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p87491528113716"><a name="p87491528113716"></a><a name="p87491528113716"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p18749182833710"><a name="p18749182833710"></a><a name="p18749182833710"></a>x</p>
</td>
</tr>
<tr id="row174932863718"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p8749128163720"><a name="p8749128163720"></a><a name="p8749128163720"></a>&lt;&lt;</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p77495289371"><a name="p77495289371"></a><a name="p77495289371"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p07491628113711"><a name="p07491628113711"></a><a name="p07491628113711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p97491128173716"><a name="p97491128173716"></a><a name="p97491128173716"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p137497284372"><a name="p137497284372"></a><a name="p137497284372"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p117498288373"><a name="p117498288373"></a><a name="p117498288373"></a>x</p>
</td>
</tr>
<tr id="row174919282376"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p11749182813377"><a name="p11749182813377"></a><a name="p11749182813377"></a>&gt;&gt;</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p107491428143715"><a name="p107491428143715"></a><a name="p107491428143715"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p10749228133717"><a name="p10749228133717"></a><a name="p10749228133717"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p474922817378"><a name="p474922817378"></a><a name="p474922817378"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p174915288375"><a name="p174915288375"></a><a name="p174915288375"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p127491728123720"><a name="p127491728123720"></a><a name="p127491728123720"></a>x</p>
</td>
</tr>
<tr id="row13749102818372"><td class="cellrowborder" rowspan="3" valign="top" width="11.75%" headers="mcps1.2.8.1.1 "><p id="p14749112820370"><a name="p14749112820370"></a><a name="p14749112820370"></a>逻辑运算符</p>
</td>
<td class="cellrowborder" valign="top" width="8.89%" headers="mcps1.2.8.1.2 "><p id="p1874910286374"><a name="p1874910286374"></a><a name="p1874910286374"></a>&amp;&amp;</p>
</td>
<td class="cellrowborder" valign="top" width="8.48%" headers="mcps1.2.8.1.3 "><p id="p1574911288371"><a name="p1574911288371"></a><a name="p1574911288371"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="31.03%" headers="mcps1.2.8.1.4 "><p id="p16749928163715"><a name="p16749928163715"></a><a name="p16749928163715"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="14.149999999999999%" headers="mcps1.2.8.1.5 "><p id="p1374932893715"><a name="p1374932893715"></a><a name="p1374932893715"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="16.13%" headers="mcps1.2.8.1.6 "><p id="p197496287379"><a name="p197496287379"></a><a name="p197496287379"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="9.569999999999999%" headers="mcps1.2.8.1.7 "><p id="p167491828193716"><a name="p167491828193716"></a><a name="p167491828193716"></a>x</p>
</td>
</tr>
<tr id="row107491528103715"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p1474992873713"><a name="p1474992873713"></a><a name="p1474992873713"></a>||</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p9749182819375"><a name="p9749182819375"></a><a name="p9749182819375"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p1474915289376"><a name="p1474915289376"></a><a name="p1474915289376"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p1974902814372"><a name="p1974902814372"></a><a name="p1974902814372"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p574962863717"><a name="p574962863717"></a><a name="p574962863717"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p18749142814370"><a name="p18749142814370"></a><a name="p18749142814370"></a>x</p>
</td>
</tr>
<tr id="row1474917283372"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p187499282375"><a name="p187499282375"></a><a name="p187499282375"></a>!</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p974962843716"><a name="p974962843716"></a><a name="p974962843716"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.3 "><p id="p1474920282378"><a name="p1474920282378"></a><a name="p1474920282378"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.4 "><p id="p6749728203713"><a name="p6749728203713"></a><a name="p6749728203713"></a>√</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.5 "><p id="p11749112813371"><a name="p11749112813371"></a><a name="p11749112813371"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.6 "><p id="p187501628133717"><a name="p187501628133717"></a><a name="p187501628133717"></a>x</p>
</td>
</tr>
<tr id="row1975042813373"><td class="cellrowborder" valign="top" width="11.75%" headers="mcps1.2.8.1.1 "><p id="p15750128193711"><a name="p15750128193711"></a><a name="p15750128193711"></a>条件运算符</p>
</td>
<td class="cellrowborder" valign="top" width="8.89%" headers="mcps1.2.8.1.2 "><p id="p117501528163717"><a name="p117501528163717"></a><a name="p117501528163717"></a>a ? b : c</p>
</td>
<td class="cellrowborder" valign="top" width="8.48%" headers="mcps1.2.8.1.3 "><p id="p175022883711"><a name="p175022883711"></a><a name="p175022883711"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="31.03%" headers="mcps1.2.8.1.4 "><p id="p37502280371"><a name="p37502280371"></a><a name="p37502280371"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="14.149999999999999%" headers="mcps1.2.8.1.5 "><p id="p375032883716"><a name="p375032883716"></a><a name="p375032883716"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="16.13%" headers="mcps1.2.8.1.6 "><p id="p117501282372"><a name="p117501282372"></a><a name="p117501282372"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="9.569999999999999%" headers="mcps1.2.8.1.7 "><p id="p1575022814377"><a name="p1575022814377"></a><a name="p1575022814377"></a>x</p>
</td>
</tr>
</tbody>
</table>

运算符使用示例如下所示：

```
// 加法运算
res[idx] = x[idx] + y[idx]; 

// 取反运算
x[idx] = (-x[idx]);

// 比较运算
if (x[idx] > y[idx]) {
    res[idx] = x[idx];
} else {
    res[idx] = y[idx];
}

// 按位与运算
res[idx] = x[idx] & y[idx];

// 逻辑或运算
if (x[idx] || y[idx]) {
    res[idx] = 1;
}

// 条件运算
res[idx] = x[idx] > y[idx] ? x[idx] : y[idx];
```

## 核函数配置<a name="section97005415463"></a>

在调用\_\_global\_\_限定符修饰的函数时必须指定执行配置。执行配置通过在函数名后带括号的参数列表之间插入，形如：

```
<<<grid_dim, block_dim, dynamic_mem_size, stream>>>
```

其中：

-   grid\_dim：int或dim3类型，用于指定网格（grid）的维度与规模，grid\_dim.x \* grid\_dim.y \* grid\_dim.z等于启动的线程块总数。
-   block\_dim：int或dim3类型，用于指定每个线程块（block）的维度与规模，block\_dim.x \* block\_dim.y \* block\_dim.z等于每个线程块包含的线程数。
-   dynamic\_mem\_size：size\_t类型，该参数指定除静态分配的内存外，本次调用为每个线程块动态分配的共享内存字节数。
-   stream：aclrtStream类型指针，指定关联的流，用于维护异步操作的执行顺序。

以下示例展示了内核函数的声明与调用方式。

```
// 声明
__global__ void add_custom(float* x, float* y, float* z, uint64_t total_length);
// 调用
add_custom<<<block_num, thread_num_per_block, dyn_ubuf_size, stream>>>(x, y, z, 1024);
```

在执行函数之前，会先对上述配置参数进行校验。如果grid\_dim或block\_dim超出设备的最大允许规模，或dynamic\_smem\_bytes超过分配静态内存后剩余的可用共享内存，该函数将会执行失败。

在多线程并发执行时，每个线程使用较少的寄存器可以让更多的线程和线程块驻留在AI处理器上，从而提升性能。因此，编译器会采用启发式算法，将寄存器溢出（register spilling）和指令数量控制在最低水平，同时尽量减少寄存器的使用量。应用程序可以通过在\_\_global\_\_函数定义中使用\_\_launch\_bounds\_\_\(\)限定符来限制启动边界（launch bounds），提供附加信息辅助编译器优化这一过程，这属于可选配置。

-   \_\_launch\_bounds\_\_\(N\) <a name="li23861114618"></a>
  
    函数标记宏，在SIMT VF入口函数上可选配置，用于在编译期指定SIMT VF启动的最大线程数。若未配置\_\_launch\_bounds\_\_，最大线程数默认为1024。参数N需要满足：
    -   N \>= dimx \* dimy \* dimz；dimx，dimy，dimz为表示线程的dim3结构体。
    -   N的取值范围为1到2048。

        最大线程数决定了每个线程可分配的寄存器数量，具体对应关系请见下表，寄存器用于存储线程中的局部变量，若局部变量的个数超出寄存器个数，容易出现栈溢出等问题。建议最大线程数与启动VF任务的dim3线程数保持一致。

        **表 5**  \_\_launch\_bounds\_\_的Thread数量与每个Thread可用寄存器数

        <a name="table1715318510594"></a>
        <table><thead align="left"><tr id="row8153550597"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p13153175175919"><a name="p13153175175919"></a><a name="p13153175175919"></a>Thread的个数(个)</p>
        </th>
        <th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p101530520595"><a name="p101530520595"></a><a name="p101530520595"></a>每个Thread可用寄存器个数(个)</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row01531535912"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p201531053591"><a name="p201531053591"></a><a name="p201531053591"></a>1025~2048</p>
        </td>
        <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1615335185910"><a name="p1615335185910"></a><a name="p1615335185910"></a>16</p>
        </td>
        </tr>
        <tr id="row415395155912"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1215375195913"><a name="p1215375195913"></a><a name="p1215375195913"></a>513~1024</p>
        </td>
        <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18153125145914"><a name="p18153125145914"></a><a name="p18153125145914"></a>32</p>
        </td>
        </tr>
        <tr id="row11153115165910"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14153257595"><a name="p14153257595"></a><a name="p14153257595"></a>257~512</p>
        </td>
        <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p115315510596"><a name="p115315510596"></a><a name="p115315510596"></a>64</p>
        </td>
        </tr>
        <tr id="row515335155914"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p815315165914"><a name="p815315165914"></a><a name="p815315165914"></a>1~256</p>
        </td>
        <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p71539519590"><a name="p71539519590"></a><a name="p71539519590"></a>127</p>
        </td>
        </tr>
        </tbody>
        </table>

        配置SIMT函数最大线程数为512，示例如下：

        ```
        __simt_vf__ __launch_bounds__(512) inline void add(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
        ```
