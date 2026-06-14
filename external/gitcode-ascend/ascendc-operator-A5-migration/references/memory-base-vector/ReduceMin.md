> **原始文档路径**: asc-devkit/docs/api/context/ReduceMin.md

# ReduceMin<a name="ZH-CN_TOPIC_0000001475218545"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
<tr id="row174192367212"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p07746191307"><a name="p07746191307"></a><a name="p07746191307"></a><span id="ph880814542020"><a name="ph880814542020"></a><a name="ph880814542020"></a>Kirin X90</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p18774619608"><a name="p18774619608"></a><a name="p18774619608"></a>√</p>
</td>
</tr>
<tr id="row423773710210"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p4336317115"><a name="p4336317115"></a><a name="p4336317115"></a><span id="ph115418716"><a name="ph115418716"></a><a name="ph115418716"></a>Kirin 9030</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p73313312117"><a name="p73313312117"></a><a name="p73313312117"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在所有的输入数据中找出最小值及最小值对应的索引位置。ReduceMin计算原理参考[ReduceMax](ReduceMax.md)。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T>
    __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T>
        __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
        ```

    -   mask连续模式

        ```
        template <typename T>
        __aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.59%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.41%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p333713610318"><a name="p333713610318"></a><a name="p333713610318"></a><span id="ph1333710362313"><a name="ph1333710362313"></a><a name="ph1333710362313"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/uint32_t/int32_t/half/float/uint64_t/int64_t</p>
<p id="p1261794045615"><a name="p1261794045615"></a><a name="p1261794045615"></a><span id="ph361715406569"><a name="ph361715406569"></a><a name="ph361715406569"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_1"><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p5161942102612"><a name="p5161942102612"></a><a name="p5161942102612"></a><span id="ph151614424266"><a name="ph151614424266"></a><a name="ph151614424266"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_1"><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p065555314"><a name="p065555314"></a><a name="p065555314"></a><span id="ph15441323310"><a name="ph15441323310"></a><a name="ph15441323310"></a>Kirin X90</span>，支持的数据类型为：half/float</p>
<p id="p082420411739"><a name="p082420411739"></a><a name="p082420411739"></a><span id="ph25359910414"><a name="ph25359910414"></a><a name="ph25359910414"></a>Kirin 9030</span>，支持的数据类型为：half/float</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p4428175618426"><a name="p4428175618426"></a><a name="p4428175618426"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p943019141428"><a name="p943019141428"></a><a name="p943019141428"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p437973612117"><a name="p437973612117"></a><a name="p437973612117"></a>LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10429155616425"><a name="p10429155616425"></a><a name="p10429155616425"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p164291756114215"><a name="p164291756114215"></a><a name="p164291756114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p96775194717"><a name="p96775194717"></a><a name="p96775194717"></a>源操作数。</p>
<p id="p1416625314471"><a name="p1416625314471"></a><a name="p1416625314471"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3681637191516"><a name="p3681637191516"></a><a name="p3681637191516"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1942985674213"><a name="p1942985674213"></a><a name="p1942985674213"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p111694654215"><a name="p111694654215"></a><a name="p111694654215"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p81161946104213"><a name="p81161946104213"></a><a name="p81161946104213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1419261951"><a name="p1419261951"></a><a name="p1419261951"></a>API执行期间，<strong id="b17399154557"><a name="b17399154557"></a><a name="b17399154557"></a>部分硬件型号</strong>需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求，具体计算方法可参考<a href="ReduceMax.md#fig911740112914">ReduceMax计算示意图</a>中的介绍。</p>
<p id="p1196185464219"><a name="p1196185464219"></a><a name="p1196185464219"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p154118565332"><a name="p154118565332"></a><a name="p154118565332"></a><span id="ph1197925693320"><a name="ph1197925693320"></a><a name="ph1197925693320"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p219462520"><a name="p219462520"></a><a name="p219462520"></a>数据类型需要与目的操作数保持一致。</p>
<p id="p1945113142551"><a name="p1945113142551"></a><a name="p1945113142551"></a><span id="ph44511914195515"><a name="ph44511914195515"></a><a name="ph44511914195515"></a>Ascend 950PR/Ascend 950DT</span>，因接口内部算法实现不同，无需使用sharedTmpBuffer，可以直接传入src或者任意大小的sharedTmpBuffer。</p>
<p id="p2045113147555"><a name="p2045113147555"></a><a name="p2045113147555"></a><span id="ph0451314155510"><a name="ph0451314155510"></a><a name="ph0451314155510"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_2"><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a>Atlas A3 推理系列产品</term></span>，需要使用sharedTmpBuffer。</p>
<p id="p5451214155517"><a name="p5451214155517"></a><a name="p5451214155517"></a><span id="ph16451121435516"><a name="ph16451121435516"></a><a name="ph16451121435516"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_2"><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a>Atlas A2 推理系列产品</term></span>，需要使用sharedTmpBuffer。</p>
<p id="p622702317911"><a name="p622702317911"></a><a name="p622702317911"></a><span id="ph10227122312915"><a name="ph10227122312915"></a><a name="ph10227122312915"></a>Kirin X90</span>，需要使用sharedTmpBuffer。</p>
<p id="p132275232918"><a name="p132275232918"></a><a name="p132275232918"></a><span id="ph32275230917"><a name="ph32275230917"></a><a name="ph32275230917"></a>Kirin 9030</span>，需要使用sharedTmpBuffer。</p>
</td>
</tr>
<tr id="row4855921113310"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p6399278339"><a name="p6399278339"></a><a name="p6399278339"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p3391275339"><a name="p3391275339"></a><a name="p3391275339"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p262616117531"><a name="p262616117531"></a><a name="p262616117531"></a>参与计算的元素个数。</p>
<p id="p163917274335"><a name="p163917274335"></a><a name="p163917274335"></a>参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。</p>
</td>
</tr>
<tr id="row103306116356"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p159578209413"><a name="p159578209413"></a><a name="p159578209413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p0554313181312"><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><span id="zh-cn_topic_0000001530181537_ph793119540147"><a name="zh-cn_topic_0000001530181537_ph793119540147"></a><a name="zh-cn_topic_0000001530181537_ph793119540147"></a>mask用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000001530181537_p121114581013"><a name="zh-cn_topic_0000001530181537_p121114581013"></a><a name="zh-cn_topic_0000001530181537_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1411059101"><a name="zh-cn_topic_0000001530181537_sup1411059101"></a><a name="zh-cn_topic_0000001530181537_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup1711155161017"><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup181195111019"><a name="zh-cn_topic_0000001530181537_sup181195111019"></a><a name="zh-cn_topic_0000001530181537_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000001530181537_p711354105"><a name="zh-cn_topic_0000001530181537_p711354105"></a><a name="zh-cn_topic_0000001530181537_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p229173384114"><a name="p229173384114"></a><a name="p229173384114"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p32933310418"><a name="p32933310418"></a><a name="p32933310418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>迭代次数。与<a href="zh-cn_topic_0000002271261208.md">通用参数说明</a>中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。</p>
</td>
</tr>
<tr id="row0863135810539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p557663119345"><a name="p557663119345"></a><a name="p557663119345"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="zh-cn_topic_0000002271261208.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
<tr id="row1597161784710"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p166821220114714"><a name="p166821220114714"></a><a name="p166821220114714"></a>calIndex</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p10682142004716"><a name="p10682142004716"></a><a name="p10682142004716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1468222074717"><a name="p1468222074717"></a><a name="p1468222074717"></a>指定是否获取最小值的索引，bool类型，默认值为false，取值：</p>
<a name="ul1168232044718"></a><a name="ul1168232044718"></a><ul id="ul1168232044718"><li>true：同时获取最小值和最小值索引。</li><li>false：不获取索引，只获取最小值。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section17124037164714"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。需要使用sharedTmpBuffer的情况下，支持dst与sharedTmpBuffer地址重叠（通常情况下dst比sharedTmpBuffer所需的空间要小），此时sharedTmpBuffer必须满足最小所需空间要求，否则不支持地址重叠。

-   dst结果存储顺序为最小值，最小值索引，若不需要索引，只会存储最小值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret\_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret\_cast<uint16\_t\*\>，若输入是float，需要使用reinterpret\_cast<uint32\_t\*\>。比如[tensor高维切分计算接口完整调用示例](#li0437172884019)中，计算结果为\[0.01034,  2.104e-05\]，2.104e-05需要使用reinterpret\_cast方法转换得到索引值353。转换示例如下：

    ```
    float minIndex = dst.GetValue(1);
    uint32_t realIndex = *reinterpret_cast<uint32_t*>(&minIndex);
    ```

-   返回最小值索引时，如果存在多个最小值，返回第一个最小值的索引。
-   当输入类型是half的时候，只支持获取最大不超过65535（uint16\_t能表示的最大值）的索引值。
-   针对Ascend 950PR/Ascend 950DT，uint64\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section642mcpsimp"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算
    int32_t mask = 128;
    AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65,mask为全部元素参与计算
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
    ```

-   tensor前n个数据计算样例

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor前n个数据计算接口
    AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, 8320, true);
    ```

-   <a name="li0437172884019"></a>tensor高维切分计算接口完整调用示例

    ```
    #include "kernel_operator.h"
    class KernelReduce {
    public:
        __aicore__ inline KernelReduce() {}
        __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)src);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
            repeat = srcDataSize / mask;
            pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
            pipe.InitBuffer(workQueue, 1, 32 * sizeof(half)); // 此处按照公式计算所需的最小work空间为32，也就是64Bytes
            pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
            AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();
     
            AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, repeat, repStride, true);
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
            workQueue.FreeTensor(sharedTmpBuffer);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, srcDataSize);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        int srcDataSize = 512;
        int dstDataSize = 512;
        int mask = 128;
        int repStride = 8;
        int repeat = 0;
    };
    extern "C" __global__ __aicore__ void kernel_ReduceMin_lv0_half_512(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    示例结果如下：

    ```
    输入数据(src_gm):
    [0.769    0.8584   0.1082   0.2715   0.1759   0.7646   0.6406   0.2944   0.4255   0.927    0.8022   0.04507  0.9688   0.919    0.3008   0.7144   0.3206   0.6753   0.8276
     0.3374   0.4636   0.3591   0.112    0.93     0.822    0.7314   0.01165  0.31     0.5586   0.2808   0.3997   0.04544  0.0931   0.8438   0.612    0.03052  0.3652   0.1153
     0.06213  0.12103  0.4421   0.8003   0.1583   0.845    0.125    0.6934   0.4592   0.871    0.573    0.4133   0.885    0.6875   0.2854   0.7007   0.1294   0.2092   0.3794
     0.7534   0.5923   0.03888  0.2412   0.8584   0.6704   0.429    0.77     0.427    0.6323   0.524    0.0519   0.514    0.2408   0.09357  0.1702   0.3694   0.665    0.2651
     0.9507   0.661    0.459    0.1317   0.7334   0.289    0.0325   0.1187   0.6626   0.2769   0.3083   0.923    0.826    0.7275   0.976    0.4854   0.724    0.7783   0.8022
     0.677    0.2401   0.377    0.839    0.2297   0.54     0.743    0.511    0.1346   0.7183   0.4775   0.3442   0.561    0.2935   0.04065  0.1001   0.753    0.6816   0.8955
     0.07324  0.5947   0.508    0.2229   0.468    0.3135   0.0898   0.5625   0.7407   0.803    0.1071   0.6724   0.797    0.8296   0.807    0.8604   0.7437   0.967    0.4307
     0.3833   0.03394  0.02478  0.9385   0.3105   0.43     0.0706   0.4363   0.05832  0.0812   0.2418   0.03967  0.557    0.2705   0.963    0.8125   0.342    0.8853   0.3047
     0.7197   0.7173   0.02887  0.7695   0.4304   0.691    0.4285   0.9917   0.3994   0.19     0.3984   0.1888   0.83     0.0644   0.9766   0.857    0.09784  0.831    0.224
     0.8228   0.8975   0.1775   0.725    0.882    0.7188   0.3257   0.05347  0.1026   0.05902  0.9697   0.445    0.728    0.626    0.3577   0.711    0.2343   0.3865   0.03888
     0.3318   0.855    0.891    0.3647   0.9297   0.5083   0.7163   0.5737   0.2155   0.804    0.2118   0.525    0.1116   0.558    0.05203  0.6343   0.5796   0.5605   0.449
     0.4475   0.3713   0.3708   0.11017  0.2048   0.087    0.265    0.937    0.933    0.4683   0.5884   0.4312   0.9326   0.839    0.592    0.566    0.4229   0.05493  0.4578
     0.353    0.2915   0.8345   0.888    0.8394   0.8774   0.3582   0.2913   0.798    0.87     0.3372   0.6914   0.9185   0.4368   0.3276   0.8125   0.782    0.885    0.6543
     0.1626   0.0965   0.8247   0.03952  0.459    0.5596   0.694    0.59     0.02153  0.3762   0.2428   0.9727   0.3672   0.732    0.2676   0.2102   0.128    0.5957   0.988
     0.583    0.9097   0.144    0.3845   0.2151   0.327    0.2925   0.974    0.771    0.9224   0.147    0.6206   0.1774   0.1415   0.7637   0.573    0.9736   0.183    0.837
     0.0753   0.098    0.8184   0.08527  0.889    0.528    0.2207   0.1852   0.5903   0.594    0.04865  0.5806   0.6006   0.2048   0.4934   0.1302   0.7217   0.949    0.04105
     0.6875   0.3975   0.845    0.6045   0.4077   0.01927  0.1505   0.4407   0.8457   0.9614   0.4504   0.7134   0.07837  0.3557   0.521    0.545    0.02188  0.581    0.3215
     0.4458   0.853    0.4656   0.928    0.2927   0.3467   0.3516   0.1686   0.88     0.1509   0.2993   0.4006   0.611    0.1251   0.0887   0.896    0.2651   0.5596   0.0359
     0.6895   0.3494   0.871    0.673    0.1486   0.7812   0.0925   0.434    0.09985  0.02402  0.2932   0.01034  0.744    0.6357   0.658    0.1487   0.3416   0.1171   0.3088
     0.557    0.837    0.10944  0.7036   0.9097   0.3706   0.73     0.2844   0.78     0.5117   0.5537   0.776    0.6553   0.128    0.3184   0.8022   0.686    0.1785   0.2212
     0.74     0.8955   0.4773   0.6084   0.7827   0.239    0.4849   0.1816   0.2854   0.166    0.012505 0.4421   0.2179   0.06094  0.2124   0.409    0.641    0.1841   0.776
     0.4685   0.2334   0.4094   0.3447   0.6836   0.434    0.10516  0.514    0.8345   0.371    0.8555   0.5396   0.844    0.7554   0.171    0.749    0.7344   0.05936  0.4482
     0.9873   0.3137   0.7627   0.871    0.5503   0.956    0.2607   0.0904   0.535    0.3079   0.762    0.793    0.545    0.889    0.8936   0.6094   0.6533   0.5737   0.945
     0.4434   0.2686   0.05872  0.0776   0.0915   0.5386   0.6777   0.3164   0.8955   0.3398   0.3801   0.3784   0.3904   0.4849   0.816    0.962    0.335    0.705    0.1871
     0.3643   0.7163   0.6484   0.4526   0.8096   0.2408   0.608    0.0215   0.7246   0.412    0.609    0.03342  0.653    0.0424   0.672    0.627    0.3025   0.9424   0.3784
     0.1012   0.4192   0.7695   0.7383   0.9395   0.06494  0.3027   0.11523  0.6035   0.1727   0.4048   0.932    0.4053   0.3528   0.8193   0.0355   0.01953  0.574    0.509
     0.1443   0.0848   0.568    0.8716   0.968    0.613    0.535    0.0389   0.84     0.0655   0.127    0.06104  0.526    0.504    0.4175   0.8027   0.482    0.304   ]
    输出数据(dst_gm):
    [0.01034,  2.104e-05], 2.104e-05需要使用reinterpret_cast方法转换得到索引值353
    ```

-   tensor前n个数据计算接口完整调用示例：

    ```
    #include "kernel_operator.h"
    class KernelReduce {
    public:
        __aicore__ inline KernelReduce() {}
        __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)src);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
            repeatTime = srcDataSize / mask;
            pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
            pipe.InitBuffer(workQueue, 1, 32 * sizeof(half)); // 此处按照公式计算所需的最小work空间为32,也就是64Bytes
            pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
            AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();
           
            // level2
            AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, srcDataSize, true);
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
            workQueue.FreeTensor(sharedTmpBuffer);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        int srcDataSize = 288;
        int dstDataSize = 16;
        int mask = 128;
        int repStride = 8;
        int repeatTime = 0;
    };
    extern "C" __global__ __aicore__ void kernel_ReduceMin_lv2_half_288(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    示例结果如下：

    ```
    示例结果
    输入数据(src_gm):
    [0.556    0.5225   0.3623   0.214    0.556    0.0643   0.769    0.594    0.261    0.3652   0.911    0.924    0.386    0.3696   0.2296   0.5957   0.1709   0.79     0.8516
     0.341    0.705    0.728    0.8135   0.7534   0.5874   0.771    0.05835  0.7456   0.1049   0.3105   0.1729   0.9253   0.8003   0.918    0.5005   0.7744   0.688    0.6807
     0.1456   0.4136   0.1055   0.12054  0.275    0.3848   0.08405  0.3843   0.3218   0.6904   0.878    0.3706   0.3586   0.3518   0.429    0.7275   0.6123   0.8096   0.563
     0.54     0.8857   0.8594   0.4143   0.525    0.2744   0.1376   0.382    0.6406   0.1534   0.134    0.2993   0.365    0.8843   0.2986   0.00393  0.6577   0.313    0.8164
     0.8706   0.7686   0.873    0.3286   0.03787  0.8145   0.4656   0.66     0.1362   0.1075   0.1376   0.9097   0.9214   0.833    0.3657   0.8438   0.006973 0.2408   0.801
     0.1862   0.864    0.8745   0.1805   0.4324   0.8647   0.844    0.8936   0.8496   0.311    0.0334   0.3967   0.579    0.43     0.2332   0.5366   0.3557   0.3542   0.945
     0.9336   0.252    0.4375   0.9727   0.859    0.6294   0.6787   0.8887   0.1884   0.524    0.787    0.04755  0.3984   0.0508   0.4065   0.716    0.3184   0.21     0.10645
     0.7544   0.2827   0.7856   0.4878   0.5903   0.12146  0.6426   0.8438   0.063    0.7617   0.6396   0.1995   0.6475   0.1464   0.7617   0.514    0.3506   0.2708   0.8643
     0.1204   0.04337  0.21     0.528    0.0644   0.2133   0.0643   0.0125   0.602    0.654    0.866    0.225    0.9473   0.408    0.4597   0.2793   0.11145  0.293    0.04156
     0.7705   0.3555   0.3977   0.7485   0.76     0.9824   0.2832   0.1239   0.4915   0.878    0.5986   0.7217   0.832    0.6206   0.6455   0.0639   0.772    0.01854  0.7437
     0.1962   0.485    0.5483   0.414    0.9253   0.2452   0.2942   0.9478   0.879    0.586    0.659    0.635    0.7197   0.933    0.08905  0.02892  0.74     0.499    0.02054
     0.2241   0.5137   0.8325   0.185    0.6196   0.949    0.935    0.5605   0.04108  0.3672   0.5566   0.3958   0.4565   0.8135   0.3015   0.46     0.1196   0.5044   0.54
     0.05203  0.687    0.8525   0.501    0.3464   0.307    0.804    0.0926   0.202    0.999    0.955    0.581    0.06216  0.271    0.9365   0.854    0.4202   0.269    0.985
     0.04547  1.       0.1208   0.5225   0.00935  0.4128   0.644    0.3826   0.6963   0.2942   0.007626 0.7144   0.609    0.3206   0.694    0.393    0.6265   0.6904   0.2487
     0.9478   0.798    0.891    0.8867   0.9414   0.395    0.11285  0.515    0.919    0.013855 0.749    0.5527   0.465    0.451    0.1458   0.59     0.893    0.0146   0.062
     0.06604  0.934    0.2242  ]
    输出数据(dst_gm):
    [0.00393,  4.3e-06], 4.3e-06需要使用reinterpret_cast方法转换得到索引值72
    ```

