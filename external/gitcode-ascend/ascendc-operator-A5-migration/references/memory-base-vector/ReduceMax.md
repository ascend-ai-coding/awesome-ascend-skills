> **原始文档路径**: asc-devkit/docs/api/context/ReduceMax.md

# ReduceMax<a name="ZH-CN_TOPIC_0000001474898269"></a>

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
<tr id="row7641213945"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p156412131643"><a name="p156412131643"></a><a name="p156412131643"></a><span id="ph20592249245"><a name="ph20592249245"></a><a name="ph20592249245"></a>Kirin X90</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1264115131946"><a name="p1264115131946"></a><a name="p1264115131946"></a>√</p>
</td>
</tr>
<tr id="row022212151644"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622221510412"><a name="p1622221510412"></a><a name="p1622221510412"></a><span id="ph13800175210420"><a name="ph13800175210420"></a><a name="ph13800175210420"></a>Kirin 9030</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1222515541"><a name="p1222515541"></a><a name="p1222515541"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在所有的输入数据中找出最大值及最大值对应的索引位置。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T>
    __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T>
        __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
        ```

    -   mask连续模式

        ```
        template <typename T>
        __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
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
<p id="p722214293126"><a name="p722214293126"></a><a name="p722214293126"></a><span id="ph6222129101217"><a name="ph6222129101217"></a><a name="ph6222129101217"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/uint32_t/int32_t/half/float/uint64_t/int64_t</p>
<p id="p1261794045615"><a name="p1261794045615"></a><a name="p1261794045615"></a><span id="ph361715406569"><a name="ph361715406569"></a><a name="ph361715406569"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_1"><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p5161942102612"><a name="p5161942102612"></a><a name="p5161942102612"></a><span id="ph151614424266"><a name="ph151614424266"></a><a name="ph151614424266"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_1"><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p854386131917"><a name="p854386131917"></a><a name="p854386131917"></a><span id="ph31931459192112"><a name="ph31931459192112"></a><a name="ph31931459192112"></a>Kirin X90</span>，支持的数据类型为：half、float。</p>
<p id="p9991425121910"><a name="p9991425121910"></a><a name="p9991425121910"></a><span id="ph158218232210"><a name="ph158218232210"></a><a name="ph158218232210"></a>Kirin 9030</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.57125712571257%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.76737673767377%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p4428175618426"><a name="p4428175618426"></a><a name="p4428175618426"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p45731031124014"><a name="p45731031124014"></a><a name="p45731031124014"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1871313861718"><a name="p1871313861718"></a><a name="p1871313861718"></a>LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10429155616425"><a name="p10429155616425"></a><a name="p10429155616425"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p164291756114215"><a name="p164291756114215"></a><a name="p164291756114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p390591211424"><a name="p390591211424"></a><a name="p390591211424"></a>源操作数。</p>
<p id="p943019141428"><a name="p943019141428"></a><a name="p943019141428"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3681637191516"><a name="p3681637191516"></a><a name="p3681637191516"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1942985674213"><a name="p1942985674213"></a><a name="p1942985674213"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p111694654215"><a name="p111694654215"></a><a name="p111694654215"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p81161946104213"><a name="p81161946104213"></a><a name="p81161946104213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>API执行期间，<strong id="b17399154557"><a name="b17399154557"></a><a name="b17399154557"></a>部分硬件型号</strong>需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求，具体计算方法可参考下文<a href="#fig911740112914">ReduceMax计算示意图</a>中的介绍。</p>
<p id="p1196185464219"><a name="p1196185464219"></a><a name="p1196185464219"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p154118565332"><a name="p154118565332"></a><a name="p154118565332"></a><span id="ph1197925693320"><a name="ph1197925693320"></a><a name="ph1197925693320"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p141161146144213"><a name="p141161146144213"></a><a name="p141161146144213"></a>数据类型需要与目的操作数保持一致。</p>
<p id="p1945113142551"><a name="p1945113142551"></a><a name="p1945113142551"></a><span id="ph44511914195515"><a name="ph44511914195515"></a><a name="ph44511914195515"></a>Ascend 950PR/Ascend 950DT</span>，因接口内部算法实现不同，无需使用sharedTmpBuffer，可以直接传入src或者任意大小的sharedTmpBuffer。</p>
<p id="p2045113147555"><a name="p2045113147555"></a><a name="p2045113147555"></a><span id="ph0451314155510"><a name="ph0451314155510"></a><a name="ph0451314155510"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_2"><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a>Atlas A3 推理系列产品</term></span>，需要使用sharedTmpBuffer。</p>
<p id="p5451214155517"><a name="p5451214155517"></a><a name="p5451214155517"></a><span id="ph16451121435516"><a name="ph16451121435516"></a><a name="ph16451121435516"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_2"><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a>Atlas A2 推理系列产品</term></span>，需要使用sharedTmpBuffer。</p>
<p id="p12793221078"><a name="p12793221078"></a><a name="p12793221078"></a><span id="ph927942217710"><a name="ph927942217710"></a><a name="ph927942217710"></a>Kirin X90</span>，需要使用sharedTmpBuffer。</p>
<p id="p72796221673"><a name="p72796221673"></a><a name="p72796221673"></a><span id="ph22791922573"><a name="ph22791922573"></a><a name="ph22791922573"></a>Kirin 9030</span>，需要使用sharedTmpBuffer。</p>
</td>
</tr>
<tr id="row138119463010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p788231418307"><a name="p788231418307"></a><a name="p788231418307"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p148821214143019"><a name="p148821214143019"></a><a name="p148821214143019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p2077285419527"><a name="p2077285419527"></a><a name="p2077285419527"></a>参与计算的元素个数。</p>
<p id="p874075111528"><a name="p874075111528"></a><a name="p874075111528"></a>参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。</p>
</td>
</tr>
<tr id="row59001123015"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10882714133016"><a name="p10882714133016"></a><a name="p10882714133016"></a>calIndex</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p15882414143017"><a name="p15882414143017"></a><a name="p15882414143017"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p2882121414309"><a name="p2882121414309"></a><a name="p2882121414309"></a>指定是否获取最大值的索引，bool类型，默认值为false，取值：</p>
<a name="ul78821714173011"></a><a name="ul78821714173011"></a><ul id="ul78821714173011"><li>true：同时获取最大值和最大值索引。</li><li>false：不获取索引，只获取最大值。</li></ul>
</td>
</tr>
<tr id="row103306116356"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p159578209413"><a name="p159578209413"></a><a name="p159578209413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p0554313181312"><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><span id="zh-cn_topic_0000001530181537_ph793119540147"><a name="zh-cn_topic_0000001530181537_ph793119540147"></a><a name="zh-cn_topic_0000001530181537_ph793119540147"></a>mask用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000001530181537_p121114581013"><a name="zh-cn_topic_0000001530181537_p121114581013"></a><a name="zh-cn_topic_0000001530181537_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1411059101"><a name="zh-cn_topic_0000001530181537_sup1411059101"></a><a name="zh-cn_topic_0000001530181537_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup1711155161017"><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup181195111019"><a name="zh-cn_topic_0000001530181537_sup181195111019"></a><a name="zh-cn_topic_0000001530181537_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000001530181537_p711354105"><a name="zh-cn_topic_0000001530181537_p711354105"></a><a name="zh-cn_topic_0000001530181537_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p229173384114"><a name="p229173384114"></a><a name="p229173384114"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p32933310418"><a name="p32933310418"></a><a name="p32933310418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>迭代次数。与<a href="zh-cn_topic_0000002271261208.md">通用参数说明</a>中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。</p>
</td>
</tr>
<tr id="row0863135810539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p557663119345"><a name="p557663119345"></a><a name="p557663119345"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.57125712571257%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.76737673767377%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="zh-cn_topic_0000002271261208.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

ReduceMax计算过程如[ReduceMax计算示意图](#fig911740112914)所示：先在每个repeat迭代中获取最大值及索引，作为中间结果存储在sharedTmpBuffer工作区中，然后在中间结果中再按照repeat迭代求最大值，以此类推，逐步求出最终的最大值和索引放在目的操作数中。需要注意的是，每次repeat迭代获取的最值索引是repeat内部索引，返回最终结果时，需要根据迭代位置和内部索引推导全量数据内的索引。

**图 1**  ReduceMax计算示意图<a name="fig911740112914"></a>  
![](figures/ReduceMax计算示意图.png "ReduceMax计算示意图")

数据量较大时通过一轮归约无法得出最终结果，需要进行多轮计算，同理，每次repeat迭代获取的最值索引是repeat内部索引，返回最终结果时，需要根据迭代位置和内部索引推导全量数据内的索引。

**图 2**  多轮计算示意图<a name="fig02269322018"></a>  
![](figures/多轮计算示意图.png "多轮计算示意图")

sharedTmpBuffer空间需要开发者申请并传入，根据是否需要获取索引，sharedTmpBuffer空间计算方式不同：需要返回索引的情况下，需要把每轮计算所需的空间进行累加，同时每轮计算的空间都要考虑UB空间32字节对齐的要求；无需返回索引的情况下，只需要提供第一轮计算所需的空间并满足32字节对齐要求即可，后续的轮次可以直接使用这块空间，此时不需要推导索引的过程，所以之前轮次的中间数据可以直接覆盖。计算最小所需空间的算法如下：

-   无需返回最大值索引

    ```
    int firstMaxRepeat = repeatTime;           // 对于tensor高维切分计算接口，firstMaxRepeat就是repeatTime；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat
    int iter1OutputCount = firstMaxRepeat * 2;                                            // 第一轮操作产生的元素个数，无论开发者是否需要返回索引，底层指令都会返回索引，所以这里要为索引预留空间，产生的元素个数为repeat次数*2
    int iter1AlignEnd = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;   // 第一轮产生的元素个数按照datablock(32字节)向上对齐
    int finalWorkLocalNeedSize = iter1AlignEnd;                                           // 第一轮计算完成后，后续可能还需要多轮迭代，但是可以复用同一块空间，所以第一轮计算所需的空间就是最终sharedTmpBuffer所需的空间大小
    ```

-   需要返回最大值索引

    ```
    int firstMaxRepeat = repeatTime;           // 对于tensor高维切分计算接口，firstMaxRepeat就是repeatTime；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat
    int iter1OutputCount = firstMaxRepeat * 2;                                            // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock; // 第二轮操作起始位置偏移，即第一轮产生的元素个数按照datablock(32字节)向上对齐的结果
    // 第一轮计算完成后，后续可能还需要多轮迭代，此时不可以复用同一块空间，因为第一轮的中间结果索引还需要再进行使用，所以需要继续准备第二轮和第三轮的空间
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;              // 第二轮操作产生的元素个数
    int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock; // 第三轮操作起始位置偏移，即第二轮产生的元素个数按照datablock(32字节)向上对齐的结果
    int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;              // 第三轮操作产生的元素个数
    int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;   // 第三轮产生的元素个数按照datablock(32字节)向上对齐的结果
    int finalWorkLocalNeedSize = iter2AlignStart + iter3AlignStart + iter3AlignEnd;       // 最终sharedTmpBuffer所需的空间大小
    ```

以上计算出来的最终的空间大小单位是元素个数，若转成Bytes数表示为finalWorkLocalNeedSize \* typeSize \(Bytes\)，具体计算示例请参考[调用示例](#section107745237168)中sharedTmpBuffer空间计算示例。

> [!NOTE]说明 
>开发者为了节省地址空间，可以选择sharedTmpBuffer空间复用源操作数的空间。此时因为sharedTmpBuffer需要的最小空间一定小于源操作数的空间，所以无需关注和计算最小空间。

## 返回值说明<a name="section17124037164714"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。需要使用sharedTmpBuffer的情况下，支持dst与sharedTmpBuffer地址重叠（通常情况下dst比sharedTmpBuffer所需的空间要小），此时sharedTmpBuffer必须满足最小所需空间要求，否则不支持地址重叠。

-   dst结果存储顺序为最大值，最大值索引，若不需要索引，只会存储最大值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret\_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret\_cast<uint16\_t\*\>，若输入是float，需要使用reinterpret\_cast<uint32\_t\*\>，比如[tensor高维切分计算接口完整示例](#li7962161711114)中，输入数据是half类型，计算结果为\[0.9985,  6.8e-06\]，6.8e-06需要使用reinterpret\_cast<uint16\_t\*\>方法转换得到索引值114。转换示例如下：

    ```
    float maxIndex = dst.GetValue(1);
    uint32_t realIndex = *reinterpret_cast<uint32_t*>(&maxIndex);
    ```

-   返回最大值索引时，如果存在多个最大值，返回第一个最大值的索引。
-   当输入类型是half的时候，只支持获取最大不超过65535（uint16\_t能表示的最大值）的索引值。
-   针对Ascend 950PR/Ascend 950DT，uint64\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section107745237168"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算
    int32_t mask = 128;
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65,mask为全部元素参与计算
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
    ```

-   tensor前n个数据计算样例

    ```
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor前n个数据计算接口
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 8320, true);
    ```

-   sharedTmpBuffer空间计算示例

    ```
    // ReduceMax接口sharedTmpBuffer计算示例一:
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320, 使用tensor高维切分计算接口, repeatTime为65, mask为128,需要索引值
    // tensor高维切分计算接口调用示例为:
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 128, 65, 8, true);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int RoundUp(int a, int b)
    { 
        return (a + b - 1) / b;
    }
    int typeSize = 2;
    int elementsPerBlock = 32 / typeSize = 16; 
    int elementsPerRepeat = 256 / typeSize = 128; 
    int firstMaxRepeat = repeatTime;
    int iter1OutputCount = firstMaxRepeat * 2 = 130;                                          // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock)*elementsPerBlock = 144; // 对第一轮操作输出个数向上取整
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat)*2 = 4;                // 第二轮操作产生的元素个数
    int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock)*elementsPerBlock = 16;  // 对第二轮操作输出个数向上取整
    int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat)*2 = 2;                // 第三轮操作产生的元素个数
    int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock = 16;  // 第三轮产生的元素个数做向上取整
    // 最终sharedTmpBuffer所需的最小空间为 iter2AlignStart + iter3AlignStart + iter3AlignEnd = 144 + 16 + 16 = 176 ,也就是352Bytes
    // ReduceMax接口sharedTmpBuffer计算示例二:
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为32640, 使用tensor高维切分计算接口,repeatTime为255, mask为128,需要索引值
    // tensor高维切分计算接口调用示例为:
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 128, 255, 8, true);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int typeSize = 2;
    int elementsPerBlock = 32 / typeSize = 16; 
    int elementsPerRepeat = 256 / typeSize = 128; 
    int firstMaxRepeat = repeatTime;
    int iter1OutputCount = firstMaxRepeat * 2 = 510;                                          // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock)*elementsPerBlock = 512; // 对第一轮操作输出个数向上取整
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat)*2 = 8;                // 第二轮操作产生的元素个数
    int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock)*elementsPerBlock = 16;  // 对第二轮操作输出个数向上取整
    int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat)*2 = 2;                // 第三轮操作产生的元素个数
    int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock = 16;  // 第三轮产生的元素个数做向上取整
    // 需要的空间为 iter2AlignStart + iter3AlignStart + iter3AlignEnd = 512 + 16 + 16 = 544 ,也就是1088Bytes
    // ReduceMax接口sharedTmpBuffer计算示例三:
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为65408,使用tensor前n个数据计算接口,count=65408,需要索引值
    // tensor前n个数据计算接口调用示例为:
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 65408, true);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int typeSize = 2;
    int elementsPerBlock = 32 / typeSize = 16; 
    int elementsPerRepeat = 256 / typeSize = 128; 
    int firstMaxRepeat = count / elementsPerRepeat = 511;
    int iter1OutputCount = firstMaxRepeat * 2 = 1022;                                          // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock)*elementsPerBlock = 1024; // 对iter1OutputCount输出个数向上取整
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat)*2 = 16;                // 第二轮操作产生的元素个数
    int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock)*elementsPerBlock = 16;   // 对iter2OutputCount输出个数向上取整
    int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat)*2 = 2;                 // 第三轮操作产生的元素个数
    int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock = 16;   // 第三轮产生的元素个数做向上取整
    // 需要的空间为 iter2AlignStart + iter3AlignStart + iter3AlignEnd = 1024 + 16 + 16 = 1056,也就是2112Bytes
    // ReduceMax接口sharedTmpBuffer计算示例四:
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的的计算数据量为512,使用tensor高维切分计算接口,repeatTime为4, mask为128,需要索引值
    // tensor高维切分计算接口调用示例为:
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 128, 4, 8, true);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int typeSize = 2;
    int elementsPerBlock = 32 / typeSize = 16; 
    int elementsPerRepeat = 256 / typeSize = 128; 
    int firstMaxRepeat = repeatTime;
    int iter1OutputCount = firstMaxRepeat * 2 = 8;                                           // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock)*elementsPerBlock = 16; // 对iter1OutputCount输出个数向上取整
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat)*2 = 2;               // 第二轮操作产生的元素个数
    // 本用例中，由于第二轮操作产生的元素个数为2，即第二轮结束就可以拿到最大值和其索引值，因此需要的空间为iter2AlignStart + RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock = 16 + 16 = 32，也就是64Bytes
    // ReduceMax接口sharedTmpBuffer计算示例五:
    // dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量count为65408,使用tensor前n个数据计算接口,count=65408，不需要索引值
    // tensor前n个数据计算接口调用示例为:
    AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, 65408, false);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int typeSize = 2;
    int elementsPerBlock = 32 / typeSize = 16; 
    int elementsPerRepeat = 256 / typeSize = 128; 
    int firstMaxRepeat = count / elementsPerRepeat = 511;
    int iter1OutputCount = firstMaxRepeat * 2 = 1022;                                          // 第一轮操作产生的元素个数
    int iter1AlignEnd = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock = 1024; // 第一轮产生的元素个数做向上取整
    // 由于calIndex为false,因此最终sharedTmpBuffer所需的最小空间大小就是对第一轮产生元素做向上取整后的结果，此处就是1024，也就是2048Bytes
    // ReduceMax接口sharedTmpBuffer计算示例六:
    // dstLocal,srcLocal和sharedTmpBuffer均为float类型,srcLocal的计算数据量为8320, 使用tensor高维切分计算接口, repeatTime为130, mask为64,需要索引值
    // tensor高维切分计算接口调用示例为:
    AscendC::ReduceMax<float>(dstLocal, srcLocal, sharedTmpBuffer, 64, 130, 8, true);
    // 此时sharedTmpBuffer所需的最小空间计算过程为:
    int typeSize = 4;
    int elementsPerBlock = 32 / typeSize = 8; 
    int elementsPerRepeat = 256 / typeSize = 64; 
    int firstMaxRepeat = repeatTime;
    int iter1OutputCount = firstMaxRepeat * 2 = 260;                                          // 第一轮操作产生的元素个数
    int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock)*elementsPerBlock = 264; // 对第一轮操作输出个数向上取整
    int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat)*2 = 10;               // 第二轮操作产生的元素个数
    int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock)*elementsPerBlock = 16;  // 对第二轮操作输出个数向上取整
    int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat)*2 = 2;                // 第三轮操作产生的元素个数
    int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock = 8;   // 第三轮产生的元素个数做向上取整
    // 最终sharedTmpBuffer所需的最小空间就是 iter2AlignStart + iter3AlignStart + iter3AlignEnd = 264 + 16 + 8 = 288,也就是1152Bytes
    ```

-   <a name="li7962161711114"></a>tensor高维切分计算接口完整示例:

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
            AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, repeat, repStride, true);
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
    extern "C" __global__ __aicore__ void kernel_ReduceMax_lv0_half_512(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    示例结果如下：

    ```
    示例结果 输入数据(src_gm):
    [0.4795   0.951    0.866    0.008545 0.8037   0.551    0.754    0.73     0.6035   0.251    0.4841   0.05914  0.9414   0.379    0.664    0.6914   0.9307   0.3853   0.4048
     0.7754   0.1265   0.709    0.7695   0.8057   0.9673   0.2566   0.8696   0.243    0.871    0.123    0.76     0.1844   0.7324   0.5757   0.0172   0.7188   0.556    0.3699
     0.7334   0.655    0.919    0.4219   0.82     0.1046   0.5796   0.4773   0.1405   0.3777   0.4421   0.983    0.728    0.642    0.37     0.9473   0.52     0.7783   0.699
     0.716    0.1791   0.1272   0.2471   0.3298   0.3518   0.9756   0.2268   0.6167   0.742    0.4185   0.8193   0.919    0.03827  0.02957  0.2598   0.798    0.3752   0.2109
     0.1753   0.7227   0.829    0.6978   0.347    0.463    0.685    0.1992   0.847    0.941    0.835    0.03336  0.1359   0.04736  0.758    0.5347   0.616    0.869    0.582
     0.694    0.2035   0.3613   0.8413   0.68     0.0896   0.3833   0.0768   0.292    0.11053  0.5586   0.578    0.3286   0.09314  0.5845   0.7124   0.2058   0.6523   0.784
     0.9985   0.6626   0.8975   0.405    0.884    0.7744   0.0258   0.484    0.768    0.7197   0.577    0.03143  0.9185   0.3608   0.3352   0.9077   0.709    0.85     0.4607
     0.61     0.4277   0.1004   0.1995   0.1608   0.2852   0.8887   0.813    0.3396   0.272    0.703    0.1312   0.734    0.2612   0.6895   0.8647   0.9165   0.1455   0.9233
     0.3027   0.7163   0.927    0.1995   0.155    0.6953   0.66     0.04163  0.99     0.544    0.4243   0.804    0.4612   0.01912  0.5127   0.8755   0.6665   0.707    0.01018
     0.874    0.8545   0.9375   0.9844   0.578    0.934    0.683    0.4668   0.63     0.2032   0.3188   0.9478   0.9375   0.03357  0.9927   0.996    0.451    0.1105   0.762
     0.82     0.8047   0.911    0.926    0.1973   0.9175   0.4521   0.4487   0.1273   0.718    0.737    0.305    0.922    0.1396   0.618    0.753    0.5913   0.874    0.08905
     0.003582 0.05252  0.674    0.3923   0.527    0.4106   0.7812   0.113    0.965    0.6157   0.4368   0.6646   0.7944   0.7964   0.531    0.6665   0.517    0.04468  0.5737
     0.752    0.4      0.4463   0.05496  0.939    0.6353   0.2036   0.667    0.3994   0.2573   0.118    0.973    0.5923   0.558    0.7114   0.785    0.714    0.7485   0.854
     0.2585   0.274    0.9824   0.4158   0.283    0.2194   0.3074   0.2793   0.531    0.8965   0.01456  0.5264   0.992    0.856    0.5986   0.734    0.908    0.12317  0.8374
     0.6665   0.1904   0.97     0.2546   0.364    0.6914   0.462    0.05353  0.02975  0.6235   0.4941   0.4714   0.788    0.06537  0.8423   0.2527   0.7734   0.591    0.443
     0.3022   0.02116  0.01605  0.772    0.6924   0.01032  0.594    0.1865   0.7393   0.8887   0.916    0.9653   0.696    0.901    0.1255   0.5513   0.2742   0.5586   0.988
     0.0954   0.4365   0.677    0.894    0.8413   0.05655  0.932    0.4426   0.336    0.848    0.9434   0.1976   0.813    0.773    0.2605   0.1543   0.8555   0.3596   0.997
     0.10315  0.5796   0.5327   0.2283   0.7583   0.3674   0.513    0.9126   0.751    0.532    0.399    0.832    0.549    0.2358   0.6655   0.477    0.5864   0.3528   0.989
     0.1412   0.748    0.3652   0.05292  0.3552   0.5767   0.826    0.4792   0.8477   0.03488  0.8267   0.2345   0.931    0.0884   0.6816   0.4685   0.618    0.09973  0.4385
     0.782    0.6465   0.03882  0.4158   0.1422   0.822    0.8203   0.95     0.3274   0.724    0.929    0.8726   0.004307 0.815    0.67     0.4368   0.7793   0.593    0.4663
     0.2207   0.01773  0.39     0.008896 0.4238   0.716    0.1155   0.601    0.9214   0.3708   0.4285   0.951    0.00431  0.726    0.977    0.1254   0.6484   0.4648   0.891
     0.723    0.6333   0.9077   0.4849   0.3008   0.0495   0.4575   0.266    0.2014   0.1106   0.6914   0.2744   0.4956   0.532    0.1752   0.709    0.3464   0.6104   0.4067
     0.1317   0.8647   0.8      0.4832   0.013855 0.6733   0.4524   0.6865   0.7017   0.9385   0.2957   0.2444   0.4167   0.55     0.8926   0.8364   0.506    0.9966   0.7207
     0.51     0.8745   0.3188   0.847    0.86     0.64     0.08453  0.59     0.2062   0.1031   0.1459   0.3806   0.2096   0.469    0.1492   0.10065  0.536    0.572    0.353
     0.068    0.07855  0.6177   0.3408   0.1538   0.2732   0.997    0.1158   0.4028   0.9536   0.7197   0.585    0.0899   0.3994   0.1835   0.737    0.4639   0.3071   0.47
     0.993    0.3862   0.293    0.1813   0.8193   0.745    0.064    0.7407   0.329    0.198    0.596    0.3      0.6562   0.819    0.2803   0.04095  0.703    0.3425   0.9224
     0.776    0.8057   0.734    0.2534   0.1824   0.793    0.3542   0.2595   0.2607   0.838    0.39     0.631    0.3542   0.1968   0.643    0.015366 0.4106   0.604   ]
    输出数据(dst_gm):
    [0.9985,  6.8e-06], 6.8e-06使用reinterpret_cast方法转换后为索引值114
    ```

-   tensor前n个数据计算接口完整调用示例:

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
    
            // level2
            AscendC::ReduceMax<half>(dstLocal, srcLocal, sharedTmpBuffer, srcDataSize, true);
    
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
    extern "C" __global__ __aicore__ void kernel_ReduceMax_lv2_half_288(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    示例结果如下：

    ```
    示例结果 输入数据(src_gm):
    [0.4778   0.5903   0.2433   0.698    0.1943   0.407    0.891    0.1766   0.5977   0.9473   0.6523   0.10913  0.0143   0.86     0.2366   0.625    0.3696   0.708    0.946
     0.538    0.3826   0.08215  0.516    0.9116   0.1548   0.507    0.8145   0.89     0.5435   0.563    0.1125   0.543    0.3142   0.8086   0.6885   0.874    0.855    0.4019
     0.1613   0.04462  0.945    0.6064   0.6904   0.00758  0.9463   0.528    0.9966   0.629    0.714    0.03134  0.4407   0.0322   0.5376   0.04443  0.03778  0.522    0.793
     0.3086   0.4      0.3984   0.5693   0.8203   0.673    0.796    0.2747   0.2246   0.468    0.1146   0.4468   0.419    0.3816   0.1636   0.1414   0.4028   0.9785   0.8984
     0.4355   0.874    0.864    0.7856   0.739    0.895    0.2487   0.5034   0.958    0.661    0.8755   0.302    0.802    0.563    0.9067   0.1562   0.1337   0.1844   0.3047
     0.543    0.3855   0.9536   0.8633   0.5435   0.002748 0.8916   0.9614   0.3665   0.1588   0.51     0.77     0.552    0.84     0.2798   0.7217   0.8633   0.3794   0.5376
     0.03     0.7783   0.9297   0.9556   0.609    0.1776   0.5957   0.2954   0.6675   0.7183   0.4182   0.8804   0.1837   0.3235   0.3486   0.43     0.8633   0.3972   0.1307
     0.7915   0.43     0.2544   0.827    0.04843  0.1637   0.3376   0.4087   0.4993   0.5923   0.3057   0.04306  0.4905   0.693    0.7393   0.777    0.01379  0.2742   0.669
     0.6826   0.04028  0.0423   0.281    0.12476  0.5366   0.2098   0.559    0.8833   0.82     0.0745   0.7485   0.04004  0.776    0.863    0.1909   0.7876   0.734    0.4727
     0.3655   0.944    0.006794 0.01872  0.687    0.5664   0.9697   0.2437   0.2014   0.0269   0.3975   0.08405  0.36     0.0751   0.02632  0.135    0.531    0.554    0.378
     0.9365   0.5254   0.8687   0.181    0.329    0.322    0.3076   0.508    0.638    0.3462   0.3882   0.7705   0.5933   0.994    0.1188   0.0782   0.94     0.00856  0.1396
     0.2191   0.00648  0.8994   0.6714   0.6724   0.57     0.3127   0.4905   0.2119   0.3938   0.5957   0.1493   0.9424   0.716    0.3699   0.829    0.647    0.8286   0.04514
     0.4028   0.5786   0.148    0.3425   0.999    0.869    0.04288  0.817    0.7075   0.03098  0.621    0.612    0.0774   0.532    0.4395   0.0711   0.4805   0.5835   0.5947
     0.1768   0.52     0.3428   0.9146   0.7324   0.5054   0.7397   0.2737   0.6313   0.1704   0.5093   0.8105   0.1312   0.752    0.3647   0.781    0.4197   0.2329   0.787
     0.762    0.63     0.9263   0.2673   0.1846   0.765    0.921    0.2913   0.3135   0.337    0.2598   0.1782   0.8013   0.641    0.6865   0.736    0.618    0.8755   0.2756
     0.9854   0.8296   0.262   ]
    输出数据(dst_gm):
    [0.999,  1.38e-05], 1.38e-05使用reinterpret_cast方法转换后为索引值232
    ```

