> **原始文档路径**: asc-devkit/docs/api/context/IsNan.md

# IsNan<a name="ZH-CN_TOPIC_0000002382849697"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
<tr id="row1834733191219"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1234716311218"><a name="p1234716311218"></a><a name="p1234716311218"></a><span id="ph434819391213"><a name="ph434819391213"></a><a name="ph434819391213"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7751240111217"><a name="p7751240111217"></a><a name="p7751240111217"></a>x</p>
</td>
</tr>
<tr id="row33481333123"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p2034813321217"><a name="p2034813321217"></a><a name="p2034813321217"></a><span id="ph334833191213"><a name="ph334833191213"></a><a name="ph334833191213"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p20751740131216"><a name="p20751740131216"></a><a name="p20751740131216"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

按元素判断输入的浮点数是否为nan，输出结果为浮点数或布尔值。当输出为浮点类型时，对于nan的输入数据，对应位置的结果为浮点类型的1，反之为0；当输出为bool类型时，对于nan的输入数据，对应位置的结果为true，反之为false。计算公式如下：

![](figures/zh-cn_formulaimage_0000002349022528.png)

-   当输出为浮点类型时：

    ![](figures/zh-cn_formulaimage_0000002353021272.png)

-   当输出为bool类型时：

    ![](figures/zh-cn_formulaimage_0000002352887900.png)

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <const IsNanConfig& config = DEFAULT_IS_NAN_CONFIG, typename T, typename U>
    __aicore__ inline void IsNan(const LocalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
    ```

-   接口框架申请临时空间

    ```
    template <const IsNanConfig& config = DEFAULT_IS_NAN_CONFIG, typename T, typename U>
    __aicore__ inline void IsNan(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
    ```

由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetIsNanMaxMinTmpSize](GetIsNanMaxMinTmpSize.md)中提供的接口获取需要预留空间的大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19360109770"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p29321815773"><a name="p29321815773"></a><a name="p29321815773"></a>IsNanConfig</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1693212152076"><a name="p1693212152076"></a><a name="p1693212152076"></a>IsNan算法的相关配置。此参数可选配，IsNanConfig类型，具体定义如下方代码所示，其中参数的含义为：</p>
<p id="p154843370346"><a name="p154843370346"></a><a name="p154843370346"></a>isReuseSource：是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row471717528218"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p125301542515"><a name="p125301542515"></a><a name="p125301542515"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p33111334203"><a name="p33111334203"></a><a name="p33111334203"></a>目的操作数的数据类型。</p>
<p id="p193112362014"><a name="p193112362014"></a><a name="p193112362014"></a><span id="ph43119317204"><a name="ph43119317204"></a><a name="ph43119317204"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、half、float。</p>
</td>
</tr>
<tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p177734589196"><a name="p177734589196"></a><a name="p177734589196"></a>源操作数的数据类型。</p>
<p id="p5773158141911"><a name="p5773158141911"></a><a name="p5773158141911"></a><span id="ph177365851918"><a name="ph177365851918"></a><a name="ph177365851918"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
</tbody>
</table>

```
struct IsNanConfig {
    bool isReuseSource;
};
```

**表 2**  接口参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p248574352318"><a name="p248574352318"></a><a name="p248574352318"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p144841743122315"><a name="p144841743122315"></a><a name="p144841743122315"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1565464814612"><a name="p1565464814612"></a><a name="p1565464814612"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p650124211103"><a name="p650124211103"></a><a name="p650124211103"></a>目的操作数的数据类型和源操作数相同或者为bool类型。当前支持的数据类型组合请见<a href="#table158181847102411">表3</a>。</p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p164821543202310"><a name="p164821543202310"></a><a name="p164821543202310"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p64811843112313"><a name="p64811843112313"></a><a name="p64811843112313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1329113341066"><a name="p1329113341066"></a><a name="p1329113341066"></a>源操作数。</p>
<p id="p3290735103018"><a name="p3290735103018"></a><a name="p3290735103018"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row85952452815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p0400131017545"><a name="p0400131017545"></a><a name="p0400131017545"></a>临时缓存。</p>
<p id="p11947511105415"><a name="p11947511105415"></a><a name="p11947511105415"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于IsNan内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetIsNanMaxMinTmpSize.md">GetIsNanMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row1212625414239"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p312613545238"><a name="p312613545238"></a><a name="p312613545238"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p10126254182313"><a name="p10126254182313"></a><a name="p10126254182313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p15110132519616"><a name="p15110132519616"></a><a name="p15110132519616"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  输入输出支持的数据类型组合

<a name="table158181847102411"></a>
<table><thead align="left"><tr id="row381964718248"><th class="cellrowborder" valign="top" width="45.910000000000004%" id="mcps1.2.3.1.1"><p id="p1681934711240"><a name="p1681934711240"></a><a name="p1681934711240"></a>srcDtype</p>
</th>
<th class="cellrowborder" valign="top" width="54.09%" id="mcps1.2.3.1.2"><p id="p48194471241"><a name="p48194471241"></a><a name="p48194471241"></a>dstDtype</p>
</th>
</tr>
</thead>
<tbody><tr id="row285104217615"><td class="cellrowborder" valign="top" width="45.910000000000004%" headers="mcps1.2.3.1.1 "><p id="p1681914742410"><a name="p1681914742410"></a><a name="p1681914742410"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="54.09%" headers="mcps1.2.3.1.2 "><p id="p1385120421762"><a name="p1385120421762"></a><a name="p1385120421762"></a>half</p>
</td>
</tr>
<tr id="row2085274214619"><td class="cellrowborder" valign="top" width="45.910000000000004%" headers="mcps1.2.3.1.1 "><p id="p471192373511"><a name="p471192373511"></a><a name="p471192373511"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="54.09%" headers="mcps1.2.3.1.2 "><p id="p1385216421761"><a name="p1385216421761"></a><a name="p1385216421761"></a>bool</p>
</td>
</tr>
<tr id="row1185215421164"><td class="cellrowborder" valign="top" width="45.910000000000004%" headers="mcps1.2.3.1.1 "><p id="p511313117714"><a name="p511313117714"></a><a name="p511313117714"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="54.09%" headers="mcps1.2.3.1.2 "><p id="p985215424614"><a name="p985215424614"></a><a name="p985215424614"></a>float</p>
</td>
</tr>
<tr id="row1881954718248"><td class="cellrowborder" valign="top" width="45.910000000000004%" headers="mcps1.2.3.1.1 "><p id="p169931327203511"><a name="p169931327203511"></a><a name="p169931327203511"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="54.09%" headers="mcps1.2.3.1.2 "><p id="p019942995719"><a name="p019942995719"></a><a name="p019942995719"></a>bool</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址偏移对齐要求请参见[通用说明和约束](通用说明和约束.md)。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
pipe.InitBuffer(tmpQue, 1, bufferSize);  // bufferSize通过Host侧tiling参数获取
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
// 输入tensor长度为1024, 算子输入的数据类型为half, 实际计算个数为512
static constexpr AscendC::IsNanConfig isNanConfig = { false };
AscendC::IsNan<isNanConfig, bool, half>(dst, src, sharedTmpBuffer, 512);
```

结果示例如下：

```
输入的数据类型为half，输出的数据类型为bool
输入数据(src):[1.0 nan 3.0 4.0 nan 6.0 nan 8.0]
输出数据(dst):[false true false false true false true false]
```

