> **原始文档路径**: asc-devkit/docs/api/context/SoftMax.md

# SoftMax<a name="ZH-CN_TOPIC_0000001621597661"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p107991559204017"><a name="p107991559204017"></a><a name="p107991559204017"></a>√</p>
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
<tr id="row045013133410"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p545003103418"><a name="p545003103418"></a><a name="p545003103418"></a><span id="ph65453251349"><a name="ph65453251349"></a><a name="ph65453251349"></a>Kirin X90</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p745011311343"><a name="p745011311343"></a><a name="p745011311343"></a>√</p>
</td>
</tr>
<tr id="row15480181392814"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p18480313182816"><a name="p18480313182816"></a><a name="p18480313182816"></a><span id="ph1193483642813"><a name="ph1193483642813"></a><a name="ph1193483642813"></a>Kirin 9030</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p124801213152817"><a name="p124801213152817"></a><a name="p124801213152817"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ...m<sub>t</sub>, n\]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor\[m, n\]按行做如下SoftMax计算：

![](figures/zh-cn_formulaimage_0000001901417037.png)

为方便理解，通过Python脚本实现的方式，表达其计算公式（以输入为ND格式为例）如下，其中src是源操作数（输入），dst、sum、max为目的操作数（输出）。

```
def softmax(src):
    #基于last轴进行rowmax（按行取最大值）处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    #基于last轴进行rowsum（按行求和）处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst, max, sum
```

当输入的数据排布格式不同时，内部的reduce过程会有所不同：当输入为ND格式时，内部的reduce过程按last轴进行；当输入为NZ格式时，内部的reduce过程按照last轴和first轴进行，reduce过程如下图所示：

**图 1**  ND格式的reduce过程<a name="fig5210550552"></a>  
![](figures/ND格式的reduce过程.png "ND格式的reduce过程")

**图 2**  NZ格式的reduce过程<a name="fig0172155842215"></a>  
![](figures/NZ格式的reduce过程.png "NZ格式的reduce过程")

## 实现原理<a name="section118311010576"></a>

以float类型，ND格式，shape为\[m, k\]的输入Tensor为例，描述SoftMax高阶API内部算法框图，如下图所示。

**图 3**  SoftMax算法框图<a name="fig03757321297"></a>  
![](figures/SoftMax算法框图.png "SoftMax算法框图")

计算过程分为如下几步，均在Vector上进行：

1.  reducemax步骤：对输入x的每一行数据求最大值得到\[m, 1\]的结果，计算结果会保存到一个临时空间temp中；
2.  broadcast步骤：对temp中的数据\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出max；
3.  sub步骤：对输入x的所有数据按行减去max；
4.  exp步骤：对sub之后的所有数据求exp；
5.  reducesum步骤：对exp后的结果的每一行数据求和得到\[m, 1\]，计算结果会保存到临时空间temp中；
6.  broadcast步骤：对temp\(\[m, 1\]\)做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出sum；
7.  div步骤：对exp后的结果的所有数据按行除以sum，得到最终结果。

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间
    -   LocalTensor的数据类型相同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   不带sumTensor和maxTensor参数

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

-   通过sharedTmpBuffer入参传入临时空间
    -   LocalTensor的数据类型相同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   不带sumTensor和maxTensor参数

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftMax/SimpleSoftMax Tiling](SoftMax-SimpleSoftMax-Tiling.md)中提供的GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p15140111103613"><a name="p15140111103613"></a><a name="p15140111103613"></a><span id="ph14851141511365"><a name="ph14851141511365"></a><a name="ph14851141511365"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
<p id="p1314412701316"><a name="p1314412701316"></a><a name="p1314412701316"></a><span id="ph11441227191318"><a name="ph11441227191318"></a><a name="ph11441227191318"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_1"><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half、float。</p>
<p id="p209372313303"><a name="p209372313303"></a><a name="p209372313303"></a><span id="ph6937237308"><a name="ph6937237308"></a><a name="ph6937237308"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_1"><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half、float。</p>
<p id="p18781235143418"><a name="p18781235143418"></a><a name="p18781235143418"></a><span id="ph14973102013517"><a name="ph14973102013517"></a><a name="ph14973102013517"></a>Kirin X90</span>，支持的数据类型为：half、float。</p>
<p id="p858714114455"><a name="p858714114455"></a><a name="p858714114455"></a><span id="ph8998224154515"><a name="ph8998224154515"></a><a name="ph8998224154515"></a>Kirin 9030</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p98212044172612"><a name="p98212044172612"></a><a name="p98212044172612"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1153245821619"><a name="p1153245821619"></a><a name="p1153245821619"></a>srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：</p>
<a name="ul353295811167"></a><a name="ul353295811167"></a><ul id="ul353295811167"><li>srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：<a name="ul09181366549"></a><a name="ul09181366549"></a><ul id="ul09181366549"><li>尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；</li><li>非尾轴长度的乘积m为8的倍数。</li></ul>
</li><li>在Tiling实现中，通过调用<a href="IsBasicBlockInSoftMax.md">IsBasicBlockInSoftMax</a>判断Tiling切分策略是否满足基本块的切分要求。</li></ul>
</td>
</tr>
<tr id="row1276635121510"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1495244819166"><a name="p1495244819166"></a><a name="p1495244819166"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p207441566176"><a name="p207441566176"></a><a name="p207441566176"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
</td>
</tr>
<tr id="row1925664115510"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p69641412246"><a name="p69641412246"></a><a name="p69641412246"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1744211392255"><a name="p1744211392255"></a><a name="p1744211392255"></a>结构体模板参数，此参数可选配，SoftmaxConfig类型，具体定义如下。</p>
<a name="screen16476116195910"></a><a name="screen16476116195910"></a><pre class="screen" codetype="Cpp" id="screen16476116195910">enum class SoftmaxMode {
    SOFTMAX_NORMAL = 0,
    SOFTMAX_OUTPUT_WITHOUT_BRC = 1,
};
struct SoftmaxConfig{
    bool isCheckTiling = true; // 是否需要检查shape和tiling的一致性；若不一致，API内会根据shape重新计算所需tiling。默认取值true：API内部会检查一致性
    uint32_t oriSrcM = 0; // 原始非尾轴长度的乘积。设置该参数后，将shape常量化，编译过程中使用常量化的shape
    uint32_t oriSrcK = 0; // 原始尾轴长度。设置该参数后，将shape常量化，编译过程中使用常量化的shape
    SoftmaxMode mode = SoftmaxMode::SOFTMAX_NORMAL; // 预留参数
};</pre>
<p id="p76421594583"><a name="p76421594583"></a><a name="p76421594583"></a>配置示例如下。</p>
<a name="screen19241326175913"></a><a name="screen19241326175913"></a><pre class="screen" codetype="Cpp" id="screen19241326175913">constexpr SoftmaxConfig SOFTMAX_DEFAULT_CFG = {true, 0, 0, SoftmaxMode::SOFTMAX_NORMAL};</pre>
<p id="p19442739102517"><a name="p19442739102517"></a><a name="p19442739102517"></a>此参数一般用于配合kernel侧tiling计算的接口使用。</p>
<p id="p693216275219"><a name="p693216275219"></a><a name="p693216275219"></a>注意：设置了oriSrcM与oriSrcK后，模板参数isBasicBlock不生效，计算数据是否为基本块由API内部判断并处理。</p>
<p id="p16541183012147"><a name="p16541183012147"></a><a name="p16541183012147"></a><span id="ph254193071416"><a name="ph254193071416"></a><a name="ph254193071416"></a>Ascend 950PR/Ascend 950DT</span>，该参数为预留参数，暂未启用，保持默认值即可。</p>
<p id="p9572132616819"><a name="p9572132616819"></a><a name="p9572132616819"></a><span id="ph757217266813"><a name="ph757217266813"></a><a name="ph757217266813"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_2"><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a>Atlas A3 推理系列产品</term></span>，支持该参数，不支持配置mode。</p>
<p id="p17572172617810"><a name="p17572172617810"></a><a name="p17572172617810"></a><span id="ph657215261283"><a name="ph657215261283"></a><a name="ph657215261283"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_2"><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a>Atlas A2 推理系列产品</term></span>，支持该参数，不支持配置mode。</p>
<p id="p13613202055516"><a name="p13613202055516"></a><a name="p13613202055516"></a>针对<span id="ph144614117566"><a name="ph144614117566"></a><a name="ph144614117566"></a>Kirin X90</span>，该参数为预留参数，暂未启用，保持默认值即可。</p>
<p id="p67861730114512"><a name="p67861730114512"></a><a name="p67861730114512"></a>针对<span id="ph1477518399457"><a name="ph1477518399457"></a><a name="ph1477518399457"></a>Kirin 9030</span>，该参数为预留参数，暂未启用，保持默认值即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p462911347151"><a name="p462911347151"></a><a name="p462911347151"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1327102402720"><a name="p1327102402720"></a><a name="p1327102402720"></a>dst的shape和源操作数src一致。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p10629534161513"><a name="p10629534161513"></a><a name="p10629534161513"></a>sumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p176295348152"><a name="p176295348152"></a><a name="p176295348152"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p17367152119417"><a name="p17367152119417"></a><a name="p17367152119417"></a>目的操作数。</p>
<p id="p799691782011"><a name="p799691782011"></a><a name="p799691782011"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p19237153711415"><a name="p19237153711415"></a><a name="p19237153711415"></a>用于保存SoftMax计算过程中reducesum的结果。</p>
<a name="ul12308151213513"></a><a name="ul12308151213513"></a><ul id="ul12308151213513"><li>sumTensor的last轴长度固定为32Byte，即一个<span id="ph68171939163014"><a name="ph68171939163014"></a><a name="ph68171939163014"></a>datablock</span>长度。该<span id="ph138891342173012"><a name="ph138891342173012"></a><a name="ph138891342173012"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph194256461302"><a name="ph194256461302"></a><a name="ph194256461302"></a>datablock</span>中的16个数均为相同的reducesum的值。</li><li>非last轴的长度与dst保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p7629183411514"><a name="p7629183411514"></a><a name="p7629183411514"></a>maxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p8629193419159"><a name="p8629193419159"></a><a name="p8629193419159"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p14929162874710"><a name="p14929162874710"></a><a name="p14929162874710"></a>目的操作数。</p>
<p id="p106669246207"><a name="p106669246207"></a><a name="p106669246207"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p296320351261"><a name="p296320351261"></a><a name="p296320351261"></a>用于保存SoftMax计算过程中reducemax的结果。</p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>maxTensor的last轴长度固定为32Byte，即一个<span id="ph1217011525305"><a name="ph1217011525305"></a><a name="ph1217011525305"></a>datablock</span>长度。该<span id="ph166671453193017"><a name="ph166671453193017"></a><a name="ph166671453193017"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph89115620307"><a name="ph89115620307"></a><a name="ph89115620307"></a>datablock</span>中的16个数均为相同的reducemax的值。</li><li>非last轴的长度与dst保持一致。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1762920347151"><a name="p1762920347151"></a><a name="p1762920347151"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1663083415154"><a name="p1663083415154"></a><a name="p1663083415154"></a>源操作数。</p>
<p id="p193795318204"><a name="p193795318204"></a><a name="p193795318204"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_3"><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_3"><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_3"><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p10724195415422"><a name="p10724195415422"></a><a name="p10724195415422"></a>last轴长度需要32Byte对齐。</p>
</td>
</tr>
<tr id="row358412552316"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_4"><a name="zh-cn_topic_0000001530181537_ph173308471594_4"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_4"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_4"><a name="zh-cn_topic_0000001530181537_ph9902231466_4"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_4"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_4"><a name="zh-cn_topic_0000001530181537_ph1782115034816_4"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftMax-SimpleSoftMax-Tiling.md">SoftMax/SimpleSoftMax Tiling</a>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p13630123491515"><a name="p13630123491515"></a><a name="p13630123491515"></a>SoftMax计算所需Tiling信息，Tiling信息的获取请参考<a href="SoftMax-SimpleSoftMax-Tiling.md">SoftMax/SimpleSoftMax Tiling</a>。</p>
</td>
</tr>
<tr id="row27339151729"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>src的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxShapeInfo {
uint32_t srcM; // 非尾轴长度的乘积
uint32_t srcK; // 尾轴长度，必须32Byte对齐
uint32_t oriSrcM; // 原始非尾轴长度的乘积
uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p1065853610376"><a name="p1065853610376"></a><a name="p1065853610376"></a>需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即<a href="#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   src和dst的Tensor空间可以复用。
-   sumTensor和maxTensor为输出，并且last轴长度必须固定32Byte，非last轴大小需要和src以及dst保持一致。
-   sumTensor和maxTensor的数据类型需要保持一致。

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   当参数softmaxShapeInfo中srcM != oriSrcM 或者 srcK != oriSrcK时，开发者需要对GM上的原始输入\(oriSrcM, oriSrcK\)在M或K方向补齐数据到\(srcM, srcK\)，补齐的数据会参与部分运算，在输入输出复用的场景下，API的计算结果会覆盖srcTensor中补齐的原始数据，在输入输出不复用的场景下，API的计算结果会覆盖dstTensor中对应srcTensor补齐位置的数据。

## 调用示例<a name="section94691236101419"></a>

本样例中输入src和输出dst的shape大小为\[320,64\]，中间计算结果sumTensor和maxTensor的shape大小为\[320,16\]，数据类型均为half，输入输出的数据排布格式为ND，src和dst空间不复用，不使能基本块。

```
AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
AscendC::LocalTensor<T> sumTempLocal = sumQueue.AllocTensor<T>();
AscendC::LocalTensor<T> maxTempLocal = maxQueue.AllocTensor<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::SoftMax<T>(dstLocal, sumTempLocal, maxTempLocal, srcLocal, tiling, srcShape);

// AscendC::SoftMax<T, false, false, false, static_config>(dstLocal, sumTempLocal,
// maxTempLocal, srcLocal, tiling, srcShape); 使用SoftmaxConfig类型的参数static_config，传入模板参数将shape常量化

outQueueDst.EnQue<T>(dstLocal);
maxQueue.FreeTensor(maxTempLocal);
sumQueue.FreeTensor(sumTempLocal);
inQueueSrc.FreeTensor(srcLocal);
```

