> **原始文档路径**: `asc-devkit/docs/guide/兼容性迁移指南/351x架构迁移指导/220x迁移351x指导/高阶API迁移指导.md`
# 高阶API迁移指导<a name="ZH-CN_TOPIC_0000002503588125"></a>

Ascend C高阶API基本兼容351x架构与220x架构，部分API进行了扩展。当前351x架构不支持卷积计算类高阶API。

## Matmul类高阶API<a name="section6925172655117"></a>

-   支持的数据类型有变化。

    **表 1**  数据类型兼容性情况

    <a name="table1534819819338"></a>
    <table><thead align="left"><tr id="row1234720819332"><th class="cellrowborder" valign="top" width="20.849999999999998%" id="mcps1.2.6.1.1"><p id="p43471486333"><a name="p43471486333"></a><a name="p43471486333"></a>A矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="19.05%" id="mcps1.2.6.1.2"><p id="p14347686337"><a name="p14347686337"></a><a name="p14347686337"></a>B矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="22.68%" id="mcps1.2.6.1.3"><p id="p13347188143313"><a name="p13347188143313"></a><a name="p13347188143313"></a>Bias矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="21.47%" id="mcps1.2.6.1.4"><p id="p03471788331"><a name="p03471788331"></a><a name="p03471788331"></a>C矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.950000000000001%" id="mcps1.2.6.1.5"><p id="p1934720814334"><a name="p1934720814334"></a><a name="p1934720814334"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1534748113317"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p19347168103314"><a name="p19347168103314"></a><a name="p19347168103314"></a>int4b_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p16347168193319"><a name="p16347168193319"></a><a name="p16347168193319"></a>int4b_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p0347985333"><a name="p0347985333"></a><a name="p0347985333"></a>int32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p1634717893315"><a name="p1634717893315"></a><a name="p1634717893315"></a>int32_t、half</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p1034788163317"><a name="p1034788163317"></a><a name="p1034788163317"></a>351x架构不支持。</p>
    </td>
    </tr>
    <tr id="row2348108183316"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p1534728183316"><a name="p1534728183316"></a><a name="p1534728183316"></a>fp8_e4m3fn_t、fp8_e5m2_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p17347138103316"><a name="p17347138103316"></a><a name="p17347138103316"></a>fp8_e4m3fn_t、fp8_e5m2_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p434819893316"><a name="p434819893316"></a><a name="p434819893316"></a>float、half、bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p9348108163320"><a name="p9348108163320"></a><a name="p9348108163320"></a>fp8_e4m3fn_t、half、bfloat16_t、float</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p133481873315"><a name="p133481873315"></a><a name="p133481873315"></a>351x架构新增。</p>
    </td>
    </tr>
    <tr id="row93481388339"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p334811853313"><a name="p334811853313"></a><a name="p334811853313"></a>hifloat8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p03481387332"><a name="p03481387332"></a><a name="p03481387332"></a>hifloat8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p734813873310"><a name="p734813873310"></a><a name="p734813873310"></a>float、half、bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p4348480335"><a name="p4348480335"></a><a name="p4348480335"></a>hifloat8_t、half、bfloat16_t、float</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p1334828163311"><a name="p1334828163311"></a><a name="p1334828163311"></a>351x架构新增。</p>
    </td>
    </tr>
    <tr id="row23485883317"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p134878103315"><a name="p134878103315"></a><a name="p134878103315"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p934813833317"><a name="p934813833317"></a><a name="p934813833317"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p3348389334"><a name="p3348389334"></a><a name="p3348389334"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p73486819337"><a name="p73486819337"></a><a name="p73486819337"></a>float、half、bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p14348387336"><a name="p14348387336"></a><a name="p14348387336"></a>351x架构新增。</p>
    </td>
    </tr>
    <tr id="row113481180333"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p123481853316"><a name="p123481853316"></a><a name="p123481853316"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p123486853310"><a name="p123486853310"></a><a name="p123486853310"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p834828183310"><a name="p834828183310"></a><a name="p834828183310"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p17348118193320"><a name="p17348118193320"></a><a name="p17348118193320"></a>float、half、bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p173488812336"><a name="p173488812336"></a><a name="p173488812336"></a>351x架构新增。</p>
    </td>
    </tr>
    <tr id="row3348128153315"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p13486803312"><a name="p13486803312"></a><a name="p13486803312"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p23482893314"><a name="p23482893314"></a><a name="p23482893314"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p33489816333"><a name="p33489816333"></a><a name="p33489816333"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p113481815337"><a name="p113481815337"></a><a name="p113481815337"></a>float、half、bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p134848153312"><a name="p134848153312"></a><a name="p134848153312"></a>351x架构新增。</p>
    </td>
    </tr>
    <tr id="row1634817819339"><td class="cellrowborder" valign="top" width="20.849999999999998%" headers="mcps1.2.6.1.1 "><p id="p03487883313"><a name="p03487883313"></a><a name="p03487883313"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.6.1.2 "><p id="p33481786334"><a name="p33481786334"></a><a name="p33481786334"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.68%" headers="mcps1.2.6.1.3 "><p id="p1134812819331"><a name="p1134812819331"></a><a name="p1134812819331"></a>int32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="21.47%" headers="mcps1.2.6.1.4 "><p id="p1334898193311"><a name="p1334898193311"></a><a name="p1334898193311"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.950000000000001%" headers="mcps1.2.6.1.5 "><p id="p1334820816331"><a name="p1334820816331"></a><a name="p1334820816331"></a>351x架构新增。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   不支持4:2稀疏特性。具体兼容方案请参考[4：2结构化稀疏功能](基础API迁移指导.md#li69092585134)。

## 其它高阶API<a name="section14195124865717"></a>

**表 2**  数学计算

<a name="table1011733113422"></a>
<table><thead align="left"><tr id="row3118531154217"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p0118031104211"><a name="p0118031104211"></a><a name="p0118031104211"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p8118143124210"><a name="p8118143124210"></a><a name="p8118143124210"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row14118113119422"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9188171317366"><a name="p9188171317366"></a><a name="p9188171317366"></a>Tanh、Asin、Sin、Acos、Cos、Log、Atan、Fmod</p>
<p id="p950434551915"><a name="p950434551915"></a><a name="p950434551915"></a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15508125252615"><a name="p15508125252615"></a><a name="p15508125252615"></a>兼容220x架构。</p>
<p id="p550845202613"><a name="p550845202613"></a><a name="p550845202613"></a>扩展支持算法配置，通过模板参数配置API使用的算法，从而提供高精度、高性能的算法选择。</p>
</td>
</tr>
<tr id="row16667103375216"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1935705813247"><a name="p1935705813247"></a><a name="p1935705813247"></a>Sinh、Cosh、Tan、Trunc、Frac、Erf、Erfc、Atanh、Asinh、</p>
<p id="p1907642152910"><a name="p1907642152910"></a><a name="p1907642152910"></a>Acosh、Floor、Ceil、Round、Axpy、Exp、Lgamma、Digamma、Xor、Cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p137711542268"><a name="p137711542268"></a><a name="p137711542268"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row1868813347468"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9688434144610"><a name="p9688434144610"></a><a name="p9688434144610"></a>Power</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p46431157182619"><a name="p46431157182619"></a><a name="p46431157182619"></a>兼容220x架构。</p>
<p id="p15643457182613"><a name="p15643457182613"></a><a name="p15643457182613"></a>扩展支持uint8_t、int8_t、uint16_t、int16_t、uint32_t、bfloat16_t数据类型。</p>
</td>
</tr>
<tr id="row69291914164916"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1392941424911"><a name="p1392941424911"></a><a name="p1392941424911"></a>Sign</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1870316072712"><a name="p1870316072712"></a><a name="p1870316072712"></a>兼容220x架构。</p>
<p id="p1670412092711"><a name="p1670412092711"></a><a name="p1670412092711"></a>扩展支持int64_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  激活函数

<a name="table192231726173811"></a>
<table><thead align="left"><tr id="row1422362623814"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1041413536446"><a name="p1041413536446"></a><a name="p1041413536446"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p9414175314412"><a name="p9414175314412"></a><a name="p9414175314412"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row522452616384"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p925911352511"><a name="p925911352511"></a><a name="p925911352511"></a>SoftMax、SimpleSoftMax、SoftmaxFlash、SoftmaxGrad、SoftmaxFlashV2、SoftmaxFlashV3、SoftmaxGradFront、AdjustSoftMaxRes、LogSoftMax、FasterGelu、FasterGeluV2、Gelu、SwiGLU、Silu、Swish、GeGLU、ReFlu、Sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p151696450316"><a name="p151696450316"></a><a name="p151696450316"></a>兼容220x架构。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  数据归一化

<a name="table1488319115396"></a>
<table><thead align="left"><tr id="row138831103911"><th class="cellrowborder" valign="top" width="49.980000000000004%" id="mcps1.2.3.1.1"><p id="p8665135824416"><a name="p8665135824416"></a><a name="p8665135824416"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50.019999999999996%" id="mcps1.2.3.1.2"><p id="p1466513582447"><a name="p1466513582447"></a><a name="p1466513582447"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1688319112397"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p1088313113916"><a name="p1088313113916"></a><a name="p1088313113916"></a>LayerNormGrad、LayerNormGradBeta、RmsNorm、BatchNorm、DeepNorm、GroupNorm</p>
</td>
<td class="cellrowborder" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p1867013611278"><a name="p1867013611278"></a><a name="p1867013611278"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row112943474533"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p429474712536"><a name="p429474712536"></a><a name="p429474712536"></a>Normalize、WelfordUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p38904912278"><a name="p38904912278"></a><a name="p38904912278"></a>兼容220x架构。</p>
<p id="p889110902716"><a name="p889110902716"></a><a name="p889110902716"></a>扩展支持bfloat16_t数据类型。</p>
</td>
</tr>
<tr id="row15971925165311"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p1059711257536"><a name="p1059711257536"></a><a name="p1059711257536"></a>LayerNorm</p>
</td>
<td class="cellrowborder" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p552213132278"><a name="p552213132278"></a><a name="p552213132278"></a>兼容220x架构。</p>
<p id="p14522151322717"><a name="p14522151322717"></a><a name="p14522151322717"></a>扩展支持求方差。</p>
</td>
</tr>
<tr id="row2090423941218"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p49051539141214"><a name="p49051539141214"></a><a name="p49051539141214"></a>WelfordFinalize</p>
</td>
<td class="cellrowborder" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p107681713273"><a name="p107681713273"></a><a name="p107681713273"></a>兼容220x架构。</p>
<p id="p37631712716"><a name="p37631712716"></a><a name="p37631712716"></a>扩展支持算法配置，通过模板参数指定在计算方差时是否使用修正系数。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  量化操作

<a name="table432310426394"></a>
<table><thead align="left"><tr id="row18324164212392"><th class="cellrowborder" valign="top" width="33.95%" id="mcps1.2.3.1.1"><p id="p1247120164516"><a name="p1247120164516"></a><a name="p1247120164516"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="66.05%" id="mcps1.2.3.1.2"><p id="p1247130184515"><a name="p1247130184515"></a><a name="p1247130184515"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row832474216394"><td class="cellrowborder" valign="top" width="33.95%" headers="mcps1.2.3.1.1 "><p id="p3324342193912"><a name="p3324342193912"></a><a name="p3324342193912"></a>AscendQuant</p>
</td>
<td class="cellrowborder" valign="top" width="66.05%" headers="mcps1.2.3.1.2 "><p id="p1857162972713"><a name="p1857162972713"></a><a name="p1857162972713"></a>兼容220x架构。</p>
<p id="p3571729132712"><a name="p3571729132712"></a><a name="p3571729132712"></a>扩展支持PRE_TOKEN量化、PRE_GROUP量化。</p>
<p id="p55711294275"><a name="p55711294275"></a><a name="p55711294275"></a>扩展支持从half、bfloat16_t、float类型到fp8_e5m2_t、fp8_e4m3fn_t、hifloat8_t、int8_t类型的量化。</p>
<p id="p35712291278"><a name="p35712291278"></a><a name="p35712291278"></a>扩展支持从half、bfloat16_t类型到fp4x2_e1m2_t、fp4x2_e2m1_t类型的量化。</p>
</td>
</tr>
<tr id="row5805358162412"><td class="cellrowborder" valign="top" width="33.95%" headers="mcps1.2.3.1.1 "><p id="p1580511586248"><a name="p1580511586248"></a><a name="p1580511586248"></a>AscendDequant</p>
</td>
<td class="cellrowborder" valign="top" width="66.05%" headers="mcps1.2.3.1.2 "><p id="p183091632112712"><a name="p183091632112712"></a><a name="p183091632112712"></a>兼容220x架构。</p>
<p id="p130963212719"><a name="p130963212719"></a><a name="p130963212719"></a>扩展支持PRE_TOKEN量化、PRE_GROUP量化。</p>
<p id="p1309832162718"><a name="p1309832162718"></a><a name="p1309832162718"></a>扩展支持从int32_t类型到half、bfloat16_t、float类型反量化。</p>
<p id="p13309532182716"><a name="p13309532182716"></a><a name="p13309532182716"></a>扩展支持从float类型到half、bfloat16_t、float类型的反量化。</p>
</td>
</tr>
<tr id="row7298212142918"><td class="cellrowborder" valign="top" width="33.95%" headers="mcps1.2.3.1.1 "><p id="p32981612142912"><a name="p32981612142912"></a><a name="p32981612142912"></a>AscendAntiQuant</p>
</td>
<td class="cellrowborder" valign="top" width="66.05%" headers="mcps1.2.3.1.2 "><p id="p568413356278"><a name="p568413356278"></a><a name="p568413356278"></a>兼容220x架构。</p>
<p id="p4684435112711"><a name="p4684435112711"></a><a name="p4684435112711"></a>扩展支持PRE_TOKEN量化、PRE_GROUP量化。</p>
<p id="p268443516276"><a name="p268443516276"></a><a name="p268443516276"></a>扩展支持从int8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t类型到half、bfloat16_t、float、half类型的伪量化。</p>
<p id="p9684153518276"><a name="p9684153518276"></a><a name="p9684153518276"></a>扩展支持从fp4x2_e1m2_t、fp4x2_e2m1_t类型到half、bfloat16_t类型的伪量化。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  归约操作

<a name="table17561457193915"></a>
<table><thead align="left"><tr id="row8756357163920"><th class="cellrowborder" valign="top" width="60.709999999999994%" id="mcps1.2.3.1.1"><p id="p102601213455"><a name="p102601213455"></a><a name="p102601213455"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="39.290000000000006%" id="mcps1.2.3.1.2"><p id="p726010113457"><a name="p726010113457"></a><a name="p726010113457"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17756175716395"><td class="cellrowborder" valign="top" width="60.709999999999994%" headers="mcps1.2.3.1.1 "><p id="p14756125703918"><a name="p14756125703918"></a><a name="p14756125703918"></a>Sum、Mean、ReduceXorSum、ReduceMean、ReduceAny、ReduceAll、ReduceProd</p>
</td>
<td class="cellrowborder" valign="top" width="39.290000000000006%" headers="mcps1.2.3.1.2 "><p id="p137995122818"><a name="p137995122818"></a><a name="p137995122818"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row1987916211103"><td class="cellrowborder" valign="top" width="60.709999999999994%" headers="mcps1.2.3.1.1 "><p id="p58799211109"><a name="p58799211109"></a><a name="p58799211109"></a>ReduceSum</p>
</td>
<td class="cellrowborder" valign="top" width="39.290000000000006%" headers="mcps1.2.3.1.2 "><p id="p20285132815"><a name="p20285132815"></a><a name="p20285132815"></a>兼容220x架构。</p>
<p id="p1620515280"><a name="p1620515280"></a><a name="p1620515280"></a>扩展支持int32_t、uint32_t、int64_t、uint64_t数据类型。</p>
</td>
</tr>
<tr id="row788583213012"><td class="cellrowborder" valign="top" width="60.709999999999994%" headers="mcps1.2.3.1.1 "><p id="p12886532504"><a name="p12886532504"></a><a name="p12886532504"></a>ReduceMax、ReduceMin</p>
</td>
<td class="cellrowborder" valign="top" width="39.290000000000006%" headers="mcps1.2.3.1.2 "><p id="p172162050162817"><a name="p172162050162817"></a><a name="p172162050162817"></a>兼容220x架构。</p>
<p id="p1321615018283"><a name="p1321615018283"></a><a name="p1321615018283"></a>扩展支持int8_t、uint8_t、int16_t、uint16_t、bfloat16_t、int32_t、uint32_t、int64_t、uint64_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 7**  排序操作

<a name="table4855685408"></a>
<table><thead align="left"><tr id="row78561987407"><th class="cellrowborder" valign="top" width="50.33%" id="mcps1.2.3.1.1"><p id="p18599528455"><a name="p18599528455"></a><a name="p18599528455"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="49.669999999999995%" id="mcps1.2.3.1.2"><p id="p115995284510"><a name="p115995284510"></a><a name="p115995284510"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1588902418535"><td class="cellrowborder" valign="top" width="50.33%" headers="mcps1.2.3.1.1 "><p id="p1477514835513"><a name="p1477514835513"></a><a name="p1477514835513"></a>Concat、Extract、GetSortOffset、GetSortLen、MrgSort</p>
</td>
<td class="cellrowborder" valign="top" width="49.669999999999995%" headers="mcps1.2.3.1.2 "><p id="p149522418295"><a name="p149522418295"></a><a name="p149522418295"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row1685613814404"><td class="cellrowborder" valign="top" width="50.33%" headers="mcps1.2.3.1.1 "><p id="p188564818400"><a name="p188564818400"></a><a name="p188564818400"></a>TopK</p>
</td>
<td class="cellrowborder" valign="top" width="49.669999999999995%" headers="mcps1.2.3.1.2 "><p id="p159901844132918"><a name="p159901844132918"></a><a name="p159901844132918"></a>兼容220x架构。</p>
<p id="p2990544112919"><a name="p2990544112919"></a><a name="p2990544112919"></a>使用RADIX_SELECT算法时，扩展支持uint8_t、int8_t、uint16_t、int16_t、uint32_t、int32_t、bfloat16_t、uint64_t、int64_t数据类型。</p>
</td>
</tr>
<tr id="row13201552343"><td class="cellrowborder" valign="top" width="50.33%" headers="mcps1.2.3.1.1 "><p id="p232075219413"><a name="p232075219413"></a><a name="p232075219413"></a>Sort</p>
</td>
<td class="cellrowborder" valign="top" width="49.669999999999995%" headers="mcps1.2.3.1.2 "><p id="p1561748172917"><a name="p1561748172917"></a><a name="p1561748172917"></a>兼容220x架构。</p>
<p id="p261144822913"><a name="p261144822913"></a><a name="p261144822913"></a>扩展支持算法配置，通过模板参数指定排序算法以及降序升序排序。</p>
</td>
</tr>
</tbody>
</table>

**表 8**  索引计算

<a name="table748063194016"></a>
<table><thead align="left"><tr id="row8480531134011"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1944815184518"><a name="p1944815184518"></a><a name="p1944815184518"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p1244845124510"><a name="p1244845124510"></a><a name="p1244845124510"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2037971217519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p10379171220517"><a name="p10379171220517"></a><a name="p10379171220517"></a>Arange</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p296915535299"><a name="p296915535299"></a><a name="p296915535299"></a>兼容220x架构。</p>
<p id="p18969653152915"><a name="p18969653152915"></a><a name="p18969653152915"></a>扩展支持int64_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 9**  数据过滤

<a name="table1937505494013"></a>
<table><thead align="left"><tr id="row03769549406"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p182236817450"><a name="p182236817450"></a><a name="p182236817450"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p422388174510"><a name="p422388174510"></a><a name="p422388174510"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1453117371552"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p107967415400"><a name="p107967415400"></a><a name="p107967415400"></a>Select</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9351156202918"><a name="p9351156202918"></a><a name="p9351156202918"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row737675474012"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p337645474010"><a name="p337645474010"></a><a name="p337645474010"></a>DropOut</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15264155902914"><a name="p15264155902914"></a><a name="p15264155902914"></a>兼容220x架构。</p>
<p id="p72651859162914"><a name="p72651859162914"></a><a name="p72651859162914"></a>扩展支持bfloat16_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 10**  张量变换

<a name="table68071919174115"></a>
<table><thead align="left"><tr id="row1880713195411"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p2655161904519"><a name="p2655161904519"></a><a name="p2655161904519"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p14655719124519"><a name="p14655719124519"></a><a name="p14655719124519"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row118071194413"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p230212473585"><a name="p230212473585"></a><a name="p230212473585"></a>Transpose</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p183391064304"><a name="p183391064304"></a><a name="p183391064304"></a>兼容220x架构。</p>
<div class="p" id="p9339126143016"><a name="p9339126143016"></a><a name="p9339126143016"></a>新增支持数据排布转换场景：<a name="ul19819193610813"></a><a name="ul19819193610813"></a><ul id="ul19819193610813"><li>二维转置或者三维的后两位转置。</li><li>三维中的第一维和第二维互换。</li><li>三维中的第一维和第三维互换。</li><li>使用交织指令对二维ND2NZ转置。</li></ul>
</div>
</td>
</tr>
<tr id="row8745113972818"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p47451139182817"><a name="p47451139182817"></a><a name="p47451139182817"></a>TransData、Pad、UnPad</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p69934108302"><a name="p69934108302"></a><a name="p69934108302"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row2715221"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p72101361211"><a name="p72101361211"></a><a name="p72101361211"></a>BroadCast</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p17188014143015"><a name="p17188014143015"></a><a name="p17188014143015"></a>兼容220x架构。</p>
<p id="p1918821433011"><a name="p1918821433011"></a><a name="p1918821433011"></a>扩展支持动态Shape。</p>
<p id="p121881814143019"><a name="p121881814143019"></a><a name="p121881814143019"></a>扩展支持int16_t、uint16_t、bfloat16_t、int32_t、uint32_t数据类型。</p>
</td>
</tr>
<tr id="row386815398281"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2942213619"><a name="p2942213619"></a><a name="p2942213619"></a>Fill</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1758771711301"><a name="p1758771711301"></a><a name="p1758771711301"></a>兼容220x架构。</p>
<p id="p12588191716305"><a name="p12588191716305"></a><a name="p12588191716305"></a>扩展支持uint8_t、int8_t、bfloat16_t、uint64_t、int64_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 11**  Hccl

<a name="table12265143614414"></a>
<table><thead align="left"><tr id="row1126583611410"><th class="cellrowborder" valign="top" width="43.35%" id="mcps1.2.3.1.1"><p id="p16138821104518"><a name="p16138821104518"></a><a name="p16138821104518"></a>AscendC 高阶API</p>
</th>
<th class="cellrowborder" valign="top" width="56.65%" id="mcps1.2.3.1.2"><p id="p61381215456"><a name="p61381215456"></a><a name="p61381215456"></a>兼容说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1326563616413"><td class="cellrowborder" valign="top" width="43.35%" headers="mcps1.2.3.1.1 "><p id="p52651936104112"><a name="p52651936104112"></a><a name="p52651936104112"></a>Hccl模板参数</p>
</td>
<td class="cellrowborder" valign="top" width="56.65%" headers="mcps1.2.3.1.2 "><p id="p1947910612319"><a name="p1947910612319"></a><a name="p1947910612319"></a>支持HCCL_SERVER_TYPE_CCU服务端类型。</p>
</td>
</tr>
<tr id="row1667811121053"><td class="cellrowborder" valign="top" width="43.35%" headers="mcps1.2.3.1.1 "><p id="p1886335314511"><a name="p1886335314511"></a><a name="p1886335314511"></a>InitV2、SetCcTilingV2、AllReduce、AllGather、ReduceScatter、AlltoAll、AlltoAllV、Commit、Wait、Finalize</p>
<p id="p1939872751318"><a name="p1939872751318"></a><a name="p1939872751318"></a></p>
</td>
<td class="cellrowborder" valign="top" width="56.65%" headers="mcps1.2.3.1.2 "><p id="p235608173114"><a name="p235608173114"></a><a name="p235608173114"></a>兼容220x架构。</p>
</td>
</tr>
<tr id="row18338107564"><td class="cellrowborder" valign="top" width="43.35%" headers="mcps1.2.3.1.1 "><p id="p63391571160"><a name="p63391571160"></a><a name="p63391571160"></a>BatchWrite、iterate、Query、InterHcclGroupSync、GetWindowsInAddr、GetWindowsOutAddr、GetRankId、GetRankDim、QueueBarrier、GetQueueNum</p>
</td>
<td class="cellrowborder" valign="top" width="56.65%" headers="mcps1.2.3.1.2 "><p id="p116241033113"><a name="p116241033113"></a><a name="p116241033113"></a>351x架构暂不支持。</p>
</td>
</tr>
<tr id="row1135194142018"><td class="cellrowborder" valign="top" width="43.35%" headers="mcps1.2.3.1.1 "><p id="p15351141142016"><a name="p15351141142016"></a><a name="p15351141142016"></a>SetReduceType、AlltoAllvWrite</p>
</td>
<td class="cellrowborder" valign="top" width="56.65%" headers="mcps1.2.3.1.2 "><p id="p23911012153113"><a name="p23911012153113"></a><a name="p23911012153113"></a>351x架构新增。</p>
</td>
</tr>
</tbody>
</table>

