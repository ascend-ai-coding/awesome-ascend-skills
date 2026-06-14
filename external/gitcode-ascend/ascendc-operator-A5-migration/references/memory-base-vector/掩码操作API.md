> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/基础API/常用操作速查指导/如何使用掩码操作API.md`
# 如何使用掩码操作API<a name="ZH-CN_TOPIC_0000002305974105"></a>

Mask用于控制矢量计算中参与计算的元素个数，支持以下工作模式及配置方式：

**表 1**  Mask工作模式

<a name="zh-cn_topic_0000002267504584_table414483923116"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002267504584_row51452039163114"><th class="cellrowborder" valign="top" width="14.14%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002267504584_p814513911319"><a name="zh-cn_topic_0000002267504584_p814513911319"></a><a name="zh-cn_topic_0000002267504584_p814513911319"></a>工作模式</p>
</th>
<th class="cellrowborder" valign="top" width="85.86%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002267504584_p14145239103115"><a name="zh-cn_topic_0000002267504584_p14145239103115"></a><a name="zh-cn_topic_0000002267504584_p14145239103115"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002267504584_row11145143963110"><td class="cellrowborder" valign="top" width="14.14%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002267504584_p614518396314"><a name="zh-cn_topic_0000002267504584_p614518396314"></a><a name="zh-cn_topic_0000002267504584_p614518396314"></a>Normal模式</p>
</td>
<td class="cellrowborder" valign="top" width="85.86%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002267504584_p770453072014"><a name="zh-cn_topic_0000002267504584_p770453072014"></a><a name="zh-cn_topic_0000002267504584_p770453072014"></a>默认模式，支持单次迭代内的Mask能力，需要开发者配置迭代次数，额外进行尾块的计算。</p>
<p id="zh-cn_topic_0000002267504584_p1584693592814"><a name="zh-cn_topic_0000002267504584_p1584693592814"></a><a name="zh-cn_topic_0000002267504584_p1584693592814"></a><strong id="zh-cn_topic_0000002267504584_b277175602011"><a name="zh-cn_topic_0000002267504584_b277175602011"></a><a name="zh-cn_topic_0000002267504584_b277175602011"></a>Normal模式下，Mask用来控制单次迭代内参与计算的元素个数。</strong></p>
<p id="zh-cn_topic_0000002267504584_p172203193391"><a name="zh-cn_topic_0000002267504584_p172203193391"></a><a name="zh-cn_topic_0000002267504584_p172203193391"></a>通过调用SetMaskNorm设置Normal模式。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002267504584_row61458393313"><td class="cellrowborder" valign="top" width="14.14%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002267504584_p914513918311"><a name="zh-cn_topic_0000002267504584_p914513918311"></a><a name="zh-cn_topic_0000002267504584_p914513918311"></a>Counter模式</p>
</td>
<td class="cellrowborder" valign="top" width="85.86%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002267504584_p376310417215"><a name="zh-cn_topic_0000002267504584_p376310417215"></a><a name="zh-cn_topic_0000002267504584_p376310417215"></a>简化模式，直接传入计算数据量，<span>自动推断迭代次数，</span>不需要开发者去感知迭代次数、处理非对齐尾块的操作；但是不具备单次迭代内的Mask能力。</p>
<p id="zh-cn_topic_0000002267504584_p714520394314"><a name="zh-cn_topic_0000002267504584_p714520394314"></a><a name="zh-cn_topic_0000002267504584_p714520394314"></a><strong id="zh-cn_topic_0000002267504584_b113012902110"><a name="zh-cn_topic_0000002267504584_b113012902110"></a><a name="zh-cn_topic_0000002267504584_b113012902110"></a>Counter模式下，Mask表示整个矢量计算参与计算的元素个数。</strong></p>
<p id="zh-cn_topic_0000002267504584_p17014342110"><a name="zh-cn_topic_0000002267504584_p17014342110"></a><a name="zh-cn_topic_0000002267504584_p17014342110"></a>通过调用SetMaskCount设置Counter模式。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  Mask配置方式

<a name="zh-cn_topic_0000002267504584_table642464733119"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002267504584_row8425134710316"><th class="cellrowborder" valign="top" width="14.12%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002267504584_p194258470311"><a name="zh-cn_topic_0000002267504584_p194258470311"></a><a name="zh-cn_topic_0000002267504584_p194258470311"></a>配置方式</p>
</th>
<th class="cellrowborder" valign="top" width="85.88%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002267504584_p442513473313"><a name="zh-cn_topic_0000002267504584_p442513473313"></a><a name="zh-cn_topic_0000002267504584_p442513473313"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002267504584_row9425114710316"><td class="cellrowborder" valign="top" width="14.12%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002267504584_p2425847103118"><a name="zh-cn_topic_0000002267504584_p2425847103118"></a><a name="zh-cn_topic_0000002267504584_p2425847103118"></a>接口传参（默认）</p>
</td>
<td class="cellrowborder" valign="top" width="85.88%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002267504584_p1342514712318"><a name="zh-cn_topic_0000002267504584_p1342514712318"></a><a name="zh-cn_topic_0000002267504584_p1342514712318"></a><span>通过矢量计算API的入参直接传递Mask值。矢量计算API的</span>模板参数isSetMask（仅部分API支持）用于控制接口传参还是外部API配置，默认值为true，表示接口传参。<span>Mask对应于高维切分计算API中的mask/mask[]参数或者tensor前n个数据计算API中的calCount参数。</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002267504584_row1425124783112"><td class="cellrowborder" valign="top" width="14.12%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002267504584_p114251471310"><a name="zh-cn_topic_0000002267504584_p114251471310"></a><a name="zh-cn_topic_0000002267504584_p114251471310"></a>外部API配置</p>
</td>
<td class="cellrowborder" valign="top" width="85.88%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002267504584_p84251647103119"><a name="zh-cn_topic_0000002267504584_p84251647103119"></a><a name="zh-cn_topic_0000002267504584_p84251647103119"></a><span>调用</span>SetVectorMask<span>接口设置Mask值，矢量计算API的模板参数isSetMask设置为false，接口入参中的Mask参数（对应于高维切分计算API中的mask/mask[]参数或者tensor前n个数据计算API中的calCount参数）</span>不生效。适用于Mask参数相同，多次重复使用的场景，无需在矢量计算API内部反复设置，会有一定的性能优势。</p>
</td>
</tr>
</tbody>
</table>

Mask操作的使用方式如下：

**表 3**  Mask操作的使用方式

<a name="zh-cn_topic_0000002267504584_table1957843418427"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002267504584_row657817341420"><th class="cellrowborder" valign="top" width="8.58%" id="mcps1.2.5.1.1"><p id="zh-cn_topic_0000002267504584_p19578634194213"><a name="zh-cn_topic_0000002267504584_p19578634194213"></a><a name="zh-cn_topic_0000002267504584_p19578634194213"></a>配置方式</p>
</th>
<th class="cellrowborder" valign="top" width="8.37%" id="mcps1.2.5.1.2"><p id="zh-cn_topic_0000002267504584_p165791734184211"><a name="zh-cn_topic_0000002267504584_p165791734184211"></a><a name="zh-cn_topic_0000002267504584_p165791734184211"></a>工作模式</p>
</th>
<th class="cellrowborder" valign="top" width="35.67%" id="mcps1.2.5.1.3"><p id="zh-cn_topic_0000002267504584_p1757923411421"><a name="zh-cn_topic_0000002267504584_p1757923411421"></a><a name="zh-cn_topic_0000002267504584_p1757923411421"></a>前n个数据计算API</p>
</th>
<th class="cellrowborder" valign="top" width="47.38%" id="mcps1.2.5.1.4"><p id="zh-cn_topic_0000002267504584_p1257915343423"><a name="zh-cn_topic_0000002267504584_p1257915343423"></a><a name="zh-cn_topic_0000002267504584_p1257915343423"></a>高维切分计算API</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002267504584_row15792034144218"><td class="cellrowborder" rowspan="2" valign="top" width="8.58%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000002267504584_p1457918343421"><a name="zh-cn_topic_0000002267504584_p1457918343421"></a><a name="zh-cn_topic_0000002267504584_p1457918343421"></a>接口传参</p>
</td>
<td class="cellrowborder" valign="top" width="8.37%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000002267504584_p1157917346420"><a name="zh-cn_topic_0000002267504584_p1157917346420"></a><a name="zh-cn_topic_0000002267504584_p1157917346420"></a>Normal模式</p>
</td>
<td class="cellrowborder" valign="top" width="35.67%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000002267504584_p35791034104218"><a name="zh-cn_topic_0000002267504584_p35791034104218"></a><a name="zh-cn_topic_0000002267504584_p35791034104218"></a>不涉及。</p>
</td>
<td class="cellrowborder" valign="top" width="47.38%" headers="mcps1.2.5.1.4 "><p id="zh-cn_topic_0000002267504584_p057913416421"><a name="zh-cn_topic_0000002267504584_p057913416421"></a><a name="zh-cn_topic_0000002267504584_p057913416421"></a>isSetMask模板参数设置为true，通过接口入参传入Mask，根据使用场景配置<span id="zh-cn_topic_0000002267504584_ph1657918349424"><a name="zh-cn_topic_0000002267504584_ph1657918349424"></a><a name="zh-cn_topic_0000002267504584_ph1657918349424"></a>dataBlockStride</span>、<span id="zh-cn_topic_0000002267504584_ph1557911348429"><a name="zh-cn_topic_0000002267504584_ph1557911348429"></a><a name="zh-cn_topic_0000002267504584_ph1557911348429"></a>repeatStride</span><span>、</span><span>repeatTime</span>参数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002267504584_row55792345422"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000002267504584_p7579534184213"><a name="zh-cn_topic_0000002267504584_p7579534184213"></a><a name="zh-cn_topic_0000002267504584_p7579534184213"></a>Counter模式</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000002267504584_p6579534194215"><a name="zh-cn_topic_0000002267504584_p6579534194215"></a><a name="zh-cn_topic_0000002267504584_p6579534194215"></a>isSetMask模板参数设置为true，通过接口入参传入Mask。</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="zh-cn_topic_0000002267504584_ul257912346428"></a><a name="zh-cn_topic_0000002267504584_ul257912346428"></a><ul id="zh-cn_topic_0000002267504584_ul257912346428"><li>isSetMask模板参数设置为true，通过接口入参传入Mask。</li><li>根据使用场景配置<span id="zh-cn_topic_0000002267504584_ph0579334194210"><a name="zh-cn_topic_0000002267504584_ph0579334194210"></a><a name="zh-cn_topic_0000002267504584_ph0579334194210"></a>dataBlockStride</span>、<span id="zh-cn_topic_0000002267504584_ph135798348424"><a name="zh-cn_topic_0000002267504584_ph135798348424"></a><a name="zh-cn_topic_0000002267504584_ph135798348424"></a>repeatStride</span>参数。<span>repeatTime</span>传入固定值即可，建议统一设置为1，该值不生效。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000002267504584_row158013415422"><td class="cellrowborder" rowspan="2" valign="top" width="8.58%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000002267504584_p258043494218"><a name="zh-cn_topic_0000002267504584_p258043494218"></a><a name="zh-cn_topic_0000002267504584_p258043494218"></a>外部API配置</p>
</td>
<td class="cellrowborder" valign="top" width="8.37%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000002267504584_p1658013346421"><a name="zh-cn_topic_0000002267504584_p1658013346421"></a><a name="zh-cn_topic_0000002267504584_p1658013346421"></a>Normal模式</p>
</td>
<td class="cellrowborder" valign="top" width="35.67%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000002267504584_p1958073434215"><a name="zh-cn_topic_0000002267504584_p1958073434215"></a><a name="zh-cn_topic_0000002267504584_p1958073434215"></a>不涉及。</p>
</td>
<td class="cellrowborder" valign="top" width="47.38%" headers="mcps1.2.5.1.4 "><div class="p" id="zh-cn_topic_0000002267504584_p1699153514480"><a name="zh-cn_topic_0000002267504584_p1699153514480"></a><a name="zh-cn_topic_0000002267504584_p1699153514480"></a>调用SetVectorMask</a>设置Mask，之后调用高维切分计算API。<a name="zh-cn_topic_0000002267504584_ul85801834154214"></a><a name="zh-cn_topic_0000002267504584_ul85801834154214"></a><ul id="zh-cn_topic_0000002267504584_ul85801834154214"><li>isSetMask模板参数设置为false，接口入参中的mask值设置为占位符MASK_PLACEHOLDER，用于占位，无实际含义。</li><li>根据使用场景配置<span>repeatTime</span>、<span id="zh-cn_topic_0000002267504584_ph1658013347423"><a name="zh-cn_topic_0000002267504584_ph1658013347423"></a><a name="zh-cn_topic_0000002267504584_ph1658013347423"></a>dataBlockStride</span>、<span id="zh-cn_topic_0000002267504584_ph2580133410427"><a name="zh-cn_topic_0000002267504584_ph2580133410427"></a><a name="zh-cn_topic_0000002267504584_ph2580133410427">repeatStride</span>参数。</li></ul>
</div>
</td>
</tr>
<tr id="zh-cn_topic_0000002267504584_row55801234104210"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000002267504584_p75801034144212"><a name="zh-cn_topic_0000002267504584_p75801034144212"></a><a name="zh-cn_topic_0000002267504584_p75801034144212"></a>Counter模式</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000002267504584_p15281754124219"><a name="zh-cn_topic_0000002267504584_p15281754124219"></a><a name="zh-cn_topic_0000002267504584_p15281754124219"></a>调用SetVectorMask设置Mask，之后调用前n个数据计算API，isSetMask模板参数设置为false；接口入参中的calCount建议设置成1。</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><div class="p" id="zh-cn_topic_0000002267504584_p13580173410556"><a name="zh-cn_topic_0000002267504584_p13580173410556"></a><a name="zh-cn_topic_0000002267504584_p13580173410556"></a>调用SetVectorMask</a>设置Mask，之后调用高维切分计算API。<a name="zh-cn_topic_0000002267504584_ul11581183434214"></a><a name="zh-cn_topic_0000002267504584_ul11581183434214"></a><ul id="zh-cn_topic_0000002267504584_ul11581183434214"><li>isSetMask模板参数设置为false；接口入参中的mask值设置为MASK_PLACEHOLDER，用于占位，无实际含义。</li><li>根据使用场景配置<span id="zh-cn_topic_0000002267504584_ph458183417420"><a name="zh-cn_topic_0000002267504584_ph458183417420"></a><a name="zh-cn_topic_0000002267504584_ph458183417420"></a>dataBlockStride</span>、<span id="zh-cn_topic_0000002267504584_ph105811034164217"><a name="zh-cn_topic_0000002267504584_ph105811034164217"></a><a name="zh-cn_topic_0000002267504584_ph105811034164217">repeatStride</span>参数。<span>repeatTime</span>传入固定值即可，建议统一设置为1，该值不生效。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

典型场景的使用示例如下：

-   场景1：Normal模式 + 外部API配置 + 高维切分计算API

    ```
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    AscendC::LocalTensor<half> src1Local;
    
    // 1、设置Normal模式
    AscendC::SetMaskNorm();
    // 2、设置Mask
    AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式
    // SetVectorMask<half, MaskMode::NORMAL>(128);  // 连续模式
    
    // 3、多次调用矢量计算API, isSetMask模板参数设置为false，接口入参中的mask值设置为占位符MASK_PLACEHOLDER，用于占位，无实际含义
    // 根据使用场景配置repeatTime、dataBlockStride、repeatStride参数
    // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内数据连续读取和写入
    // dstRepStride, src0RepStride, src1RepStride = 8, 相邻迭代间数据连续读取和写入
    AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
    AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
    AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
    // 4、恢复Mask值为默认值
    AscendC::ResetMask();
    ```

-   场景2：Counter模式 + 外部API配置 + 高维切分计算API

    ```
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    AscendC::LocalTensor<half> src1Local;
    int32_t len = 128;  // 参与计算的元素个数
    // 1、设置Counter模式
    AscendC::SetMaskCount();
    // 2、设置Mask
    AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(len);
    // 3、多次调用矢量计算API, isSetMask模板参数设置为false；接口入参中的mask值设置为MASK_PLACEHOLDER，用于占位，无实际含义
    // 根据使用场景正确配置dataBlockStride、repeatStride参数。repeatTime传入固定值即可，建议统一设置为1，该值不生效
    AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    // 4、恢复工作模式
    AscendC::SetMaskNorm();
    // 5、恢复Mask值为默认值
    AscendC::ResetMask();
    ```

-   场景3：Counter模式 + 外部API配置 + 前n个数据计算接口配合使用

    ```
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    half num = 2; 
    // 1、设置Mask
    AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(128); // 参与计算的元素个数为128
    // 2、调用前n个数据计算API，isSetMask模板参数设置为false；接口入参中的calCount建议设置成1。
    AscendC::Adds<half, false>(dstLocal, src0Local, num, 1);
    AscendC::Muls<half, false>(dstLocal, src0Local, num, 1);
    // 3、恢复工作模式
    AscendC::SetMaskNorm();
    // 4、恢复Mask值为默认值
    AscendC::ResetMask();
    ```

>[!NOTE]说明 
>-   前n个数据计算API接口内部会设置工作模式为Counter模式，所以如果前n个数据计算API配合Counter模式使用时，无需手动调用[SetMaskCount](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/SetMaskCount.md)设置Counter模式。
>-   所有手动使用Counter模式的场景，使用完毕后，需要调用[SetMaskNorm](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/SetMaskNorm.md)恢复工作模式。
>-   调用[SetVectorMask](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/SetVectorMask.md)设置Mask，使用完毕后，需要调用[ResetMask](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/context/ResetMask.md)恢复Mask值为默认值。
>-   使用高维切分计算API配套Counter模式使用时，比前n个数据计算API增加了可间隔的计算，支持dataBlockStride、repeatStride参数。

