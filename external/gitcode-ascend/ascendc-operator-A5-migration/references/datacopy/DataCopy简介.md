> **原始文档路径**: asc-devkit/docs/api/context/DataCopy简介.md

# DataCopy简介<a name="ZH-CN_TOPIC_0000002417123961"></a>

DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。

下表展示了DataCopy各项功能的描述和其通路的支持情况。

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row3507155622811"><th class="cellrowborder" align="center" valign="top" id="mcps1.1.9.1.1"><p id="p15507105619289"><a name="p15507105619289"></a><a name="p15507105619289"></a>功能</p>
</th>
<th class="cellrowborder" align="center" valign="top" id="mcps1.1.9.1.2"><p id="p450715566283"><a name="p450715566283"></a><a name="p450715566283"></a>描述</p>
</th>
<th class="cellrowborder" colspan="2" align="center" valign="top" id="mcps1.1.9.1.3"><p id="p11321143964118"><a name="p11321143964118"></a><a name="p11321143964118"></a>Local Memory -&gt; Global Memory</p>
</th>
<th class="cellrowborder" colspan="2" align="center" valign="top" id="mcps1.1.9.1.4"><p id="p12456144213412"><a name="p12456144213412"></a><a name="p12456144213412"></a>Global Memory -&gt; Local Memory</p>
</th>
<th class="cellrowborder" colspan="2" align="center" valign="top" id="mcps1.1.9.1.5"><p id="p92331244154110"><a name="p92331244154110"></a><a name="p92331244154110"></a>Local Memory -&gt; Local Memory</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><a href="基础数据搬运.md">基础数据搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a><span id="ph5281655142914"><a name="ph5281655142914"></a><a name="ph5281655142914"></a>提供基础的数据搬运能力，数据在传输过程中保持原始格式和内容不变，支持连续和非连续的数据搬运。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p1854115683315"><a name="p1854115683315"></a><a name="p1854115683315"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p13445193515"><a name="p13445193515"></a><a name="p13445193515"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p73610563511"><a name="p73610563511"></a><a name="p73610563511"></a>√</p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><a href="增强数据搬运.md">增强数据搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p15269162994511"><a name="p15269162994511"></a><a name="p15269162994511"></a><span id="ph65385444614"><a name="ph65385444614"></a><a name="ph65385444614"></a>对数据搬运能力进行增强，相比于基础数据搬运接口，增加了CO1-&gt;CO2通路的随路计算。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p753396133515"><a name="p753396133515"></a><a name="p753396133515"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p9534186193515"><a name="p9534186193515"></a><a name="p9534186193515"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p1253520613355"><a name="p1253520613355"></a><a name="p1253520613355"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><a href="切片数据搬运.md">切片数据搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a><span id="ph1188502514571"><a name="ph1188502514571"></a><a name="ph1188502514571"></a>支持数据的切片搬运，提取多维Tensor数据的子集进行搬运。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p1350517207369"><a name="p1350517207369"></a><a name="p1350517207369"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p050692063612"><a name="p050692063612"></a><a name="p050692063612"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p35419619334"><a name="p35419619334"></a><a name="p35419619334"></a>×</p>
</td>
</tr>
<tr id="row103361763242"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p1902158683"><a name="p1902158683"></a><a name="p1902158683"></a><a href="随路转换ND2NZ搬运.md">随路转换ND2NZ搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p1695483941817"><a name="p1695483941817"></a><a name="p1695483941817"></a><span id="ph92961431014"><a name="ph92961431014"></a><a name="ph92961431014"></a>支持在数据搬运时进行ND到NZ格式的转换。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p1228215423615"><a name="p1228215423615"></a><a name="p1228215423615"></a>×</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p1848012253617"><a name="p1848012253617"></a><a name="p1848012253617"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p8481522133617"><a name="p8481522133617"></a><a name="p8481522133617"></a>√</p>
</td>
</tr>
<tr id="row18403312418"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p226110614915"><a name="p226110614915"></a><a name="p226110614915"></a><a href="随路转换NZ2ND搬运.md">随路转换NZ2ND搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p1695443971810"><a name="p1695443971810"></a><a name="p1695443971810"></a><span id="ph997516715484"><a name="ph997516715484"></a><a name="ph997516715484"></a>支持在数据搬运时进行NZ到ND格式的转换。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p209841425143610"><a name="p209841425143610"></a><a name="p209841425143610"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p65417611332"><a name="p65417611332"></a><a name="p65417611332"></a>×</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p165511012379"><a name="p165511012379"></a><a name="p165511012379"></a>×</p>
</td>
</tr>
<tr id="row17253142120252"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="p3317192721517"><a name="p3317192721517"></a><a name="p3317192721517"></a><a href="随路量化激活搬运.md">随路量化激活搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p68258134271"><a name="p68258134271"></a><a name="p68258134271"></a><span id="ph1073053771815"><a name="ph1073053771815"></a><a name="ph1073053771815"></a>支持在数据搬运过程中进行量化和Relu激活等操作，同时支持Local Memory到Global Memory通路NZ到ND格式的转换。</span></p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p5911033183616"><a name="p5911033183616"></a><a name="p5911033183616"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p74967123714"><a name="p74967123714"></a><a name="p74967123714"></a>×</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p590783413615"><a name="p590783413615"></a><a name="p590783413615"></a>√</p>
</td>
</tr>
<tr id="row1550572453212"><td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.1 "><p id="zh-cn_topic_0000002299156297_p5831180161811"><a name="zh-cn_topic_0000002299156297_p5831180161811"></a><a name="zh-cn_topic_0000002299156297_p5831180161811"></a><a href="多维数据搬运（ISASI）.md">多维数据搬运</a></p>
</td>
<td class="cellrowborder" align="center" valign="top" headers="mcps1.1.9.1.2 "><p id="p171162612257"><a name="p171162612257"></a><a name="p171162612257"></a>多维数据搬运接口，相比于基础数据搬运接口，可更加自由配置搬入的维度信息以及对应的Stride。</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.3 "><p id="p154156153315"><a name="p154156153315"></a><a name="p154156153315"></a>×</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.4 "><p id="p65653753619"><a name="p65653753619"></a><a name="p65653753619"></a>√</p>
</td>
<td class="cellrowborder" colspan="2" align="center" valign="top" headers="mcps1.1.9.1.5 "><p id="p15411760337"><a name="p15411760337"></a><a name="p15411760337"></a>×</p>
</td>
</tr>
</tbody>
</table>

