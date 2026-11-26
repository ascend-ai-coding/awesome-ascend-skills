# torch_npu接口列表

本章节包含常用自定义接口，包括创建Tensor及计算类操作。

**表1** torch_npu API

<a name="table1849611717116"></a>

<table><thead align="left"><tr id="row10496101716111"><th class="cellrowborder" valign="top" width="38.61%" id="mcps1.2.3.1.1"><p id="p1649713174119"><a name="p1649713174119"></a><a name="p1649713174119"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="61.39%" id="mcps1.2.3.1.2"><p id="p9497217151115"><a name="p9497217151115"></a><a name="p9497217151115"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1149711715114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13497217191119"><a name="p13497217191119"></a><a name="p13497217191119"></a><a href="（beta）torch_npu-_npu_dropout.md">（beta）torch_npu._npu_dropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p349751701118"><a name="p349751701118"></a><a name="p349751701118"></a>不使用种子（seed）进行dropout结果计数。</p>
</td>
</tr>
<tr id="row949712178110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5497817161113"><a name="p5497817161113"></a><a name="p5497817161113"></a><a href="（beta）torch_npu-fast_gelu.md">（beta）torch_npu.fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9497101717113"><a name="p9497101717113"></a><a name="p9497101717113"></a>快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function）对输入的每个元素计算FastGelu的前向结果。</p>
</td>
</tr>
<tr id="row17497217181119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16497617191116"><a name="p16497617191116"></a><a name="p16497617191116"></a><a href="（beta）torch_npu-npu_alloc_float_status.md">（beta）torch_npu.npu_alloc_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p6497141711115"><a name="p6497141711115"></a><a name="p6497141711115"></a>申请一个专门用于存储浮点运算状态标志的tensor。该tensor用于后续记录计算过程中的溢出状态。</p>
</td>
</tr>
<tr id="row104977172114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1849701710119"><a name="p1849701710119"></a><a name="p1849701710119"></a><a href="（beta）torch_npu-npu_anchor_response_flags.md">（beta）torch_npu.npu_anchor_response_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p69023133817"><a name="zh-cn_topic_0000001655404257_p69023133817"></a><a name="zh-cn_topic_0000001655404257_p69023133817"></a>在单个特征图中生成锚点的责任标志。</p>
</td>
</tr>
<tr id="row1920533291116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13205173251112"><a name="p13205173251112"></a><a name="p13205173251112"></a><a href="（beta）torch_npu-npu_batch_nms.md">（beta）torch_npu.npu_batch_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p820523261113"><a name="p820523261113"></a><a name="p820523261113"></a>根据batch分类计算输入框评分，通过评分排序，删除评分高于阈值（iou_threshold）的框，支持多批多类处理。通过NonMaxSuppression（nms）操作可有效删除冗余的输入框，提高检测精度。NonMaxSuppression：抑制不是极大值的元素，搜索局部的极大值，常用于计算机视觉任务中的检测类模型。</p>
</td>
</tr>
<tr id="row162057328112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11205193211112"><a name="p11205193211112"></a><a name="p11205193211112"></a><a href="（beta）torch_npu-npu_bert_apply_adam.md">（beta）torch_npu.npu_bert_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12061132181116"><a name="p12061132181116"></a><a name="p12061132181116"></a>针对BERT模型，获取Adam优化器的计算结果。</p>
</td>
</tr>
<tr id="row182061132151115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p52061332161110"><a name="p52061332161110"></a><a name="p52061332161110"></a><a href="（beta）torch_npu-npu_bmmV2.md">（beta）torch_npu.npu_bmmV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1582323323616"><a name="zh-cn_topic_0000001655404257_p1582323323616"></a><a name="zh-cn_topic_0000001655404257_p1582323323616"></a>将矩阵“a”乘以矩阵“b”，生成“a*b”。</p>
</td>
</tr>
<tr id="row520615328117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p020653281114"><a name="p020653281114"></a><a name="p020653281114"></a><a href="（beta）torch_npu-npu_bounding_box_decode.md">（beta）torch_npu.npu_bounding_box_decode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p2640134717250"><a name="zh-cn_topic_0000001655404257_p2640134717250"></a><a name="zh-cn_topic_0000001655404257_p2640134717250"></a>根据rois和deltas生成标注框。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row657717509111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2577105001114"><a name="p2577105001114"></a><a name="p2577105001114"></a><a href="（beta）torch_npu-npu_bounding_box_encode.md">（beta）torch_npu.npu_bounding_box_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2577105019119"><a name="p2577105019119"></a><a name="p2577105019119"></a>计算标注框和ground truth真值框之间的坐标变化。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row589165316115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1489155321111"><a name="p1489155321111"></a><a name="p1489155321111"></a><a href="（beta）torch_npu-npu_ciou.md">（beta）torch_npu.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p749621223015"><a name="zh-cn_topic_0000001655404257_p749621223015"></a><a name="zh-cn_topic_0000001655404257_p749621223015"></a>基于NPU的CIoU操作，计算预测边界框与真实边界框之间的CIoU（Complete Intersection over Union）损失。</p>
</td>
</tr>
<tr id="row138914536113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1289453191113"><a name="p1289453191113"></a><a name="p1289453191113"></a><a href="（beta）torch_npu-npu_clear_float_status.md">（beta）torch_npu.npu_clear_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p289105341110"><a name="p289105341110"></a><a name="p289105341110"></a>清除溢出检测相关标志位。</p>
</td>
</tr>
<tr id="row158945317117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5901953141111"><a name="p5901953141111"></a><a name="p5901953141111"></a><a href="（beta）torch_npu-npu_confusion_transpose.md">（beta）torch_npu.npu_confusion_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p7908532119"><a name="p7908532119"></a><a name="p7908532119"></a>融合reshape和transpose运算。</p>
</td>
</tr>
<tr id="row109045310113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p49035381110"><a name="p49035381110"></a><a name="p49035381110"></a><a href="（beta）torch_npu-npu_conv3d.md">（beta）torch_npu.npu_conv3d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18901253111116"><a name="p18901253111116"></a><a name="p18901253111116"></a>在由多个输入平面组成的输入图像上应用一个3D卷积。</p>
</td>
</tr>
<tr id="row75771250101115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p15771150151111"><a name="p15771150151111"></a><a name="p15771150151111"></a><a href="（beta）torch_npu-npu_deformable_conv2d.md">（beta）torch_npu.npu_deformable_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1888192943717"><a name="zh-cn_topic_0000001655404257_p1888192943717"></a><a name="zh-cn_topic_0000001655404257_p1888192943717"></a>使用预期输入计算变形卷积输出（deformed convolution output）。</p>
</td>
</tr>
<tr id="row125771550121117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16577135019112"><a name="p16577135019112"></a><a name="p16577135019112"></a><a href="（beta）torch_npu-npu_diou.md">（beta）torch_npu.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p12796833183010"><a name="zh-cn_topic_0000001655404257_p12796833183010"></a><a name="zh-cn_topic_0000001655404257_p12796833183010"></a>该接口用于实现基于NPU的DIoU（距离交并比）计算。该算法综合考虑预测框与真实框中心点之间的距离以及边界框的重叠率，使边界框回归趋于稳定，提升目标定位精度。</p>
</td>
</tr>
<tr id="row195782506118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1857885061113"><a name="p1857885061113"></a><a name="p1857885061113"></a><a href="（beta）torch_npu-npu_format_cast.md">（beta）torch_npu.npu_format_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p68591917123"><a name="zh-cn_topic_0000001655404257_p68591917123"></a><a name="zh-cn_topic_0000001655404257_p68591917123"></a>修改NPU张量的格式。</p>
</td>
</tr>
<tr id="row457855012117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p557813506114"><a name="p557813506114"></a><a name="p557813506114"></a><a href="（beta）torch_npu-npu_format_cast_.md">（beta）torch_npu.npu_format_cast_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125789503117"><a name="p125789503117"></a><a name="p125789503117"></a>原地修改self张量格式，与src格式保持一致。</p>
</td>
</tr>
<tr id="row1757895091111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17578195010114"><a name="p17578195010114"></a><a name="p17578195010114"></a><a href="（beta）torch_npu-npu_get_float_status.md">（beta）torch_npu.npu_get_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p8144193193519"><a name="zh-cn_topic_0000001655404257_p8144193193519"></a><a name="zh-cn_topic_0000001655404257_p8144193193519"></a>获取溢出检测结果。</p>
</td>
</tr>
<tr id="row3206113212114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p42065329112"><a name="p42065329112"></a><a name="p42065329112"></a><a href="（beta）torch_npu-npu_giou.md">（beta）torch_npu.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p0835111286"><a name="zh-cn_topic_0000001655404257_p0835111286"></a><a name="zh-cn_topic_0000001655404257_p0835111286"></a>首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。</p>
</td>
</tr>
<tr id="row5377163616111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p937723691115"><a name="p937723691115"></a><a name="p937723691115"></a><a href="（beta）torch_npu-npu_grid_assign_positive.md">（beta）torch_npu.npu_grid_assign_positive</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p258994112512"><a name="zh-cn_topic_0000001655404257_p258994112512"></a><a name="zh-cn_topic_0000001655404257_p258994112512"></a><span>执行position-sensitive的候选区域池化梯度计算。</span></p>
</td>
</tr>
<tr id="row557919470115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p157994710118"><a name="p157994710118"></a><a name="p157994710118"></a><a href="（beta）torch_npu-npu_indexing.md">（beta）torch_npu.npu_indexing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p191786291"><a name="zh-cn_topic_0000001655404257_p191786291"></a><a name="zh-cn_topic_0000001655404257_p191786291"></a>以begin为起始索引，end为结束索引，strides为步长，对输入张量进行切片。</p>
</td>
</tr>
<tr id="row11579164761115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p195798473119"><a name="p195798473119"></a><a name="p195798473119"></a><a href="（beta）torch_npu-npu_iou.md">（beta）torch_npu.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p10544194202310"><a name="zh-cn_topic_0000001655404257_p10544194202310"></a><a name="zh-cn_topic_0000001655404257_p10544194202310"></a>根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。</p>
</td>
</tr>
<tr id="row11579104721117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p135794473113"><a name="p135794473113"></a><a name="p135794473113"></a><a href="（beta）torch_npu-npu_linear.md">（beta）torch_npu.npu_linear</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8579144781110"><a name="p8579144781110"></a><a name="p8579144781110"></a>将矩阵“a”乘以矩阵“b”，生成“a*b”。</p>
</td>
</tr>
<tr id="row9579184715119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p85793478116"><a name="p85793478116"></a><a name="p85793478116"></a><a href="（beta）torch_npu-npu_lstm.md">（beta）torch_npu.npu_lstm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p757912479111"><a name="p757912479111"></a><a name="p757912479111"></a>计算LSTM（长短期记忆网络）前向传播。</p>
</td>
</tr>
<tr id="row3579194701112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16579184717113"><a name="p16579184717113"></a><a name="p16579184717113"></a><a href="（beta）torch_npu-npu_max.md">（beta）torch_npu.npu_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1758054719115"><a name="p1758054719115"></a><a name="p1758054719115"></a>使用dim对最大结果进行计算。类似于torch.max，优化NPU设备实现。</p>
</td>
</tr>
<tr id="row558010477110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17580147201113"><a name="p17580147201113"></a><a name="p17580147201113"></a><a href="（beta）torch_npu-npu_nms_rotated.md">（beta）torch_npu.npu_nms_rotated</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p2907120192218"><a name="zh-cn_topic_0000001655404257_p2907120192218"></a><a name="zh-cn_topic_0000001655404257_p2907120192218"></a>按分数降序选择旋转标注框的子集。</p>
</td>
</tr>
<tr id="row1158054718119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1858004714112"><a name="p1858004714112"></a><a name="p1858004714112"></a><a href="（beta）torch_npu-npu_nms_v4.md">（beta）torch_npu.npu_nms_v4</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p296295602119"><a name="zh-cn_topic_0000001655404257_p296295602119"></a><a name="zh-cn_topic_0000001655404257_p296295602119"></a>按分数降序选择标注框的子集。</p>
</td>
</tr>
<tr id="row85801547131119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p105802478112"><a name="p105802478112"></a><a name="p105802478112"></a><a href="（beta）torch_npu-npu_nms_with_mask.md">（beta）torch_npu.npu_nms_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1358054721113"><a name="p1358054721113"></a><a name="p1358054721113"></a>生成值0或1，用于NMS算子确定有效位。</p>
</td>
</tr>
<tr id="row858014718113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p25801047101117"><a name="p25801047101117"></a><a name="p25801047101117"></a><a href="（beta）torch_npu-npu_one_hot.md">（beta）torch_npu.npu_one_hot</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5580547191120"><a name="p5580547191120"></a><a name="p5580547191120"></a>返回一个one-hot张量。input中每个元素的值v作为类别索引：输出在one_hot维（深度为depth）索引v处填充on_value，其余位置填充off_value。input应为整数索引张量，元素取值建议在[0, depth)范围内，越界的元素对应的输出全部填充off_value。</p>
</td>
</tr>
<tr id="row2037715364114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p173771360112"><a name="p173771360112"></a><a name="p173771360112"></a><a href="（beta）torch_npu-npu_pad.md">（beta）torch_npu.npu_pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93771036191114"><a name="p93771036191114"></a><a name="p93771036191114"></a>填充张量。</p>
</td>
</tr>
<tr id="row1537743614119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p737793615117"><a name="p737793615117"></a><a name="p737793615117"></a><a href="（beta）torch_npu-npu_ps_roi_pooling.md">（beta）torch_npu.npu_ps_roi_pooling</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1037763620111"><a name="p1037763620111"></a><a name="p1037763620111"></a>执行Position Sensitive ROI Pooling。</p>
</td>
</tr>
<tr id="row2037743613112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p12377203601118"><a name="p12377203601118"></a><a name="p12377203601118"></a><a href="（beta）torch_npu-npu_ptiou.md">（beta）torch_npu.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1593155184019"><a name="p1593155184019"></a><a name="p1593155184019"></a>根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。</p>
</td>
</tr>
<tr id="row1337793616112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p133781436181115"><a name="p133781436181115"></a><a name="p133781436181115"></a><a href="（beta）torch_npu-npu_random_choice_with_mask.md">（beta）torch_npu.npu_random_choice_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p219311346269"><a name="zh-cn_topic_0000001655404257_p219311346269"></a><a name="zh-cn_topic_0000001655404257_p219311346269"></a>混洗非零元素的index。</p>
</td>
</tr>
<tr id="row11378336161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83782360112"><a name="p83782360112"></a><a name="p83782360112"></a><a href="（beta）torch_npu-npu_roi_align.md">（beta）torch_npu.npu_roi_align</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11378133620112"><a name="p11378133620112"></a><a name="p11378133620112"></a>从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row168411042171118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11841184214110"><a name="p11841184214110"></a><a name="p11841184214110"></a><a href="（beta）torch_npu-npu_rotated_iou.md">（beta）torch_npu.npu_rotated_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p384134214117"><a name="p384134214117"></a><a name="p384134214117"></a>计算旋转框的IoU。</p>
</td>
</tr>
<tr id="row1284118427112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p0841184211115"><a name="p0841184211115"></a><a name="p0841184211115"></a><a href="（beta）torch_npu-npu_rotated_overlaps.md">（beta）torch_npu.npu_rotated_overlaps</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20841164221110"><a name="p20841164221110"></a><a name="p20841164221110"></a>计算旋转框的重叠面积。</p>
</td>
</tr>
<tr id="row68411842121111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1384134221114"><a name="p1384134221114"></a><a name="p1384134221114"></a><a href="（beta）torch_npu-npu_sign_bits_pack.md">（beta）torch_npu.npu_sign_bits_pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p484114213112"><a name="p484114213112"></a><a name="p484114213112"></a>将float类型1位Adam打包为`torch.uint8`。</p>
</td>
</tr>
<tr id="row128412042131111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p784116424113"><a name="p784116424113"></a><a name="p784116424113"></a><a href="（beta）torch_npu-npu_sign_bits_unpack.md">（beta）torch_npu.npu_sign_bits_unpack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p58411242191112"><a name="p58411242191112"></a><a name="p58411242191112"></a>将`torch.uint8`类型1位Adam拆包为float。</p>
</td>
</tr>
<tr id="row68428425112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1842204218113"><a name="p1842204218113"></a><a name="p1842204218113"></a><a href="（beta）torch_npu-npu_slice.md">（beta）torch_npu.npu_slice</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p0842124210117"><a name="p0842124210117"></a><a name="p0842124210117"></a>从张量中提取切片。</p>
</td>
</tr>
<tr id="row7842204261115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13842442161112"><a name="p13842442161112"></a><a name="p13842442161112"></a><a href="（beta）torch_npu-npu_softmax_cross_entropy_with_logits.md">（beta）torch_npu.npu_softmax_cross_entropy_with_logits</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1842144291119"><a name="p1842144291119"></a><a name="p1842144291119"></a>计算softmax的交叉熵损失。</p>
</td>
</tr>
<tr id="row884254219118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1784284271118"><a name="p1784284271118"></a><a name="p1784284271118"></a><a href="（beta）torch_npu-npu_transpose.md">（beta）torch_npu.npu_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1884294261114"><a name="p1884294261114"></a><a name="p1884294261114"></a>返回原始张量视图，其维度已permute，结果连续。</p>
</td>
</tr>
<tr id="row118421942161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p188423429111"><a name="p188423429111"></a><a name="p188423429111"></a><a href="（beta）torch_npu-npu_yolo_boxes_encode.md">（beta）torch_npu.npu_yolo_boxes_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p68421942191116"><a name="p68421942191116"></a><a name="p68421942191116"></a>根据YOLO的锚点框（anchor box）和真值框（ground-truth box）生成标注框。自定义mmdetection算子。</p>
</td>
</tr>
<tr id="row6842164251115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p19842134241119"><a name="p19842134241119"></a><a name="p19842134241119"></a><a href="（beta）torch_npu-npu_fused_attention_score.md">（beta）torch_npu.npu_fused_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p98421642101113"><a name="p98421642101113"></a><a name="p98421642101113"></a>实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。</p>
</td>
</tr>
<tr id="row2378193620119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9378836161118"><a name="p9378836161118"></a><a name="p9378836161118"></a><a href="（beta）torch_npu-npu_multi_head_attention.md">（beta）torch_npu.npu_multi_head_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14378143615111"><a name="p14378143615111"></a><a name="p14378143615111"></a>实现Transformer模块中的MultiHeadAttention计算逻辑。</p>
</td>
</tr>
<tr id="row14545161172716"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p854661182710"><a name="p854661182710"></a><a name="p854661182710"></a><a href="（beta）torch_npu-npu_rms_norm.md">（beta）torch_npu.npu_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125461318273"><a name="p125461318273"></a><a name="p125461318273"></a>RMSNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。</p>
</td>
</tr>
<tr id="row2791419328"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1990671211278"><a name="p1990671211278"></a><a name="p1990671211278"></a><a href="（beta）torch_npu-npu_dropout_with_add_softmax.md">（beta）torch_npu.npu_dropout_with_add_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p15906141216278"><a name="p15906141216278"></a><a name="p15906141216278"></a>实现axpy_v2、softmax_v2、drop_out_domask_v3功能。</p>
</td>
</tr>
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_quant.md">torch_npu.npu_rms_norm_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16917252327389"><a name="p16917252327389"></a><a name="p16917252327389"></a>融合RmsNorm算子和Quantize算子，减少搬入搬出的操作。</p>
</td>
</tr>
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_quant_v2.md">torch_npu.npu_rms_norm_quant_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16917252327389"><a name="p16917252327389"></a><a name="p16917252327389"></a>融合RmsNorm算子和Quantize算子，减少搬入搬出的操作，v2版本新增了rstd的输出。</p>
</td>
</tr>
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_dynamic_mx_quant.md">torch_npu.npu_rms_norm_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16917252327389"><a name="p16917252327389"></a><a name="p16917252327389"></a>实现RmsNorm算子与DynamicMxQuant算子融合，先对输入进行RmsNorm归一化，将RmsNorm结果与beta相加后再进行动态MX量化，减少搬入搬出操作。其中RmsNorm是大模型常用的归一化操作，相比LayerNorm其去掉了减去均值的部分。</p>
</td>
</tr>
<tr id="row17801219526"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p89062124277"><a name="p89062124277"></a><a name="p89062124277"></a><a href="torch_npu-npu_rotary_mul.md">torch_npu.npu_rotary_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p69061812102715"><a name="p69061812102715"></a><a name="p69061812102715"></a>实现RotaryEmbedding旋转位置编码。</p>
</td>
</tr>
<tr id="row58015194219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1590618121275"><a name="p1590618121275"></a><a name="p1590618121275"></a><a href="torch_npu-npu_scaled_masked_softmax.md">torch_npu.npu_scaled_masked_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1290691218270"><a name="p1290691218270"></a><a name="p1290691218270"></a>计算输入张量x经缩放和mask遮蔽后的Softmax结果。</p>
</td>
</tr>
<tr id="row480181910211"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p137651452352"><a name="p137651452352"></a><a name="p137651452352"></a><a href="（beta）torch_npu-npu_swiglu.md">（beta）torch_npu.npu_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14325153814812"><a name="p14325153814812"></a><a name="p14325153814812"></a>提供swiglu的激活函数。</p>
</td>
</tr>
<tr id="row106843515114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p168510515113"><a name="p168510515113"></a><a name="p168510515113"></a><a href="torch_npu-npu_group_norm_swish.md">torch_npu.npu_group_norm_swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p19342196105315"><a name="p19342196105315"></a><a name="p19342196105315"></a>计算输入input的组归一化结果y、均值mean、标准差的倒数rstd，以及swish的输出。</p>
</td>
</tr>
<tr id="row10636159121815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p763620598183"><a name="p763620598183"></a><a name="p763620598183"></a><a href="torch_npu-npu_cross_entropy_loss.md">torch_npu.npu_cross_entropy_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18198201124816"><a name="p18198201124816"></a><a name="p18198201124816"></a>将原生CrossEntropyLoss中的log_softmax和nll_loss融合，降低计算时使用的内存。接口允许计算zloss。</p>
</td>
</tr>
<tr id="row142536475869"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p142536475870"><a name="p142536475870"></a><a name="p142536475870"></a><a href="torch_npu-npu_add_rms_norm_quant.md">torch_npu.npu_add_rms_norm_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p142536475871"><a name="p142536475871"></a><a name="p142536475871"></a>将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。</p>
</td>
</tr>
<tr id="row724843692712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p162481336102714"><a name="p162481336102714"></a><a name="p162481336102714"></a><a href="torch_npu-npu_advance_step_flashattn.md">torch_npu.npu_advance_step_flashattn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p13248936102717"><a name="p13248936102717"></a><a name="p13248936102717"></a>在NPU上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新input_tokens，input_positions，seq_lens和slot_mapping。</p>
</td>
</tr>
<tr id="row1526115194711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9360134415216"><a name="p9360134415216"></a><a name="p9360134415216"></a><a href="torch_npu-npu_all_gather_base_mm.md">torch_npu.npu_all_gather_base_mm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001694916914_p444492882116"><a name="zh-cn_topic_0000001694916914_p444492882116"></a><a name="zh-cn_topic_0000001694916914_p444492882116"></a>TP切分场景下，实现allgather和matmul的融合，融合算子内部实现通信和计算流水并行。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_all_gather_quant_mm.md">torch_npu.npu_all_gather_quant_mm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>TP切分的量化场景下，实现allgather和matmul的融合，输出反量化结果。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_all_to_all_matmul.md">torch_npu.npu_all_to_all_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>完成AlltoAll通信、Permute（保证通信后地址连续）和Matmul计算的融合，先通信后计算。</p>
</td>
</tr>
<tr id="row950000000021"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000022"><a name="p950000000022"></a><a name="p950000000022"></a><a href="torch_npu-npu_all_to_all_quant_matmul.md">torch_npu.npu_all_to_all_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000023"><a name="p950000000023"></a><a name="p950000000023"></a>完成量化的Matmul计算、Permute（保证通信后地址连续）和AlltoAll通信的融合，先通信后计算。支持K-C动态量化模式和mx量化模式。</p>
</td>
</tr>
<tr id="npu_anti_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_anti_mx_quant"><a name="p_npu_anti_mx_quant"></a><a name="p_npu_anti_mx_quant"></a><a href="torch_npu-npu_anti_mx_quant.md">torch_npu.npu_anti_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_anti_mx_quant_desc"><a name="p_npu_anti_mx_quant_desc"></a><a name="p_npu_anti_mx_quant_desc"></a><span>将调用npu_dynamic_mx_quant量化得到的float4/float8的Tensor反量化为`torch.float16`/`torch.bfloat16`/`torch.float32`格式，是npu_dynamic_mx_quant的逆过程。</span></p></td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_anti_quant.md">torch_npu.npu_anti_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001850161621_p0996174814315"><a name="zh-cn_topic_0000001850161621_p0996174814315"></a><a name="zh-cn_topic_0000001850161621_p0996174814315"></a>对张量x进行反量化操作，即将量化后的低精度数据恢复为浮点数。反量化是量化的逆过程，用于将低精度数据（如`torch.int8`、`torch_npu.int4`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`）转换回高精度浮点数。</p>
</td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_attention_to_ffn.md">torch_npu.npu_attention_to_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001850161621_p0996174814315"><a name="zh-cn_topic_0000001850161621_p0996174814315"></a><a name="zh-cn_topic_0000001850161621_p0996174814315"></a>将Attention节点上数据发往FFN节点。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_attention_update.md">torch_npu.npu_attention_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>将各SP域PA算子输出的中间结果lse、local_out两个局部变量结果更新成全局结果。</p>
</td>
</tr>
<tr id="row_block_sparse_attention"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10326218132921"><a name="p10326218132921"></a><a name="p10326218132921"></a><a href="torch_npu-npu_block_sparse_attention.md">torch_npu.npu_block_sparse_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10326218132922"><a name="p10326218132922"></a><a name="p10326218132922"></a>BlockSparseAttention 稀疏注意力正向计算，支持块级稀疏模式，通过 block_sparse_mask 指定每个 Q 块选择的 KV 块，实现高效稀疏注意力。</p>
</td>
</tr>
<tr id="row_chunk_gated_delta_rule"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10326218132923"><a name="p10326218132923"></a><a name="p10326218132923"></a><a href="torch_npu-npu_chunk_gated_delta_rule.md">torch_npu.npu_chunk_gated_delta_rule</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10326218132924"><a name="p10326218132924"></a><a name="p10326218132924"></a>Chunked Gated Delta Rule(CGDR)是GDR的chunk版实现，它通过将输入序列切块，实现了一定的并行效果，在长上下文场景其计算效率相对Recurrent Gated Delta Rule更高，适用于prefill阶段。</p>
</td>
</tr>
<tr id="row16527145114477"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1338142581"><a name="p1338142581"></a><a name="p1338142581"></a><a href="torch_npu-npu_convert_weight_to_int4pack.md">torch_npu.npu_convert_weight_to_int4pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16338242582"><a name="p16338242582"></a><a name="p16338242582"></a>将数据类型为`torch.int32`的输入tensor打包为`torch_npu.int4`存放，每8个`torch_npu.int4`数据通过一个`torch.int32`数据承载，并进行交叠排放。</p>
</td>
</tr>
<tr id="row194591911194818"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p633812455816"><a name="p633812455816"></a><a name="p633812455816"></a><a href="torch_npu-npu_dynamic_quant.md">torch_npu.npu_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93381541583"><a name="p93381541583"></a><a name="p93381541583"></a>对输入的张量进行动态量化，支持pertoken、pertensor、perchannel和MoE（Mixture of Experts，混合专家模型）场景。</p>
</td>
</tr>
<tr id="row44591911184819"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p820225218412"><a name="p820225218412"></a><a name="p820225218412"></a><a href="torch_npu-npu_dynamic_quant_asymmetric.md">torch_npu.npu_dynamic_quant_asymmetric</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p271341016220"><a name="p271341016220"></a><a name="p271341016220"></a>对输入的张量进行动态非对称量化。支持pertoken、pertensor、perchannel和MoE（Mixture of Experts，混合专家模型）场景。</p>
</td>
</tr>
<tr id="npu_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant"><a name="p_npu_dynamic_mx_quant"></a><a name="p_npu_dynamic_mx_quant"></a><a href="torch_npu-npu_dynamic_mx_quant.md">torch_npu.npu_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_desc"><a name="p_npu_dynamic_mx_quant_desc"></a><a name="p_npu_dynamic_mx_quant_desc"></a><span>对输入张量进行目的数据类型为float4、float8的动态MX量化，在给定的轴上根据每block_size个数计算出对应的量化尺度，再对组内数据除以该尺度后转换为目的数据类型。</span></p></td>
</tr>
<tr id="npu_dynamic_block_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_block_mx_quant"><a name="p_npu_dynamic_block_mx_quant"></a><a name="p_npu_dynamic_block_mx_quant"></a><a href="torch_npu-npu_dynamic_block_mx_quant.md">torch_npu.npu_dynamic_block_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_block_mx_quant_desc"><a name="p_npu_dynamic_block_mx_quant_desc"></a><a name="p_npu_dynamic_block_mx_quant_desc"></a><span>对输入变量，以数据块（32*32）为基本块进行MX量化并转换为目标数据类型。</span></p></td>
</tr>
<tr id="npu_gelu_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_gelu_quant"><a name="p_npu_gelu_quant"></a><a name="p_npu_gelu_quant"></a><a href="torch_npu-npu_gelu_quant.md">torch_npu.npu_gelu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_gelu_quant_desc"><a name="p_npu_gelu_quant_desc"></a><a name="p_npu_gelu_quant_desc"></a><span>对张量进行GELU（Gaussian Error Linear Unit，高斯误差线性单元）激活操作，再对结果进行静态/动态量化。</span></p></td>
</tr>
<tr id="row1245931134815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1055135819577"><a name="p1055135819577"></a><a name="p1055135819577"></a><a href="torch_npu-npu_fast_gelu.md">torch_npu.npu_fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p135525810574"><a name="p135525810574"></a><a name="p135525810574"></a>快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算FastGelu；输入是具有任何有效形状的张量。</p>
</td>
</tr>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn.md">torch_npu.npu_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（expert_tokens为空）时是FFN，有专家分组时是MoeFFN。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score.md">torch_npu.npu_fused_infer_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>适配增量&amp;全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。</p>
</td>
</tr>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn_to_attention.md">torch_npu.npu_ffn_to_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>将FFN节点上的数据发往Attention节点。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score_v2.md">torch_npu.npu_fused_infer_attention_score_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>适配增量&amp;全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。新增支持MLA（MultiHead Latent Attention）全量化功能。</p>
</td>
</tr>
<tr id="row_npu_fused_matmul"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_fused_matmul.md">torch_npu.npu_fused_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>矩阵乘与通用向量计算融合，减少数据搬运提升性能。</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention.md">torch_npu.npu_fusion_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>实现“Transformer Attention Score”的融合计算。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_fusion_attention_v2.md">torch_npu.npu_fusion_attention_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>实现“Transformer Attention Score”的融合计算，支持Rope扩展，偏置sink以及GQA。</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention_v3.md">torch_npu.npu_fusion_attention_v3</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>实现“Transformer Attention Score”的融合计算，支持图模式。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p1720095217417"></a><a name="p1720095217417"></a><a href="torch_npu-npu_gelu.md">torch_npu.npu_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9269284217"><a name="p9269284217"></a><a name="p9269284217"></a>计算高斯误差线性单元（GELU）激活函数。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p1720095217417"></a><a name="p1720095217417"></a><a href="torch_npu-npu_gelu_mul.md">torch_npu.npu_gelu_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9269284217"><a name="p9269284217"></a><a name="p9269284217"></a>当输入tensor的尾轴为32B对齐场景时，使用该API可对输入tensor进行GELU与MUL结合的复合计算操作以提高算子性能，若尾轴为非32B对齐场景时，建议走小算子拼接逻辑，即按照下述公式分步拼接计算。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p5338134195811"></a><a name="p5338134195811"></a><a href="torch_npu-npu_group_norm_silu.md">torch_npu.npu_group_norm_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5338546582"><a name="p5338546582"></a><a name="p5338546582"></a>对输入张量input依次执行GroupNorm和SiLU激活，返回三个张量：out（SiLU激活后的输出）、meanOut（归一化均值）、rstdOut（归一化标准差的倒数）。</p>
</td>
</tr>
<tr id="row174571042727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10203752943"><a name="p10203752943"></a><a name="p10203752943"></a><a href="torch_npu-npu_group_quant.md">torch_npu.npu_group_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p4382954217"><a name="p4382954217"></a><a name="p4382954217"></a>对输入的张量进行分组量化操作。</p>
</td>
</tr>
<tr id="npu_grouped_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_dynamic_mx_quant"><a name="p_npu_grouped_dynamic_mx_quant"></a><a name="p_npu_grouped_dynamic_mx_quant"></a><a href="torch_npu-npu_grouped_dynamic_mx_quant.md">torch_npu.npu_grouped_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_dynamic_mx_quant_desc"><a name="p_npu_grouped_dynamic_mx_quant_desc"></a><a name="p_npu_grouped_dynamic_mx_quant_desc"></a><span>根据传入的分组索引的起始值（group_index）对各个group以基本块（blocksize）为粒度进行目标数据类型为float8/float4的动态MX量化，并输出量化尺度mxscale（`torch_npu.float8_e8m0fnu`）。</span></p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul_add.md">torch_npu.npu_grouped_matmul_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul_add是一种对多个矩阵乘法（matmul）操作进行分组计算并与累加矩阵相加的高效方法，支持K轴分组。</span></p>
</td>
</tr>
<tr id="row_npu_grouped_matmul_add"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_grouped_matmul_add_.md">torch_npu.npu_grouped_matmul_add_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatmul操作接InplaceAdd操作的融合场景。本算子（GroupedMatmulAdd）在非量化场景中将上述算子融合起来，以提高网络性能。</p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul.md">torch_npu.npu_grouped_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。</span></p>
</td>
</tr>
<tr id="npu_grouped_matmul_finalize_routing"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_matmul_finalize_routing"><a name="p_npu_grouped_matmul_finalize_routing"></a><a name="p_npu_grouped_matmul_finalize_routing"></a><a href="torch_npu-npu_grouped_matmul_finalize_routing.md">torch_npu.npu_grouped_matmul_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_matmul_finalize_routing_desc"><a name="p_npu_grouped_matmul_finalize_routing_desc"></a><a name="p_npu_grouped_matmul_finalize_routing_desc"></a><span>GroupedMatMul与MoeFinalizeRouting的融合算子。</span></p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul_swiglu_quant.md">torch_npu.npu_grouped_matmul_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul_swiglu_quant是一种融合分组矩阵乘法（GroupedMatmul）、反量化（dequant）、SwiGLu混合激活函数、量化（quant）的计算方法，用于模型性能优化。</span></p>
</td>
</tr>
<tr id="row1231245645712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p548217963045654"><a name="p548217963045654"></a><a name="p548217963045654"></a><a href="torch_npu-npu_grouped_matmul_swiglu_quant_v2.md">torch_npu.npu_grouped_matmul_swiglu_quant_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5420573896217596"><a name="p5420573896217596"></a><a name="p5420573896217596"></a><span>npu_grouped_matmul_swiglu_quant_v2是一种融合分组矩阵乘法（GroupedMatmul）、反量化（dequant）、SwiGLu混合激活函数、量化（quant）的计算方法，用于模型性能优化。</span></p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_incre_flash_attention.md">torch_npu.npu_incre_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>增量FA实现。</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_kv_quant_sparse_flash_attention.md">torch_npu.npu_kv_quant_sparse_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Sparse Flash Attention伪量化实现。</p>
</td>
</tr>
<tr id="row_npu_kronecker_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kronecker_quant.md">torch_npu.npu_kronecker_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对输入张量依次进行两次矩阵乘法，并对矩阵乘结果量化至`torch_npu.int4`类型或`torch_npu.float4_e2m1fn_x2`类型。</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_lightning_indexer.md">torch_npu.npu_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>获取每一个token对应的Top-k个位置（全量化）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_matmul_all_to_all.md">torch_npu.npu_matmul_all_to_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>完成Matmul计算、Permute（保证通信后地址连续）和AlltoAll通信的融合，先计算后通信。</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog.md">torch_npu.npu_mla_prolog</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作</span>。</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog_v2.md">torch_npu.npu_mla_prolog_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作</span>。</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog_v3.md">torch_npu.npu_mla_prolog_v3</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作。v3版本相比v2新增query_norm和dequant_scale_q_norm输出、pertoken-pergroup量化模式、尺度矫正因子、多种cache_mode格式支持及多个量化配置参数，并调整cache_index为可选参数。</span>。</p>
</td>
</tr>
<tr id="row125419366217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18998182119294"><a name="p18998182119294"></a><a name="p18998182119294"></a><a href="torch_npu-npu_mm_all_reduce_base.md">torch_npu.npu_mm_all_reduce_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001721582972_p2122194522714"><a name="zh-cn_topic_0000001721582972_p2122194522714"></a><a name="zh-cn_topic_0000001721582972_p2122194522714"></a>TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。</p>
</td>
</tr>
<tr id="row102221513433"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p273718419525"><a name="p273718419525"></a><a name="p273718419525"></a><a href="torch_npu-npu_mm_reduce_scatter_base.md">torch_npu.npu_mm_reduce_scatter_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1385231745515"><a name="p1385231745515"></a><a name="p1385231745515"></a>TP切分场景下，实现matmul和reduce_scatter的融合，融合算子内部实现计算和通信流水并行。</p>
</td>
</tr>
<tr id="row192221013132"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1474331220582"><a name="p1474331220582"></a><a name="p1474331220582"></a><a href="torch_npu-npu_moe_compute_expert_tokens.md">torch_npu.npu_moe_compute_expert_tokens</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12743131265818"><a name="p12743131265818"></a><a name="p12743131265818"></a>MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。</p>
</td>
</tr>
<tr id="row622320135311"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2743141295810"><a name="p2743141295810"></a><a name="p2743141295810"></a><a href="torch_npu-npu_moe_finalize_routing.md">torch_npu.npu_moe_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1974341219588"><a name="p1974341219588"></a><a name="p1974341219588"></a>MoE计算中，最后处理合并MoE FFN的输出结果。</p>
</td>
</tr>
<tr id="row11308231313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p57431312165820"><a name="p57431312165820"></a><a name="p57431312165820"></a><a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14743151275819"><a name="p14743151275819"></a><a name="p14743151275819"></a>在MoE（混合专家模型，Mixture of Experts）架构的门控机制中，用于专家路由计算。对输入x做Softmax计算，再做topk操作，以挑选出权重最高的前K个专家。</p>
</td>
</tr>
<tr id="row133111231730"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p147431112105811"><a name="p147431112105811"></a><a name="p147431112105811"></a><a href="torch_npu-npu_moe_init_routing.md">torch_npu.npu_moe_init_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1174351212586"><a name="p1174351212586"></a><a name="p1174351212586"></a>MoE的routing计算，根据<a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>的计算结果做routing处理。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute.md">torch_npu.npu_moe_token_permute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE计算中，根据索引indices将输入tokens扩展，并按照专家索引排序，返回排序后的token及其与原始token的映射关系。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute.md">torch_npu.npu_moe_token_unpermute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE计算中，torch_npu.npu_moe_token_permute的逆操作，根据sorted_indices将扩展排序后的token特征映射回原始token位置并累加，支持probs加权。</p>
</td>
</tr>
<tr id="row1531023733"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7819527225"><a name="p7819527225"></a><a name="p7819527225"></a><a href="torch_npu-npu_prefetch.md">torch_npu.npu_prefetch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11819162142215"><a name="p11819162142215"></a><a name="p11819162142215"></a>提供网络weight预取功能，用于在计算执行前将指定的权重数据预先加载到L2 Cache中，减少算子访问这些权重时的访存等待时间。例如，在MatMul等算子之前进行预取，算子执行时可直接从低时延的L2 Cache中读取权重，进而提升算子数据访问与计算效率。实际性能收益取决于用户采用的并行方式和配置。</p>
</td>
</tr>
<tr id="row13116234313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35366207112"><a name="p35366207112"></a><a name="p35366207112"></a><a href="torch_npu-npu_prompt_flash_attention.md">torch_npu.npu_prompt_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20536122017115"><a name="p20536122017115"></a><a name="p20536122017115"></a>全量FA实现。</p>
</td>
</tr>
<tr id="row950000000031"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000032"><a name="p950000000032"></a><a name="p950000000032"></a><a href="torch_npu-npu_quant_gmm_alltoallv.md">torch_npu.npu_quant_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000033"><a name="p950000000033"></a><a name="p950000000033"></a>实现路由专家GroupedMatmul和AlltoAllv的融合，先计算后通信，同时与共享专家MatMul计算并行融合。支持T-T量化模式和mx量化模式。</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_quant_lightning_indexer.md">torch_npu.npu_quant_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>QuantLightningIndexer是推理场景下SparseFlashAttention（SFA）的前处理计算模块，用于选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。</p>
</td>
</tr>
<tr id="row9719124019218"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p14326218132919"><a name="p14326218132919"></a><a name="p14326218132919"></a><a href="torch_npu-npu_quant_matmul.md">torch_npu.npu_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001814195101_p156512056161014"><a name="zh-cn_topic_0000001814195101_p156512056161014"></a><a name="zh-cn_topic_0000001814195101_p156512056161014"></a>完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。</p>
</td>
</tr>
<tr id="row950000000041"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000042"><a name="p950000000042"></a><a name="p950000000042"></a><a href="torch_npu-npu_quant_matmul_all_to_all.md">torch_npu.npu_quant_matmul_all_to_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000043"><a name="p950000000043"></a><a name="p950000000043"></a>完成量化的Matmul计算、Permute（保证通信后地址连续）和AlltoAll通信的融合，先通信后计算。支持K-C量化模式和mx量化模式。</p>
</td>
</tr>
<tr id="row9719124019219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p104326218132920"><a name="p104326218132920"></a><a name="p104326218132920"></a><a href="torch_npu-npu_quant_matmul_gelu.md">torch_npu.npu_quant_matmul_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2026022511100607"><a name="p2026022511100607"></a><a name="p2026022511100607"></a>完成量化的矩阵乘和GELU激活函数的融合计算，支持A8W8和A4W4量化。</p>
</td>
</tr>
<tr id="row202508121056216"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p202508121056216"><a name="p202508121056216"></a><a name="p202508121056216"></a><a href="torch_npu-npu_quant_matmul_reduce_sum.md">torch_npu.npu_quant_matmul_reduce_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2025081210578767"><a name="p2025081210578767"></a><a name="p2025081210578767"></a>完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。</p>
</td>
</tr>
<tr id="row202508121056216"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p202508121056216"><a name="p202508121056216"></a><a name="p202508121056216"></a><a href="torch_npu-npu_quant_mm_reduce_scatter.md">torch_npu.npu_quant_mm_reduce_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2025081210578767"><a name="p2025081210578767"></a><a name="p2025081210578767"></a>融合MatMul与ReduceScatter集合通信计算，对矩阵乘结果按卡数做归约散射通信，支持低精度量化和反量化。</p>
</td>
</tr>
<tr id="row9368201625615"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p432973317361"><a name="p432973317361"></a><a name="p432973317361"></a><a href="torch_npu-npu_quant_scatter.md">torch_npu.npu_quant_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p133291433113614"><a name="p133291433113614"></a><a name="p133291433113614"></a>先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices更新self中的值，并将结果保存到输出tensor，self本身的数据不变。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p20338144125814"></a><a name="p20338144125814"></a><a href="torch_npu-npu_recurrent_gated_delta_rule.md">torch_npu.npu_recurrent_gated_delta_rule</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p33382046580"></a><a name="p33382046580"></a>完成变步长的Recurrent Gated Delta Rule计算。</p>
</td>
</tr>
<tr id="row159729564415"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p109331430113612"><a name="p109331430113612"></a><a name="p109331430113612"></a><a href="torch_npu-npu_quant_scatter_.md">torch_npu.npu_quant_scatter_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p693315307360"><a name="p693315307360"></a><a name="p693315307360"></a>先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices更新self中的值，self中的数据被改变。</p>
</td>
</tr>
<tr id="row1432501813299"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p20338144125815"><a name="p20338144125815"></a><a name="p20338144125815"></a><a href="torch_npu-npu_quantize.md">torch_npu.npu_quantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p33382046581"><a name="p33382046581"></a><a name="p33382046581"></a>对输入的张量进行量化处理。</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_nd_update.md">torch_npu.npu_scatter_nd_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>将updates中的值按指定的索引indices更新input中的值，并将结果保存到输出tensor，input本身的数据不变。</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_sparse_flash_attention.md">torch_npu.npu_sparse_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Sparse Flash Attention实现。</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>该接口实现了npu_lightning_indexer的反向功能，并融合了Loss的计算。</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_top_k_top_p.md">torch_npu.npu_top_k_top_p</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>对原始输入<code>logits</code>进行<code>top-k</code>和<code>top-p</code>采样过滤。</p>
</td>
</tr>
<tr id="row16875125182712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_top_k_top_p_sample.md">torch_npu.npu_top_k_top_p_sample</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>根据输入词频<code>logits</code>、<code>top_k</code>/<code>top_p</code>采样参数、随机采样权重分布<code>q</code>，进行topK-topP-Sample采样计算，输出每个batch的最大词频<code>logits_select_idx</code>，以及topK-topP采样后的词频分布<code>logits_top_kp_select</code>。</p>
</td>
</tr>
<tr id="row16875125182713"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_pa_cache.md">torch_npu.npu_scatter_pa_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>更新KCache中指定位置的<code>key</code>。</p>
</td>
</tr>
<tr id="row16875125182713"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_pa_kv_cache.md">torch_npu.npu_scatter_pa_kv_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>更新KVCache中指定位置的<code>key</code>和<code>value</code>。</p>
</td>
</tr>
<tr id="row9470191314519"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83781723143618"><a name="p83781723143618"></a><a name="p83781723143618"></a><a href="torch_npu-npu_scatter_nd_update_.md">torch_npu.npu_scatter_nd_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001816704932_p12276121818501"><a name="zh-cn_topic_0000001816704932_p12276121818501"></a><a name="zh-cn_topic_0000001816704932_p12276121818501"></a>将updates中的值按指定的索引indices更新input中的值，并将结果保存到输出tensor，input中的数据被改变。</p>
</td>
</tr>
<tr id="row1243753818365"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1332519186295"><a name="p1332519186295"></a><a name="p1332519186295"></a><a href="torch_npu-npu_trans_quant_param.md">torch_npu.npu_trans_quant_param</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10325181832913"><a name="p10325181832913"></a><a name="p10325181832913"></a>完成量化计算参数scale数据类型的转换。</p>
</td>
</tr>
<tr id="row1432853315360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p34567614299"><a name="p34567614299"></a><a name="p34567614299"></a><a href="torch_npu-npu_weight_quant_batchmatmul.md">torch_npu.npu_weight_quant_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p154556613293"><a name="p154556613293"></a><a name="p154556613293"></a>该接口用于实现矩阵乘计算中的weight输入和输出的量化操作，支持pertensor，perchannel，pergroup多场景量化。</p>
</td>
</tr>
<tr id="row250825154001"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p250825154002"><a name="p250825154002"></a><a name="p250825154002"></a><a href="torch_npu-npu_weight_quant_preprocess.md">torch_npu.npu_weight_quant_preprocess</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p250825154003"><a name="p250825154003"></a><a name="p250825154003"></a>该接口针对Matmul类算子的伪量化参数进行预处理。</p>
</td>
</tr>
<tr id="row10933133063618"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5743101211585"><a name="p5743101211585"></a><a name="p5743101211585"></a><a href="torch_npu-scatter_update.md">torch_npu.scatter_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p874318124582"><a name="p874318124582"></a><a name="p874318124582"></a>将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。</p>
</td>
</tr>
<tr id="row57012280360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1274381212587"><a name="p1274381212587"></a><a name="p1274381212587"></a><a href="torch_npu-scatter_update_.md">torch_npu.scatter_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p3743712125811"><a name="p3743712125811"></a><a name="p3743712125811"></a>将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据被改变。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-empty_with_swapped_memory.md">torch_npu.empty_with_swapped_memory</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>申请一个device信息为NPU且实际内存在host侧的特殊tensor。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-erase_stream.md">torch_npu.erase_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>tensor通过record_stream在内存池上添加标记后，该标记已被stream使用，可以通过该接口移除该标记。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gather_sparse_index.md">torch_npu.npu_gather_sparse_index</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>从输入tensor的指定维度，按照<code>index</code>中的下标序号提取元素，保存到输出tensor中。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine.md">torch_npu.npu_moe_distribute_combine</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>先进行reduce_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch.md">torch_npu.npu_moe_distribute_dispatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对Token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_v2.md">torch_npu.npu_moe_distribute_combine_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>先进行reduce_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。v2版本新增特殊专家场景（零专家、拷贝专家、常量专家）支持，并配合dispatch_v2的assist_info_for_combine实现更高效的全卡同步。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch_v2.md">torch_npu.npu_moe_distribute_dispatch_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对Token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。v2版本将返回值expand_idx替换为assist_info_for_combine以包含更详细的token信息，并新增comm_alg参数用于代替HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE环境变量。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_gating_top_k.md">torch_npu.npu_moe_gating_top_k</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_init_routing_v2.md">torch_npu.npu_moe_init_routing_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE（Mixture of Experts）的routing计算，根据<a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>的计算结果做routing处理，支持不量化和动态量化模式。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_swiglu_quant.md">torch_npu.npu_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>在swiglu激活函数后添加quant操作，实现输入x的SwiGluQuant计算，支持`torch.int8`或`torch_npu.int4`量化输出，支持MoE场景和非MoE场景（group_index为空），支持分组量化，支持动态/静态量化。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dequant_swiglu_quant.md">torch_npu.npu_dequant_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对张量x做dequant反量化+swiglu+quant量化操作，同时支持分组。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_swiglu_mx_quant.md">torch_npu.npu_swiglu_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>在Swiglu激活函数后添加DynamicMxQuant操作，实现输入x的SwigluMxQuant计算，支持MoE场景和非MoE场景（group_index为空），支持多种swiglu计算模式。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache.md">torch_npu.npu_kv_rmsnorm_rope_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache_v2.md">torch_npu.npu_kv_rmsnorm_rope_cache_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_interleave_rope.md">torch_npu.npu_interleave_rope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>针对单输入x进行旋转位置编码。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_mrope.md">torch_npu.npu_mrope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>推理场景下将预计算的sin、cos位置编码cache作用于query与key，实现旋转位置编码（RoPE），并支持多模态MRoPE。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_re_routing.md">torch_npu.npu_moe_re_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家序重新排列。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-matmul_checksum.md">torch_npu.matmul_checksum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>提供基于原生torch.matmul和Tensor.matmul接口的aicore错误硬件故障接口，内部执行矩阵计算结果校验过程，并对校验误差和实时计算的校验门限进行对比，判断校验误差是否超过门限，若超过则认为发生了aicore错误。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_gmm.md">torch_npu.npu_alltoallv_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gmm_alltoallv.md">torch_npu.npu_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，完成路由专家GroupedMatMul、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_quant_gmm.md">torch_npu.npu_alltoallv_quant_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>实现路由专家AlltoAllv和GroupedMatMul的量化融合，先通信后计算。同时与共享专家MatMul计算并行融合。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_add_rms_norm.md">torch_npu.npu_moe_distribute_combine_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>完成moe_distribute_combine+add+rms_norm融合。需与torch_npu.npu_moe_distribute_dispatch配套使用，相当于按npu_moe_distribute_dispatch算子收集数据的路径原路返回后对数据进行add_rms_norm操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_transpose_batchmatmul.md">torch_npu.npu_transpose_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>完成张量input与张量weight的矩阵乘计算。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_update_expert.md">torch_npu.npu_moe_update_expert</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>本API支持负载均衡和专家剪枝功能。经过映射后的专家表和mask可传入MoE层进行数据分发和处理。</p>
</td>
</tr>
<tr id="row285193313382"><td class="cellrowborder" valign="top" width="38.22%" headers="mcps1.2.3.1.1 "><p id="p2851433193819"><a name="p2851433193819"></a><a name="p2851433193819"></a><a href="torch_npu-npu_dynamic_block_quant.md">torch_npu.npu_dynamic_block_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.78%" headers="mcps1.2.3.1.2 "><p id="p785143323817"><a name="p785143323817"></a><a name="p785143323817"></a>对输入张量，通过给定的row_block_size和col_block_size将输入划分成多个数据块，以数据块为基本粒度进行量化。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_device_limit.md">torch_npu.set_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>设置一个进程上指定device执行算子时所使用的cube和vector核数。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_device_limit.md">torch.npu.get_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>获取指定Device上的资源限制。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_stream_limit.md">torch.npu.set_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>设置指定流上算子在执行时所使用的cube和vector核数。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-reset_stream_limit.md">torch.npu.reset_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>恢复指定流上算子在执行时所使用的cube和vector核数为默认配置。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_stream_limit.md">torch.npu.get_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>获取指定流上算子执行的cube和vector核数。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_sim_exponential_.md">torch_npu.npu_sim_exponential_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>原地获取输入张量中元素的指数分布采样。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_softmax_lse.md">torch_npu.npu_dense_lightning_indexer_softmax_lse</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>是npu_dense_lightning_indexer_grad_kl_loss接口的前置接口，通过把Lightning Indexer组件的Softmax求最大值和求和运算提前来降低接口的显存占用。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_grad_kl_loss.md">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>该接口实现了Lightning Indexer组件warmup阶段训练的反向梯度计算，并融合了Loss的计算。</p>
</td>
</tr>
<tr id="batch_norm_reduce"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_batch_norm_reduce"><a name="p_batch_norm_reduce"></a><a name="p_batch_norm_reduce"></a><a href="torch_npu-batch_norm_reduce.md">torch_npu.batch_norm_reduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_batch_norm_reduce_desc"><a name="p_batch_norm_reduce_desc"></a><a name="p_batch_norm_reduce_desc"></a><span>该接口用于按BatchNorm通道维对输入Tensor进行归约，计算通道维以外维度上的元素和 `sum` 与平方和 `square_sum` 。</span></p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_apply_adam_w.md">torch_npu.npu_apply_adam_w</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>实现AdamW优化器的参数更新计算，在Adam的基础上将权重衰减与梯度更新解耦，常用于大模型训练场景下的参数优化。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm.md">torch_npu.npu_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>将Add计算与RMSNorm归一化融合，常用于大模型中将残差连接后的张量进行归一化处理。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_v2.md">torch_npu.npu_add_rms_norm_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>AddRmsNorm算子是Add算子和RmsNorm算子的融合。其中RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，去掉了减去均值的部分。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_cast.md">torch_npu.npu_add_rms_norm_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_dynamic_quant.md">torch_npu.npu_add_rms_norm_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuant算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_dynamic_mx_quant.md">torch_npu.npu_add_rms_norm_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicMxQuant算子则是在尾轴上按blocksize分组进行动态MX量化的算子。AddRmsNormDynamicMxQuant算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的DynamicMxQuant算子融合起来，减少搬入搬出操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_fused_floyd_attention.md">torch_npu.npu_fused_floyd_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>训练场景下，npu_fused_floyd_attention相较于传统FA(npu_fusion_attention)主要是计算QK/PV注意力时，会额外将维度(seq)作为batch轴处理，从而将注意力计算转换为批量矩阵乘法(batchMatmul)。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_masked_causal_conv1d.md">torch_npu.npu_masked_causal_conv1d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对hidden层的token之间进行带mask的因果一维分组卷积操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_clipped_swiglu.md">torch_npu.npu_clipped_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>带截断的Swish门控线性单元激活函数，实现x的变体SwiGlu计算。</p>
</td>
</tr>
<tr id="row_rotate_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_rotate_quant"><a name="p_rotate_quant"></a><a name="p_rotate_quant"></a><a href="torch_npu-npu_rotate_quant.md">torch_npu.npu_rotate_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_rotate_quant_desc"><a name="p_rotate_quant_desc"></a><a name="p_rotate_quant_desc"></a><span>npu_rotate_quant是一种融合旋转（Rotate）和量化（Quant）的计算方法，用于模型性能优化。</span></p>
</td>
</tr>
<tr id="save_npugraph_tensor"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_save_npugraph_tensor"><a name="p_save_npugraph_tensor"></a><a name="p_save_npugraph_tensor"></a><a href="torch_npu-save_npugraph_tensor.md">torch_npu.save_npugraph_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_save_npugraph_tensor_desc"><a name="p_save_npugraph_tensor_desc"></a><a name="p_save_npugraph_tensor_desc"></a><span>当前接口提供了类似原生PyTorch的torch.save接口特性且不影响aclgraph捕获、重放的tensor dump能力，允许将aclgraph中间节点的tensor数据、数据类型、shape信息保存到指定二进制文件中以便用户观察aclgraph的执行过程的tensor数据，以快速定位问题。</span></p>
</td>
</tr>
<tr id="print_npugraph_tensor"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_print_npugraph_tensor"><a name="p_print_npugraph_tensor"></a><a name="p_print_npugraph_tensor"></a><a href="torch_npu-print_npugraph_tensor.md">torch_npu.print_npugraph_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_print_npugraph_tensor_desc"><a name="p_print_npugraph_tensor_desc"></a><a name="p_print_npugraph_tensor_desc"></a><span>print_npugraph_tensor接口提供了类似原生Python的print接口特性且不影响aclgraph捕获、重放的tensor打印能力，允许将aclgraph中间节点的tensor数据、数据类型、shape信息直接打印出来，以便用户观察aclgraph的执行过程中的tensor数据，以快速定位问题。</span></p>
</td>
</tr>
<tr id="npu_apply_rotary_pos_emb">
<td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_apply_rotary_pos_emb"><a name="p_npu_apply_rotary_pos_emb"></a><a name="p_npu_apply_rotary_pos_emb"></a><a href="torch_npu-apply_rotary_pos_emb.md">torch_npu.npu_apply_rotary_pos_emb</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_apply_rotary_pos_emb_desc"><a name="p_npu_apply_rotary_pos_emb_desc"></a><a name="p_npu_apply_rotary_pos_emb_desc"></a><span>为提升推理网络性能，将query和key两路算子融合为单路，在旋转位置编码计算中直接对结果执行原地更新。</span></p>
</td>
</tr>
<tr id="npu_add_quant_gmm_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_gmm_"><a name="p_npu_add_quant_gmm_"></a><a name="p_npu_add_quant_gmm_"></a><a href="torch_npu-npu_add_quant_gmm_.md">torch_npu.npu_add_quant_gmm_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_gmm__desc"><a name="p_npu_add_quant_gmm__desc"></a><a name="p_npu_add_quant_gmm__desc"></a><span>在micro-batch训练场景下，将量化GroupedMatmul与InplaceAdd操作融合，以提高网络性能。</span></p>
</td>
</tr>
<tr id="npu_add_quant_matmul_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_matmul_"><a name="p_npu_add_quant_matmul_"></a><a name="p_npu_add_quant_matmul_"></a><a href="torch_npu-npu_add_quant_matmul_.md">torch_npu.npu_add_quant_matmul_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_matmul__desc"><a name="p_npu_add_quant_matmul__desc"></a><a name="p_npu_add_quant_matmul__desc"></a><span>训练场景下micro-batch的梯度累积会存在大量QuantBatchMatmul操作接InplaceAdd操作，本接口将操作融合起来，以提高网络性能。</span></p>
</td>
</tr>
<tr id="npu_dynamic_dual_level_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_dual_level_mx_quant"><a name="p_npu_dynamic_dual_level_mx_quant"></a><a name="p_npu_dynamic_dual_level_mx_quant"></a><a href="torch_npu-npu_dynamic_dual_level_mx_quant.md">torch_npu.npu_dynamic_dual_level_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_dual_level_mx_quant_desc"><a name="p_npu_dynamic_dual_level_mx_quant_desc"></a><a name="p_npu_dynamic_dual_level_mx_quant_desc"></a><span>实现目的数据类型为FLOAT4类的MX量化。只对输入张量的尾轴量化，其他轴均按合轴处理。</span></p>
</td>
</tr>
<tr id="npu_dynamic_mx_quant_with_dual_axis"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant_with_dual_axis"><a name="p_npu_dynamic_mx_quant_with_dual_axis"></a><a name="p_npu_dynamic_mx_quant_with_dual_axis"></a><a href="torch_npu-npu_dynamic_mx_quant_with_dual_axis.md">torch_npu.npu_dynamic_mx_quant_with_dual_axis</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_with_dual_axis_desc"><a name="p_npu_dynamic_mx_quant_with_dual_axis_desc"></a><a name="p_npu_dynamic_mx_quant_with_dual_axis_desc"></a><span>在输入张量的-1轴和-2轴上同时进行目标数据类型为float4类、float8类的MX量化，输出两轴各自的量化结果y1、y2与量化尺度mxscale1、mxscale2。</span></p>
</td>
</tr>
<tr id="npu_fused_causal_conv1d"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_causal_conv1d"><a name="p_npu_fused_causal_conv1d"></a><a name="p_npu_fused_causal_conv1d"></a><a href="torch_npu-npu_fused_causal_conv1d.md">torch_npu.npu_fused_causal_conv1d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_causal_conv1d_desc"><a name="p_npu_fused_causal_conv1d_desc"></a><a name="p_npu_fused_causal_conv1d_desc"></a><span>对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持 APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。</span></p>
</td>
</tr>
<tr id="npu_grouped_dynamic_block_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_dynamic_block_quant"><a name="p_npu_grouped_dynamic_block_quant"></a><a name="p_npu_grouped_dynamic_block_quant"></a><a href="torch_npu-npu_grouped_dynamic_block_quant.md">torch_npu.npu_grouped_dynamic_block_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_dynamic_block_quant_desc"><a name="p_npu_grouped_dynamic_block_quant_desc"></a><a name="p_npu_grouped_dynamic_block_quant_desc"></a><span>根据传入的分组索引的起始值（group_list）对各个group以基本块的粒度进行量化，量化为（FP8/HiFP8），并输出量化参数scale（`torch.float32`）。</span></p></td>
</tr>
<tr id="npu_mhc_post"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_mhc_post"><a name="p_npu_mhc_post"></a><a name="p_npu_mhc_post"></a><a href="torch_npu-npu_mhc_post.md">torch_npu.npu_mhc_post</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_mhc_post_desc"><a name="p_npu_mhc_post_desc"></a><a name="p_npu_mhc_post_desc"></a><span>对mHC(Manifold-Constrained Hyper-Connections)架构中第l层输出h<sub>l</sub><sup>out</sup>进行Post Mapping，对第l层的输入x<sub>l</sub>进行Res Mapping，然后对二者进行残差连接，得到第(l+1)层的输入x<sub>l+1</sub>。当`h_res`传入`None`时，跳过Res Mapping，直接进行残差相加。</span></p></td>
</tr>
<tr id="npu_mhc_sinkhorn"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_mhc_sinkhorn"><a name="p_npu_mhc_sinkhorn"></a><a name="p_npu_mhc_sinkhorn"></a><a href="torch_npu-npu_mhc_sinkhorn.md">torch_npu.npu_mhc_sinkhorn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_mhc_sinkhorn_desc"><a name="p_npu_mhc_sinkhorn_desc"></a><a name="p_npu_mhc_sinkhorn_desc"></a><span>对mHC架构中的H′<sub>res</sub>矩阵（即网络中mHC层的输入数据）执行Sinkhorn迭代归一化变换，最终得到双随机矩阵H<sub>res</sub>，支持输出迭代过程中的中间归一化结果（norm_out）和求和结果（sum_out），用于反向梯度计算。</span></p></td>
</tr>
<tr id="npu_transpose_quant_batchmatmul"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_transpose_quant_batchmatmul"><a name="p_npu_transpose_quant_batchmatmul"></a><a name="p_npu_transpose_quant_batchmatmul"></a><a href="torch_npu-npu_transpose_quant_batchmatmul.md">torch_npu.npu_transpose_quant_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_transpose_quant_batchmatmul_desc"><a name="p_npu_transpose_quant_batchmatmul_desc"></a><a name="p_npu_transpose_quant_batchmatmul_desc"></a><span>完成张量x1与张量x2量化的矩阵乘计算。仅支持三维的tensor传入。tensor支持转置，转置序列根据传入的序列进行变更。perm_x1代表张量x1的转置序列，perm_x2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。</span></p></td>
</tr>
<tr id="npu_fused_linear_online_max_sum"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_linear_online_max_sum"><a name="p_npu_fused_linear_online_max_sum"></a><a name="p_npu_fused_linear_online_max_sum"></a><a href="torch_npu-npu_fused_linear_online_max_sum.md">torch_npu.npu_fused_linear_online_max_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_linear_online_max_sum_desc"><a name="p_npu_fused_linear_online_max_sum_desc"></a><a name="p_npu_fused_linear_online_max_sum_desc"></a><span>词汇表并行场景下融合矩阵乘与交叉熵前处理算子。功能等价于Megatron的MatMul与fused_vocab_parallel_cross_entropy的实现，支持vocabulary_size维度切分融合MatMul与CELoss。本接口为通信前部分，需与npu_fused_cross_entropy_loss_with_max_sum配合使用。</span></p></td>
</tr>
<tr id="npu_fused_cross_entropy_loss_with_max_sum"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_cross_entropy_loss_with_max_sum"><a name="p_npu_fused_cross_entropy_loss_with_max_sum"></a><a name="p_npu_fused_cross_entropy_loss_with_max_sum"></a><a href="torch_npu-npu_fused_cross_entropy_loss_with_max_sum.md">torch_npu.npu_fused_cross_entropy_loss_with_max_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_cross_entropy_loss_with_max_sum_desc"><a name="p_npu_fused_cross_entropy_loss_with_max_sum_desc"></a><a name="p_npu_fused_cross_entropy_loss_with_max_sum_desc"></a><span>词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题。本接口为计算Loss与Softmax结果的部分，需配合npu_fused_linear_online_max_sum使用。在多卡场景下，需先对logits_max和sum_exp_logits执行全局通信（all-reduce），再调用本接口。</span></p></td>
</tr>
<tr id="npu_fused_linear_cross_entropy_loss_with_max_sum_backward"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"></a><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"></a><a href="torch_npu-npu_fused_linear_cross_entropy_loss_with_max_sum_backward.md">torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"></a><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"></a><span>词汇表并行场景下交叉熵损失计算的梯度算子。用于计算叶子节点input和weight的梯度。需要获得npu_fused_linear_online_max_sum和npu_fused_cross_entropy_loss_with_max_sum的相关输出作为本接口输入。支持高性能模式（传入Softmax）和省显存模式（传入logits_max和sum_exp_logits）。</span></p></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dual_level_quant_matmul.md">torch_npu.npu_dual_level_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><span>QuantMatmul的二级量化功能，减少精度损失。</span></p></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute.md">torch_npu.npu_moe_token_permute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE的permute计算，根据索引indices将tokens广播并排序。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute_with_routing_map.md">torch_npu.npu_moe_token_permute_with_routing_map</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE的permute计算，将token和expert的标签作为routing_map传入，根据routing_map将tokens和可选probs广播后排序。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute.md">torch_npu.npu_moe_token_unpermute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE的unpermute计算，根据sorted_indices存储的下标获取permuted_tokens中的输入数据，与可选probs相乘后进行累加求和，输出还原后的计算结果。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute_with_routing_map.md">torch_npu.npu_moe_token_unpermute_with_routing_map</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对经过permute计算处理的permuted_tokens，根据sorted_indices和routing_map累加回原unpermuted_tokens，支持drop_and_pad填充模式。</p>
</td>
</tr>
<tr id="npu_add_quant_gmm_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_gmm_"><a name="p_npu_add_quant_gmm_"></a><a name="p_npu_add_quant_gmm_"></a><a href="torch_npu-npu_add_quant_gmm_.md">torch_npu.npu_add_quant_gmm_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_gmm__desc"><a name="p_npu_add_quant_gmm__desc"></a><a name="p_npu_add_quant_gmm__desc"></a><span>训练场景下micro-batch的梯度累计会存在大量GroupedMatMul操作接InplaceAdd操作，本接口将操作融合起来，实现分组矩阵乘计算和加法计算，仅支持mx量化、T-C量化和T-T量化场景。</span></p>
</td>
</tr>
<tr id="npu_apply_adam_w"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_apply_adam_w"><a name="p_npu_apply_adam_w"></a><a name="p_npu_apply_adam_w"></a><a href="torch_npu-npu_apply_adam_w.md">torch_npu.npu_apply_adam_w</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_apply_adam_w_desc"><a name="p_npu_apply_adam_w_desc"></a><a name="p_npu_apply_adam_w_desc"></a><span>实现AdamW优化器功能，将权重衰减与基于梯度的参数更新解耦，原地更新权重参数以及一阶、二阶矩估计。</span></p></td>
</tr>
<tr id="npu_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant"><a name="p_npu_dynamic_mx_quant"></a><a name="p_npu_dynamic_mx_quant"></a><a href="torch_npu-npu_dynamic_mx_quant.md">torch_npu.npu_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_desc"><a name="p_npu_dynamic_mx_quant_desc"></a><a name="p_npu_dynamic_mx_quant_desc"></a><span>目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的轴axis上，每block_size个数为一组，计算出这组数对应的量化尺度mxscale，再根据round_mode将组内元素转换到对应的dst_type。</span></p>
</td>
</tr>
</tbody>
</table>
