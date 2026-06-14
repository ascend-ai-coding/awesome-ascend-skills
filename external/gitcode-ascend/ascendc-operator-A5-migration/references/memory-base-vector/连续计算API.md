> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/基础API/接口分类说明/连续计算API.md`
# 连续计算API<a name="ZH-CN_TOPIC_0000002491670826"></a>

连续计算API，支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维tensor的连续计算问题。

```
Add(dst, src1, src2, n);
```

下图以矢量加法为例，展示了**连续计算API**的特点。

**图 1** **连续计算API**<a name="zh-cn_topic_0000001762058545_fig6847134062319"></a>  
![](../../../../figures/连续计算API.png "连续计算API")

