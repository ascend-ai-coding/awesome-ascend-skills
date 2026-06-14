> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/基础API/接口分类说明/高维切分API.md`
# 高维切分API<a name="ZH-CN_TOPIC_0000002271261208"></a>

>[!NOTE]说明 
>-   本章节对**矢量计算基础API**中的**tensor高维切分计算**接口做解释说明。如果您不需要使用此类接口，可略过该章节。
>-   下文中的repeatTime、dataBlockStride、repeatStride、mask为通用描述，其命名不一定与具体指令中的参数命名完全对应。
>    比如，单次迭代内不同datablock间地址步长dataBlockStride参数，在单目API中，对应为dstBlkStride、srcBlkStride参数；在双目API中，对应为dstBlkStride、src0BlkStride、src1BlkStride参数。
>    您可以在具体接口的参数说明中，找到参数含义的描述。

使用**tensor高维切分计算**API可充分发挥硬件优势，支持开发者控制指令的**迭代执行**和操作数的**地址间隔**，功能更加灵活。

矢量计算通过Vector计算单元完成，矢量计算的源操作数和目的操作数均通过Unified Buffer（UB）来进行存储。Vector计算单元每个迭代会从UB中取出**8**个**datablock**（每个datablock数据块内部地址连续，长度32Byte），进行计算，并写入对应的8个datablock中。下图为单次迭代内的8个datablock进行Exp计算的示意图。

**图 1**  单次迭代内的8个datablock进行Exp计算示意图<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig143499419171"></a>  
![](../../../../figures/单次迭代内的8个datablock进行Exp计算示意图.png "单次迭代内的8个datablock进行Exp计算示意图")

-   矢量计算API支持开发者通过**repeatTime**来配置迭代次数，从而控制指令的多次迭代执行。假设repeatTime设置为2，矢量计算单元会进行2个迭代的计算，可计算出2 \* 8（每个迭代8个datablock） \* 32Byte（每个datablock32Byte） = 512Byte的结果。如果数据类型为half，则计算了256个元素。下图展示了2次迭代Exp计算的示意图。由于硬件限制，**repeatTime不能超过255。**

    **图 2**  2次迭代Exp计算<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig148871676180"></a>  
    ![](../../../../figures/2次迭代Exp计算.png "2次迭代Exp计算")

-   针对同一个迭代中的数据，可以通过**mask**参数进行掩码操作来控制实际参与计算的个数。下图为进行Abs计算时通过mask逐比特模式按位控制哪些元素参与计算的示意图，1表示参与计算，0表示不参与计算。

    **图 3**  通过**mask**参数进行掩码操作示意图（以float数据类型为例）<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig7630131541814"></a>  
    ![](../../../../figures/通过mask参数进行掩码操作示意图（以float数据类型为例）.png "通过mask参数进行掩码操作示意图（以float数据类型为例）")

-   矢量计算单元还支持带间隔的向量计算，通过**dataBlockStride**（单次迭代内不同datablock间地址步长）和**repeatStride**（相邻迭代间相同datablock的地址步长）来进行配置。
    -   **dataBlockStride**

        如果需要控制单次迭代内，数据处理的步长，可以通过设置同一迭代内不同datablock的地址步长dataBlockStride来实现。下图给出了单次迭代内非连续场景的示意图，示例中源操作数的dataBlockStride配置为2，表示单次迭代内不同datablock间地址步长（起始地址之间的间隔）为2个datablock。

        **图 4**  单次迭代内非连续场景的示意图<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig744519103222"></a>  
        ![](../../../../figures/单次迭代内非连续场景的示意图.png "单次迭代内非连续场景的示意图")

    -   **repeatStride**

        当repeatTime大于1，需要多次迭代完成矢量计算时，您可以根据不同的使用场景合理设置相邻迭代间相同datablock的地址步长repeatStride的值。

        下图给出了多次迭代间非连续场景的示意图，示例中源操作数和目的操作数的**repeatStride**均配置为9，表示相邻迭代间相同datablock起始地址之间的间隔为9个datablock。相同datablock是指datablock在迭代内的位置相同，比如下图中的src1和src9处于相邻迭代，在迭代内都是第一个datablock的位置，其间隔即为**repeatStride**的数值。

        **图 5**  多次迭代间非连续场景的示意图<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig7376324392"></a>  
        ![](../../../../figures/多次迭代间非连续场景的示意图.png "多次迭代间非连续场景的示意图")

下文中给出了dataBlockStride、repeatStride、mask的详细配置说明和示例。

## dataBlockStride<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416"></a>

dataBlockStride是指同一迭代内不同datablock的地址步长。

-   连续计算，dataBlockStride设置为1，对同一迭代内的8个datablock数据连续进行处理。
-   非连续计算，dataBlockStride值大于1（如取2），同一迭代内不同datablock之间在读取数据时出现一个datablock的间隔，如下图所示。

    **图 6**  dataBlockStride不同取值举例<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_fig6550124115203"></a>  
    ![](../../../../figures/dataBlockStride不同取值举例.png "dataBlockStride不同取值举例")

## repeatStride<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420"></a>

repeatStride是指相邻迭代间相同datablock的地址步长。

-   **连续计算场景**：假设定义一个Tensor供目的操作数和源操作数同时使用（即地址重叠），repeatStride取值为8。此时，矢量计算单元第一次迭代读取连续8个datablock，第二轮迭代读取下一个连续的8个datablock，通过多次迭代即可完成所有输入数据的计算。

    ![](../../../../figures/repeat_stride1.png)

-   **非连续计算场景**：repeatStride取值大于8（如取10）时，则相邻迭代间矢量计算单元读取的数据在地址上不连续，出现2个datablock的间隔。

    ![](../../../../figures/repeat_stride2.png)

-   **反复计算场景**：repeatStride取值为0时，矢量计算单元会对首个连续的8个datablock进行反复读取和计算。

    ![](../../../../figures/repeat_stride3.png)

-   **部分重复计算**：repeatStride取值大于0且小于8时，相邻迭代间部分数据会被矢量计算单元重复读取和计算，此种情形一般场景不涉及。

    ![](../../../../figures/repeat_stride4.png)

## mask参数<a name="zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182"></a>

mask用于控制每次迭代内参与计算的元素。可通过连续模式和逐bit模式两种方式进行设置。

-   连续模式：表示前面连续的多少个元素参与计算。数据类型为uint64\_t。取值范围和源操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同（当前数据类型单次迭代时能处理的元素个数最大值为：256 / sizeof\(数据类型\)）。当操作数的数据类型占bit位16位时（如half/uint16\_t），mask∈\[1, 128\]；当操作数为32位时（如float/int32\_t），mask∈\[1, 64\]。

    具体样例如下：

    ```
    // int16_t数据类型单次迭代能处理的元素个数最大值为256/sizeof(int16_t) = 128，mask = 64，mask∈[1, 128]，所以是合法输入
    // repeatTime = 1, 共128个元素，单次迭代能处理128个元素，故repeatTime = 1
    // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
    // dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
    uint64_t mask = 64;
    AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据(src0Local): [1 2 3 ... 64 ...128]
    输入数据(src1Local): [1 2 3 ... 64 ...128]
    输出数据(dstLocal): [2 4 6 ... 128 undefined...undefined]
    ```

    ```
    // int32_t数据类型单次迭代能处理的元素个数最大值为256/sizeof(int32_t) = 64，mask = 64，mask∈[1, 64]，所以是合法输入
    // repeatTime = 1, 共64个元素，单次迭代能处理64个元素，故repeatTime = 1
    // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
    // dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
    uint64_t mask = 64;
    AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据(src0Local): [1 2 3 ... 64]
    输入数据(src1Local): [1 2 3 ... 64]
    输出数据(dstLocal): [2 4 6 ... 128]
    ```

-   逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

    mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask\[0\]、mask\[1\]∈\[0, 2<sup>64</sup>-1\]并且不同时为0；当操作数为32位时，数组长度为1，mask\[0\]∈\(0, 2<sup>64</sup>-1\]；当操作数为64位时，数组长度为1，mask\[0\]∈\(0, 2<sup>32</sup>-1\]。

    具体样例如下:

    ```
    // 数据类型为int16_t
    uint64_t mask[2] = {6148914691236517205, 6148914691236517205};
    // repeatTime = 1, 共128个元素，单次迭代能处理128个元素，故repeatTime = 1。
    // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据。
    // dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入。
    AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据(src0Local): [1 2 3 ... 64 ...127 128]
    输入数据(src1Local): [1 2 3 ... 64 ...127 128]
    输出数据(dstLocal): [2 undefined 6 ... undefined ...254 undefined]
    ```

    mask过程如下：

    mask=\{6148914691236517205, 6148914691236517205\}（注：6148914691236517205表示64位二进制数0b010101....01，mask按照低位到高位的顺序排布）

    ![](../../../../figures/mask.png)

    ```
    // 数据类型为int32_t
    uint64_t mask[1] = {6148914691236517205};
    // repeatTime = 1, 共64个元素，单次迭代能处理64个元素，故repeatTime = 1。
    // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据。
    // dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入。
    AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据(src0Local): [1 2 3 ... 63 64]
    输入数据(src1Local): [1 2 3 ... 63 64]
    输出数据(dstLocal): [2 undefined 6 ... 126 undefined]
    ```

    mask过程如下：

    mask=\{6148914691236517205, 0\}（注：6148914691236517205表示64位二进制数0b010101....01）

    ![](../../../../figures/mask_32bit.png)

