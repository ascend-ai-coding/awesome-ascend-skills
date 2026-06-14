> **原始文档路径**: `asc-devkit/docs/guide/编程指南/编程模型/AI-Core-SIMT编程/同步机制.md`
# 同步机制<a name="ZH-CN_TOPIC_0000002554691683"></a>

SIMT是一种单指令多线程的编程模式，其异步编程模型旨在通过多线程并发执行达到内存操作加速的目的。在这一编程模型中，线程作为执行计算或操作内存的最小抽象单位，其操作是相互独立的。然而，在某些应用场景中，需要支持线程间的同步，或防止不同线程对同一内存区域的读写操作引发的数据竞争。为此，Ascend C提供了相应的同步接口，这些接口允许开发者根据需求选择合适的同步机制，以确保异步操作的正确性和性能。

<a name="table1153875685516"></a>
<table><thead align="left"><tr id="row16538115615511"><th class="cellrowborder" valign="top" width="19.88%" id="mcps1.1.3.1.1"><p id="p125381156115512"><a name="p125381156115512"></a><a name="p125381156115512"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="80.12%" id="mcps1.1.3.1.2"><p id="p17538756175510"><a name="p17538756175510"></a><a name="p17538756175510"></a>功能说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row82556463018"><td class="cellrowborder" valign="top" width="19.88%" headers="mcps1.1.3.1.1 "><p id="p13255194620010"><a name="p13255194620010"></a><a name="p13255194620010"></a><a href="https://gitcode.com/munanhw/asc-devkit/blob/master/docs/api/context/asc_syncthreads.md">asc_syncthreads</a></p>
</td>
<td class="cellrowborder" valign="top" width="80.12%" headers="mcps1.1.3.1.2 "><p id="p32556461908"><a name="p32556461908"></a><a name="p32556461908"></a>等待当前线程块内所有线程代码都执行到该函数位置。</p>
</td>
</tr>
<tr id="row155387562553"><td class="cellrowborder" valign="top" width="19.88%" headers="mcps1.1.3.1.1 "><p id="p20538185685510"><a name="p20538185685510"></a><a name="p20538185685510"></a><a href="https://gitcode.com/munanhw/asc-devkit/blob/master/docs/api/context/asc_threadfence.md">asc_threadfence</a></p>
</td>
<td class="cellrowborder" valign="top" width="80.12%" headers="mcps1.1.3.1.2 "><p id="p135393567552"><a name="p135393567552"></a><a name="p135393567552"></a>用于保证不同线程对同一份全局、共享内存的访问过程中，写入操作的时序性。它不会阻塞线程，仅保证内存操作的可见性顺序。</p>
</td>
</tr>
<tr id="row1590833918293"><td class="cellrowborder" valign="top" width="19.88%" headers="mcps1.1.3.1.1 "><p id="p190843918297"><a name="p190843918297"></a><a name="p190843918297"></a>asc_threadfence_block</p>
</td>
<td class="cellrowborder" valign="top" width="80.12%" headers="mcps1.1.3.1.2 "><p id="p189121251102716"><a name="p189121251102716"></a><a name="p189121251102716"></a>用于协调同一线程块（Thread Block）内线程之间的内存操作顺序，<span>确保某一线程在asc_</span>threadfence_block()<span>之前</span><span>的所有内存操作（读写），对</span><span>同一线程块内的其他线程</span><span>是可见的</span>。</p>
</td>
</tr>
</tbody>
</table>

