# MC2 算子分类速查与可信源

> 生成需求文档前应查阅本索引，技术信息必须来自可信源，禁止编造。

## MC2 算子分类速查

> 参考仓库：[ops-transformer/mc2/](https://gitcode.com/cann/ops-transformer/tree/master/mc2)

| 通信原语 | 已有算子 | 说明 |
|---------|---------|------|
| AllReduce | `matmul_all_reduce`、`matmul_all_reduce_add_rms_norm`、`inplace_matmul_all_reduce_add_rms_norm`、`grouped_mat_mul_all_reduce` | Matmul + AllReduce 融合，部分叠加 Add/RmsNorm |
| AllToAll | `allto_all_matmul`、`matmul_allto_all`、`allto_allv_grouped_mat_mul`、`allto_allv_quant_grouped_mat_mul`、`grouped_mat_mul_allto_allv` | AllToAll + Matmul 融合，含量化/分组变体 |
| AllGather | `all_gather_matmul`、`all_gather_matmul_v2` | AllGather + Matmul 融合 |
| ReduceScatter | `matmul_reduce_scatter`、`matmul_reduce_scatter_v2`、`batch_mat_mul_reduce_scatter_allto_all` | ReduceScatter + Matmul 融合 |
| MOE/组合 | `mega_moe`、`attention_to_ffn`、`ffn_to_attention` | MOE 场景或跨层融合 |
| 通信辅助 | `distribute_barrier`、`distribute_barrier_extend`、`engram_fetch`、`engram_fetch_wait` | 分布式同步与数据搬运 |

## 可信源清单

| 来源 | URL / 路径 | 用途 |
|------|-----------|------|
| ops-transformer MC2 算子仓 | https://gitcode.com/cann/ops-transformer/tree/master/mc2 | 查阅已有算子的 README、aclnn 接口文档、约束 |
| 两段式接口说明 | https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md | aclnn 两段式调用机制 |
| 基本概念索引 | https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/basic_concept.md | 数据类型/格式/推导关系 |
| HCCL API (C) 文档 | https://hiascend.com/document/redirect/CannCommunityHcclCppApi | HCCL 接口定义、通信域管理 |
| SHMEM 文档 | https://shmem-doc.pages.dev/ | SHMEM/UDMA API、用法 |
| 本地 MC2 开发最佳实践 | `ops/ascendc-mc2-best-practice/` | 各底座开发约束与架构：SHMEM+Blaze、apace、MTE/MoE、HCCL+Matmul（910B） |
