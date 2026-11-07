# [算子名称] 设计文档

## 1. 算子接口与 Assumptions

### 1.1 用户确认项

| 确认项 | 内容 | 确认状态 |
| --- | --- | --- |
| 算子名称 `op_name` | [用户确认的算子名称] | confirmed |
| 算子数据类型 `dtype/dtypes` | [逐项列出输入、输出、累加/中间 dtype] | confirmed |

### 1.2 功能描述

[一句话描述算子功能，明确 local compute、communication、final result。]

### 1.3 Assumptions 与 Open Questions

| 类型 | 内容 | 是否阻塞代码生成 |
| --- | --- | --- |
| assumption | [可接受假设] | no |
| open question | [必须用户确认的问题] | yes/no |

## 2. Canonical DSL

DSL 是本设计文档的唯一结构化信息源。后续代码生成、走读和性能采集均按 DSL 字段执行。必须保留完整 fenced `yaml` block，供 `shmem-ops-code-gen` 直接消费。

```yaml
dsl_version: "0.3"

meta:
  op_name: "replace_me"
  op_kind: "transport"             # transport | collective | compute | fused_compute_comm
  build_mode: "independent_project"   # independent_project（默认）| in_tree_example（须 Phase 0 用户明确要求）
  op_root: "custom-ops/replace_me"   # custom-ops/<op_name>/（默认）| examples/<op_name>/
  target_soc: ["ascend910b"]
  scope: "intra_node"             # intra_node | inter_node | both
  torch_required: false           # Phase 0 用户确认；true 才执行 Phase 5.5
  performance_required: false       # Phase 0 #4；true 才执行 Phase 6
  performance_auto_optim: false     # Phase 0 #5；true 且未达标才自动 Phase 6.5
  docker_container: null            # Phase 0；非空时全部 build/run/perf 进容器
  docker_workdir: ""                # 容器内 SHMEM_REPO 绝对路径
  docker_exec_required: false       # docker_container 非空时为 true
  skills_root: ""                    # MUST；用户 @ skill 时上溯写入
  shmem_repo: ""                    # MUST；Phase 0 定位后写入；见 shmem-repo-resolution.md

source:
  kind: "natural_language"        # natural_language | user_pseudocode | heterogeneous_reference | mixed
  summary: "replace_me"
  user_confirmations:
    op_name: "confirmed_by_user"
    dtypes: []
  references: []

interface:
  inputs: []
  # - name: "input"
  #   dtype: "fp16"
  #   shape: "[n_pes, shard_elems]"
  #   role: "local_input"
  outputs: []
  # - name: "output"
  #   dtype: "fp16"
  #   shape: "[total_elems]"
  #   visibility: "replicated"    # replicated | sharded | owner_only
  attrs: []
  # - name: "block_dim"
  #   type: "int"
  #   default: "2 * n_pes"
  shape_relations: []

semantics:
  local_compute: "replace_me"
  communication: "replace_me"
  finalize: "replace_me"
  result: "replace_me"

topology:
  group: "replace_me"
  peer_model: "replace_me"
  addressing: "symmetric_memory"

memory:
  buffers: []
  # - name: "staging"
  #   type: "symmetric"
  #   size: "shard_elems * sizeof(dtype)"
  symmetric_layout: "replace_me"
  output_ownership: "replace_me"

schedule:
  # 设计顺序：先分核 → 再 tiling → 最后 phases
  core_partition:
    mode: "replace_me"              # 引用 core-allocation.md 模式：A（全员协作）/ B（按PE分组）/ C1（生产者消费者）/ C2（维度分组）/ CoC（通算融合）
    groups:
      []
      # - name: "send"
      #   aiv_range: "[0, aiv_num/2)"
      #   role: "本地 input 搬运到 GVA"
      #   data_interface: "aclshmemx_mte_put_nbi"
      # - name: "recv"
      #   aiv_range: "[aiv_num/2, aiv_num)"
      #   role: "从远端 GVA 拉取数据到 output"
      #   data_interface: "aclshmemx_mte_get_nbi"
    overlap: "replace_me"           # 组间如何重叠：并行 / 2-stage pipeline / 无
    sync_between_groups: "replace_me"  # 组间同步方式：cross-core flag / barrier / signal-wait / quiet
    rationale: "replace_me"         # 分核依据，引用 core-allocation.md 或 shmem/examples 路径
    cross_pe_dataplane: "replace_me"  # 跨 PE 数据面接口，必须是 aclshmem_*/aclshmemx_*
  tiling:
    tile: "replace_me"
    chunk: "replace_me"
    tail: "replace_me"
  phases:
    []
    # - name: "staging"
    #   action: "所有 AIV 将本地 input 拷贝到 symmetric memory"
    # - name: "remote_fetch"
    #   action: "Recv 组从远端 GVA 拉取数据到 output"
    # - name: "finalize"
    #   action: "回写最终结果到输出 buffer"
  compute:                          # 仅 fused_compute_comm 必填
    compute_path: "device_catlass_coc"  # device_catlass_coc（推荐）| host_aclnn（过渡，须 rationale）
    reference_examples:
      - "examples/matmul_allreduce"
    upgrade_plan: "replace_me"      # host_aclnn 时写明升级至 CATLASS 的条件

correctness:
  oracle: "replace_me"
  tolerance:
    rtol: "replace_me"
    atol: "replace_me"
  invariants:
    []
    # - invariant: "symmetric allocation 顺序和大小一致"
    #   test_method: "rank pattern"
    #   case: "all cases"
  case_matrix: []
  # 功能 case 和 stress/边界 case 统一放在此处：
  # - case: "boundary_tail"
  #   shape: "不整除 chunk 的 shape"
  #   category: "stress"
  # - case: "medium_8pe"
  #   shape: "[8, 32M]"
  #   category: "functional"

performance:
  metric: ["e2e_latency_us", "kernel_latency_us", "algo_bandwidth_GBps", "e2e_bus_bandwidth_GBps", "kernel_bus_bandwidth_GBps"]
  baseline: "replace_me"
  baseline_search: "replace_me"       # 已检查哪些 HCCL/aclnn 方案，为何不适用（无 baseline 时必填）
  baseline_target: "replace_me"       # 四算子: 有 baseline: "current >= 80% baseline"；无 baseline: metric_only
  target_cases: []
  min_scale: "replace_me"             # 集合通信 >= 256MB；计算/通算融合 hidden size >= 1000
  profiling_method: "replace_me"
  max_opt_rounds: 5
```

## 3. Capability Mapping & Gap Analysis

| Requirement | SHMEM capability/source | Classification | Adaptation / Gap | Risk |
| --- | --- | --- | --- | --- |
| [需求点] | [API/example/template/source path] | [可复用/需适配/需新实现] | [可复用：无；需适配：适配点；需新实现：gap 原因 + 设计方案 + 验证方式] | [正确性/性能风险] |

`Classification` 只能是 `可复用`、`需适配`、`需新实现`。

映射至少覆盖六类：lifecycle、memory、transport、sync、compute、scheduler。

如果没有"需新实现"项，在表后写明 `No known gap`。

## 4. 实现边界

| 位置 | 职责 |
| --- | --- |
| Device kernel | [通信、计算、搬运、同步等算子核心语义] |
| `main.cpp` | [单 PE Host 编排：参数、初始化、资源、读写、launch、cleanup] |
| Host helper `.cpp/.h` | [可选：运行时 shape/layout/tiling/route/payload/packing 计划] |
| Python scripts | [输入/golden/checker、测试用 route/payload、误差报告] |

约束：`main.cpp` 必须单进程单 PE，不 fork/spawn 多 PE；复杂 Host 逻辑必须拆到独立 `.cpp/.h`；golden 和 checker 不进入 Host C++。

## 5. Design Review Before Handoff

| Check | Result | Fix or rationale |
| --- | --- | --- |
| 并发/分核思路是否在首版 correctness 交付中落地 | [pass/fail] | [说明] |
| 是否存在不必要的 Host/global barrier | [pass/fail] | [说明] |
| tile/chunk/tail 是否支持并发且不靠单核兜底 | [pass/fail] | [说明] |
| route/offset/prefix 计算是否避免明显 O(N²) 扫描 | [pass/fail] | [说明] |
| 是否避免多余 GM scratch 写读，或说明无法避免原因 | [pass/fail] | [说明] |
| 对于 reduce-scatter / allreduce RS 等多 source→同 destination 场景，累加路径是否使用 `SetAtomicAdd<T>()`（MTE 批量）而非串行 UB 累加（必须符合 `atomic-add-pattern.md §12` 优先级1；仅 MTE 不可用时允许降级为 SDMA fallback） | [pass/fail] | [说明] |
| 高性能传输路径、并发分核或 overlap 是否属于本轮实现范围 | [pass/fail] | [说明] |
| Capability Mapping compute 行是否引用了 `atomic-add-pattern.md` 评估累加方式（非仅列出 AscendC Add） | [pass/fail] | [说明] |
| **拓扑与并发匹配**：设计的并发模型是否能发挥目标拓扑的全部链路并行度？（如 full-mesh 8 卡有 7 条独立 HCCS 链路，设计中是否让多链路**同时**活跃而非串行遍历 peer？） | [pass/fail] | [给出同时活跃链路数 vs 拓扑总链路数] |
| **链路带宽利用**：每个通信 phase 中，单 PE 同时活跃链路数 × 单链路带宽是否接近拓扑理论聚合带宽？（差距 >50% 须解释原因） | [pass/fail] | [给出估算数据和推导] |
| **分核与通信并行度**：AIV 分组是否支撑了并发目标？是否存在"分核了但 peer 维度仍是串行"的假并发？ | [pass/fail] | [说明每个 group 内 AIV 的通信是否真并行] |
| **性能瓶颈预判**：根据上述分析，标注设计预期的性能瓶颈位置（链路串行 / peer 串行 / 分核不足 / 同步开销 / UB 容量等） | [pass/fail] | [列出瓶颈类型和位置] |

## 6. Compile/Test Contract

### 6.1 Compile Contract

| 字段 | 内容 |
| --- | --- |
| build_mode | **`independent_project`（默认）** 或 `in_tree_example`（须 Phase 0 用户明确要求） |
| cann_env | [用户指定的 CANN set_env.sh 路径；不得硬编码默认路径] |
| shmem_repo | [绝对路径；见 shmem-repo-resolution.md；与 meta.shmem_repo 一致] |
| op_dir | [`custom-ops/<op_name>/`（默认）或 `examples/<op_name>/`] |
| target | [binary/shared library] |
| command | independent: [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1 编译；in-tree: 同文件 §0 + SHMEM build -examples |
| run_command | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2 运行 |
| matrix_command | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §3 case matrix |
| perf_command | [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) §1 阶段 B |
| torch_build_command | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §4 Torch 编译 |
| required_flags | [-examples/-debug/-enable_rdma/-enable_ascendc_dump/-soc_type ...] |
| expected_outputs | [build/bin 或 build/lib 产物] |

### 6.2 Functional/Correctness Contract

| 字段 | 内容 |
| --- | --- |
| command | [`scripts/run.sh`/checker 命令] |
| python_env | [python_cmd 或 conda/venv 激活命令；用于 gen_data.py、golden、check_result.py、baseline/profiler] |
| python_deps | [numpy/torch/torch_npu 或脚本实际 import 的依赖] |
| determinism | [seed、repeats、输出路径隔离] |

Oracle、tolerance、case matrix（含 stress/边界 case）已在 DSL `correctness` 中定义，此处不再重复。

## 7. Handoff Checklist for shmem-ops-code-gen

- [ ] Canonical DSL 完整，且无阻塞 open question。
- [ ] 用户已确认 `op_name` 和 dtype/dtypes，且记录在 `source.user_confirmations`。
- [ ] Capability Mapping 覆盖 lifecycle、memory、transport、sync、compute、scheduler；"需新实现"项有设计方案和验证方式。
- [ ] DSL `schedule` 的 core_partition、tiling、phases 可实现。
- [ ] DSL `correctness.invariants` 每项有 test_method 和 case；`case_matrix` 含 stress/边界 case。
- [ ] DSL `performance` 的 baseline_search、baseline_target、min_scale 已填写。
- [ ] Design review（Section 5）**12 项**全部填写，无 FAIL；拓扑/带宽/分核专项检查已给出量化数据（同时活跃链路数、带宽估算、瓶颈位置）。
- [ ] 实现边界清楚：`main.cpp` 只做单 PE 编排，复杂 Host 逻辑已归入独立 `.cpp/.h`，golden/checker 归入 Python。
- [ ] compile/test contract 可执行，test contract 写明 Python 环境或依赖。
