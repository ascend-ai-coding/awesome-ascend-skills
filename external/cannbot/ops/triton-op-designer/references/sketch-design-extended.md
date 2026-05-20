# UnifiedSketch DSL 扩展规范
本文档在 UnifiedSketch DSL 基础上，扩展出更完整的算子设计表达能力。

---

## 新增核心概念

### 1. 需求分析区块 (`requirements`)

在草图中增加功能定义

```python
sketch <op_name> {
  requirements {
    // 功能定义
    function: "[算子功能说明]"
    formula: "[数学公式]"
    
    // 变量说明表
    variables: [
      {name: "Input",  type: "Tensor", shape: "[...]", dtype: "f16", constraint: "[约束]"},
      {name: "Output", type: "Tensor", shape: "[...]", dtype: "f16", constraint: "[约束]"},
    ]
  }
}
```

### 2. 规格约束区块 (`constraints`)

明确硬件和输入输出限制

```python
sketch <op_name> {
  constraints {
    // 输入约束
    input: {
      shape: "[具体约束]"
      dtype: "FLOAT16, BF16, FLOAT"
      format: "ND"
      alignment: "32B"
      min_elements: 1
    }
    
    // 输出约束
    output: {
      shape: "与输入相同"
      dtype: "与输入相同"
      format: "与输入相同"
    }
    
    // 硬件约束
    hardware: {
      ub_size: 192*1024       // UB 缓冲区大小（A2/A3）
      alignment: 32           // 内存对齐要求（字节）
      single_value_buffer: 32 // 单值缓冲区分配（字节）
      num_aicores: "[具体数量]"
    }
  }
}
```

### 3. Tiling 策略区块 (`tiling`)

"Tiling 切分"详细设计

```python
sketch <op_name> {
  tiling {
    // 核间切分策略
    inter_core {
      strategy: "[切分策略名称]"  // 如 "batch_split", "feature_split", "row_split"
      split_dim: "[切分维度]"     // 如 "B", "M", "N"
      
      // 切分计算方法（必须给出具体公式）
      compute: """
        per_core = ceil(total / num_cores)
        start = core_id * per_core
        end = min((core_id + 1) * per_core, total)
      """
      
      // 负载均衡处理
      load_balance: "[动态分配 / 向上取整]"
      
      // 边界处理
      boundary: "最后一个 Core 处理剩余数据"
    }
    
    // 核内切分策略
    intra_core {
      // UB 空间分配（必须逐项列出，禁止模糊）
      buffers: [
        {name: "input",    size: "D * dtype_size",     dtype: "f16", purpose: "输入缓存"},
        {name: "upcast",   size: "D * 4",              dtype: "f32", purpose: "升精度计算"},
        {name: "mean",     size: "32",                 dtype: "f32", purpose: "归约结果（32B对齐）"},
        {name: "output",   size: "D * dtype_size",     dtype: "f16", purpose: "输出缓存"},
      ]
      
      // 总空间计算
      total_buffer_size: "sum(buffer.size for buffer in buffers)"
      
      // 单次循环处理量
      batch_per_iteration: "UB_SIZE // total_buffer_size"
      
      // 对齐检查
      alignment_check: "所有缓冲区地址按 32B 对齐"
      
      // 精度转换策略
      precision: {
        storage: "f16"     // UB 中存储类型
        compute: "f32"     // 计算时升精度
        output:  "f16"     // 输出时降精度
      }
    }
    
    // 大矩阵优化（可选）
    large_matrix_opt: {
      diagonal_grid: {
        condition: "NUM_BLOCKS_M >= 4 and NUM_BLOCKS_N >= 4"
        task_m: "block_idx % NUM_BLOCKS_M"
        task_n: "(block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N"
      }
    }
  }
}
```

### 4. 数据流图区块 (`dataflow`)

明确 GM↔UB 数据传输

```python
sketch <op_name> {
  dataflow {
    // 数据流节点定义
    nodes: [
      {id: "gm_input",  type: "GM",  dtype: "f16", desc: "输入张量"},
      {id: "ub_input",  type: "UB",  dtype: "f16", desc: "输入缓冲区"},
      {id: "ub_upcast", type: "UB",  dtype: "f32", desc: "升精度缓冲区"},
      {id: "ub_output", type: "UB",  dtype: "f16", desc: "输出缓冲区"},
      {id: "gm_output", type: "GM",  dtype: "f16", desc: "输出张量"},
    ]
    
    // 数据流边定义
    edges: [
      {from: "gm_input",  to: "ub_input",  op: "load",   desc: "GM → UB"},
      {from: "ub_input",  to: "ub_upcast", op: "cast",   desc: "f16 → f32"},
      {from: "ub_upcast", to: "ub_output", op: "compute", desc: "核心计算"},
      {from: "ub_output", to: "gm_output", op: "store",  desc: "UB → GM"},
    ]
    
    // 数据类型转换标注
    dtype_transitions: [
      {stage: "load",   from: "f16", to: "f16"},
      {stage: "cast",   from: "f16", to: "f32"},
      {stage: "compute", from: "f32", to: "f32"},
      {stage: "store",  from: "f16", to: "f16"},
    ]
  }
}
```

---

## 扩展 @llm_hint 类型

### 新增硬件约束类 hint

| hint | 含义 | 来源 |
|------|------|------|
| `"ub_limit:192KB"` | UB 容量限制 | triton-operator-design 硬件约束 |
| `"align:32B"` | 32 字节对齐要求 | triton-operator-design 硬件约束 |
| `"single_value:32B"` | 单值缓冲区分配 32B | triton-operator-design 硬件约束 |
| `"upcast:fp32"` | 计算前升精度到 FP32 | triton-operator-design 精度策略 |
| `"downcast:fp16"` | 输出前降精度到 FP16 | triton-operator-design 精度策略 |
| `"core_split:batch"` | 按 batch 维度核间切分 | triton-operator-design Tiling 策略 |
| `"core_split:feature"` | 按特征维度核间切分 | triton-operator-design Tiling 策略 |
| `"diagonal_grid"` | 对角线 Grid 调度（大矩阵） | triton-operator-design Tiling 策略 |
| `"vector_core"` | 使用 Vector Core 计算 | triton-operator-design 核心区分 |
| `"cube_core"` | 使用 Cube Core 计算 | triton-operator-design 核心区分 |

### 新增数据流类 hint

| hint | 含义 |
|------|------|
| `"gm2ub"` | 数据从 Global Memory 加载到 UB |
| `"ub2gm"` | 数据从 UB 存储到 Global Memory |
| `"cast_up"` | 升精度转换（如 f16→f32） |
| `"cast_down"` | 降精度转换（如 f32→f16） |
| `"reuse_buffer"` | 缓冲区复用 |

---

## 扩展核心操作

### 新增 Tiling 相关操作

```python
// 核间切分计算
tile_split(total, num_cores, strategy="ceil")
// 返回: {per_core, start, end}

// UB 空间分配（带对齐）
alloc_aligned(size, align=32, llm_hint=[...])
// 返回: 对齐后的缓冲区

// 动态缓冲区复用
reuse_buffer(old_buffer, new_purpose, new_dtype)
// 返回: 复用的缓冲区

// 精度转换
cast(src, dst_dtype)
// 返回: 转换后的数据
```

### 新增归约操作（强制升精度）

```python
// 归约操作自动升精度到 FP32
reduce_sum_upcast(src, axis, dst)
reduce_max_upcast(src, axis, dst)
reduce_mean_upcast(src, axis, dst)

// 内部实现隐含:
// 1. src 升精度到 FP32
// 2. 执行归约
// 3. 结果存储到 dst（保持 FP32，由调用方决定是否降精度）
```

---

## RMSNorm 草图完整示例：

```python
sketch rmsnorm {
  // ========== 需求分析 ==========
  requirements {
    function: "Root Mean Square Layer Normalization"
    formula: "y = x / sqrt(mean(x^2) + eps) * gamma"
    
    variables: [
      {name: "x",      type: "Tensor", shape: "[B, D]", dtype: "f16", constraint: "ND格式"},
      {name: "gamma",  type: "Tensor", shape: "[D]",    dtype: "f16", constraint: "ND格式"},
      {name: "y",      type: "Tensor", shape: "[B, D]", dtype: "f16", constraint: "ND格式"},
      {name: "eps",    type: "scalar",                    dtype: "f32", constraint: "默认1e-6"},
    ]
  }
  
  // ========== 规格约束 ==========
  constraints {
    input: {
      shape: "[B, D], B>=1, D>=1"
      dtype: "FLOAT16, BF16, FLOAT"
      format: "ND"
      alignment: "32B"
    }
    output: {
      shape: "与输入相同"
      dtype: "与输入相同"
    }
    hardware: {
      ub_size: 192*1024
      alignment: 32
      single_value_buffer: 32
    }
  }
  
  // ========== Tiling 策略 ==========
  tiling {
    inter_core {
      strategy: "batch_split"
      split_dim: "B"
      compute: """
        batch_per_core = ceil(B / num_cores)
        batch_start = core_id * batch_per_core
        batch_end = min((core_id + 1) * batch_per_core, B)
      """
      load_balance: "向上取整，最后一个 Core 处理剩余"
    }
    
    intra_core {
      buffers: [
        {name: "input",    size: "D * 2",   dtype: "f16", purpose: "输入缓存"},
        {name: "upcast_x", size: "D * 4",   dtype: "f32", purpose: "升精度输入"},
        {name: "square",   size: "D * 4",   dtype: "f32", purpose: "平方值"},
        {name: "mean",     size: "32",      dtype: "f32", purpose: "均值结果（32B对齐）"},
        {name: "rms",      size: "32",      dtype: "f32", purpose: "RMS值（32B对齐）"},
        {name: "gamma",    size: "D * 4",   dtype: "f32", purpose: "升精度gamma"},
        {name: "output",   size: "D * 2",   dtype: "f16", purpose: "输出缓存"},
      ]
      total_buffer_size: "D*2 + D*4 + D*4 + 32 + 32 + D*4 + D*2"
      batch_per_iteration: "192*1024 // total_buffer_size"
      alignment_check: "所有缓冲区按32B对齐"
      precision: {
        storage: "f16"
        compute: "f32"
        output: "f16"
      }
    }
  }
  
  // ========== 数据流图 ==========
  dataflow {
    nodes: [
      {id: "gm_x",      type: "GM", dtype: "f16", desc: "输入x"},
      {id: "gm_gamma",  type: "GM", dtype: "f16", desc: "gamma参数"},
      {id: "ub_x",      type: "UB", dtype: "f16", desc: "输入缓冲区"},
      {id: "ub_x_f32",  type: "UB", dtype: "f32", desc: "升精度x"},
      {id: "ub_sq",     type: "UB", dtype: "f32", desc: "平方值"},
      {id: "ub_mean",   type: "UB", dtype: "f32", desc: "均值"},
      {id: "ub_rms",    type: "UB", dtype: "f32", desc: "RMS值"},
      {id: "ub_gamma",  type: "UB", dtype: "f32", desc: "升精度gamma"},
      {id: "ub_out",    type: "UB", dtype: "f16", desc: "输出缓冲区"},
      {id: "gm_y",      type: "GM", dtype: "f16", desc: "输出y"},
    ]
    edges: [
      {from: "gm_x",     to: "ub_x",     op: "load",    desc: "GM→UB"},
      {from: "gm_gamma", to: "ub_gamma", op: "load",    desc: "GM→UB"},
      {from: "ub_x",     to: "ub_x_f32", op: "cast_up", desc: "f16→f32"},
      {from: "ub_x_f32", to: "ub_sq",    op: "mul",     desc: "平方"},
      {from: "ub_sq",    to: "ub_mean",  op: "reduce_mean_upcast", desc: "均值"},
      {from: "ub_mean",  to: "ub_rms",   op: "sqrt_add_eps", desc: "sqrt(mean+eps)"},
      {from: "ub_x_f32", to: "ub_out",   op: "div_mul", desc: "x/rms*gamma"},
      {from: "ub_out",   to: "gm_y",     op: "store",   desc: "UB→GM"},
    ]
  }
  
  // ========== 核心计算逻辑 ==========
  symbols: B, D;
  tensors: X[B, D]: f16; Gamma[D]: f16; Y[B, D]: f16;
  constexpr: BLOCK_SIZE = 1024;
  
  @llm_hint("parallel", "core_split:batch")
  for b in range(0, B, batch_per_iteration):
    // 分配 UB 缓冲区
    x_tile = alloc([D], llm_hint=["fast", "input_cache"])
    x_f32  = alloc([D], llm_hint=["fast", "temp_workspace"])
    sq_tile = alloc([D], llm_hint=["fast", "temp_workspace"])
    mean_val = alloc([1], llm_hint=["fastest", "accumulator"])
    rms_val = alloc([1], llm_hint=["fastest", "accumulator"])
    gamma_f32 = alloc([D], llm_hint=["fast", "input_cache"])
    y_tile = alloc([D], llm_hint=["fast", "output_buffer"])
    
    // 加载数据
    load(X[b, 0:D] -> x_tile)
    load(Gamma[0:D] -> gamma_f32)
    
    // 升精度
    @llm_hint("cast_up")
    cast(x_tile, f32 -> x_f32)
    cast(gamma_f32, f32 -> gamma_f32)
    
    // 计算平方
    mul(x_f32, x_f32, sq_tile)
    
    // 归约（自动升精度）
    @llm_hint("upcast:fp32")
    reduce_mean_upcast(sq_tile, axis=0, mean_val)
    
    // 计算 RMS
    sqrt_add_eps(mean_val, eps, rms_val)
    
    // 归一化并乘 gamma
    div(x_f32, rms_val, x_f32)
    mul(x_f32, gamma_f32, x_f32)
    
    // 降精度输出
    @llm_hint("cast_down")
    cast(x_f32, f16 -> y_tile)
    
    // 存储结果
    store(y_tile -> Y[b, 0:D])
}
```

---

## 使用建议

1. **简单算子**（Elementwise）：可继续使用基础 UnifiedSketch DSL，无需完整扩展
2. **复杂算子**（Reduction、MatMul、融合算子）：建议使用完整扩展，确保 Tiling 和精度策略正确
3. **性能敏感算子**：必须包含 `tiling {}` 和 `dataflow {}` 区块
4. **多数据类型支持**：必须在 `constraints {}` 中明确各数据类型的处理差异