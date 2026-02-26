---
name: ascendc
description: Guides the agent to develop AscendC transformer GMM-style custom ops (such as grouped_matmul_finalize_routing) and their CANN aclnn examples by following existing patterns under ops-transformer/gmm and attention/softmax_ops/examples. Use when adding or modifying these ops, their kernels, tiling/infershape logic, or CANN API examples.
keywords:
    - ascend
    - ascendc
    - kernel
    - npu
    - development
    - 开发环境
    - 算子
    - 昇腾
---
# AscendC Transformer Operator Development

This skill guides the agent to develop/modify AscendC transformer-related operators according to existing patterns, including:

- FFN (Feed Forward Network) operators
- GMM (Grouped Matrix Multiplication) type operators
- MoE (Mixture of Experts) routing operators
  And the corresponding CANN `aclnn_*` example code.

## When to Use

Apply this skill in the following scenarios:

- Need to add or modify FFN (Feed Forward Network) related AscendC operators
- Need to add or modify GMM (Grouped Matrix Multiplication) type AscendC operators
- Need to add or modify MoE (Mixture-of-Experts) routing type AscendC operators
- Need to supplement `op_host` definitions, tiling/infershape logic, or `op_kernel` implementations for existing AscendC operators
- Need to write CANN `aclnn_*` examples similar to `ffn/ffn/examples/test_aclnn_ffn.cpp`
- Need to align, refactor, or bug-fix these operators while maintaining consistency with existing operator styles

---

## Overall Workflow

When users request to develop/modify such operators, follow these steps (order matters):

1. **Locate Reference Operators/Examples**
   - Based on operator type, search in corresponding directories:
     - FFN operators: `ops-transformer/ffn/`
     - GMM operators: `ops-transformer/gmm/`
     - MoE operators: `ops-transformer/moe/`
   - Look for the following file types:
     - `*_def.cpp` (operator definition)
     - `*_tiling*.h/.cpp` (tiling, scheduling logic)
     - `op_kernel/*.h` (AscendC kernel implementation)
   - Find CANN `aclnn_*` examples in the `examples/` subdirectory under the corresponding operator directory, e.g.:
     - FFN: `ffn/ffn/examples/test_aclnn_ffn.cpp`
     - GMM: `gmm/grouped_matmul/examples/`
     - MoE: `moe/moe_init_routing/examples/`
2. **Define Graph operator interface in op_host**
3. **Implement AscendC kernel in op_kernel (including quantization/routing logic)**
4. **Complete/reuse tiling, infershape and registration logic (if relevant files exist)**
5. **Write or update CANN API examples and unit tests**

Subsequent sections will detail what to do in each step and which details to pay attention to.

---

## Step 1: Reuse Existing Patterns

### Required Reading References

- FFN Operator Reference:
  - Graph definition: `ops-transformer/ffn/ffn/op_host/ffn_def.cpp`
  - Tiling implementation: `ops-transformer/ffn/ffn/op_host/ffn_tiling.cpp`
  - CANN API example: `ops-transformer/ffn/ffn/examples/test_aclnn_ffn.cpp`
- GMM Operator Reference:
  - Graph definition: `ops-transformer/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp`
  - AscendC kernel implementation: `ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul.h`
- MoE Operator Reference:
  - Graph definition: `ops-transformer/moe/moe_init_routing/op_host/moe_init_routing_def.cpp`
  - CANN API example: `ops-transformer/moe/moe_init_routing/examples/test_aclnn_moe_init_routing.cpp`
- General CANN API Example Reference:
  - `ops-transformer/attention/softmax_ops/examples/test_aclnn_softmax_ops.cpp`

### Behavioral Guidelines

- **Always copy the skeleton from existing similar operators first, then make minimal necessary modifications**
- Maintain:
  - Naming style (file names, class names, namespaces)
  - Macro usage patterns (`ASCEND_IS_AIC`, `ASCEND_IS_AIV`, etc.)
  - Queue and UB buffer management patterns (`TQue`, `TPipe`)
  - AICore configuration and support for different chips (e.g., `ascend910b` / `ascend910_95`)

---

## Step 2: Define Operator Interface in op_host

### Key Patterns

Inherit from `OpDef`, define the class within the `namespace ops`, and register using `OP_ADD`:

- Inputs:
  - Use `Input("name")` + `.ParamType(REQUIRED/OPTIONAL)`
  - Explicitly define `.DataType({ ... })`, `.Format({ ... })`, and `.UnknownShapeFormat({ ... })`
  - For multi-scenario/multi-type support, list all combinations using vector format
- Outputs:
  - Use `Output("y")`, similarly configure DataType / Format
- Attributes:
  - Use `.Attr("attr_name").AttrType(OPTIONAL/REQUIRED).Int/Float/Bool/ListInt(...)` to set default values
- AICore Configuration:
  - Construct `OpAICoreConfig`, setting:
    - `DynamicCompileStaticFlag(true)`
    - `DynamicFormatFlag(true)`
    - `DynamicRankSupportFlag(true)`
    - `DynamicShapeSupportFlag(true)`
    - `NeedCheckSupportFlag(false)` (if reference operators do so)
    - Necessary `ExtendCfgInfo(...)`, e.g., `"softsync.flag"`, `"prebuildPattern.value"`, `"coreType.value"`, `"aclnnSupport.value"`
  - Call `this->AICore().AddConfig("ascend910b", config);` etc. based on chip model
- Register at the end with `OP_ADD(YourOpClassName);`

### Operator-Specific Examples

#### FFN Operator (Refer to `ffn_def.cpp`)

The FFN operator supports Feed Forward Network computation with optional activation functions:

cpp

```
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight1")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight2")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias1")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});
// Output definitions
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("activation").AttrType(OPTIONAL).Int({0}); // 0: GELU, 1: RELU, 2: FASTGELU, 3: SILU, 4: SIGMOID, 5: TANH
Attr("inner_precise").AttrType(OPTIONAL).Int({0}); // 0: BF16, 1: FLOAT32
```

#### GMM Operator (Refer to `grouped_matmul_def.cpp`)

The GMM operator supports grouped matrix multiplication with configurable grouping and data types:

cpp

```
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});

// Output definitions
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32, DT_INT8})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("split_item").AttrType(OPTIONAL).ListInt({}); // Grouping information
Attr("dtype").AttrType(OPTIONAL).Int({0}); // 0: FLOAT16, 1: BF16, 2: INT8
Attr("transpose_weight").AttrType(OPTIONAL).Int({0}); // 0: No transpose, 1: Transpose
```

#### MoE Operator (Refer to `moe_init_routing_def.cpp`)

The MoE operator supports Mixture-of-Experts routing logic:

cpp

```
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Input("rowIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Input("expertIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// Output definitions
Output("expandedXOut")
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Output("expandedRowIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Output("expandedExpertIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("activeNum").AttrType(OPTIONAL).Int({0}); // Number of active experts
```

### Agent Key Points

- When creating new operators:
  - **Completely copy the class declaration and constructor body from reference operators**, then only modify:
    - Class name / file name
    - Input/output names and counts
    - Supported `DataType` / `Format`
    - Specific attributes and default values
  - Unless there are special reasons, do not arbitrarily change the AICore flags and `ExtendCfgInfo` structure from reference operators
- If `aclnn` support is needed:
  - Follow the `"aclnnSupport.value", "support_aclnn"` configuration in reference operators

---

## Step 3: Implement AscendC Kernel in op_kernel

### Common Characteristics

- Use the same namespace as the operator (e.g., `namespace FFN`, `namespace GroupedMatmul`, `namespace MoeInitRouting`)
- Include necessary headers:
  - `kernel_operator.h`
  - `lib/matmul_intf.h` (for matrix multiplication related operators)
  - Your own utility headers (e.g., `ffn_utils.h`, `grouped_matmul_utils.h`)
- Define type aliases:
  - `using aT = MatmulType<...>;`
  - `using bT = MatmulType<...>;`
  - `using BiasT = ...;`
  - `using cT = ...;`
  - `using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;`
- Use template parameters to control different scenarios (data types, quantization modes, activation functions, etc.)

### Operator-Specific Implementations

#### FFN Operator Implementation

The FFN operator implements Feed Forward Network computation, containing two linear transformations and an activation function:

cpp

```
namespace FFN {

// Define activation type enum
enum ActiveType {
    ACTIVE_GELU = 0,
    ACTIVE_RELU = 1,
    ACTIVE_FASTGELU = 2,
    ACTIVE_SILU = 3,
    ACTIVE_SIGMOID = 4,
    ACTIVE_TANH = 5
};

// Define parameter structure
template <typename T, ActiveType ACTIVE, bool WITH_BIAS>
struct Param {
    using InputType = T;
    using OutputType = T;
    static constexpr ActiveType kActive = ACTIVE;
    static constexpr bool kWithBias = WITH_BIAS;
};

// Main computation class
template <class P> class FfnCompute {
public:
    using InputType = typename P::InputType;
    using OutputType = typename P::OutputType;

    // Initialization function
    void Init(const InitParams &initParams, const FFNTiling *tiling) {
        // Initialize global tensors, UB buffer, queues, etc.
    }

    // Processing function
    void Process() {
        // First linear transformation: x * weight1 + bias1
        // Apply activation function
        // Second linear transformation: (x * weight1 + bias1) * weight2 + bias2
        // Write back results
    }

private:
    // Implement activation function
    void ApplyActivation(InputType *src, OutputType *dst, uint32_t size) {
        switch (P::kActive) {
            case ACTIVE_GELU:
                // Implement GELU activation
                break;
            case ACTIVE_FASTGELU:
                // Implement FASTGELU activation
                break;
            // Other activation function implementations
        }
    }
};

} // namespace FFN
```

#### GMM Operator Implementation

The GMM operator implements grouped matrix multiplication:

cpp

```
namespace GroupedMatmul {

// Define parameter structure
template <typename T, typename WeightT, typename BiasT, typename OutputT>
struct Param {
    using InputType = T;
    using WeightType = WeightT;
    using BiasType = BiasT;
    using OutputType = OutputT;
};

// Main computation class
template <class P> class GroupedMatmulCompute {
public:
    using InputType = typename P::InputType;
    using WeightType = typename P::WeightType;
    using BiasType = typename P::BiasType;
    using OutputType = typename P::OutputType;

    // Initialization function
    void Init(const InitParams &initParams, const GroupedMatmulTiling *tiling) {
        // Initialize global tensors, grouping information, UB buffer, queues, etc.
    }

    // Processing function
    void Process() {
        // Loop through each group
        for (uint32_t groupIdx = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
            // Compute matrix multiplication for current group
            ComputeGroup(groupIdx);
        }
    }

private:
    // Group computation function
    void ComputeGroup(uint32_t groupIdx) {
        // Set input, weight, output offsets for current group
        // Execute matrix multiplication
        // Add bias (if any)
        // Write back current group results
    }
};

} // namespace GroupedMatmul
```

#### MoE Operator Implementation

The MoE operator implements Mixture-of-Experts routing logic:

cpp

```
namespace MoeInitRouting {

// Define parameter structure
template <typename T, typename IndexT>
struct Param {
    using InputType = T;
    using IndexType = IndexT;
};

// Main computation class
template <class P> class MoeInitRoutingCompute {
public:
    using InputType = typename P::InputType;
    using IndexType = typename P::IndexType;

    // Initialization function
    void Init(const InitParams &initParams, const MoeInitRoutingTiling *tiling) {
        // Initialize global tensors, UB buffer, queues, etc.
    }

    // Processing function
    void Process() {
        // Process routing logic
        // Expand input x based on rowIdx and expertIdx
        // Generate expanded rowIdx and expertIdx
        // Write back results
    }

private:
    // Expand input tensor
    void ExpandInput(const InputType *x, IndexType *rowIdx, IndexType *expertIdx,
                    InputType *expandedX, IndexType *expandedRowIdx, IndexType *expandedExpertIdx) {
        // Implement expansion logic
    }
};

} // namespace MoeInitRouting
```

### Typical Structure (Reference Only, Don't Memorize Rigidly)

- Utility functions:
  - e.g., `DataCopyPad2D`, with two GM↔UB overloads, carrying `DataCopy2DDimParams`
- Main class:
  - Contains `Init(...)` method: initializes global tensors, UB buffer, queues, etc.
  - Contains `Process()` method: overall execution flow, including computation logic and result writing
  - Contains private helper methods: implements specific computation logic (e.g., activation functions, group processing)
  - If new operator logic is similar, **try to reuse this entire structure as much as possible, making only necessary changes**

### Agent Key Points

- When creating new operators/variants:
  - First confirm:
    - Whether it's still based on `MatmulImpl`, and which GM tensors are needed
    - Which fields are in the tiling structure (e.g., `matmulTiling.baseM/baseN/k`, `groupNum`, etc.)
  - Modify only:
    - Add/remove GM inputs (e.g., additional scale/bias/logits)
    - Adjust tensor combinations used in `ComputeDequantAndActivate` / `PerTokenScaleBrcb`, etc.
    - Modify business-logic-specific initialization in `InitOutputWithZeros`, `PreProcess`
  - Maintain:
    - Patterns for queue/UB allocation, `PipeBarrier`, `DataCopyPad`, `SetAtomicAdd` - do not change these unless there are clear bugs or requirements

---

## Step 4: Tiling / Infershape / Other Host Logic

Although this skill example doesn't expand all files, the agent should follow these patterns in the codebase:

1. Search under `op_host/` for:
   - `*_tiling*.h/.cpp`
   - `*_infershape.cpp`
   - Other `${op_name}_*.cpp` files
2. Analyze in reference operators:
   - Tiling parameters (batch, M/N/K, group count, deterministic flag, etc.)
   - How to convert Graph-level shape/attr to the `tiling` structure needed by kernel
3. When creating new operators:
   - If semantics are similar, prioritize copying reference tiling/infershape code, then rename and modify fields
   - Ensure:
     - Graph attributes/shapes correctly map to `tiling->...` fields accessed in kernel
     - Deterministic switches, workspace size, coreNum/parallNum calculation logic maintain consistent style

---

## Step 5: CANN aclnn Examples (examples)

Refer to `test_aclnn_softmax_ops.cpp`, the pattern is as follows:

1. **Common Utility Functions**
   - `GetShapeSize`: calculates product of shape dimensions
   - `PrintOutResult<T>`: copies device results back to host and prints
   - `Init(deviceId, &stream)`:
     - `aclInit`
     - `aclrtSetDevice`
     - `aclrtCreateStream`
   - `CreateAclTensor<T>`:
     - Use `aclrtMalloc` to allocate device memory
     - `aclrtMemcpy` to copy from host to device
     - Calculate `strides` for contiguous tensors
     - Call `aclCreateTensor` to create `aclTensor*`
2. **Main() Workflow**
   - Initialize ACL runtime
   - For each input/output construct:
     - Host-side data (`std::vector<T>`)
     - Corresponding shape (`std::vector<int64_t>`)
     - Call `CreateAclTensor(...)` to create `aclTensor*` and device addr
   - Get workspace size and executor:
     - Call `aclnnYourOpGetWorkspaceSize(...)`
   - If `workspaceSize > 0`:
     - `aclrtMalloc` to allocate workspace
   - Call actual operator:
     - `aclnnYourOp(workspaceAddr, workspaceSize, executor, stream);`
   - `aclrtSynchronizeStream(stream);`
   - Copy back output and call `PrintOutResult` to print results
   - Destroy tensors, free device memory, destroy stream, reset device, `aclFinalize`

### Agent Key Points

- When creating new `aclnn_*` examples:
  - **Completely copy the structure of `test_aclnn_softmax_ops.cpp`**, then modify:
    - Header includes (`aclnnop/aclnn_xxx.h`)
    - Number of tensors, shapes, dtypes, and fill data
    - Function names and parameter lists for `aclnnXxxGetWorkspaceSize` / `aclnnXxx`
  - Maintain:
    - Error checking macros (`CHECK_RET`) and logging output macros
    - Paired allocation/release of all `acl*` resources

---

## Step 6: Testing and Verification (If Python Frontend Exists)

If Python tests exist in the project (e.g., `op-plugin/test/test_custom_ops/test_*.py`):

1. Use existing test files as templates:
   - Usually named `test_npu_<op_name>_*.py`
   - Use NPU device, call op encapsulated by frontend API
2. When creating new operators:
   - Construct typical input shapes (including boundary scenarios)
   - If there's a comparable reference implementation (e.g., CPU version or simple Python algorithm), use it to calculate expected outputs
   - Assert:
     - Shape and dtype are correct
     - Numerical errors are within reasonable range (especially in quantization/dequantization scenarios)

---

## Additional Constraints for the Agent

- **Don't lazily invent patterns from scratch**: Always first search and align with implementations and examples of adjacent operators
- Before modifying any existing file:
  - Read the entire file first to understand its role in the overall operator
- For logic involving chip support / dynamic shapes / deterministic behavior:
  - Prioritize maintaining consistency with existing operators, unless bugs or requirements demand changes
- For examples and tests:
  - Better to have small, clear examples (single shape, easy to manually verify) rather than complex scenarios from the beginning

---

## Brief Usage Example

- **User**: Add a new GMM routing operator similar to `grouped_matmul_finalize_routing`, but with different scale/bias combinations
- **Agent Behavior (following this skill)**:
  1. Find and read `grouped_matmul_finalize_routing_*` related files in the `gmm` directory
  2. Copy `*_def.cpp`, rename and modify interface, adjust inputs/attributes
  3. Copy the core class from `op_kernel/grouped_matmul_finalize_routing.h`, adjust GM tensors and dequantization flow according to new requirements
  4. Refer to related tiling/infershape files to ensure correct parameter mapping from Graph to kernel
  5. Refer to `test_aclnn_softmax_ops.cpp` to write a new `aclnn` example, and supplement unit tests if needed

---

## Generic Operator Example Generation

### Overview

This section provides a general guide for generating CANN `aclnn_*` examples for any AscendC operator, including how to automatically extract operator information from `op_host` or `op_kernel` files to create accurate example code.

### How to Generate Generic Operator Examples

#### Step 1: Locate Operator Definition Files

1. **Find op_host files**: Look for `*_def.cpp` or similar files in the `op_host` directory of the operator.
2. **Find op_kernel files**: Look for implementation files in the `op_kernel` directory.
3. **Extract operator information**: From these files, extract:
   - Input parameters (names, data types, shapes)
   - Output parameters (names, data types, shapes)
   - Attribute parameters (names, types, default values)
   - Supported data types and formats

#### Step 2: Generate Example Code Structure

Use the following template to structure your example code:

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_[operator_name].h"
#include <iostream>
#include <vector>

// Define data types based on operator requirements
#define Kernel_dtype [appropriate_data_type]
#define Acl_dtpe [corresponding_acl_data_type]

// Error checking and logging macros
#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

// Utility functions
int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

template <typename T>
void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<T> resultData(size, 0);

  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(T), *deviceAddr,
      size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("idx[%ld] (offset: %ld Bytes) : %f\n", i, i * sizeof(T), resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize ACL
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs (based on operator definition)
  // [Input tensor declarations and initializations]

  // 3. Construct outputs (based on operator definition)
  // [Output tensor declarations and initializations]

  // 4. Get workspace size and executor
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnn[OperatorName]GetWorkspaceSize([input_tensors], [output_tensors], &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnn[OperatorName]GetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 5. Allocate workspace if needed
  void *workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    std::cout << "workspaceSize: " << workspaceSize << " bytes" << std::endl;
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 6. Execute the operator
  ret = aclnn[OperatorName](workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnn[OperatorName] failed. ERROR: %d\n", ret);
            return ret);

  // 7. Synchronize stream
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 8. Print results
  // [Print output tensors]

  // 9. Clean up resources
  // [Destroy tensors and free memory]

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

#### Step 3: Extract Operator Information

To automatically generate accurate examples, follow these steps to extract information from operator definition files:

1. **From op_host files**:
   - Look for `Input("name")` and `Output("name")` declarations
   - Extract data types from `.DataType({...})`
   - Extract formats from `.Format({...})`
   - Extract attributes from `.Attr("name")`

2. **From op_kernel files**:
   - Look for parameter structures and data types
   - Extract computation logic to understand input/output relationships
   - Identify any special handling or requirements

#### Step 4: Customize Example for Specific Operator

Using the extracted information, customize the template by:

1. **Updating includes**: Change `aclnnop/aclnn_[operator_name].h` to the correct header
2. **Setting data types**: Adjust `Kernel_dtype` and `Acl_dtpe` based on supported types
3. **Defining input tensors**: Create tensors for each input parameter
4. **Defining output tensors**: Create tensors for each output parameter
5. **Calling the API**: Update the function names and parameter lists
6. **Printing results**: Add code to print output tensors
7. **Cleaning up**: Ensure all resources are properly released

### Example: moe_init_routing_grouped_matmul_grad

#### Overview

The `moe_init_routing_grouped_matmul_grad` operator is a specialized AscendC operator for Mixture-of-Experts (MoE) models, designed to compute gradients for the routing and grouped matrix multiplication operations in the backward pass.

#### Implementation Principle

This operator performs the following key computations:

1. **Gradient Propagation**: Computes gradients with respect to the input tensor(s) and weight matrix based on the gradient of the output.
2. **Grouped Matrix Multiplication**: Efficiently handles grouped matrix operations to support expert-specific computations.
3. **Routing Gradient Handling**: Uses routing indices to properly aggregate gradients back to the original input space.

#### Generated Example Code

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_[operator_name].h"
#include <iostream>
#include <vector>

```#define Kernel_dtype float
#define Acl_dtpe aclDataType::ACL_FLOAT
#define Index_dtype int32_t
#define Acl_index_dtpe aclDataType::ACL_INT32

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

template <typename T>
void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<T> resultData(size, 0);

  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(T), *deviceAddr,
      size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("idx[%ld] (offset: %ld Bytes) : %f\n", i, i * sizeof(T), resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize ACL
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs (based on operator definition)
  // Input 1: input_tensor_1
  aclTensor *inputTensor1 = nullptr;
  void *inputTensor1DeviceAddr = nullptr;
  std::vector<int64_t> inputTensor1Shape = {1, 1, 1, 1}; // FILL IN actual shape
  std::cout << "inputTensor1Shape: ";
  for (size_t i = 0; i < inputTensor1Shape.size(); i++) {
    std::cout << inputTensor1Shape[i] << " ";
  }
  std::cout << std::endl;
  std::vector<Kernel_dtype> inputTensor1HostData(GetShapeSize(inputTensor1Shape), 0.0f);
  // FILL IN actual data
  ret = CreateAclTensor(inputTensor1HostData, inputTensor1Shape, &inputTensor1DeviceAddr, Acl_dtpe, &inputTensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Input 2: input_tensor_2
  aclTensor *inputTensor2 = nullptr;
  void *inputTensor2DeviceAddr = nullptr;
  std::vector<int64_t> inputTensor2Shape = {1, 1, 1, 1}; // FILL IN actual shape
  std::cout << "inputTensor2Shape: ";
  for (size_t i = 0; i < inputTensor2Shape.size(); i++) {
    std::cout << inputTensor2Shape[i] << " ";
  }
  std::cout << std::endl;
  std::vector<Kernel_dtype> inputTensor2HostData(GetShapeSize(inputTensor2Shape), 0.0f);
  // FILL IN actual data
  ret = CreateAclTensor(inputTensor2HostData, inputTensor2Shape, &inputTensor2DeviceAddr, Acl_dtpe, &inputTensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Add more inputs as needed

  // 3. Construct outputs (based on operator definition)
  // Output 1: output_tensor_1
  aclTensor *outputTensor1 = nullptr;
  void *outputTensor1DeviceAddr = nullptr;
  std::vector<int64_t> outputTensor1Shape = {1, 1, 1, 1}; // FILL IN actual shape
  std::cout << "outputTensor1Shape: ";
  for (size_t i = 0; i < outputTensor1Shape.size(); i++) {
    std::cout << outputTensor1Shape[i] << " ";
  }
  std::cout << std::endl;
  std::vector<Kernel_dtype> outputTensor1HostData(GetShapeSize(outputTensor1Shape), 0.0f);
  ret = CreateAclTensor(outputTensor1HostData, outputTensor1Shape, &outputTensor1DeviceAddr, Acl_dtpe, &outputTensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Add more outputs as needed

  // 4. Get workspace size and executor
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnn[OperatorName]GetWorkspaceSize([input_tensors], [output_tensors], &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnn[OperatorName]GetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 5. Allocate workspace if needed
  void *workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    std::cout << "workspaceSize: " << workspaceSize << " bytes" << std::endl;
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 6. Execute the operator
  ret = aclnn[OperatorName](workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnn[OperatorName] failed. ERROR: %d\n", ret);
            return ret);

  // 7. Synchronize stream
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 8. Print results
  std::cout << "\n\n\noutputTensor1[after]: " << std::endl;
  PrintOutResult<Kernel_dtype>(outputTensor1Shape, &outputTensor1DeviceAddr);
  // Print other outputs as needed

  // 9. Clean up resources
  aclDestroyTensor(inputTensor1);
  aclDestroyTensor(inputTensor2);
  // Destroy other tensors as needed
  aclDestroyTensor(outputTensor1);
  // Destroy other output tensors as needed

  aclrtFree(inputTensor1DeviceAddr);
  aclrtFree(inputTensor2DeviceAddr);
  // Free other input device addresses as needed
  aclrtFree(outputTensor1DeviceAddr);
  // Free other output device addresses as needed
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

### Handling Missing Operator Information

If the skill cannot find corresponding information in `op_host` or `op_kernel` files:

1. **Check file paths**: Ensure you're looking in the correct directories
2. **Look for alternative files**: Check for other files that might contain operator definitions
3. **Use placeholder values**: If information is missing, use generic placeholder values in the example
4. **Add comments**: Clearly mark sections where information is missing with comments
5. **Provide instructions**: Add instructions for users to fill in the missing information

#### Example of Placeholder Usage

```cpp
// Input tensors - FILL IN based on operator definition
aclTensor *input1 = nullptr;
void *input1DeviceAddr = nullptr;
std::vector<int64_t> input1Shape = {1, 1, 1, 1}; // FILL IN actual shape
std::vector<Kernel_dtype> input1HostData(input1Shape[0] * input1Shape[1] * input1Shape[2] * input1Shape[3], 0.0f);
ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, Acl_dtpe, &input1);
```

### Best Practices for Example Generation

1. **Start with a template**: Use the generic template as a starting point
2. **Extract accurate information**: Carefully read operator definitions to get correct parameter names, types, and shapes
3. **Use meaningful values**: Initialize tensors with values that make sense for the operator's functionality
4. **Add comments**: Explain the purpose of each section and any assumptions made
5. **Test the example**: Verify that the generated example compiles and runs correctly
6. **Handle edge cases**: Consider different input shapes and data types
7. **Follow existing patterns**: Maintain consistency with other examples in the codebase

By following this guide, you can efficiently generate accurate CANN `aclnn_*` examples for any AscendC operator, whether it's a simple arithmetic operation or a complex transformer-related operator like `moe_init_routing_grouped_matmul_grad`.
