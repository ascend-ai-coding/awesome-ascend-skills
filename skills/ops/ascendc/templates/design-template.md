# [Operator Name] Design Document

## 1. Operator interface

### 1.1 Function signature
```cpp
at::Tensor [operator_name](
    const at::Tensor &input1,
    const at::Tensor &input2
    /* other params */
);
```

### 1.2 Parameters
| Name | Type | In/Out | Supported dtypes | Description | Constraints |
|---|---|---|---|---|---|
| input1 | at::Tensor | input | bfloat16/float16/float32 | input tensor 1 | ND |
| input2 | at::Tensor | input | bfloat16/float16/float32 | input tensor 2 | ND |
| output | at::Tensor | output | bfloat16/float16/float32 | output tensor | ND |

### 1.3 Supported dtypes
- [ ] bfloat16
- [ ] float16
- [ ] float32

---

## 2. Compute logic

### 2.1 Algorithm
[Describe the compute steps and data flow.]

### 2.2 Pseudocode
```
for each tile in input:
    load tile to UB
    compute on tile (fp16/bf16: cast up to fp32 first)
    store result to GM
```

### 2.3 Implementation path
- [ ] AscendC Kernel (pure vector)
- [ ] CATLASS template lib (matmul-class)
- [ ] aclnn wrap (CANN built-in)

**Reason:** [why this path]

---

## 3. Tiling strategy

Two-level tiling: block-level (inter-core) + UB-level (intra-core).

**Operator type:** elementwise / reduction(row) / index / sort / pool

### 3.1 Tiling struct
```cpp
struct [OperatorName]TilingData {
    int64_t totalLength;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t tileLength;     // UB single-pass length
};
```

### 3.2 Block-level tiling (inter-core)
| Param | Formula | Value |
|---|---|---|
| totalLengthCore | (totalLength + CORE_NUM - 1) / CORE_NUM | [ ] |
| totalLengthCoreAlign | (totalLengthCore + 511) / 512 * 512 | [ ] |
| usedCoreNum | (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign | [ ] |
| formerNum | usedCoreNum - 1 | [ ] |
| formerLength | totalLengthCoreAlign | [ ] |
| tailNum | 1 | [ ] |
| tailLength | totalLength - formerNum * formerLength | [ ] |

Validate: `formerNum*formerLength + tailNum*tailLength == totalLength`.

### 3.3 UB-level tiling (intra-core)

Precision: the NPU vector unit cannot compute on fp16/bf16 directly — up-cast to fp32.

| Input dtype | Handling | Compute precision | UB impact |
|---|---|---|---|
| float16 | up-cast to float32 | float32 | extra fp32 buffer |
| bfloat16 | up-cast to float32 | float32 | extra fp32 buffer |
| float32 | direct | float32 | none |

UB allocation table:

| Buffer | Size (bytes) | Use | Count | Total |
|---|---|---|---|---|
| inQueueX | tileLength * dtypeSize | input | BUFFER_NUM | [ ] |
| outQueueZ | tileLength * dtypeSize | output | BUFFER_NUM | [ ] |
| tmp fp32 (fp16/bf16) | tileLength * 4 | fp32 compute | 1 | [ ] |
| **Total** | — | — | — | **[ ]** |

tileLength:

| Param | Formula | Value |
|---|---|---|
| bufferCoefficient | from UB allocation table | [ ] |
| maxTileElements | UB_SIZE_LIMIT / bufferCoefficient (UB size via platform API) | [ ] |
| alignElements | 32 / dtypeSize | [ ] |
| tileLength | (maxTileElements / alignElements) * alignElements | [ ] |

---

## 4. Workspace

| Operator class | Workspace size |
|---|---|
| elementwise | SYSTEM_WORKSPACE_SIZE (typically 16 MB) |
| others | sizeof([OperatorName]TilingData) |

---

## 5. Performance notes
- [ ] double buffer to hide memory latency
- [ ] cache-line alignment for inter-core balance
- [ ] reduce GM accesses
- Compute mode: [memory-bound / compute-bound / balanced]

---

## 6. Kernel implementation notes

```cpp
__aicore__ inline void Process() {
    int64_t coreLength = (AscendC::GetBlockIdx() == usedCoreNum - 1) ? tailLength : formerLength;
    int64_t tileNum = (coreLength + tileLength - 1) / tileLength;
    int64_t tailTileLength = coreLength - (tileNum - 1) * tileLength;
    for (int64_t i = 0; i < tileNum - 1; ++i) {
        CopyIn(i, tileLength); Compute(i, tileLength); CopyOut(i, tileLength);
    }
    CopyIn(tileNum - 1, tailTileLength); Compute(tileNum - 1, tailTileLength); CopyOut(tileNum - 1, tailTileLength);
}
```

---

## 7. Implementation checklist
- [ ] op_host: TilingData struct, block-level + UB-level tiling, bufferCoefficient, workspace, kernel launch
- [ ] op_kernel: Init offsets, CopyIn, Compute (fp16/bf16 up-cast), CopyOut, Process tail handling
- [ ] framework: ops.h declaration, register.cpp schema + impl, CMakeLists sources

---

## Usage

Replace every `[placeholder]`, tick the applicable boxes, fill values, adjust code to the
real operator, and delete sections that do not apply. After completing, generate code with
the code-gen phase.
