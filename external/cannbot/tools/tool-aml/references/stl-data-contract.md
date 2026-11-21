# STL 输入数据契约：aclDataType / StlDataMode / StlInputSpec 白名单

生成 `<TYPE>_<NAME>.cpp` 的输入规格时，`StlInputSpec` 的 `dtype` 与 `mode` **只能使用本页列出的枚举值**。本页是可信源快照，来源：

- `aclDataType`：`runtime/include/external/acl/acl_base_rt.h`
- `StlDataMode` / `StlInputSpec`：`oam-tools-diag` 的 `src/aml/aml/aml_diagnose/aicore_stl_detect/common/aml_aicore_stl_data_gen.h`
- dtype 支持范围：`common/aml_aicore_stl_data_gen.cpp` 的 `ConvertHostDataToBytes` / `GetStlDtypeSize`

任何本页未列出的枚举值都是**虚构的**，写进生成代码必然编译报错。用户需求超出本页范围时，拒绝生成（见第五节）。

## 一、aclDataType 完整枚举（acl_base_rt.h）

```c
typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,           // fp32    ← 数据生成支持
    ACL_FLOAT16 = 1,         // fp16    ← 数据生成支持
    ACL_INT8 = 2,            //         ← 数据生成支持
    ACL_INT32 = 3,
    ACL_UINT8 = 4,           //         ← 数据生成支持
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,           //         ← 数据生成支持
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
    ACL_HIFLOAT8 = 34,
    ACL_FLOAT8_E5M2 = 35,    //         ← 数据生成支持
    ACL_FLOAT8_E4M3FN = 36, //         ← 数据生成支持
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT6_E3M2 = 38,
    ACL_FLOAT6_E2M3 = 39,
    ACL_FLOAT4_E2M1 = 40,    // 半字节  ← 数据生成支持
    ACL_FLOAT4_E1M2 = 41,
    ACL_HIFLOAT4 = 42,
    ACL_HIFLOAT4_SCALE = 43,
} aclDataType;
```

历史教训：曾臆造 `ACL_BFLOAT16` 导致编译失败——**该枚举不存在**，bf16 的正确枚举名是 `ACL_BF16`。

## 二、dtype 支持白名单（数据生成函数的编码范围）

| 用户说法 | 枚举值 | 元素字节 | 编码实现 |
|---|---|---|---|
| fp32 | `ACL_FLOAT` | 4 | 原样 memcpy |
| fp16 | `ACL_FLOAT16` | 2 | IEEE half（`FloatToFp16`） |
| bf16 | `ACL_BF16` | 2 | fp32 高 16 位 RNE（`FloatToBf16`） |
| fp8_e4m3 | `ACL_FLOAT8_E4M3FN` | 1 | `FloatToMiniFloat(4, 3)` |
| fp8_e5m2 | `ACL_FLOAT8_E5M2` | 1 | `FloatToMiniFloat(5, 2)` |
| uint8 | `ACL_UINT8` | 1 | `static_cast<uint8_t>` |
| int8 | `ACL_INT8` | 1 | `static_cast<int8_t>` |
| fp4_e2m1 | `ACL_FLOAT4_E2M1` | **0.5** | 半字节打包（元素 2k→低半字节，2k+1→高半字节） |

**不在表内的 dtype 一律不支持**：`ConvertHostDataToBytes` 无对应分支会返回空、`GetStlDtypeSize` 返回 0，`CreateStlTensor` 返回 `AML_ERROR_INVALID_PARAM`。常见不支持需求：int32、int64、uint16、fp64、hifloat8 等。

fp4 特殊性：

- `GetStlDtypeSize(ACL_FLOAT4_E2M1)` 返回 **0**（无整字节尺寸），不能用「元素数 × dtypeSize」算字节数；
- 字节数必须经 `GetStlTensorByteSize`：`(元素数 + 1) / 2`（向上取整，两元素一字节）；
- 生成 `spec.shape` 时按「元素数 = size(字节) × 2」反推。

## 三、StlDataMode 完整枚举（aml_aicore_stl_data_gen.h）

```c++
enum class StlDataMode {
    STL_DATA_FIXED = 0,      // 固定数值(按 dtype 逐元素编码 fixedValue)
    STL_DATA_NORMAL = 1,     // 正态分布
    STL_DATA_FILE = 2,       // 从文件读取原始字节(文件在当前形态子目录下,见 fileName)
    STL_DATA_SEQUENTIAL = 3, // 递增序列:seqStart, seqStart+seqStep, seqStart+2*seqStep, ...
    STL_DATA_RANDOM = 4,     // 均匀随机分布:在 [randMin, randMax] 内随机生成
};
```

**只有这 5 种**。生成代码中出现任何其他 `STL_DATA_*` 均为虚构，编译必报错。

用户口语 → 模式映射：

| 用户说法 | 模式 | 关键字段 |
|---|---|---|
| 固定值 / 全 x | `STL_DATA_FIXED` | `fixedValue` |
| 递增 / 等差序列 | `STL_DATA_SEQUENTIAL` | `seqStart`、`seqStep` |
| 正态 / 高斯分布 | `STL_DATA_NORMAL` | `normalMean`、`normalStddev`（可选 `clampEnabled`/`clampMin`/`clampMax`） |
| 随机（均匀分布） | `STL_DATA_RANDOM` | `randMin`、`randMax`（须 randMin ≤ randMax） |
| 从文件读取 | `STL_DATA_FILE` | `fileName`（不含路径） |

用户要求泊松分布、指数分布、稀疏数据等 → 无对应模式，**拒绝生成**。

## 四、StlInputSpec 完整字段（aml_aicore_stl_data_gen.h）

```c++
struct StlInputSpec {
    aclDataType dtype = ACL_FLOAT;                   // 数据类型(白名单见第二节)
    std::vector<int64_t> shape;                      // tensor shape
    StlDataMode mode = StlDataMode::STL_DATA_NORMAL; // 生成方式
    float fixedValue = 0.0F;                         // mode == STL_DATA_FIXED 时使用的固定值
    float normalMean = 0.0F;                         // 正态分布均值
    float normalStddev = 1.0F;                       // 正态分布标准差(决定数据范围)
    bool clampEnabled = false;                       // 是否将正态分布结果限制到 [clampMin, clampMax]
    float clampMin = 0.0F;                           // 数据范围下界(clampEnabled 时生效)
    float clampMax = 0.0F;                           // 数据范围上界(clampEnabled 时生效)
    std::string fileName;                            // mode == STL_DATA_FILE 时的文件名(不含路径)
    float seqStart = 1.0F;                          // mode == STL_DATA_SEQUENTIAL 时的起始值
    float seqStep = 1.0F;                           // mode == STL_DATA_SEQUENTIAL 时的相邻元素步长
    float randMin = 0.0F;                           // mode == STL_DATA_RANDOM 时的随机下界(含)
    float randMax = 1.0F;                           // mode == STL_DATA_RANDOM 时的随机上界(含)
};
```

除本 struct 已有字段外，**不得发明新字段**（如 `poissonLambda`、`sparsity`）。

字段填写矩阵（✅ 必填 / ○ 可选 / — 不用）：

| 字段 | FIXED | SEQUENTIAL | NORMAL | RANDOM | FILE |
|---|---|---|---|---|---|
| dtype | ✅ | ✅ | ✅ | ✅ | ✅\* |
| shape | ✅ | ✅ | ✅ | ✅ | ✅ |
| mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| fixedValue | ✅ | — | — | — | — |
| seqStart / seqStep | — | ✅ | — | — | — |
| normalMean / normalStddev | — | — | ✅ | — | — |
| clampEnabled / clampMin / clampMax | — | — | ○ | — | — |
| randMin / randMax | — | — | — | ✅ | — |
| fileName | — | — | — | — | ✅ |

\* FILE 模式**不按 dtype 编码内容**（读原始字节），但 dtype×shape 仍决定申请/读取字节数：
`CreateStlTensor` 用 `GetStlTensorByteSize(spec.dtype, spec.shape)` 算 byteSize，
`LoadStlRawBytesFromFile` 按该 byteSize 读取，读不足视为失败（不回退随机）。
建议 `dtype = ACL_UINT8` 且 `shape = {字节数}`（1 元素 = 1 字节，元素数即字节数，参照 `aic_cube_mxfp4_covery` 的 mx 输入）。

## 五、拒绝生成（超范围需求的标准处理）

用户要求的 dtype 或填充方式超出上述范围时，**不要生成 `.cpp`，更不得用虚构枚举「绕过」**，按以下要点回复：

1. 明确指出哪一项不支持（dtype 名 / 填充方式名）；
2. 列出当前支持范围：
   - dtype：fp32 / fp16 / bf16 / fp8_e4m3 / fp8_e5m2 / fp4_e2m1 / uint8 / int8；
   - 填充方式：fixed / sequential / normal / random / file；
3. 给出可行替代，例如：
   - int32 输入 → 改用 `file` 模式（把 int32 数据写成二进制文件由框架读入），或先扩展 `aml_aicore_stl_data_gen.cpp` 的 `ConvertHostDataToBytes` / `GetStlDtypeSize` 再生成；
   - 泊松分布 → 用 `random` / `normal` 近似（须提示精度差异并由用户确认），或扩展 `GenStlTensorHostData`；
4. 待用户确认方案后再生成。

> 若确实需要新 dtype / 新模式，正确路径是**先在 oam-tools-diag 仓扩展 `aml_aicore_stl_data_gen.{h,cpp}` 并同步更新本页白名单**，再生成算子——而不是在算子 `.cpp` 里臆造枚举。
