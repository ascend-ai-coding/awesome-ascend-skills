# MatMul Shape 提取

当 `op_high_time_selector.py` 脚本的输出或者`op_pivot_table_analyzer.py`或者`op_perf_pivot_table.py`的脚本输出中包含 `MatMul`、`MatMulV2` 或 `MatMulV3` 等算子时，可以使用 `extract_op_shapes.py` 脚本从 `op_analysis_details.csv` 文件或者`op_total_duration.csv`文件中提取 MatMul 算子的 M, N, K 维度信息。

## 使用方法及示例

```bash
python ./awesome-ascend-skills/profiling-analysis/profiling-computing/scripts/extract_op_shapes.py --input /path/to/your/op_analysis_details.csv --output /path/to/your/matmul_mnk_results.csv --op matmul
```

### 参数说明

*   `--input`: `op_analysis_details.csv` 或者`op_total_duration.csv`的文件的路径。
*   `--output`: 输出结果的 CSV 文件路径。
*   `--op`: 算子类型，这里固定为 `matmul`。


## 形状规则

### 规则 `basic-2x2`

输入case1:

```text
a,b;c,b
```

解释:

- 左边部分是 `[m, k]`
- 右边部分是 `[n, k]`
- 结果是 `m=a`, `n=c`, `k=b`

输入case2:

```text
a,b;b,c
```

解释:

- 左边部分是 `[m, k]`
- 右边部分是 `[k, n]`
- 结果是 `m=a`, `n=c`, `k=b`

### 规则 `packed-2x4`

输入case1:

```text
a,b;c,d,e,f
```

解释:

- 第一部分是 `[m, k]`
- 第二部分是 `[n_因子_1, k_因子_1, k_因子_2, n_因子_2]`
- 结果是:
  - `m = a`
  - `k = d * e = b`
  - `n = c * f`

输入case2:

```text
a,b;c,d,e,f
```

解释:

- 第一部分是 `[m, k]`
- 第二部分是 `[k_因子_1, n_因子_1, n_因子_2, k_因子_2]`
- 结果是:
  - `m = a`
  - `n = d * e`
  - `k = c * f = b`

