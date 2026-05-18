# `test_aclnn_<op>` 测试示例写法

## aclnn 头文件与职责边界

- 工程构建会根据 OpDef **生成** aclnn 对外头文件（如 `aclnn_<op_name>.h`）。
- 本 skill **只交付测试源码**：在 `examples/` 下提供可编译运行的 `test_aclnn_<op_name>.cpp`。

---

## 推荐代码内容（须覆盖）

1. ACL 运行时：`aclInit`、`aclrtSetDevice`、`aclrtCreateStream`（及对应释放/销毁，按工程惯例）。
2. 按 **OpDef** 的 Input/Output 构造 `aclTensor`：shape、dtype、format 与定义一致；至少一组可跑数据。Catlass **Matmul + 固定 COMPUTE_LENGTH 的 Tile Epilogue** 时，**不宜**用过小的 \(M,N\)（如个位数），易与尾块/向量长组合触发 AIV **UB 越界**；宜选 **L1 分块 M/N 的整数倍**（常见 M 为 128 的倍数、N 为 256 的倍数，以实际 Kernel 中 `GemmShape` 为准）。若 OpDef 含 **转置等 BOOL 属性**，`GetWorkspaceSize` / 执行接口须传入 **与 tensor 布局一致** 的 bool，**参数顺序以 opbuild 生成的 aclnn 头文件为准**（勿手写猜测签名）。
3. **两段式调用**：`GetWorkspaceSize` → `aclrtMalloc(ws)` → 执行接口；workspace 大小由 tiling 固定写法决定，见 [tiling-rules.md](./tiling-rules.md)。
4. 同步后把输出从 device 取回 host，打印或写文件；**建议写文件**，便于后续精度对比。

Include 工程生成的 aclnn 头。**以安装后的 `op_api/include` 为准**：常见为 `#include "aclnn_<op_name>.h"`（与 `aclnn_catlass_basic_matmul.h` 同在 include **根目录**）。`build.sh --run_example ... cust` 会加 `-I $ASCEND_HOME_PATH/opp/vendors/<vendor>_math/op_api/include`，需**先安装** `build_out` 下打好的 `*.run` 包，否则找不到头文件。跑 **`cust` 示例前**建议在已 `source $ASCEND_HOME_PATH/set_env.sh` 之外再 **`source $ASCEND_HOME_PATH/opp/vendors/<vendor>_math/bin/set_env.bash`**，确保链接到刚安装的自定义 `op_api`。若文档示例写 `aclnnop/...`，依赖对 `include/aclnnop` 的符号链接；**优先**使用与生成物一致的 `aclnn_<op_name>.h` 单文件 include，减少路径歧义。

---

## 编译安装与运行示例

**环境**：已 `source $ASCEND_HOME_PATH/set_env.sh`。

```bash
# 打包安装自定义算子（vendor/soc 按工程要求）
bash build.sh --pkg --ops=<op_name> --vendor_name=custom --soc=ascend910b

# 编译并运行 test_aclnn（eager = aclnn 两段式那条示例）
bash build.sh --run_example <op_name> eager cust --vendor_name=custom
```

说明：

- `<op_name>`：与 `ops/<op_name>/` 目录名一致（如 `catlass_matmul`）。
- `eager`：走 `ops/<op_name>/examples/test_aclnn_<op>.cpp`；`graph` 则对应 `test_geir_*`（若存在）。
- `cust`：使用 vendor 自定义算子包路径（如 `${ASCEND_HOME_PATH}/opp/vendors/...`）。

`--run_example` 会定位 `test_aclnn_<op>.cpp`并编译执行。

**注意**：`build.sh --run_example` **单独**执行时，脚本会关闭 custom 编译路径，**不会**重新编译/打包算子；验证 **kernel/tiling 修改** 须先 **`bash build.sh --pkg --ops=<op_name> --vendor_name=...`**，将 `build_out` 下 **`*.run` 安装到 CANN**（`--install-path=$ASCEND_HOME_PATH/opp`），再跑 `--run_example`，否则会误以为仍失败（实则为旧二进制）。

---

## 测试示例检查清单

- [ ] 路径为 `ops/<op_name>/examples/test_aclnn_<op_name>.cpp`
- [ ] 包含工程生成的 aclnn 头文件
- [ ] 含完整两段式：`GetWorkspaceSize` + 执行接口
- [ ] tensor 与 OpDef 一致；输出建议可落盘

---

## CANN 9.x `aclrtMemcpy` 签名

部分环境为五参数：`aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)`，与旧四参数顺序不同；样例需与当前 `acl/acl_rt.h` 一致。

---

## 未安装自定义包时的 aclnn

若未将 `build_out` 下 `*.run` 安装到 `ASCEND_HOME_PATH/opp`，`GetWorkspaceSize` 可能出现 **161001**（算子二进制未在 `binary_info_config` 中注册）。先 **`bash build.sh --pkg ...`** 并安装 run 包，再跑 `--run_example`。