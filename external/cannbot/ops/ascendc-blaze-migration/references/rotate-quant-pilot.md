# RotateQuant 完整迁移试点

这是第一版的 RotateQuant 闭环记录。

## 任务身份与完成线

| 项目 | 固定值 |
|---|---|
| 原算子 | `cann/ops-nn:matmul/rotate_quant` |
| ops-nn 原实现提交 | `0919397b1f0a69cf96ec41746175ed5247db63ee` |
| ops-nn 迁移实现提交 | `01127fde9f0fc667e01b8796fc5509dbcf39e2a9` |
| ops-tensor 原实现提交 | `2de22b293b38d7624a184d62219c30033b8bd22e` |
| ops-tensor 迁移实现提交 | `34059053d9f48e9c74aa7457c35107de8c71f56a` |
| 设备与软件 | Ascend 950 / DAV_3510；CANN 9.1.0；BiSheng/CCEC clang 15.0.5 |
| 实现 PR | [cann/ops-tensor!168](https://gitcode.com/cann/ops-tensor/merge_requests/168) |
| 接入 PR | [cann/ops-nn!7782](https://gitcode.com/cann/ops-nn/merge_requests/7782)，依赖前一个 PR |

完成线：PR 最新提交本地验证通过，两个 PR 已创建。编译流水线、CI 和评审不在本次范围内。

## G0：冻结事实

- 原实现和迁移实现都从 ACLNN 两阶段接口进入 RotateQuant 主机侧 tiling、核函数下发和 Arch35 核函数。
- 主机侧 tiling 结构、tiling key、工作空间、`blockDim`、`KERNEL_TYPE_MIX_AIC_1_2`、核函数 ABI、任务映射和支持域保持不变。
- 支持域覆盖 FP16/BF16 输入，MX FP4 E2M1、FP8 E4M3FN/E5M2 输出，K 32/64/128，维数 1-7，2D/3D 旋转、`alpha`、舍入模式、`scaleAlg`、完整块、尾块和特殊值。
- 正确性输入由仓内 `matmul/rotate_quant/examples/test_aclnn_rotate_quant_mx.cpp` 确定性生成；该入口保存全部 `y`/`scale` 字节，并检查重复运行和独立 256 字节物理越界防护区。

## G1：组件复用与扩展决定

固定 ops-tensor 原实现无法组合出 M/批次交错调度、A 行交错与 B 全载复用的 Fixpipe MMAD、单 AIC 对两个 AIV 的交替单消费者协议，以及分组截断后的 MX FP4/FP8 量化。用户逐层批准新增组件。

| 层级 | 最终组件 | 职责与适用边界 |
|---|---|---|
| Scheduler | `BlockSchedulerMatmulInterleavedBatch` | 只消费普通 Params，输出形状和坐标；不读取 RotateQuant tiling，不计算 GM 偏移 |
| BlockMmad | `BlockMmadMatmulBFullLoadInterleavedAFixpipe` | FP16/BF16 ND，B 全载或按批次重载，固定单个 L0 分块，Fixpipe 输出 BF16 |
| Kernel | `KernelMatmulFixpipeAlternatingAiv` | DAV_3510 1:2 MIX，AIC 每轮只通知一个交替 AIV；负责地址偏移、UB 视图和跨核同步 |
| Epilogue | `BlockEpilogueMxQuantGroupClamp` | BF16 输入，分组截断与 MX FP4/FP8 量化、`y`/`scale` 写回；不执行旋转 |

原算子仓只把冻结 tiling 数据映射到 `problemShape`、MMAD、Epilogue 和 Scheduler Params。组件文档声明适用与不适用范围、资源和同步约束；公共头组装测试与 Scheduler 坐标测试覆盖支持和拒绝边界。

## G2：彻底接入与全新检出

- DAV_3510 入口为 `rotate_quant.cpp` → `rotate_quant_blaze.h` → Blaze 公共组件。旧 `rotate_quant_basic_cmct.h` 和算子私有 Scheduler 被删除。
- 迁移实现的活动依赖闭包无 CMCT/CGMCT；新增或修改的实现文件在文件名、头文件引用、命名空间、类型、基类和选择器中也无 CMCT/CGMCT。
- ops-nn 固定 ops-tensor 迁移实现 SHA。同级检出目录的 `HEAD` 必须等于该 SHA；无同级目录或缓存时从 GitCode 获取。
- 全新检出环境从空依赖目录取得该 SHA，并构建 FP16/BF16 x FP4/E4/E5 全部 6 个核函数。ops-tensor 公共组件单元测试 3/3 通过。

最小全新构建复现：

```bash
git clone https://gitcode.com/cann/ops-nn.git ops-nn-candidate
cd ops-nn-candidate
git fetch origin 01127fde9f0fc667e01b8796fc5509dbcf39e2a9 --depth=1
git checkout --detach FETCH_HEAD
source /usr/local/Ascend/cann-9.1.0/cann-9.1.0/set_env.sh
mkdir -p third_party
bash build.sh --pkg --soc=ascend950 --ops=rotate_quant -j16 \
  --cann_3rd_lib_path="$PWD/third_party"
test "$(git -C third_party/ops-tensor rev-parse HEAD)" = \
  34059053d9f48e9c74aa7457c35107de8c71f56a
```

预期结果：生成 6 个 Ascend 950 RotateQuant 核函数和 `build_out/cann-ops-nn-custom_linux-x86_64.run`。

ops-tensor 组件单元测试复现：

```bash
git clone https://gitcode.com/cann/ops-tensor.git ops-tensor
cd ops-tensor
git fetch origin 34059053d9f48e9c74aa7457c35107de8c71f56a --depth=1
git checkout --detach FETCH_HEAD
source /usr/local/Ascend/cann-9.1.0/cann-9.1.0/set_env.sh
bash build.sh --opkernel -u --ops=rotate_quant_blaze -j8
```

## G3：逐字节一致性、性能与规范检查

8 个正确性用例覆盖 6 个数据类型组合对应的二进制、K 32/64/128、维数 1/2/3/7、完整块与尾块、N<64、2D/3D 旋转、`alpha`、三种 FP4 舍入模式和 BF16 Inf/NaN。原实现与迁移实现的 `y.bin`、`scale.bin` 8/8 逐字节一致；双方重复运行一致，越界防护区完整。

仓内测试程序的安装、编译和单个用例运行方式：

```bash
./build_out/cann-ops-nn-custom_linux-x86_64.run --quiet \
  --install-path=/absolute/path/to/candidate-opp
export ASCEND_CUSTOM_OPP_PATH=/absolute/path/to/candidate-opp/vendors/custom_nn
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  matmul/rotate_quant/examples/test_aclnn_rotate_quant_mx.cpp \
  -I${ASCEND_HOME_PATH}/include -L${ASCEND_HOME_PATH}/lib64 \
  -L${ASCEND_CUSTOM_OPP_PATH}/op_api/lib \
  -lcust_opapi -lopapi_math -lascendcl -lnnopbase \
  -o test_aclnn_rotate_quant_mx
./test_aclnn_rotate_quant_mx bf16_e4_alg0_k64_tail artifacts/candidate 2 5
```

原实现包使用同一源码、用例和参数运行到另一目录。对两个目录的 `y.bin`、`scale.bin` 执行 `cmp`；`metadata.json` 中 `repeat_bitwise_equal` 和 `guards_intact` 均为 `true`。无参数运行测试程序可列出全部用例。

最终性能采用同设备、同 CANN、同 tiling 的交错顺序，每个用例每进程预热 5 次、采集 20 条目标任务、运行 3 个独立进程：

| 用例 | 原实现进程中位数（微秒） | 迁移实现进程中位数（微秒） | 迁移实现/原实现 |
|---|---:|---:|---:|
| BF16/E4, M1024 N256 K64 | 27.652 / 27.721 / 27.698 | 27.938 / 27.826 / 27.570 | 1.0032 |
| BF16/FP4, M1024 N256 K128 | 72.409 / 74.223 / 73.179 | 73.677 / 72.899 / 72.242 | 0.9955 |

BF16/E4 是稳定验收用例。正反顺序详细性能采集的任务中位数均值为原实现 29.159 us、迁移实现 29.305 us，回退约 `0.50%`。MTE2/MMAD/Fixpipe 高低模式随采集顺序切换；FP16/E5 数据也未显示迁移实现的 MMAD、MTE2 或 Fixpipe 流水活跃时间增加。性能负责人将结果按例外验收通过，结论不是“无差异”。

## G4：双仓交付

- ops-tensor PR 最新提交为 `34059053d9f48e9c74aa7457c35107de8c71f56a`，包含公共 Blaze 组件、测试和组件文档。
- ops-nn PR 最新提交为 `01127fde9f0fc667e01b8796fc5509dbcf39e2a9`，包含原仓接入、依赖绑定和测试程序，并依赖 ops-tensor PR。
- 两个分支各只有一个迁移提交。PR 复现说明包含仓库 SHA、CANN/设备、全新依赖环境、构建、安装、运行、逐字节比对和性能协议。
- PR 创建后停止，不触发编译，不跟踪 CI 或评审。
