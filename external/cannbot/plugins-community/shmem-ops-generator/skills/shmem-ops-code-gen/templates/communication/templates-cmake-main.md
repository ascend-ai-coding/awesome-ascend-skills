# Communication 算子 — CMake + Host 入口模板

> **阅读时机**：渐进式生成步骤 1（目录结构、CMake、Host 入口）时读取本文件。
> 索引与约束见 [GUIDE.md](GUIDE.md)。

提取规则：按下方 `## <relative_path> → <target>` 标题匹配，提取 fenced code block，写入目标路径，替换 `<op_name>` / `<OpName>` / `<OP_NAME>`。

---

## CMakeLists.txt → `<op_name>/CMakeLists.txt`

```cmake
# ==== 定制：替换 <op_name> 为算子名 ====
# 独立工程构建模式：显式引用 src/ 下源码，include src/ 和 examples/utils/
cmake_minimum_required(VERSION 3.16)
project(<op_name> LANGUAGES CXX)

set(CANN_ENV "$ENV{ASCEND_HOME_PATH}")
if(NOT CANN_ENV)
    message(FATAL_ERROR "ASCEND_HOME_PATH not set; source install/set_env.sh and CANN set_env.sh first")
endif()

set(SHMEM_ROOT "${CANN_ENV}/shmem")
set(EXAMPLE_UTILS "${SHMEM_ROOT}/examples/utils")

include_directories(
    ${CANN_ENV}/include
    ${SHMEM_ROOT}/include
    ${SHMEM_ROOT}/examples/utils
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(<op_name>
    src/main.cpp
    src/<op_name>_kernel.cpp
)
target_link_libraries(<op_name> shmem ascendcl)
```

---

## src/main.cpp → `<op_name>/src/main.cpp`

```cpp
// ============================================================
// SHMEM Communication 算子 Host 模板
// 来源: shmem/examples/allgather/main.cpp 简化
// 使用: 写入 <op_name>/src/main.cpp，替换 <op_name>/<OpName> 占位符
// ============================================================

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cstdint>

#include "acl/acl.h"
#include "shmem.h"

#include "<op_name>_kernel.h"

#define CHECK_ACL(call)                                                     \
    do {                                                                    \
        auto _ret = (call);                                                 \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_SHMEM(call)                                                   \
    do {                                                                    \
        int _ret = (call);                                                  \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ---- SHMEM init helper（内联，不依赖 examples/utils）----
static void init_set_attr(int pe_id, int n_pes, uint64_t local_mem_size,
                          const char *ip_port,
                          aclshmemx_uniqueid_t &uid,
                          aclshmemx_init_attr_t &attr)
{
    attr.my_pe      = pe_id;
    attr.n_pes      = n_pes;
    attr.local_mem_size = local_mem_size;
    strncpy(attr.ip_port, ip_port, sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    if (pe_id == 0) {
        aclshmemx_get_uniqueid(&uid);
    }
    attr.uniqueid = uid;
}

// ---- FFTS config helper ----
static uint64_t get_ffts_config()
{
    uint64_t fftsAddr = 0;
    CHECK_ACL(aclrtGetMemInfo(ACL_HBM_MEM, &fftsAddr, nullptr));
    return fftsAddr;
}

// ---- 文件读写工具 ----
template <typename T>
void ReadBinaryFile(const std::string &path, T *dst, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    f.read(reinterpret_cast<char *>(dst), count * sizeof(T));
}

template <typename T>
void WriteBinaryFile(const std::string &path, const T *src, size_t count)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot create " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    f.write(reinterpret_cast<const char *>(src), count * sizeof(T));
}

// ---- 参数 ----
struct Options {
    int n_pes = 2;
    int pe_id = 0;
    std::string ip_port = "tcp://127.0.0.1:8998";
    std::string data_dir = "./data";
    int perf_times = 0;  // 0 = correctness only
    int warmup = 5;
    // ==== 定制：添加算子特有参数（shape、dtype 等）====
    int elements = 1024;

    int Parse(int argc, char **argv)
    {
        if (argc < 5) {
            std::cerr << "Usage: <op_name> <n_pes> <pe_id> <ip_port> <data_dir>"
                      << " [elements] [perf_times]" << std::endl;
            return -1;
        }
        n_pes     = std::atoi(argv[1]);
        pe_id     = std::atoi(argv[2]);
        ip_port   = argv[3];
        data_dir  = argv[4];
        if (argc > 5) elements   = std::atoi(argv[5]);
        if (argc > 6) perf_times = std::atoi(argv[6]);
        return 0;
    }
};

int main(int argc, char *argv[])
{
    Options opts;
    if (opts.Parse(argc, argv) != 0) return 1;

    int pe_id = opts.pe_id;
    int n_pes = opts.n_pes;
    int32_t device_id = pe_id;  // ==== 定制：多卡映射 ====

    // ---- ACL & SHMEM init ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_uniqueid_t uid;
    aclshmemx_init_attr_t attributes;
    init_set_attr(pe_id, n_pes, local_mem_size, opts.ip_port.c_str(), uid, attributes);
    CHECK_ACL(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint64_t fftsAddr = get_ffts_config();

    // ---- 读取输入 ----
    // ==== 定制：按 design.md 的 interface.inputs 调整 ====
    size_t input_bytes = opts.elements * sizeof(int32_t);
    void *input_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&input_dev, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<int32_t> input_host(opts.elements);
    std::string input_path = opts.data_dir + "/input_pe" + std::to_string(pe_id) + ".bin";
    ReadBinaryFile(input_path, input_host.data(), opts.elements);
    CHECK_ACL(aclrtMemcpy(input_dev, input_bytes, input_host.data(), input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 分配输出 ----
    // ==== 定制：按 design.md 的 interface.outputs 和 visibility 调整 ====
    size_t output_elems = opts.elements * n_pes;
    size_t output_bytes = output_elems * sizeof(int32_t);
    void *output_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&output_dev, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- 分配 symmetric buffer ----
    // ==== 定制：按 design.md 的 memory.buffers 调整大小 ====
    size_t symm_size = opts.elements * sizeof(int32_t) + 16 * 16 * sizeof(int32_t);
    if (symm_size == 0) {
        std::cerr << "symmetric buffer size is zero" << std::endl;
        return 1;
    }
    void *symm_ptr = aclshmem_malloc(symm_size);
    if (symm_ptr == nullptr) {
        std::cerr << "aclshmem_malloc failed, size=" << symm_size << std::endl;
        return 1;
    }

    // ---- 执行 ----
    uint32_t block_num = 2 * n_pes;  // ==== 定制：按 schedule.core_partition ====
    int magic = 1;

    // ==== 定制：staging 拷贝函数 —— 将 input_dev 拷贝到 symmetric buffer ====
    // staging 是 e2e 性能循环的必要组成，按 design.md 的 memory.staging 选择路径
    auto staging_copy = [&](void *src, void *dst, size_t bytes) {
        CHECK_ACL(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
    };

    if (opts.perf_times > 0) {
        // warmup（含 staging + barrier + kernel）；msprof 解析时按 WARMUP 跳过
        for (int i = 0; i < opts.warmup; i++) {
            magic++;
            staging_copy(input_dev, symm_ptr, input_bytes);
            aclshmem_barrier_all();
            <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
                (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
                opts.elements, magic);
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }

        // ---- inline_event timing：保留原有 [PERF] 指标；同一 runner 也可被 msprof 包裹 ----
        double kernel_us_acc = 0;
        auto t0_e2e = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < opts.perf_times; i++) {
            magic++;
            staging_copy(input_dev, symm_ptr, input_bytes);
            aclshmem_barrier_all();
            auto t0_kern = std::chrono::high_resolution_clock::now();
            <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
                (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
                opts.elements, magic);
            CHECK_ACL(aclrtSynchronizeStream(stream));
            auto t1_kern = std::chrono::high_resolution_clock::now();
            kernel_us_acc += std::chrono::duration<double, std::micro>(t1_kern - t0_kern).count();
        }
        auto t1_e2e = std::chrono::high_resolution_clock::now();
        double e2e_us = std::chrono::duration<double, std::micro>(t1_e2e - t0_e2e).count()
                        / opts.perf_times;
        double kernel_us = kernel_us_acc / opts.perf_times;

        // ==== 定制：按算子类型设置 bus_factor ====
        // AllReduce: 2*(n-1)/n, ReduceScatter/AllGather: (n-1)/n, AllToAll/Shuffle: (n-1)/n, P2P: 1
        double bus_factor = (double)(n_pes - 1) / n_pes;
        double payload_bytes = (double)input_bytes;
        double algo_bw = payload_bytes / (e2e_us * 1e-6) / 1e9;
        double e2e_bus_bandwidth_GBps  = algo_bw * bus_factor;
        double kernel_algo_bw = payload_bytes / (kernel_us * 1e-6) / 1e9;
        double kernel_bus_bandwidth_GBps  = kernel_algo_bw * bus_factor;
        double peak_bw = 196.0;  // ==== 定制：按通信模式和拓扑调整（P2P: 28, 集合 8卡 full-mesh: 196）====
        double util_pct = kernel_bus_bandwidth_GBps / peak_bw * 100.0;

        if (pe_id == 0) {
            std::cout << "[PERF] source=inline_event"
                      << " warmup=" << opts.warmup
                      << " measure=" << opts.perf_times
                      << " e2e_us=" << e2e_us
                      << " kernel_us=" << kernel_us
                      << " algo_bandwidth_GBps=" << algo_bw
                      << " e2e_bus_bandwidth_GBps=" << e2e_bus_bandwidth_GBps
                      << " kernel_bus_bandwidth_GBps=" << kernel_bus_bandwidth_GBps
                      << " bandwidth_utilization_pct=" << util_pct
                      << " payload_bytes=" << payload_bytes
                      << " bus_factor=" << bus_factor
                      << " peak_bandwidth_GBps=" << peak_bw
                      << std::endl;
        }

        aclshmemx_get_prof(nullptr, true);
    } else {
        staging_copy(input_dev, symm_ptr, input_bytes);
        aclshmem_barrier_all();
        magic++;
        <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
            (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
            opts.elements, magic);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    // ---- 拷回输出并写文件 ----
    std::vector<int32_t> output_host(output_elems);
    CHECK_ACL(aclrtMemcpy(output_host.data(), output_bytes, output_dev, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    std::string output_path = opts.data_dir + "/output_pe" + std::to_string(pe_id) + ".bin";
    WriteBinaryFile(output_path, output_host.data(), output_elems);

    // ---- 清理 ----
    aclshmem_free(symm_ptr);
    CHECK_ACL(aclrtFree(input_dev));
    CHECK_ACL(aclrtFree(output_dev));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_SHMEM(aclshmem_finalize());
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());

    std::cout << "[SUCCESS] <op_name> completed on PE " << pe_id << std::endl;
    return 0;
}
```