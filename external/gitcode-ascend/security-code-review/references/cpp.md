# C++ 安全审查详细参考

> 本文件包含 C++ 安全审查的完整代码示例和审查要点。由 SKILL.md 按需引用。

## 目录

1. [缓冲区溢出](#1-缓冲区溢出)
2. [内存管理](#2-内存管理)
3. [整数溢出](#3-整数溢出)
4. [格式化字符串漏洞](#4-格式化字符串漏洞)
5. [未初始化变量](#5-未初始化变量)
6. [RAII 与资源泄漏](#6-raii-与资源泄漏)
7. [线程安全](#7-线程安全)
8. [类型转换安全](#8-类型转换安全)
9. [JSON 请求嵌套深度校验](#9-json-请求嵌套深度校验)
10. [服务化请求资源上限校验](#10-服务化请求资源上限校验配置参数组合内存炸弹)
11. [特殊 Token 注入与多模态输入校验](#11-特殊-token-注入与多模态输入校验)

---

## 1. 缓冲区溢出

```cpp
// ❌ 不安全：未检查边界
char buf[64];
strcpy(buf, user_input);  // 缓冲区溢出
sprintf(buf, "Hello %s", user_input);  // 同样不安全

// ✅ 安全：使用安全函数或 std::string
std::string buf(user_input);  // 自动管理内存

// 如果必须用 C 字符串：
char buf[64];
strncpy(buf, user_input, sizeof(buf) - 1);
buf[sizeof(buf) - 1] = '\0';
snprintf(buf, sizeof(buf), "Hello %s", user_input);
```

## 2. 内存管理

```cpp
// ❌ 不安全：裸指针 + 手动内存管理
int* ptr = new int[100];
// ... 异常发生 → 内存泄漏
delete[] ptr;  // 可能不会执行

// ❌ 不安全：use-after-free
int* p = new int(42);
delete p;
*p = 10;  // 未定义行为

// ❌ 不安全：double-free
delete p;
delete p;  // 未定义行为

// ✅ 安全：使用智能指针
auto ptr = std::make_unique<int[]>(100);  // 自动释放
auto shared = std::make_shared<MyClass>();  // 共享所有权
```

## 3. 整数溢出

```cpp
// ❌ 不安全：未检查整数溢出
int size = get_user_size();  // 可能为负数或极大值
char* buf = new char[size];  // 整数溢出 → 分配小缓冲区

// ✅ 安全：检查范围
size_t size = get_user_size();
if (size == 0 || size > MAX_ALLOWED_SIZE) {
    throw std::invalid_argument("Invalid size");
}
auto buf = std::make_unique<char[]>(size);
```

## 4. 格式化字符串漏洞

```cpp
// ❌ 不安全：用户控制格式字符串
printf(user_input);  // 格式化字符串攻击，可读写内存
fprintf(stderr, user_input);

// ✅ 安全：始终使用固定格式字符串
printf("%s", user_input);
fprintf(stderr, "%s", user_input);

// ✅ 更好：使用 C++ 流
std::cout << user_input << std::endl;
// 或 C++20 std::format
std::string msg = std::format("User: {}", user_input);
```

## 5. 未初始化变量

```cpp
// ❌ 不安全：未初始化变量
int status;  // 未初始化
if (condition) {
    status = 0;
}
return status;  // 未初始化时为未定义行为

// ✅ 安全：始终初始化
int status = -1;
if (condition) {
    status = 0;
}
return status;
```

## 6. RAII 与资源泄漏

```cpp
// ❌ 不安全：手动资源管理
FILE* fp = fopen("data.txt", "r");
// ... 如果异常发生，fp 不会被关闭
fclose(fp);

std::mutex mtx;
mtx.lock();
// ... 如果异常发生，锁不会释放
mtx.unlock();

// ✅ 安全：RAII
{
    std::ifstream file("data.txt");  // 析构时自动关闭
    // ...
}

{
    std::lock_guard<std::mutex> lock(mtx);  // 析构时自动释放
    // ...
}
```

## 7. 线程安全

```cpp
// ❌ 不安全：数据竞争
int counter = 0;
void increment() { counter++; }  // 多线程下数据竞争

// ✅ 安全：使用原子操作或锁
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1); }

// 或使用互斥锁
std::mutex mtx;
int counter = 0;
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    counter++;
}
```

## 8. 类型转换安全

```cpp
// ❌ 不安全：C 风格强制转换（向下转型无类型检查）
Derived* derived = (Derived*)base;   // 不安全：base 可能并非 Derived 类型，无运行时检查
int* int_ptr = (int*)void_ptr;       // 不安全：void_ptr 实际类型未知，可能导致类型混淆

// ✅ 安全：使用 C++ 类型转换
auto* derived = dynamic_cast<Derived*>(base);  // 运行时类型检查
if (derived == nullptr) {
    // 处理转换失败
}
auto value = static_cast<int>(float_value);  // 明确意图
```

## 9. JSON 请求嵌套深度校验

```cpp
// ❌ 不安全：直接解析未限制深度的 JSON（nlohmann::json 默认无深度限制）
#include <nlohmann/json.hpp>

void handleRequest(const std::string& rawBody) {
    auto data = nlohmann::json::parse(rawBody);  // 恶意深层嵌套可导致栈溢出崩溃
    process(data);
}

// ❌ 不安全：递归遍历 JSON 无深度保护
void walk(const nlohmann::json& j) {
    if (j.is_object()) {
        for (auto& [k, v] : j.items()) {
            walk(v);  // 深层嵌套导致栈溢出
        }
    } else if (j.is_array()) {
        for (auto& item : j) {
            walk(item);  // 同上
        }
    }
}
```

```cpp
// ✅ 安全：解析前扫描嵌套深度（O(n) 时间，O(1) 空间）
#include <stdexcept>
#include <string>

constexpr int MAX_JSON_DEPTH = 32;

void checkJsonDepth(const std::string& raw, int maxDepth = MAX_JSON_DEPTH) {
    int depth = 0;
    bool inString = false;
    bool escape = false;
    for (char ch : raw) {
        if (escape) { escape = false; continue; }
        if (ch == '\\') { escape = true; continue; }
        if (ch == '"') { inString = !inString; continue; }
        if (inString) continue;
        if (ch == '{' || ch == '[') {
            if (++depth > maxDepth) {
                throw std::invalid_argument(
                    "JSON nesting depth " + std::to_string(depth) +
                    " exceeds maximum allowed " + std::to_string(maxDepth));
            }
        } else if (ch == '}' || ch == ']') {
            --depth;
        }
    }
}

void handleRequest(const std::string& rawBody) {
    checkJsonDepth(rawBody);  // 先检查深度
    auto data = nlohmann::json::parse(rawBody);  // 再解析
    process(data);
}
```

```cpp
// ✅ 安全：递归遍历时传递深度计数器
void walk(const nlohmann::json& j, int depth = 0) {
    constexpr int MAX_DEPTH = 32;
    if (depth > MAX_DEPTH) {
        throw std::runtime_error("JSON traversal depth exceeds limit");
    }
    if (j.is_object()) {
        for (auto& [k, v] : j.items()) {
            walk(v, depth + 1);
        }
    } else if (j.is_array()) {
        for (auto& item : j) {
            walk(item, depth + 1);
        }
    }
}
```

```cpp
// ✅ 安全：使用 RapidJSON 自定义 Handler 实现深度限制
// 注意：RapidJSON 标准 API 默认不限制解析深度，需通过自定义 Handler 实现
#include <rapidjson/document.h>
#include <rapidjson/reader.h>

template <unsigned MaxDepth = 32>
struct DepthLimitedHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>> {
    unsigned depth = 0;
    bool StartObject() { return ++depth <= MaxDepth; }
    bool EndObject(rapidjson::SizeType) { --depth; return true; }
    bool StartArray() { return ++depth <= MaxDepth; }
    bool EndArray(rapidjson::SizeType) { --depth; return true; }
    // ... 其他 handler 方法
};

void handleRequest(const std::string& rawBody) {
    DepthLimitedHandler<32> handler;
    rapidjson::Reader reader;
    rapidjson::StringStream ss(rawBody.c_str());
    auto result = reader.Parse(ss, handler);
    if (result.IsError()) {
        throw std::invalid_argument("JSON parse failed: depth exceeded or malformed");
    }
}
```

**审查要点：**
- 搜索所有 `nlohmann::json::parse()`、`rapidjson::Document::Parse()`、`Json::Reader::parse()` 调用
- 检查 HTTP/gRPC 请求解析入口是否在解析前校验嵌套深度
- 检查递归遍历 JSON 的函数是否传递了深度参数并设定上限
- 注意栈大小限制：Linux 默认线程栈 8MB，深嵌套 JSON 解析每层约消耗数百字节栈空间，超过 ~10000 层即可能崩溃
- 相关漏洞标准：CWE-674 (Uncontrolled Recursion)、CWE-400 (Uncontrolled Resource Consumption)、CWE-120 (Stack-based Buffer Overflow via recursion)

## 10. 服务化请求资源上限校验（配置参数组合内存炸弹）

```cpp
// ❌ 不安全：各配置参数独立设置，未校验组合后的总资源消耗
// 典型推理服务配置：
constexpr int MAX_REQS = 10000;              // 最大并发请求数
constexpr size_t BODY_LIMIT = 10 * 1024 * 1024;  // 单请求体上限 10MB
constexpr size_t HEADER_LIMIT = 8 * 1024;    // 请求头上限 8KB

// 问题：nlohmann::json 将字符串解析为 JSON 结构体时，内存放大约 33 倍
// 最坏情况峰值内存 = MAX_REQS × BODY_LIMIT × JSON放大系数
// = 10000 × 10MB × 33 ≈ 3.3TB >> Pod 可用内存（通常 64~512GB）
// 攻击者构造大量合法但体积接近上限的请求，即可耗尽内存导致 OOM 服务崩溃

void startServer() {
    server.setMaxRequests(MAX_REQS);          // 单独看合理
    server.setBodyLimit(BODY_LIMIT);          // 单独看合理
    server.setHeaderLimit(HEADER_LIMIT);      // 单独看合理
    // 但三者组合后的峰值内存远超物理内存 → 服务挂掉
}
```

```cpp
// ✅ 安全：基于可用内存反推配置参数的安全上限
#include <sys/sysinfo.h>  // Linux
#include <algorithm>

// 方法 1：启动时校验配置参数组合是否超出内存预算
struct ServerResourceConfig {
    size_t maxReqs;
    size_t bodyLimit;
    size_t headerLimit;
    double jsonAmplificationFactor;  // JSON 解析内存放大系数

    size_t estimatePeakMemory() const {
        // 峰值内存 = 并发请求数 × (请求体 × JSON放大 + 请求头 + 每请求固定开销)
        const size_t perRequestOverhead = 4096;  // 连接管理、上下文等固定开销
        return maxReqs * (
            static_cast<size_t>(bodyLimit * jsonAmplificationFactor)
            + headerLimit
            + perRequestOverhead
        );
    }

    bool validate(size_t availableMemoryBytes, double maxUsageRatio = 0.6) const {
        size_t peakMem = estimatePeakMemory();
        size_t memBudget = static_cast<size_t>(availableMemoryBytes * maxUsageRatio);
        if (peakMem > memBudget) {
            LOG_ERROR("Resource config UNSAFE: estimated peak memory %.2f GB "
                      "exceeds budget %.2f GB (%.0f%% of %.2f GB available). "
                      "maxReqs=%zu, bodyLimit=%zu, jsonAmplification=%.1fx",
                      peakMem / 1e9, memBudget / 1e9,
                      maxUsageRatio * 100,
                      availableMemoryBytes / 1e9,
                      maxReqs, bodyLimit, jsonAmplificationFactor);
            return false;
        }
        LOG_INFO("Resource config OK: peak memory %.2f GB within budget %.2f GB",
                 peakMem / 1e9, memBudget / 1e9);
        return true;
    }
};

void startServer(const ServerResourceConfig& config) {
    // 获取可用内存
    struct sysinfo si;
    sysinfo(&si);
    size_t availableMem = si.totalram * si.mem_unit;

    // 启动前校验：峰值内存不得超过可用内存的 60%
    if (!config.validate(availableMem, 0.6)) {
        throw std::runtime_error("Server config exceeds safe memory limits, "
                                 "reduce maxReqs or bodyLimit");
    }
    // ... 启动服务
}
```

```cpp
// ✅ 安全：方法 2 — 基于内存预算反推 maxReqs 安全值
size_t computeSafeMaxReqs(size_t availableMemory,
                          size_t bodyLimit,
                          double jsonAmplification,
                          double maxUsageRatio = 0.6) {
    size_t memBudget = static_cast<size_t>(availableMemory * maxUsageRatio);
    size_t perReqMem = static_cast<size_t>(bodyLimit * jsonAmplification) + 8192;
    size_t safeMax = memBudget / perReqMem;
    LOG_INFO("Safe maxReqs = %zu (memBudget=%.2fGB, perReq=%.2fMB)",
             safeMax, memBudget / 1e9, perReqMem / 1e6);
    return safeMax;
}

// 使用：让系统自动计算安全上限
size_t maxReqs = std::min(userConfigMaxReqs,
                          computeSafeMaxReqs(availableMem, bodyLimit, 33.0));
```

```cpp
// ✅ 安全：方法 3 — 运行时动态内存水位监控 + 请求准入控制
#include <atomic>
#include <fstream>
#include <string>

class MemoryGuard {
public:
    static constexpr double HIGH_WATERMARK = 0.80;  // 内存使用超 80% 开始拒绝新请求
    static constexpr double LOW_WATERMARK  = 0.60;  // 降到 60% 恢复接收

    // 检查当前系统内存使用率（Linux /proc/meminfo）
    static double getMemoryUsageRatio() {
        std::ifstream meminfo("/proc/meminfo");
        size_t totalMem = 0, availMem = 0;
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0)
                totalMem = std::stoull(line.substr(10)) * 1024;
            else if (line.find("MemAvailable:") == 0)
                availMem = std::stoull(line.substr(14)) * 1024;
        }
        return (totalMem > 0) ? 1.0 - static_cast<double>(availMem) / totalMem : 1.0;
    }

    // 请求准入决策
    bool admitRequest() {
        double usage = getMemoryUsageRatio();
        if (usage > HIGH_WATERMARK) {
            rejectMode_.store(true, std::memory_order_release);
            LOG_WARN("Memory usage %.1f%% > %.1f%%, rejecting new requests",
                     usage * 100, HIGH_WATERMARK * 100);
            return false;  // 返回 HTTP 503 Service Unavailable
        }
        if (usage < LOW_WATERMARK) {
            rejectMode_.store(false, std::memory_order_release);
        }
        return !rejectMode_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> rejectMode_{false};
};

// 在 HTTP handler 入口处调用
Status handleInferenceRequest(const Request& req) {
    if (!memoryGuard.admitRequest()) {
        return Status(503, "Server memory pressure, try again later");
    }
    auto body = nlohmann::json::parse(req.body());  // 准入通过后才解析
    // ... 推理处理
}
```

**审查要点：**
- **核心公式**：`峰值内存 = maxReqs × bodyLimit × JSON放大系数 + 基础内存开销`
- 搜索所有 HTTP/gRPC 服务的配置参数：`maxReqs`/`maxConnections`、`bodyLimit`/`maxBodySize`、`headerLimit` 等
- 计算各参数组合后的**最坏情况峰值内存**，与部署目标机器/Pod 的内存做对比
- 检查是否存在**启动时校验**：配置加载后是否验证组合参数不超出内存预算
- 检查是否存在**运行时保护**：内存水位监控、请求准入控制、背压机制
- JSON 解析库的内存放大系数参考：nlohmann::json ≈ 20~40x，RapidJSON (DOM) ≈ 4~8x，simdjson (on-demand) ≈ 0x（流式解析无放大）
- 不仅是 JSON，也适用于 XML/Protobuf/MessagePack 等所有需要将网络字节流反序列化为内存结构体的场景
- 相关漏洞标准：CWE-400 (Uncontrolled Resource Consumption)、CWE-770 (Allocation of Resources Without Limits)、CWE-789 (Memory Allocation with Excessive Size Value)

## 11. 特殊 Token 注入与多模态输入校验

```cpp
// ❌ 不安全：假设特殊 Token 成对出现，直接用索引取配对位置
void processMultimodalInput(const std::vector<int>& inputIds,
                            const ModelConfig& config) {
    auto boiPositions = findTokenPositions(inputIds, config.boiTokenId);
    auto eoiPositions = findTokenPositions(inputIds, config.eoiTokenId);

    for (size_t i = 0; i < boiPositions.size(); ++i) {
        // 致命：若 eoi 缺失或数量不匹配，eoiPositions[i] 越界 → 崩溃
        size_t eoiPos = eoiPositions[i];
        processImage(inputIds, boiPositions[i], eoiPos);
    }
}

// ❌ 不安全：假设 vision_start 后的序列一定遵循内部协议格式
void processVisionSequence(const std::vector<int>& inputIds,
                           size_t startPos,
                           const ModelConfig& config) {
    // 假设 startPos+1 是 image_count, startPos+2 是 width, ...
    int imageCount = inputIds[startPos + 1];   // ← 可能越界
    int width      = inputIds[startPos + 2];   // ← 可能越界
    int height     = inputIds[startPos + 3];   // ← 可能越界
    // 用户发送 <|vision_start|><|video_pad|><|vision_end|> 代替预期的 image_pad 序列
    // 偏移计算全部错误 → 越界访问 → 段错误 → 服务崩溃
}
```

```cpp
// ✅ 安全：严格校验特殊 Token 配对完整性

#include <vector>
#include <stdexcept>
#include <utility>

struct TokenPair {
    size_t beginPos;
    size_t endPos;
};

std::vector<TokenPair> validateSpecialTokenPairs(
    const std::vector<int>& inputIds,
    int beginTokenId,
    int endTokenId,
    const std::string& tokenName)
{
    std::vector<size_t> beginPositions, endPositions;
    for (size_t i = 0; i < inputIds.size(); ++i) {
        if (inputIds[i] == beginTokenId) beginPositions.push_back(i);
        else if (inputIds[i] == endTokenId) endPositions.push_back(i);
    }

    // 校验 1：数量必须相等
    if (beginPositions.size() != endPositions.size()) {
        throw std::invalid_argument(
            tokenName + " token count mismatch: " +
            std::to_string(beginPositions.size()) + " begin vs " +
            std::to_string(endPositions.size()) + " end");
    }

    std::vector<TokenPair> pairs;
    for (size_t i = 0; i < beginPositions.size(); ++i) {
        // 校验 2：end 必须在 begin 之后
        if (endPositions[i] <= beginPositions[i]) {
            throw std::invalid_argument(
                tokenName + " pair " + std::to_string(i) +
                ": end not after begin");
        }
        pairs.push_back({beginPositions[i], endPositions[i]});
    }

    // 校验 3：相邻对不交叉
    for (size_t i = 0; i + 1 < pairs.size(); ++i) {
        if (pairs[i].endPos > pairs[i + 1].beginPos) {
            throw std::invalid_argument(
                tokenName + " pairs " + std::to_string(i) +
                " and " + std::to_string(i + 1) + " overlap");
        }
    }
    return pairs;
}
```

```cpp
// ✅ 安全：校验序列内容类型，不信任隐式格式假设

void parseVisionSequenceSafe(const std::vector<int>& inputIds,
                             size_t startPos,
                             size_t endPos,
                             const ModelConfig& config) {
    if (endPos <= startPos + 1) {
        throw std::invalid_argument("Empty vision sequence");
    }

    // 校验序列中每个 Token 是否在合法集合内（白名单）
    for (size_t i = startPos + 1; i < endPos; ++i) {
        int tokenId = inputIds[i];
        if (tokenId != config.imagePadId &&
            tokenId != config.videoPadId &&
            tokenId != config.audioPadId) {
            throw std::invalid_argument(
                "Illegal token " + std::to_string(tokenId) +
                " in vision sequence at position " + std::to_string(i));
        }
    }
    // 按实际内容分派，不硬假设格式
    // ...
}
```

```cpp
// ✅ 安全：框架层兜底异常捕获

Status handleInferenceRequest(const Request& req) {
    try {
        auto inputIds = tokenize(req.prompt());
        // 多模态 Token 校验 — 在 model forward 之前
        validateMultimodalTokens(inputIds, modelConfig);
        auto result = model.forward(inputIds);
        return SuccessResponse(result);
    } catch (const std::invalid_argument& e) {
        LOG_WARN("Malformed multimodal input rejected: %s", e.what());
        return ErrorResponse(400, std::string("Bad request: ") + e.what());
    } catch (const std::out_of_range& e) {
        LOG_ERROR("Input validation gap (out_of_range): %s", e.what());
        return ErrorResponse(400, "Malformed input");
    } catch (const std::exception& e) {
        // 最终兜底：任何未预期异常不导致进程退出
        LOG_ERROR("Unexpected error: %s", e.what());
        return ErrorResponse(500, "Internal server error");
    }
}
```

**审查要点：**
- 搜索所有通过 Token ID 查找位置的代码（`std::find`、`findTokenPositions`、循环比较等），检查结果为空或数量不匹配时是否安全处理
- 检查多模态 Token 处理是否对序列格式做了**隐式假设**（如"start 后一定跟 N 个 pad Token"），这些假设能否被用户构造的非预期 Token 组合打破
- 检查 model forward / pre-processing 路径上是否有 `std::out_of_range`、`std::invalid_argument` 等异常的**框架层 catch 兜底**
- 相关漏洞标准：CWE-129 (Improper Validation of Array Index)、CWE-248 (Uncaught Exception)、CWE-20 (Improper Input Validation)

---

## C++ 安全工具

| 工具 | 用途 | 命令/用法 |
|------|------|----------|
| **AddressSanitizer** | 内存错误检测 | `-fsanitize=address` |
| **ThreadSanitizer** | 数据竞争检测 | `-fsanitize=thread` |
| **UBSanitizer** | 未定义行为检测 | `-fsanitize=undefined` |
| **Valgrind** | 内存泄漏检测 | `valgrind --leak-check=full ./app` |
| **cppcheck** | 静态分析 | `cppcheck --enable=all src/` |
| **clang-tidy** | Linter + 安全规则 | `clang-tidy -checks='*' src/*.cpp` |
| **Coverity** | 企业级静态分析 | CI 集成 |
