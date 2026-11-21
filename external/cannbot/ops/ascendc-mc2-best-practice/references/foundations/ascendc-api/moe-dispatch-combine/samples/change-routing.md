# MoE Dispatch/Combine 改动定位

当任务已经进入生成 / 改造路径，但还没决定先改哪个文件时，按变化类型定位：

- 构建方式、源文件组织、编译选项变化：先改 `CMakeLists.txt`、`build.sh`
- host 侧输入输出、tiling、launch 参数、通信资源创建方式变化：先改 `test/test_moe_dispatch.cpp`
- 通信资源结构体、tiling 字段、kernel 入参语义变化：先改 `include/tiling_data.h`
- `mc2Context` 字段解释方式、基础地址辅助函数变化：先改 `include/moe_dispatch_base_compat.h`
- window 布局、状态区布局、状态写入/等待协议变化：先改 `kernel/mte_dispatch_comm.h`
- 路由规则、slot 计算、发窗组织、输出回搬、`expandIdx` 写回顺序、输出统计变化：再改 `kernel/moe_dispatch.h`
- kernel 入口签名、模板实例或 launch 包装变化：再改 `moe_dispatch.cpp`
- 输出结果校验规则变化：最后改 `scripts/verify_dispatch.py`

## 避免以下做法

- 不要跳过样例工程直接从空文件开始写 dispatch 工程
- 不要一开始同时重写 host、context 辅助层、comm、kernel 全链路
- 没有明确差异前，不要先改 `kernel/moe_dispatch.h` 主流程