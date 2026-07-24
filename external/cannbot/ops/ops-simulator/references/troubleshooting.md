# 问题排查指南

## 常见问题

### Q1: 仿真失败，提示设备设置错误

**原因**：仿真环境仅支持单卡场景，代码中只能设置为 0 卡

**解决方案**：检查代码中的 deviceId 设置，确保为 0

```cpp
int32_t deviceId = 0;  // 必须为 0
CHECK_ACL(aclrtSetDevice(deviceId));
```

---

### Q2: 编译成功但仿真运行报错

**原因**：可能未正确设置 LD_LIBRARY_PATH

**解决方案**：
```bash
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
```

---

### Q3: 无法生成性能报告

**原因**：未添加 `--gen-report` 参数

**解决方案**：
```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
```

---

### Q4: 编译选项错误

**错误信息**：
```
error: unrecognized command line option
```

**解决方案**：
检查编译选项是否正确：
```cmake
target_compile_options(ascendc_kernels_bbit PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
)
```

---

## 日志分析

### 日志位置

```
build/cannsim_*/cannsim.log
```

### 常用分析命令

```bash
# 查看完整日志
cat cannsim_*/cannsim.log

# 查看错误信息
grep -i "error" cannsim_*/cannsim.log

# 查看警告信息
grep -i "warning" cannsim_*/cannsim.log
```

---

## 权限问题

### 文件权限要求

- 输入文件要求 other 用户不可写
- 高安全场景需确保 group 用户不可写

### 修复权限

```bash
# 移除 other 写权限
chmod o-w input_file.bin

# 移除 group 写权限（高安全场景）
chmod g-w input_file.bin
```

---

## 获取帮助

如果以上方法无法解决问题：

1. 检查 CANN 版本是否支持当前芯片型号
2. 确认系统架构为 X86_64、ARM，不支持其他架构
3. 查看 CANN 官方文档获取最新信息
