#!/bin/bash
# 带宽测试命令模板
# 用法参考: references/parameters/bandwidth.md

# ========== 基础测试 ==========

# 默认 h2d/d2h/d2d 全测（步长模式）
ascend-dmi --bw -q

# 指定设备的 d2d 测试
ascend-dmi --bw -t d2d -d 0 -q

# 指定设备的 h2d 测试
ascend-dmi --bw -t h2d -d 0 -q

# 指定设备的 d2h 测试
ascend-dmi --bw -t d2h -d 0 -q


# ========== 定长模式测试 ==========

# h2d 定长 128M，迭代 100 次
ascend-dmi --bw -t h2d -d 0 -s 134217728 --et 100 -q

# d2h 定长 128M，迭代 100 次
ascend-dmi --bw -t d2h -d 0 -s 134217728 --et 100 -q

# d2d 定长模式（注意：A2/A3 系列不支持 -s）
ascend-dmi --bw -t d2d -d 0 --et 100 -q


# ========== P2P 测试 ==========

# 全量 P2P 带宽矩阵
ascend-dmi --bw -t p2p -q

# 指定源和目标设备的 P2P 测试
ascend-dmi --bw -t p2p --ds 0 --dd 1 -s 134217728 --et 100 -q


# ========== 多设备测试 ==========

# 测试多个设备
ascend-dmi --bw -t d2d -d 0,1,2 -q
