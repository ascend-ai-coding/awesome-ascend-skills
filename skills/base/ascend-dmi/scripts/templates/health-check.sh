#!/bin/bash
# 健康检查命令模板
# 用法参考: references/parameters/diagnosis.md

# ========== 一键式健康检查 ==========

# 执行 healthCheck 场景全部诊断
ascend-dmi --dg --se healthCheck -q


# ========== 单项诊断 ==========

# 驱动健康诊断
ascend-dmi --dg -i driver -q

# CANN 与驱动兼容性诊断
ascend-dmi --dg -i cann -q

# 芯片诊断
ascend-dmi --dg -i device -d 0 -q

# 网络健康诊断
ascend-dmi --dg -i network -q

# 片上内存诊断
ascend-dmi --dg -i hbm -d 0 -q

# 眼图诊断
ascend-dmi --dg -i signalQuality -d 0 -q


# ========== 组合诊断 ==========

# 软件类诊断
ascend-dmi --dg -i driver,cann -q

# 硬件类诊断
ascend-dmi --dg -i device,network,hbm,signalQuality -d 0 -q


# ========== 保存结果 ==========

# 指定结果保存路径
ascend-dmi --dg --se healthCheck -r /test -q
