#!/bin/bash
# 诊断测试命令模板（性能规格）
# 用法参考: references/parameters/diagnosis.md

# ========== 一键式性能规格测试 ==========

# 执行 performanceCheck 场景全部诊断
ascend-dmi --dg --se performanceCheck -q


# ========== 单项诊断 ==========

# 带宽诊断
ascend-dmi --dg -i bandwidth -d 0 -q

# 算力诊断
ascend-dmi --dg -i aiflops -d 0 -q

# NIC 诊断
ascend-dmi --dg -i nic -q

# PRBS 码流诊断
ascend-dmi --dg -i prbs -d 0 -q


# ========== AICORE 诊断 ==========

# AICORE 诊断（非压测）
ascend-dmi --dg -i aicore -d 0 -q


# ========== 组合诊断 ==========

# 全部性能规格诊断
ascend-dmi --dg -i bandwidth,aiflops,nic -d 0 -q


# ========== 一键式全场景诊断 ==========

# 执行全部诊断场景
ascend-dmi --dg --se healthCheck,performanceCheck,stressTest -q
