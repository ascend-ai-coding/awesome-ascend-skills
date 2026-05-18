#!/bin/bash
# 压力测试命令模板
# 用法参考: references/parameters/diagnosis.md

# ========== AICORE 压测 ==========

# 单设备 AICORE 压测
ascend-dmi --dg -s -i aicore -d 0 -q

# 多设备 AICORE 压测
ascend-dmi --dg -s -i aicore -d 0,1,2,3 -q


# ========== 片上内存压测 ==========

# 标准片上内存压测
ascend-dmi --dg -s -i hbm -d 0 -q

# 片上内存高危地址压测
ascend-dmi --dg -s -i random -d 0 -q


# ========== P2P 压测 ==========

# P2P 带宽压测
ascend-dmi --dg -s -i bandwidth -d 0 -q


# ========== 功耗压测 ==========

# EDP 功耗压测
ascend-dmi --dg -s -i edp -d 0 -q

# TDP 功耗压测
ascend-dmi --dg -s -i tdp -d 0 -q


# ========== AICPU 压测 ==========

# AICPU 压测（不支持与其他诊断项一起使用）
ascend-dmi --dg -s -i aicpu -d 0 -q


# ========== 一键式压力测试 ==========

# 执行 stressTest 场景全部压测
ascend-dmi --dg --se stressTest -q
