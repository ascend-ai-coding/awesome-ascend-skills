#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
昇腾CANN 8.5.0 NPU驱动/固件包校验脚本
由Shell脚本调用，入参：第1个参数为驱动固件包所在的文件夹完整路径
校验规则：1.文件夹存在 2.含1个符合格式的驱动.run 3.含1个符合格式的固件.run
          4.包为文件而非目录 5.包拥有可执行权限
退出码：0-校验成功 1-入参错误 2-文件夹不存在 3-无驱动包/多个驱动包 4-无固件包/多个固件包
        5-包非文件 6-包无执行权限 7-包命名格式不匹配
"""
import os
import sys
import re

# 定义驱动/固件包的命名正则（严格匹配昇腾指定格式）
DRIVER_PATTERN = re.compile(r"Ascend-hdk-.+-npu-driver_.+_linux-.+\.run")  # 驱动包正则
FIRMWARE_PATTERN = re.compile(r"Ascend-hdk-.+-npu-firmware_.+\.run")      # 固件包正则

def get_matched_pkg(folder_path, pattern):
    """遍历文件夹，返回匹配正则的.run包列表"""
    pkg_list = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and pattern.match(file):
            pkg_list.append(file)
    return pkg_list

def check_npu_pkgs(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"[ERROR] 驱动固件包所在文件夹不存在: {folder_path}", file=sys.stderr)
        return 2

    # 匹配驱动包和固件包
    driver_pkgs = get_matched_pkg(folder_path, DRIVER_PATTERN)
    firmware_pkgs = get_matched_pkg(folder_path, FIRMWARE_PATTERN)

    # 检查驱动包数量（必须唯一）
    if len(driver_pkgs) == 0:
        print(f"[ERROR] 文件夹{folder_path}中未找到符合格式的NPU驱动.run包", file=sys.stderr)
        return 3
    elif len(driver_pkgs) > 1:
        print(f"[ERROR] 文件夹{folder_path}中找到多个驱动包：{driver_pkgs}", file=sys.stderr)
        return 3

    # 检查固件包数量（必须唯一）
    if len(firmware_pkgs) == 0:
        print(f"[ERROR] 文件夹{folder_path}中未找到符合格式的NPU固件.run包", file=sys.stderr)
        return 4
    elif len(firmware_pkgs) > 1:
        print(f"[ERROR] 文件夹{folder_path}中找到多个固件包：{firmware_pkgs}", file=sys.stderr)
        return 4

    # 拼接完整包路径
    driver_pkg = os.path.join(folder_path, driver_pkgs[0])
    firmware_pkg = os.path.join(folder_path, firmware_pkgs[0])
    all_pkgs = [("驱动", driver_pkg), ("固件", firmware_pkg)]

    # 检查包是否为文件、是否有可执行权限
    for pkg_type, pkg_path in all_pkgs:
        if not os.path.isfile(pkg_path):
            print(f"[ERROR] {pkg_type}包非文件（是目录）: {pkg_path}", file=sys.stderr)
            return 5
        if not os.access(pkg_path, os.X_OK):
            print(f"[ERROR] {pkg_type}包无执行权限，请执行chmod +x {pkg_path}", file=sys.stderr)
            return 6

    # 所有校验通过，输出匹配到的包名（供Shell脚本提取）
    print(f"[SUCCESS] 包校验通过！\n驱动包：{driver_pkg}\n固件包：{firmware_pkg}")
    return 0

if __name__ == "__main__":
    # 检查Shell传入的入参（仅接收1个：文件夹路径）
    if len(sys.argv) != 2:
        print(f"[ERROR] 入参错误！仅接收1个参数：包所在文件夹完整路径", file=sys.stderr)
        sys.exit(1)
    # 执行校验
    folder_path = sys.argv[1]
    exit_code = check_npu_pkgs(folder_path)
    sys.exit(exit_code)
