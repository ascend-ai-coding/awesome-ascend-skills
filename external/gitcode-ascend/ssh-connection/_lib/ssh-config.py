#!/usr/bin/env python3
"""ssh-config.py — 解析 SSH 连接配置
用法: python3 ssh-config.py <profile_name> [--ssh-opts]
  默认输出 JSON 格式的完整连接信息
  --ssh-opts  输出可直接拼接到 ssh 命令的参数字符串
依赖: PyYAML (pip install pyyaml)
"""

import sys
import os
import json
import platform
import shlex
import yaml

# config.yaml 位于 suite 根目录（_lib 的上级目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SUITE_DIR, "config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 配置文件不存在: {CONFIG_PATH}", file=sys.stderr)
        print("请先运行 agent 引导配置，或复制 config.example.yaml 为 config.yaml", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_profile(config, profile_name):
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(profiles.keys()) if profiles else "无"
        print(f"错误: profile '{profile_name}' 不存在。可用: {available}", file=sys.stderr)
        sys.exit(1)

    defaults = config.get("defaults", {})
    profile = profiles[profile_name]

    result = {
        "host": profile.get("host", ""),
        "port": profile.get("port", defaults.get("port", 22)),
        "username": profile.get("username", ""),
        "identity_file": profile.get("identity_file", ""),
        "password": "",
        "jump_host": profile.get("jump_host", ""),
        "container": profile.get("container", ""),
        "container_runtime": profile.get("container_runtime", "docker"),
        "connect_timeout": profile.get("connect_timeout", defaults.get("connect_timeout", 10)),
        "control_persist": profile.get("control_persist", defaults.get("control_persist", 600)),
        "retry_count": profile.get("retry_count", defaults.get("retry_count", 3)),
    }

    # 从环境变量读取密码
    env_password = profile.get("env_password", "")
    if env_password:
        result["password"] = os.environ.get(env_password, "")

    return result

def to_ssh_opts(profile):
    opts = []
    opts.append(f"-p {profile['port']}")
    opts.append(f"-o ConnectTimeout={profile['connect_timeout']}")

    # ControlMaster 在 Windows 上不可用（无 Unix domain socket 支持）
    if platform.system() != "Windows":
        opts.append(f"-o ControlMaster=auto")
        opts.append(f"-o ControlPath=~/.ssh/sockets/%r@%h-%p")
        opts.append(f"-o ControlPersist={profile['control_persist']}")

    if profile["identity_file"]:
        opts.append(f"-i {shlex.quote(profile['identity_file'])}")
    if profile["jump_host"]:
        opts.append(f"-J {shlex.quote(profile['jump_host'])}")

    opts.append(f"{profile['username']}@{profile['host']}")
    return " ".join(opts)

def main():
    if len(sys.argv) < 2:
        print("用法: python3 ssh-config.py <profile_name> [--ssh-opts]", file=sys.stderr)
        sys.exit(1)

    profile_name = sys.argv[1]
    output_ssh_opts = "--ssh-opts" in sys.argv

    config = load_config()
    profile = resolve_profile(config, profile_name)

    if output_ssh_opts:
        print(to_ssh_opts(profile))
    else:
        print(json.dumps(profile, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
