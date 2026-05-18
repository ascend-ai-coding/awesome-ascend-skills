#!/usr/bin/env python3
"""
故障诊断日志采集脚本
用于在多节点集群上采集训练及推理前/中/后的日志
"""

import argparse
import os
import re
import shlex
import subprocess
import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

IP_FIELD = "ip"
FILES_FIELD = "files"
ERROR_FIELD = "errors"
PASSWORD_FIELD = "password"
KEY_FILE_FIELD = "key_file"
O_FIELD = "-o"

ALLOWED_COMMANDS = {
    "ascend-fd version",
    "hccn_tool",
    "npu-smi",
    "cat /proc/cpuinfo",
    "dmesg",
    "cat /var/log",
    "msnpureport",
    "ls /var/log",
}

SENSITIVE_PATTERNS = [
    r"password\s*=\s*\S+",
    r"passwd\s*=\s*\S+",
    r"secret\s*=\s*\S+",
    r"key\s*=\s*\S+",
    r"token\s*=\s*\S+",
]


def setup_logging(log_file: str = None) -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger("LogCollector")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_utc8_timestamp() -> str:
    """获取 UTC+8 时区的时间戳"""
    utc8 = ZoneInfo("Asia/Shanghai")
    return datetime.datetime.now(utc8).strftime("%Y%m%d_%H%M%S")


def validate_ip(ip: str) -> bool:
    """验证 IP 地址格式"""
    if not ip or not isinstance(ip, str):
        return False
    pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(pattern, ip):
        return False
    parts = ip.split(".")
    return all(0 <= int(part) <= 255 for part in parts)


def validate_username(username: str) -> bool:
    """验证用户名格式"""
    if not username or not isinstance(username, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", username)) and len(username) <= 32


def validate_password(password: str) -> bool:
    """验证密码安全性"""
    if not password or not isinstance(password, str):
        return False
    if len(password) > 256:
        return False
    dangerous_chars = ['\x00', '\n', '\r', "'", '"', '\\', '`', '$', '|', ';', '&', '(', ')', '<', '>']
    if any(char in password for char in dangerous_chars):
        return False
    return True


def validate_path(path: str) -> bool:
    """验证路径安全性"""
    if not path or not isinstance(path, str):
        return False
    if ".." in path:
        return False
    if any(char in path for char in ["|", ";", "&", "$", "`", "(", ")"]):
        return False
    try:
        Path(path).resolve()
        return True
    except (ValueError, OSError):
        return False


def sanitize_string(s: str, max_length: int = 256) -> str:
    """清理字符串，移除危险字符"""
    if not s:
        return ""
    s = str(s)[:max_length]
    s = re.sub(r"[;&|`$(){}[\]]", "", s)
    return s.strip()


def filter_sensitive_info(text: str) -> str:
    """过滤敏感信息"""
    if not text:
        return text
    for pattern in SENSITIVE_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def is_command_allowed(command: str) -> bool:
    """检查命令是否在允许列表中"""
    if not command:
        return False
    command_lower = command.lower()
    for allowed in ALLOWED_COMMANDS:
        if command_lower.startswith(allowed.lower()):
            return True
    return False


def validate_node_config(node: Dict, logger: logging.Logger) -> Tuple[bool, str]:
    """验证节点配置"""
    if not isinstance(node, dict):
        return False, "Node config must be a dictionary"
    
    ip = node.get(IP_FIELD)
    if not validate_ip(ip):
        return False, f"Invalid IP address: {ip}"
    
    user = node.get("user", "root")
    if not validate_username(user):
        return False, f"Invalid username: {user}"
    
    password = node.get(PASSWORD_FIELD)
    if password and not validate_password(password):
        return False, "Invalid password: contains dangerous characters"
    
    key_file = node.get(KEY_FILE_FIELD)
    if key_file and not validate_path(key_file):
        return False, f"Invalid key_file path: {key_file}"
    
    name = node.get("name", ip)
    if name and not isinstance(name, str):
        return False, f"Invalid node name: {name}"
    
    return True, ""


class LogCollector:
    """日志采集器"""
    
    ASCEND_FD_INSTALL_URL = "https://www.hiascend.com/document/detail/zh/mindcluster/730/faultdiag/faultdiagug/mindxdlFDUG008.html"
    
    def __init__(self, output_dir: str, nodes: List[Dict], logger: logging.Logger = None):
        self.output_dir = Path(output_dir).resolve()
        self.nodes = self._validate_nodes(nodes)
        self.timestamp = get_utc8_timestamp()
        self.logger = logger or logging.getLogger("LogCollector")
        
    def check_ascend_fd_installed(self, node: Dict) -> Dict:
        """检查节点是否安装了 ascend-fd"""
        ip = node.get(IP_FIELD)
        node_name = node.get("name", ip)
        
        returncode, stdout, stderr = self.ssh_execute(node, "ascend-fd version")
        
        res = {
            "node": node_name,
            IP_FIELD: ip,
            "installed": False,
            "version": None,
        }
        if returncode == 0 and stdout.strip():
            version = stdout.strip().split("\n")[0] if stdout.strip() else "unknown"
            res["installed"] = True
            res["version"] = sanitize_string(version)
        else:
            res["message"] = f"ascend-fd 未安装，请参考以下链接安装：\n{self.ASCEND_FD_INSTALL_URL}"
        return res
    
    def check_all_nodes_ascend_fd(self) -> List[Dict]:
        """检查所有节点是否安装了 ascend-fd"""
        results = []
        for node in self.nodes:
            self.logger.info(f"Checking ascend-fd on node: {node.get(IP_FIELD)}")
            result = self.check_ascend_fd_installed(node)
            results.append(result)
            if not result["installed"]:
                self.logger.warning(f"  ascend-fd not installed")
            else:
                self.logger.info(f"  ascend-fd version: {result['version']}")
        return results
        
    def ssh_execute(self, node: Dict, command: str) -> Tuple[int, str, str]:
        """通过 SSH 执行命令（安全版本）"""
        ip = node.get(IP_FIELD)
        user = node.get("user", "root")
        key_file = node.get(KEY_FILE_FIELD)
        password = node.get(PASSWORD_FIELD)
        
        if not is_command_allowed(command):
            self.logger.error(f"Command not allowed: {filter_sensitive_info(command)}")
            return -1, "", "Command not allowed"
        
        ssh_cmd = []
        
        if password:
            ssh_cmd.extend(["sshpass", "-p", password])
        
        ssh_cmd.extend([
            "ssh",
            O_FIELD, "StrictHostKeyChecking=no",
            O_FIELD, "ConnectTimeout=30",
            O_FIELD, "BatchMode=no" if password else "BatchMode=yes",
        ])
        
        if key_file and validate_path(key_file):
            ssh_cmd.extend(["-i", key_file])
        
        ssh_cmd.append(f"{user}@{ip}")
        ssh_cmd.append(command)
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                shell=False,
            )
            stdout = filter_sensitive_info(result.stdout)
            stderr = filter_sensitive_info(result.stderr)
            return result.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"SSH connection timeout to {ip}")
            return -1, "", "SSH connection timeout"
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SSH execution failed: {filter_sensitive_info(str(e))}")
            return -1, "", f"SSH error: {e.returncode}"
        except FileNotFoundError:
            self.logger.error("sshpass not found. Please install: apt-get install sshpass")
            return -1, "", "sshpass not installed"
        except Exception as e:
            self.logger.error(f"SSH execution failed: {filter_sensitive_info(str(e))}")
            return -1, "", "SSH execution failed"
    
    def scp_download(self, node: Dict, remote_path: str, local_path: str) -> bool:
        """通过 SCP 下载文件（安全版本）"""
        if not validate_path(remote_path) or not validate_path(local_path):
            self.logger.error("Invalid path for SCP download")
            return False
        
        ip = node.get(IP_FIELD)
        user = node.get("user", "root")
        key_file = node.get(KEY_FILE_FIELD)
        password = node.get(PASSWORD_FIELD)
        
        scp_cmd = []
        
        if password:
            scp_cmd.extend(["sshpass", "-p", password])
        
        scp_cmd.extend([
            "scp",
            O_FIELD, "StrictHostKeyChecking=no",
            O_FIELD, "ConnectTimeout=30",
        ])
        
        if key_file and validate_path(key_file):
            scp_cmd.extend(["-i", key_file])
        
        scp_cmd.append(f"{user}@{ip}:{remote_path}")
        scp_cmd.append(local_path)
        
        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                timeout=600,
                shell=False,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.error(f"SCP download timeout from {ip}")
            return False
        except FileNotFoundError:
            self.logger.error("sshpass not found. Please install: apt-get install sshpass")
            return False
        except Exception as e:
            self.logger.error(f"SCP download failed: {filter_sensitive_info(str(e))}")
            return False
    
    def collect_before_training(self, node: Dict) -> Dict:
        """采集训练及推理前日志"""
        ip = node.get(IP_FIELD)
        node_name = node.get("name", ip)
        node_dir = self.output_dir / sanitize_string(node_name) / "environment_check"
        
        results = {"node": node_name, IP_FIELD: ip, FILES_FIELD: [], ERROR_FIELD: []}
        
        commands = {
            "net_health": "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -net_health -g",
            "link": "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -link -g",
            IP_FIELD: "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -ip -g",
            "stat": "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -stat -g",
            "link_stat": "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -link_stat -g",
            "npu_info": "/usr/local/bin/npu-smi info",
            "npu_ecc": "/usr/local/bin/npu-smi info -i 0 -t ecc",
            "npu_board": "/usr/local/bin/npu-smi info -i 0 -t board",
            "npu_usages": "/usr/local/bin/npu-smi info -i 0 -t usages",
            "npu_health": "/usr/local/bin/npu-smi info -i 0 -c 0 -t health",
        }
        
        output_lines = [
            f"# NPU Environment Check - Before Training",
            f"# Node: {node_name}, IP: {ip}",
            f"# Time: {self.timestamp}",
            "",
        ]
        
        for name, cmd in commands.items():
            output_lines.append(f"# Command: {cmd}")
            returncode, stdout, stderr = self.ssh_execute(node, cmd)
            if returncode == 0:
                output_lines.append(stdout)
                output_lines.append("")
            else:
                output_lines.append(f"Error: {stderr}")
                output_lines.append("")
                results[ERROR_FIELD].append({"command": name, "error": stderr})
                self.logger.warning(f"Command '{name}' failed on {ip}: {stderr}")
        
        output_file = node_dir / "npu_info_before.txt"
        if self._safe_write_file(output_file, "\n".join(output_lines)):
            results[FILES_FIELD].append(str(output_file))
        
        self.logger.debug(f"Collected before-training logs from {ip}")
        return results
    
    def collect_during_training(self, node: Dict) -> Dict:
        """采集训练及推理中日志"""
        ip = node.get(IP_FIELD)
        node_name = node.get("name", ip)
        node_dir = self.output_dir / sanitize_string(node_name) / "environment_check"
        
        results = {"node": node_name, IP_FIELD: ip, FILES_FIELD: [], ERROR_FIELD: []}
        
        returncode, stdout, stderr = self.ssh_execute(node, "cat /proc/cpuinfo | grep 'processor' | wc -l")
        core_num = stdout.strip() if returncode == 0 else "0"
        
        npu_stat_cmd = "/usr/local/Ascend/driver/tools/hccn_tool -i 0 -stat -g"
        output_file = node_dir / "npu_0_details.csv"
        
        returncode, stdout, stderr = self.ssh_execute(node, npu_stat_cmd)
        if returncode == 0:
            content = "timestamp," + stdout.replace(":", ",").replace("\n", "").replace(" ", "")
            if self._safe_write_file(output_file, content):
                results[FILES_FIELD].append(str(output_file))
        
        npu_smi_cmd = "/usr/local/bin/npu-smi info -t common -i 0"
        output_file = node_dir / "npu_smi_0_details.csv"
        
        returncode, stdout, stderr = self.ssh_execute(node, npu_smi_cmd)
        if returncode == 0:
            content = f"time,dev_id,hbm_rate,aicore_rate,rated_freq,freq,temp,power\n{self.timestamp},0,0,0,0,0,0,0\n"
            if self._safe_write_file(output_file, content):
                results[FILES_FIELD].append(str(output_file))
        
        host_metrics_file = node_dir / f"host_metrics_{sanitize_string(core_num)}.json"
        if self._safe_write_file(host_metrics_file, json.dumps({"node_mem_used": [], "node_rss": [], "node_cpu": []})):
            results[FILES_FIELD].append(str(host_metrics_file))
        
        self.logger.debug(f"Collected during-training logs from {ip}")
        return results
    
    def collect_after_training(self, node: Dict) -> Dict:
        """采集训练及推理后日志"""
        ip = node.get(IP_FIELD)
        node_name = node.get("name", ip)
        results = {"node": node_name, IP_FIELD: ip, FILES_FIELD: [], ERROR_FIELD: []}
        
        before_results = self.collect_before_training(node)
        results[FILES_FIELD].extend(before_results[FILES_FIELD])
        results[ERROR_FIELD].extend(before_results[ERROR_FIELD])
        
        node_dir = self.output_dir / sanitize_string(node_name)
        
        commands = {
            "dmesg": "dmesg -T | tail -n 100000",
            "messages": "cat /var/log/messages 2>/dev/null || echo 'No messages log'",
            "sysmonitor": "cat /var/log/sysmonitor.log 2>/dev/null || echo 'No sysmonitor log'",
        }
        
        for name, cmd in commands.items():
            output_file = node_dir / name
            returncode, stdout, stderr = self.ssh_execute(node, cmd)
            if returncode == 0:
                if self._safe_write_file(output_file, stdout):
                    results[FILES_FIELD].append(str(output_file))
            else:
                results[ERROR_FIELD].append({"command": name, "error": stderr})
                self.logger.warning(f"Command '{name}' failed on {ip}: {stderr}")
        
        device_log_dir = node_dir / "device_log"
        device_log_dir.mkdir(parents=True, exist_ok=True)
        
        returncode, stdout, stderr = self.ssh_execute(node, "msnpureport")
        if returncode == 0:
            results[FILES_FIELD].append(str(device_log_dir))
        
        dl_log_dir = node_dir / "dl_log"
        dl_log_dir.mkdir(parents=True, exist_ok=True)
        
        dl_paths = [
            "/var/log/mindx-dl/devicePlugin",
            "/var/log/mindx-dl/noded",
            "/var/log/ascend-docker-runtime",
            "/var/log/mindx-dl/volcano-scheduler",
            "/var/log/mindx-dl/volcano-controller",
            "/var/log/mindx-dl/npu-exporter",
        ]
        
        for path in dl_paths:
            returncode, stdout, stderr = self.ssh_execute(node, f"ls {path} 2>/dev/null")
            if returncode == 0 and stdout.strip():
                results[FILES_FIELD].append(f"{dl_log_dir}/{Path(path).name}")
        
        self.logger.debug(f"Collected after-training logs from {ip}")
        return results
    
    def collect_all(self, stage: str) -> Dict:
        """在所有节点上采集日志"""
        all_results = {
            "stage": sanitize_string(stage),
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
            "nodes": []
        }
        
        collect_func = {
            "before": self.collect_before_training,
            "during": self.collect_during_training,
            "after": self.collect_after_training,
        }.get(stage, self.collect_after_training)
        
        for node in self.nodes:
            self.logger.info(f"Collecting logs from node: {node.get(IP_FIELD)}")
            result = collect_func(node)
            all_results["nodes"].append(result)
        
        report_file = self.output_dir / "collection_report.json"
        if self._safe_write_file(report_file, json.dumps(all_results, indent=2, ensure_ascii=False)):
            self.logger.info(f"Collection report saved to: {report_file}")
        
        return all_results
 
    def _validate_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """验证并清理节点配置"""
        validated = []
        for node in nodes:
            is_valid, error = validate_node_config(node, self.logger if hasattr(self, 'logger') else logging.getLogger())
            if is_valid:
                validated_node = {
                    IP_FIELD: sanitize_string(node[IP_FIELD]),
                    "user": sanitize_string(node.get("user", "root")),
                    "name": sanitize_string(node.get("name", node[IP_FIELD])),
                    KEY_FILE_FIELD: sanitize_string(node.get(KEY_FILE_FIELD, "")) if node.get(KEY_FILE_FIELD) else None,
                }
                if node.get(PASSWORD_FIELD):
                    validated_node[PASSWORD_FIELD] = node[PASSWORD_FIELD]
                validated.append(validated_node)
        return validated

    def _safe_write_file(self, filepath: Path, content: str) -> bool:
        """安全写入文件"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except (OSError, IOError) as e:
            self.logger.error(f"Failed to write file {filepath}: {e}")
            return False

def load_nodes_from_file(filepath: str) -> List[Dict]:
    """从文件加载节点配置"""
    if not validate_path(filepath):
        raise ValueError(f"Invalid file path: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.endswith(".json"):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of nodes")
            return data
        else:
            nodes = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = shlex.split(line)
                    if len(parts) >= 1:
                        node = {IP_FIELD: parts[0]}
                        if len(parts) >= 2:
                            node["user"] = parts[1]
                        nodes.append(node)
            return nodes


def main():
    parser = argparse.ArgumentParser(description="故障诊断日志采集脚本")
    parser.add_argument("-s", "--stage", required=True, choices=["before", "during", "after"],
                        help="采集阶段: before(训练前), during(训练中), after(训练后)")
    parser.add_argument("-n", "--nodes", required=True, help="节点配置文件(JSON格式)")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--skip-check", action="store_true", help="跳过 ascend-fd 安装检查")
    parser.add_argument("--log-file", help="日志文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出模式")
    
    args = parser.parse_args()
    
    if args.log_file and not validate_path(args.log_file):
        return 1
    
    logger = setup_logging(args.log_file)
    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    try:
        nodes = load_nodes_from_file(args.nodes)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load nodes config: {e}")
        return 1
    
    if not nodes:
        logger.error("No nodes found in configuration file")
        return 1
    
    output_dir = Path(args.output) / f"log_collection_{get_utc8_timestamp()}"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        return 1
    
    if not args.log_file:
        log_file = output_dir / "collector.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
    
    collector = LogCollector(str(output_dir), nodes, logger)
    
    if not args.skip_check:
        logger.info("=== Checking ascend-fd installation on all nodes ===")
        check_results = collector.check_all_nodes_ascend_fd()
        
        not_installed = [r for r in check_results if not r["installed"]]
        if not_installed:
            logger.error(f"{len(not_installed)} node(s) do not have ascend-fd installed:")
            for r in not_installed:
                logger.error(f"  - {r['node']} ({r[IP_FIELD]})")
            logger.error(f"Please install ascend-fd on the above nodes before collecting logs.")
            logger.error(f"Installation guide: {LogCollector.ASCEND_FD_INSTALL_URL}")
            logger.error(f"Or use --skip-check to skip this check.")
            return 1
        logger.info("=== All nodes have ascend-fd installed ===")
    
    results = collector.collect_all(args.stage)
    
    logger.info("Collection completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Nodes processed: {len(results['nodes'])}")
    
    total_errors = sum(len(n.get(ERROR_FIELD, [])) for n in results['nodes'])
    if total_errors > 0:
        logger.warning(f"Errors encountered: {total_errors}")
    
    return 0


if __name__ == "__main__":
    exit(main())
