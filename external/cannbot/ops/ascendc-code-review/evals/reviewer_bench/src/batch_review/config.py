#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""配置加载模块 — JSON 配置解析 + 环境变量校验"""
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional


class TargetConfig:
    """单个目标配置"""
    
    def __init__(self, data: dict, defaults: dict):
        self.name: str = data["name"]
        self.path: Optional[str] = data.get("path")  # 可选：PR 检视不需要
        self.prompt: str = data["prompt"]
        self.model: str = data.get("model", defaults.get("model", ""))
        self.api_key_env: str = data["api_key_env"]
        self.provider_key: str = data.get("provider_key", "")  # 显式声明的 provider 环境变量名，如 ANTHROPIC_API_KEY
        self.base_url: str = data.get("base_url", defaults.get("base_url", ""))  # 第三方网关地址（claude 引擎）
        self.extra_files: List[str] = data.get("extra_files", [])
        self.agent: Optional[str] = data.get("agent", defaults.get("agent"))
        self.report_file: Optional[str] = data.get("report_file")


class Config:
    """完整配置"""
    
    def __init__(self, data: dict):
        self._data = data
        
        # 引擎选择（顶层，与 execution 同级）
        self.engine: str = data.get("engine", "opencode")
        
        # review section
        review = data.get("review", {})
        self.agent: Optional[str] = review.get("agent")
        self.model: str = review.get("model", "")
        self.timeout_sec: int = review.get("timeout_sec", 1800)
        self.idle_timeout_sec: int = review.get("idle_timeout_sec", 300)
        self.format: str = review.get("format", "text")
        self.skill_prompt: str = review.get("skill_prompt", "")  # PR 检视的 prompt 模板
        
        # 全局 base_url（review 级别，作默认值）
        self.base_url: str = review.get("base_url", "")
        
        # execution section
        execution = data.get("execution", {})
        self.max_parallel: int = execution.get("max_parallel", 3)
        self.retry_on_failure: int = execution.get("retry_on_failure", 0)
        self.retry_delay_sec: int = execution.get("retry_delay_sec", 30)
        self.output_dir: str = execution.get("output_dir", "./review-reports")
        
        # key_pool section — API Key 池（自动轮转分配）
        # 支持两种格式：
        #   1. 字符串列表：["ANTHROPIC_API_KEY_1", "ANTHROPIC_API_KEY_2"]
        #   2. 对象列表：[{"env": "MY_KEY", "provider_key": "ANTHROPIC_API_KEY", "base_url": "..."}]
        raw_pool = data.get("key_pool", [])
        self.key_pool: List[dict] = []
        for item in raw_pool:
            if isinstance(item, str):
                # 字符串格式：自动推断 provider_key，base_url 为空
                self.key_pool.append({
                    "env": item,
                    "provider_key": _infer_provider_key(item),
                    "base_url": "",
                })
            elif isinstance(item, dict):
                env_name = item.get("env", "")
                self.key_pool.append({
                    "env": env_name,
                    "provider_key": item.get("provider_key", _infer_provider_key(env_name)),
                    # key_pool 级别 base_url 覆盖全局 base_url
                    "base_url": item.get("base_url", "") or self.base_url,
                })
        
        # targets section
        targets_data = data.get("targets", [])
        self.targets: List[TargetConfig] = [
            TargetConfig(t, review) for t in targets_data
        ]
    
    def validate(self, has_pr_file: bool = False):
        """校验配置完整性"""
        errors = []
        
        # 校验 engine 值
        if self.engine not in ("opencode", "claude"):
            errors.append(f"不支持的 engine '{self.engine}'，仅支持 'opencode' 或 'claude'")
            raise ConfigError(errors)
        
        # 根据 engine 检查对应可执行文件
        if not shutil.which(self.engine):
            errors.append(f"{self.engine} 可执行文件未找到，请确认已安装并加入 PATH")
        
        if has_pr_file:
            # PR 列表文件模式：校验 key_pool 和 skill_prompt
            if not self.key_pool:
                errors.append("使用 PR 列表文件时，必须在配置中设置 key_pool（API Key 环境变量名列表）")
            if not self.skill_prompt:
                errors.append("使用 PR 列表文件时，必须在 review.skill_prompt 中配置 prompt 模板")
            if not self.model:
                errors.append("使用 PR 列表文件时，必须在 review.model 中配置默认模型")
            
            # 检查 key_pool 中的环境变量是否已设置
            for item in self.key_pool:
                env_name = item.get("env", "")
                if not env_name:
                    errors.append("key_pool 中存在没有 env 字段的条目")
                elif env_name not in os.environ:
                    errors.append(f"key_pool 中引用的环境变量 '{env_name}' 未设置")
                
                # opencode 引擎需要 provider_key
                if self.engine == "opencode" and not item.get("provider_key"):
                    errors.append(f"key_pool 中 '{env_name}' 未设置 provider_key，请显式指定（如 ANTHROPIC_API_KEY）")
        else:
            # 手动 targets 模式
            if not self.targets:
                errors.append("配置文件中没有定义任何 targets（也没有提供 PR 列表文件）")
            
            for t in self.targets:
                if not t.name:
                    errors.append("target 缺少 name 字段")
                if not t.prompt:
                    errors.append(f"target '{t.name}' 缺少 prompt 字段")
                if not t.model:
                    errors.append(f"target '{t.name}' 缺少 model 字段（且 review 中未设置默认 model）")
                if not t.api_key_env:
                    errors.append(f"target '{t.name}' 缺少 api_key_env 字段")
                elif t.api_key_env not in os.environ:
                    errors.append(f"target '{t.name}' 引用的环境变量 '{t.api_key_env}' 未设置")
                if t.path and not t.path.startswith("http"):
                    path = Path(t.path)
                    if not path.exists():
                        errors.append(f"target '{t.name}' 的路径 '{t.path}' 不存在")
        
        if errors:
            raise ConfigError(errors)
        
        return True
    
    def apply_cli_overrides(self, output: Optional[str] = None, max_parallel: Optional[int] = None):
        """应用 CLI 参数覆盖"""
        if output:
            self.output_dir = output
        if max_parallel:
            self.max_parallel = max_parallel


class ConfigError(Exception):
    """配置错误"""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def load_config(config_path: str, has_pr_file: bool = False) -> Config:
    """加载 JSON 配置文件"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError([f"配置文件 JSON 格式错误: {e}"])
    
    config = Config(data)
    config.validate(has_pr_file=has_pr_file)
    config._config_path = str(path.absolute())
    return config


def load_pr_list(pr_file: str) -> List[str]:
    """从文件加载 PR URL 列表（每行一个 URL，支持 # 注释和空行）"""
    path = Path(pr_file)
    if not path.exists():
        raise FileNotFoundError(f"PR 列表文件不存在: {pr_file}")
    
    pr_urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # 跳过空行和注释
                pr_urls.append(line)
    
    return pr_urls


def generate_targets_from_prs(
    pr_urls: List[str],
    key_pool: List[dict],
    skill_prompt: str,
    model: str,
    agent: Optional[str] = None,
) -> List[TargetConfig]:
    """从 PR 列表和 key_pool 自动生成 targets（round-robin 分配 key）"""
    targets = []
    for idx, pr_url in enumerate(pr_urls):
        # 从 URL 提取 PR 编号生成 name
        match = re.search(r'/pull/(\d+)', pr_url)
        if match:
            pr_number = match.group(1)
            # 提取仓库名
            repo_match = re.search(r'gitcode\.com/[^/]+/([^/]+)', pr_url)
            repo_name = repo_match.group(1) if repo_match else "unknown"
            name = f"{repo_name}-pr-{pr_number}"
        else:
            name = f"pr-{idx+1}"
        
        # Round-robin 分配 key
        key_item = key_pool[idx % len(key_pool)]
        api_key_env = key_item["env"]
        provider_key = key_item.get("provider_key", "")
        base_url = key_item.get("base_url", "")
        
        # 替换 {pr_url} 占位符
        prompt = skill_prompt.replace("{pr_url}", pr_url)
        
        target_data = {
            "name": name,
            "prompt": prompt,
            "api_key_env": api_key_env,
            "provider_key": provider_key,
            "base_url": base_url,
            "model": model,
        }
        if agent:
            target_data["agent"] = agent
        
        targets.append(TargetConfig(target_data, {}))
    
    return targets


def _infer_provider_key(api_key_env: str) -> Optional[str]:
    """从 api_key_env 名称推断标准 provider key 名称
    
    供 opencode 引擎使用（字符串简写 key_pool 时自动推断）。
    """
    env_upper = api_key_env.upper()
    
    if "ANTHROPIC" in env_upper:
        return "ANTHROPIC_API_KEY"
    elif "OPENAI" in env_upper:
        return "OPENAI_API_KEY"
    elif "DEEPSEEK" in env_upper:
        return "DEEPSEEK_API_KEY"
    elif "GOOGLE" in env_upper or "GEMINI" in env_upper:
        return "GOOGLE_API_KEY"
    elif "AZURE" in env_upper:
        return "AZURE_API_KEY"
    
    return None