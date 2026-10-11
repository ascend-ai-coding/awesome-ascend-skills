#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared configuration loading and initialization for the Issue Handler."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from runtime_paths import (
        CLASSIFY_CONFIG,
        CONFIG_DIR,
        LEGACY_CLASSIFY_CONFIG,
        LEGACY_OPERATOR_OWNERS_CONFIG,
        OPERATOR_OWNERS_CONFIG,
    )
except ModuleNotFoundError:  # direct import through importlib.util in tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_paths import (  # type: ignore[no-redef]
        CLASSIFY_CONFIG,
        CONFIG_DIR,
        LEGACY_CLASSIFY_CONFIG,
        LEGACY_OPERATOR_OWNERS_CONFIG,
        OPERATOR_OWNERS_CONFIG,
    )

try:
    from config_validation import problems, path_problems
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_validation import problems, path_problems

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_TEMPLATE_NAMES = {"classify_config", "operator_owners"}
_RESPONSIBILITY_LEVELS = {"handle", "list-only", "ignore"}
_AUTOMATION_KEYS = ("auto-response", "auto-assign")


class ConfigError(ValueError):
    """Raised when a handler YAML document is malformed."""

    def __init__(self, message, *, errors=None):
        super().__init__(message)
        self.errors = errors or [{"message": message}]


class _MarkedMapping(dict):
    """Keep field positions without exposing source values in diagnostics."""

    pass


def _raise_problems(errors, source, document):
    for error in errors:
        error["file"] = str(source)
        node = document
        mark = None
        for key in error.get("field", "").split("."):
            if not isinstance(node, dict):
                break
            mark = getattr(node, "marks", {}).get(key, mark)
            node = node.get(key)
        if mark:
            error.update(line=mark.line + 1, column=mark.column + 1)
    if errors:
        raise ConfigError("; ".join(f"{e.get('field', '')}: {e['message']}" for e in errors), errors=errors)


def _require_yaml():
    if yaml is None:
        raise ConfigError("PyYAML is required; install it with: pip install pyyaml")
    yaml.SafeDumper.add_representer(_MarkedMapping, yaml.SafeDumper.represent_dict)
    return yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    y = _require_yaml()

    class UniqueLoader(y.SafeLoader):
        pass

    def mapping(loader, node):
        # Reject duplicate explicit keys before YAML merge expansion.
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag not in ("tag:yaml.org,2002:str", "tag:yaml.org,2002:merge"):
                raise y.constructor.ConstructorError(
                    None, None, "configuration keys must be strings", key_node.start_mark
                )
            key = key_node.value
            if key in seen:
                raise y.constructor.ConstructorError(None, None, "duplicate YAML key", key_node.start_mark)
            seen.add(key)
        loader.flatten_mapping(node)
        result = _MarkedMapping()
        result.marks = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise y.constructor.ConstructorError(
                    None, None, "configuration keys must be strings", key_node.start_mark
                )
            result[key] = loader.construct_object(value_node, deep=True)
            result.marks[key] = key_node.start_mark
        return result

    UniqueLoader.add_constructor(y.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        raw = y.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (OSError, UnicodeError, y.YAMLError) as exc:
        error = {
            "file": str(path),
            "message": "cannot read YAML config: " + (getattr(exc, "problem", None) or type(exc).__name__),
        }
        mark = getattr(exc, "problem_mark", None)
        if mark:
            error.update(line=mark.line + 1, column=mark.column + 1)
        raise ConfigError(error["message"], errors=[error]) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML config {path} must contain a mapping")
    return raw


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict) and value:
            result[key] = _merge(result[key], value)
        else:
            # Empty mappings/lists are intentional overrides and must be kept.
            result[key] = value
    return result


def _validate_responsibility(document: dict[str, Any], source: Path) -> None:
    responsibility = document.get("responsibility")
    if responsibility is None:
        return
    if not isinstance(responsibility, dict):
        raise ConfigError(f"responsibility in {source} must be a mapping")
    unknown = set(responsibility) - _RESPONSIBILITY_LEVELS
    if unknown:
        raise ConfigError("responsibility contains invalid level key(s): " + ", ".join(map(str, sorted(unknown))))
    for level, conditions in responsibility.items():
        invalid_list = not isinstance(conditions, list)
        invalid_item = not invalid_list and any(not isinstance(condition, str) for condition in conditions)
        if invalid_list or invalid_item:
            raise ConfigError(f"responsibility.{level} must be a list of strings")


def _validate_automation(
    document: dict[str, Any],
    source: Path,
    require_automation: bool,
) -> None:
    fallback = document.get("auto-assign-fallback-user")
    fallback_is_string = isinstance(fallback, str)
    fallback_has_valid_shape = fallback_is_string and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", fallback))
    fallback_invalid = fallback is not None and (
        not fallback_is_string or (fallback != "" and (not fallback_has_valid_shape or fallback.casefold() == "direct"))
    )
    if fallback_invalid:
        raise ConfigError("auto-assign-fallback-user must be empty or a GitCode login (not a URL or nickname)")
    for key in _AUTOMATION_KEYS:
        if key not in document:
            if require_automation:
                raise ConfigError(f"{key} in {source} must be a boolean (true or false)")
            continue
        if not isinstance(document.get(key), bool):
            raise ConfigError(f"{key} in {source} must be a boolean (true or false)")
    automation_complete = "auto-assign" in document and "auto-response" in document
    assignment_without_response = automation_complete and document["auto-assign"] and not document["auto-response"]
    if assignment_without_response:
        raise ConfigError("启用 auto-assign 时必须同时启用 auto-response：auto-assign: true 需要 auto-response: true")


def _validate(
    document: dict[str, Any],
    source: Path,
    *,
    validate_automation: bool = False,
    require_automation: bool = False,
) -> None:
    errors = []
    try:
        _validate_responsibility(document, source)
    except ConfigError as exc:
        errors.extend(exc.errors)
    ignored = document.get("ignored_issue_ids", [])
    if not isinstance(ignored, list) or any(
        isinstance(i, bool) or not str(i).isdigit() or int(i) <= 0 for i in ignored
    ):
        errors.append(
            {"field": "ignored_issue_ids", "message": "ignored_issue_ids must be a list of positive Issue numbers"}
        )
    if validate_automation:
        try:
            _validate_automation(document, source, require_automation)
        except ConfigError as exc:
            errors.extend(exc.errors)
        errors.extend(problems(document, complete=require_automation))
    _raise_problems(errors, source, document)


def load_template(name: str = "classify_config") -> dict[str, Any]:
    """Load and validate a packaged ``assets/<name>.yaml.template``."""
    if name not in _TEMPLATE_NAMES:
        raise ConfigError(f"unknown handler config template: {name}")
    path = _ASSETS / f"{name}.yaml.template"
    if not path.exists():
        raise ConfigError(f"handler config template does not exist: {path}")
    document = _read_yaml(path)
    _validate(
        document, path, validate_automation=name == "classify_config", require_automation=name == "classify_config"
    )
    return document


def load_handler_config(path: str | Path | None = None, *, allow_legacy: bool = True) -> dict[str, Any]:
    """Return template defaults recursively overridden by repository config."""
    return load_named_config("classify_config", path, allow_legacy=allow_legacy)


def load_named_config(name: str, path: str | Path | None = None, *, allow_legacy: bool = True) -> dict[str, Any]:
    """Load one named template with a canonical or legacy repository override."""
    defaults = load_template(name)
    if name == "operator_owners":
        canonical, legacy = OPERATOR_OWNERS_CONFIG, LEGACY_OPERATOR_OWNERS_CONFIG
    elif name == "classify_config":
        canonical, legacy = CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG
    else:
        raise ConfigError(f"unknown handler config template: {name}")
    if path is None:
        config_path = canonical if canonical.exists() else (legacy if allow_legacy and legacy.exists() else canonical)
        explicit = False
    else:
        config_path, explicit = Path(path), True
    if explicit and not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")
    if config_path.exists():
        user = _read_yaml(config_path)
        _validate(user, config_path, validate_automation=name == "classify_config")
        merged = _merge(defaults, user)
        _validate(
            merged,
            config_path,
            validate_automation=name == "classify_config",
            require_automation=name == "classify_config",
        )
        if name == "classify_config":
            _raise_problems(path_problems(merged, config_path), config_path, user)
        return merged
    return defaults


def get_automation_policy(config: dict[str, Any]) -> dict[str, bool]:
    """Return normalized automation flags from ``load_handler_config``.

    The YAML/API keys retain their documented hyphenated names.  Callers use
    this small normalized view so automation policy is read consistently:
    ``{"auto_response": bool, "auto_assign": bool}``.
    """
    if not isinstance(config, dict):
        raise ConfigError("handler config must be a mapping")
    try:
        auto_response = config["auto-response"]
        auto_assign = config["auto-assign"]
    except KeyError as exc:
        raise ConfigError(f"missing automation config key: {exc.args[0]}") from exc
    if not isinstance(auto_response, bool) or not isinstance(auto_assign, bool):
        raise ConfigError("automation config values must be booleans")
    if auto_assign and not auto_response:
        raise ConfigError("启用 auto-assign 时必须同时启用 auto-response：auto-assign: true 需要 auto-response: true")
    return {"auto_response": auto_response, "auto_assign": auto_assign}


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    y = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(y.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")


def initialize_config(repository_root: str | Path) -> dict[str, str]:
    """Instantiate both packaged templates under a repository's canonical config."""
    root = Path(repository_root)
    written: dict[str, str] = {}
    for name, canonical, legacy in (
        ("classify_config", CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG),
        ("operator_owners", OPERATOR_OWNERS_CONFIG, LEGACY_OPERATOR_OWNERS_CONFIG),
    ):
        destination = root / canonical
        # A dangling symlink is still an existing user choice; never replace it.
        if destination.exists() or destination.is_symlink():
            continue
        template = load_template(name)
        legacy_path = root / legacy
        if legacy_path.exists():
            legacy_doc = _read_yaml(legacy_path)
            _validate(legacy_doc, legacy_path)
            document = _merge(template, legacy_doc)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write((_ASSETS / f"{name}.yaml.template").read_text(encoding="utf-8"))
            written[name] = str(destination)
            continue
        _write_yaml(destination, document)
        written[name] = str(destination)
    return written
