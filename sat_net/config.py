from __future__ import annotations

from pathlib import Path
from typing import Any

from sat_net.util import NamedDict, deep_merge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_CONFIG = "configs/main.json"
DEFAULT_AGENT_CONFIG = "configs/agents/madqn.json"
DEFAULT_SPF_AGENT_CONFIG = "configs/agents/spf.json"


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def load_config(path: str | Path) -> NamedDict:
    return NamedDict.load(resolve_repo_path(path))


def to_plain(value: Any) -> Any:
    if isinstance(value, NamedDict):
        return value.to_dict()
    return value


def merge_section(defaults: dict[str, Any], main_config: NamedDict, section: str, overrides: dict[str, Any]) -> NamedDict:
    section_defaults = main_config.get(section, NamedDict({}))
    data = deep_merge(defaults, to_plain(section_defaults))
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return NamedDict(deep_merge(data, clean_overrides))


def load_env_config(main_config: NamedDict) -> NamedDict:
    return NamedDict(main_config.to_dict())


def load_agent_config(main_config: NamedDict, override_path: str | None = None, default_path: str = DEFAULT_AGENT_CONFIG) -> NamedDict:
    return load_config(override_path or main_config.get("agent", default_path))


def eval_agent_paths(main_config: NamedDict, overrides: list[str] | None = None) -> list[str]:
    if overrides:
        return list(overrides)
    eval_config = main_config.get("eval", NamedDict({}))
    paths = eval_config.get("agents", [DEFAULT_SPF_AGENT_CONFIG])
    return list(paths)
