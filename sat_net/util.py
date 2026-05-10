import argparse
import json
from enum import IntEnum
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two JSON-like dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(file_path: str | Path, parent_dir: Path | None = None) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path

    if parent_dir is not None:
        candidate = parent_dir / path
        if candidate.exists():
            return candidate

    return Path.cwd() / path


def load_json_config(file_path: str | Path, seen: set[Path] | None = None) -> dict:
    """Load a JSON config, resolving optional recursive `extends` references."""
    path = _resolve_config_path(file_path).resolve()
    branch = set() if seen is None else set(seen)
    if path in branch:
        raise ValueError(f"Circular config extends detected at {path}")
    branch.add(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    parents = data.get("extends")
    if parents is None:
        return data

    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise TypeError(f"Config extends must be a string or list: {path}")

    merged: dict = {}
    for parent in parents:
        parent_path = _resolve_config_path(parent, parent_dir=path.parent)
        merged = deep_merge(merged, load_json_config(parent_path, seen=branch))
    return deep_merge(merged, data)


def ms2str(ms: float) -> str:
    """Convert milliseconds to a string of the form HH:MM:SS.mmm."""
    ms_int = int(ms)
    total_seconds = ms_int // 1000  # Convert milliseconds to seconds
    milliseconds = ms_int % 1000  # Extract remaining milliseconds
    hours = total_seconds // 3600  # Extract hours
    remaining_seconds = total_seconds % 3600  # Seconds left after hours
    minutes = remaining_seconds // 60  # Extract minutes
    seconds = remaining_seconds % 60  # Extract seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class NamedDict:
    def __init__(self, data=None):
        if isinstance(data, argparse.Namespace):
            data = vars(data)
        elif data is None:
            data = {}

        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = NamedDict(value)
            else:
                self._data[key] = value

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, other):
        for key, value in other.items():
            self._data[key] = value

    def __contains__(self, item):
        return item in self._data

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'DynamicObj' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self._data[key] = NamedDict(value)
        else:
            self._data[key] = value

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def to_dict(self):
        data_dict = {}
        for key, value in self._data.items():
            if isinstance(value, NamedDict):
                data_dict[key] = value.to_dict()
            else:
                data_dict[key] = value
        return data_dict

    def to_string(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def load(cls, file_path):
        """Loads configuration from a JSON file."""
        return cls(load_json_config(file_path))

    def save(self, file_path):
        """Saves configuration to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


class NetworkError(IntEnum):
    """
    Reason for failing to deliver a flowlet in the network.
    - SUCCESS: The operation was successful
    - TTL_EXPIRED: The operation failed because the TTL expired
    - LINK_FULL: The operation failed because the link is full
    - INVALID_NEXT_HOP: The operation failed because the selected next hop is invalid
    - FAILED_TO_FIND_NEXT_HOP: The operation failed because no route exists
    - LINK_DISCONNECTED: The operation failed because the link is disconnected (e.g., due to satellite movement)
    - NO_AVAILABLE_SAT: The operation failed because no access satellite is visible
    """

    SUCCESS = 0
    TTL_EXPIRED = 1
    LINK_FULL = 2
    INVALID_NEXT_HOP = 3
    FAILED_TO_FIND_NEXT_HOP = 4
    LINK_DISCONNECTED = 5
    NO_AVAILABLE_SAT = 6
