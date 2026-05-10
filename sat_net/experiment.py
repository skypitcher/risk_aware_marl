from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Small file-based experiment manager for training and evaluation runs."""

    def __init__(self, log_dir: str | os.PathLike[str]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str) -> Path:
        path = self.log_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        path = self.path(relative_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_json_ready(payload), f, indent=4, sort_keys=True)

    def append_jsonl(self, relative_path: str, payload: dict[str, Any]) -> None:
        path = self.path(relative_path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")

    def append_csv(self, relative_path: str, row: dict[str, Any]) -> None:
        path = self.path(relative_path)
        flat_row = flatten_scalars(row)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat_row)

    def save_manifest(
        self,
        args: Any,
        env_config: Any,
        agent_config: Any,
        notes: str | None = None,
    ) -> None:
        self.write_json(
            "manifest.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "command": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "git": git_state(),
                "notes": notes or "",
                "args": _to_plain(args),
                "env_config": _to_plain(env_config),
                "agent_config": _to_plain(agent_config),
            },
        )


def episode_record(epoch: int, phase: str, result: Any) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "phase": phase,
        "seed": result.seed,
        "train": result.train,
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": result.metrics.to_dict(),
        "env": result.info,
        "steps": getattr(result, "step_stats", {}),
        "agent": getattr(result, "agent_stats", {}),
    }


def flatten_scalars(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        value = _to_plain(value)
        if isinstance(value, dict):
            row.update(flatten_scalars(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[name] = value
    return row


def git_state() -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run_git(["status", "--porcelain"])),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return _to_plain(value)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value
