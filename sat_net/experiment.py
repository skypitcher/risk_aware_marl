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


def rollout_record(
    sampling_step: int,
    rollout: int,
    phase: str,
    result: Any,
    simulated_time_ms: float | None = None,
    cumulative: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_stats = getattr(result, "step_stats", {})
    window_steps = int(step_stats.get("env_interaction_steps", step_stats.get("steps", 0)) or 0)
    elapsed_seconds = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
    window_duration_ms = float(step_stats.get("duration_ms", 0.0) or 0.0)
    aggregate_duration_ms = float(step_stats.get("aggregate_duration_ms", window_duration_ms) or 0.0)
    decisions = int(step_stats.get("decisions", 0) or 0)
    metrics = result.metrics
    packet_step_denominator = max(window_steps, 1)
    return {
        "step": sampling_step,
        "env_interaction_steps": sampling_step,
        "sampling_step": sampling_step,
        "rollout": rollout,
        "phase": phase,
        "seed": result.seed,
        "train": result.train,
        "global_simulated_time_ms": simulated_time_ms,
        "window_start_time_ms": step_stats.get("start_time_ms"),
        "window_end_time_ms": step_stats.get("end_time_ms"),
        "window_duration_ms": window_duration_ms,
        "window_duration_seconds": window_duration_ms / 1000.0,
        "elapsed_seconds": elapsed_seconds,
        "sim_speed": step_stats.get("sim_speed"),
        "sample_efficiency": {
            "env_steps": window_steps,
            "vector_steps": step_stats.get("vector_steps"),
            "num_envs": step_stats.get("num_envs", 1),
            "env_steps_per_wall_second": window_steps / max(elapsed_seconds, 1e-12),
            "decisions": decisions,
            "decisions_per_env_step": decisions / max(window_steps, 1),
            "generated_packets_per_env_step": metrics.generated / packet_step_denominator,
            "delivered_packets_per_env_step": metrics.delivered / packet_step_denominator,
            "dropped_packets_per_env_step": metrics.dropped / packet_step_denominator,
            "simulated_seconds_per_env_step": (aggregate_duration_ms / 1000.0) / max(window_steps, 1),
        },
        "metrics": metrics.to_dict(),
        "env": result.info,
        "steps": step_stats,
        "agent": getattr(result, "agent_stats", {}),
        "cumulative": cumulative,
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
