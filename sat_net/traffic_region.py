from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from sat_net.geometric import geo_to_ecef_position, great_circle_distance
from sat_net.util import NamedDict


@dataclass(slots=True)
class TrafficRegion:
    """Aggregated terrestrial traffic source/sink represented by one lat/lon point."""

    id: int
    name: str
    latitude: float
    longitude: float
    weight: float
    position: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.position = geo_to_ecef_position(lat=self.latitude, lon=self.longitude, alt=0.0)

    def get_projected_position(self) -> tuple[float, float]:
        return self.longitude, self.latitude

    def distance_to(self, other: "TrafficRegion") -> float:
        return great_circle_distance(self.longitude, self.latitude, other.longitude, other.latitude)


def _resolve_path(path_value: str | None, project_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _regions_from_records(records: Iterable[dict]) -> list[TrafficRegion]:
    regions = []
    for item in records:
        weight = float(item.get("weight", item.get("population", 1.0)))
        if weight <= 0:
            continue
        idx = len(regions)
        regions.append(
            TrafficRegion(
                id=idx,
                name=str(item.get("name", f"region_{idx}")),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                weight=weight,
            )
        )
    return regions


def _load_records_from_file(region_file: Path) -> list[dict]:
    with region_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("regions", [])
    if not isinstance(data, list):
        raise ValueError(f"Region file must contain a list or a dict with 'regions': {region_file}")
    return data


def _population_array_from_file(path: Path, channel: str, max_value: float | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path).astype(np.float64, copy=False)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            key = "population" if "population" in data else data.files[0]
            values = data[key].astype(np.float64, copy=False)
    else:
        values = _population_array_from_image(path, channel)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values[values < 0] = 0
    if max_value is not None:
        values[values > max_value] = 0
    return values


def _population_array_from_image(path: Path, channel: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)

    if arr.ndim == 2:
        values = arr.astype(np.float64)
    elif arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        alpha = arr[..., 3].astype(np.float64) / 255.0 if arr.shape[-1] >= 4 else 1.0
        if channel == "alpha" and arr.shape[-1] >= 4:
            values = arr[..., 3].astype(np.float64)
        elif channel == "red":
            values = rgb[..., 0]
        elif channel == "green":
            values = rgb[..., 1]
        elif channel == "blue":
            values = rgb[..., 2]
        else:
            values = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * alpha
    else:
        raise ValueError(f"Unsupported population map shape {arr.shape}: {path}")

    return values


def _regions_from_population_map(
    map_path: Path,
    lat_bins: int,
    lon_bins: int,
    max_regions: int | None,
    min_weight: float,
    channel: str,
    max_value: float | None,
    area_scale_weights: bool,
) -> list[TrafficRegion]:
    values = _population_array_from_file(map_path, channel=channel, max_value=max_value)
    height, width = values.shape

    y_edges = np.linspace(0, height, lat_bins + 1, dtype=int)
    x_edges = np.linspace(0, width, lon_bins + 1, dtype=int)

    records = []
    for iy in range(lat_bins):
        y0, y1 = y_edges[iy], y_edges[iy + 1]
        if y1 <= y0:
            continue
        lat = 90.0 - ((y0 + y1) * 0.5 / height) * 180.0
        area_scale = max(math.cos(math.radians(lat)), 1e-3)
        for ix in range(lon_bins):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            if x1 <= x0:
                continue
            weight = float(values[y0:y1, x0:x1].sum())
            if area_scale_weights:
                weight *= area_scale
            if weight <= min_weight:
                continue
            lon = ((x0 + x1) * 0.5 / width) * 360.0 - 180.0
            records.append(
                {
                    "name": f"grid_{iy}_{ix}",
                    "latitude": lat,
                    "longitude": lon,
                    "weight": weight,
                }
            )

    records.sort(key=lambda item: item["weight"], reverse=True)
    if max_regions:
        records = records[:max_regions]
    return _regions_from_records(records)


class TrafficRegionModel:
    """Population/region driven flowlet source and destination sampler."""

    def __init__(self, regions: list[TrafficRegion]):
        if len(regions) < 2:
            raise ValueError("TrafficRegionModel requires at least two regions")
        self.regions = regions
        weights = np.array([max(region.weight, 0.0) for region in regions], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(regions), dtype=np.float64)
        self.weights = weights / weights.sum()
        self._region_by_id = {region.id: region for region in regions}

    @classmethod
    def from_config(cls, traffic_config: NamedDict, project_root: str | Path) -> "TrafficRegionModel":
        project_root = Path(project_root)
        region_file = _resolve_path(traffic_config.get("region_file", None), project_root)
        map_path = _resolve_path(traffic_config.get("population_map_path", None), project_root)

        if map_path is not None and map_path.exists():
            regions = _regions_from_population_map(
                map_path=map_path,
                lat_bins=int(traffic_config.get("grid_lat_bins", 36)),
                lon_bins=int(traffic_config.get("grid_lon_bins", 72)),
                max_regions=traffic_config.get("max_regions", 512),
                min_weight=float(traffic_config.get("min_region_weight", 0.0)),
                channel=str(traffic_config.get("population_channel", "luma")),
                max_value=traffic_config.get("population_max_value", None),
                area_scale_weights=bool(traffic_config.get("population_area_scale", True)),
            )
        elif region_file is not None and region_file.exists():
            regions = _regions_from_records(_load_records_from_file(region_file))
        else:
            inline_regions = traffic_config.get("regions", [])
            regions = _regions_from_records(inline_regions)

        return cls(regions)

    def get(self, region_id: int) -> TrafficRegion:
        return self._region_by_id[region_id]

    def sample_source_ids(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(len(self.regions), size=size, p=self.weights)

    def sample_target_ids(self, rng: np.random.Generator, source_ids: np.ndarray) -> np.ndarray:
        target_ids = rng.choice(len(self.regions), size=len(source_ids), p=self.weights)
        same = target_ids == source_ids
        while same.any():
            target_ids[same] = rng.choice(len(self.regions), size=int(same.sum()), p=self.weights)
            same = target_ids == source_ids
        return target_ids

    @property
    def total_weight(self) -> float:
        return float(sum(region.weight for region in self.regions))
