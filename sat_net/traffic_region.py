from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from sat_net.geometric import EARTH_R_KM, geo_to_ecef_position, great_circle_distance
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

    records: list[tuple[float, str, float, float]] = []
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
            records.append((weight, f"grid_{iy}_{ix}", lat, lon))

    records.sort(key=lambda item: item[0], reverse=True)
    if max_regions:
        records = records[:max_regions]
    return [
        TrafficRegion(id=idx, name=name, latitude=lat, longitude=lon, weight=weight)
        for idx, (weight, name, lat, lon) in enumerate(records)
    ]


def _region_distance_matrix_km(regions: list[TrafficRegion]) -> np.ndarray:
    lat = np.radians(np.array([region.latitude for region in regions], dtype=np.float64))
    lon = np.radians(np.array([region.longitude for region in regions], dtype=np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon * 0.5) ** 2
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_R_KM * central_angle


class TrafficRegionModel:
    """Population/region driven flowlet source and destination sampler."""

    def __init__(self, regions: list[TrafficRegion], traffic_config: NamedDict | None = None):
        if len(regions) < 2:
            raise ValueError("TrafficRegionModel requires at least two regions")
        traffic_config = NamedDict({}) if traffic_config is None else traffic_config
        self.regions = regions
        weights = np.array([max(region.weight, 0.0) for region in regions], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(regions), dtype=np.float64)
        self.weights = weights / weights.sum()
        self._region_by_id = {region.id: region for region in regions}
        self.distance_matrix_km = _region_distance_matrix_km(regions)
        self.source_population_power = float(traffic_config.get("od_source_population_power", 1.0))
        self.target_population_power = float(traffic_config.get("od_target_population_power", 1.0))
        self.distance_decay = float(traffic_config.get("od_distance_decay", 0.9))
        self.distance_offset_km = float(traffic_config.get("od_distance_offset_km", 500.0))
        self.min_distance_km = float(traffic_config.get("od_min_distance_km", 50.0))
        self.diurnal_peak_hour = float(traffic_config.get("od_diurnal_peak_hour", 20.0))
        self.diurnal_floor = float(traffic_config.get("od_diurnal_floor", 0.25))
        self.source_diurnal_amplitude = float(traffic_config.get("od_source_diurnal_amplitude", 0.55))
        self.target_diurnal_amplitude = float(traffic_config.get("od_target_diurnal_amplitude", 0.20))
        self._longitudes = np.array([region.longitude for region in regions], dtype=np.float64)
        self._base_od_matrix = self._build_base_od_matrix()
        self._base_flat_od = self._base_od_matrix.reshape(-1)
        self._base_source_marginal = self._base_od_matrix.sum(axis=1)

    @classmethod
    def from_config(cls, traffic_config: NamedDict, project_root: str | Path) -> "TrafficRegionModel":
        project_root = Path(project_root)
        map_path = _resolve_path(traffic_config.get("population_map_path", None), project_root)
        if map_path is None:
            raise ValueError("traffic.population_map_path is required")
        if not map_path.exists():
            raise FileNotFoundError(f"Population map not found: {map_path}")

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

        return cls(regions, traffic_config=traffic_config)

    def get(self, region_id: int) -> TrafficRegion:
        return self._region_by_id[region_id]

    def sample_od_pairs(
        self,
        rng: np.random.Generator,
        size: int,
        time_ms: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if size <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        if time_ms is None:
            flat_probs = self._base_flat_od
        else:
            matrix = self._time_adjusted_od_matrix(time_ms)
            flat_probs = matrix.reshape(-1)

        pair_ids = self._sample_flat_probs(rng, flat_probs, size)
        num_regions = len(self.regions)
        source_ids = pair_ids // num_regions
        target_ids = pair_ids % num_regions
        return source_ids.astype(np.int64, copy=False), target_ids.astype(np.int64, copy=False)

    def sample_source_ids(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(len(self.regions), size=size, p=self._base_source_marginal)

    def sample_target_ids(self, rng: np.random.Generator, source_ids: np.ndarray) -> np.ndarray:
        source_ids = np.asarray(source_ids, dtype=np.int64)
        target_ids = np.empty(len(source_ids), dtype=np.int64)
        for source_id in np.unique(source_ids):
            rows = source_ids == source_id
            probs = self._base_od_matrix[int(source_id)]
            row_sum = probs.sum()
            if row_sum <= 0:
                probs = np.ones(len(self.regions), dtype=np.float64)
                probs[int(source_id)] = 0.0
                probs /= probs.sum()
            else:
                probs = probs / row_sum
            target_ids[rows] = rng.choice(len(self.regions), size=int(rows.sum()), p=probs)
        return target_ids

    @property
    def total_weight(self) -> float:
        return float(sum(region.weight for region in self.regions))

    def od_summary(self) -> dict[str, float | int]:
        nonzero = self._base_od_matrix > 0
        return {
            "regions": len(self.regions),
            "pairs": int(nonzero.sum()),
            "distance_decay": self.distance_decay,
            "distance_offset_km": self.distance_offset_km,
            "mean_pair_distance_km": float((self._base_od_matrix * self.distance_matrix_km).sum()),
            "max_pair_probability": float(self._base_od_matrix.max()),
        }

    def _build_base_od_matrix(self) -> np.ndarray:
        weights = np.maximum(np.array([region.weight for region in self.regions], dtype=np.float64), 0.0)
        if weights.sum() <= 0:
            weights = np.ones(len(self.regions), dtype=np.float64)
        weights = weights / weights.sum()

        source = np.power(weights, self.source_population_power)
        target = np.power(weights, self.target_population_power)
        distance = np.maximum(self.distance_matrix_km, self.min_distance_km)
        distance_weight = np.power(distance + max(self.distance_offset_km, 0.0), -self.distance_decay)
        matrix = source[:, None] * target[None, :] * distance_weight
        np.fill_diagonal(matrix, 0.0)
        return self._normalize_od_matrix(matrix)

    def _time_adjusted_od_matrix(self, time_ms: float) -> np.ndarray:
        source_activity = self._local_time_activity(time_ms, self.source_diurnal_amplitude)
        target_activity = self._local_time_activity(time_ms, self.target_diurnal_amplitude)
        matrix = self._base_od_matrix * source_activity[:, None] * target_activity[None, :]
        return self._normalize_od_matrix(matrix)

    def _local_time_activity(self, time_ms: float, amplitude: float) -> np.ndarray:
        utc_hour = (float(time_ms) / 3_600_000.0) % 24.0
        local_hour = (utc_hour + self._longitudes / 15.0) % 24.0
        phase = 2.0 * math.pi * (local_hour - self.diurnal_peak_hour) / 24.0
        activity = 1.0 + float(amplitude) * np.cos(phase)
        return np.maximum(activity, self.diurnal_floor)

    @staticmethod
    def _normalize_od_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        total = matrix.sum()
        if total <= 0:
            raise ValueError("OD demand matrix has zero total mass")
        return matrix / total

    @staticmethod
    def _sample_flat_probs(rng: np.random.Generator, probs: np.ndarray, size: int) -> np.ndarray:
        cdf = np.cumsum(probs, dtype=np.float64)
        cdf[-1] = 1.0
        return np.searchsorted(cdf, rng.random(size), side="right").astype(np.int64, copy=False)
