from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import numpy as np
import tifffile


SERVICE_URL = "https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_1km/ImageServer/exportImage"
OUTPUT_DIR = Path("assets/population")
OUTPUT_GRID = OUTPUT_DIR / "worldpop_total_population_2020_1440x720.npy"
OUTPUT_META = OUTPUT_DIR / "worldpop_total_population_2020_1440x720.json"
MAX_VALID_POP_PER_PIXEL = 1_000_000.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "f": "json",
        "bbox": "-180,-90,180,90",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": "1440,720",
        "format": "tiff",
        "pixelType": "F32",
        "time": "2020-01-01",
        "noData": 0,
    }
    export_url = f"{SERVICE_URL}?{urlencode(params)}"
    with urlopen(export_url, timeout=120) as response:
        export_meta = json.loads(response.read())

    tmp_tif = OUTPUT_DIR / "_worldpop_export_tmp.tif"
    urlretrieve(export_meta["href"], tmp_tif)
    population = tifffile.imread(tmp_tif).astype(np.float32, copy=False)
    tmp_tif.unlink(missing_ok=True)

    valid = np.isfinite(population) & (population >= 0) & (population <= MAX_VALID_POP_PER_PIXEL)
    population = np.where(valid, population, 0.0).astype(np.float32, copy=False)
    np.save(OUTPUT_GRID, population)

    metadata = {
        "source": "WorldPop Total Population 1km ImageServer",
        "source_url": "https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_1km/ImageServer",
        "export_url": export_url,
        "download_href": export_meta["href"],
        "year": 2020,
        "shape": list(population.shape),
        "bbox": [-180, -90, 180, 90],
        "max_valid_pop_per_pixel": MAX_VALID_POP_PER_PIXEL,
        "population_sum_in_downsampled_grid": float(population.sum()),
        "license": "WorldPop datasets are licensed under Creative Commons Attribution 4.0 International.",
        "citation": (
            "WorldPop (www.worldpop.org - School of Geography and Environmental Science, "
            "University of Southampton; Department of Geography and Geosciences, University of Louisville; "
            "Departement de Geographie, Universite de Namur) and CIESIN, Columbia University (2018). "
            "Global High Resolution Population Denominators Project."
        ),
    }
    OUTPUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_GRID} shape={population.shape} sum={population.sum():.2f}")
    print(f"wrote {OUTPUT_META}")


if __name__ == "__main__":
    main()
