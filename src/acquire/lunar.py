"""Lunar predictor surfaces (LOLA SLDEM2015, Diviner DLRE).

Stage:  parse the PDS3 parcels delivered by the PDS Geosciences node and write
them onto the same (360, 720) 0.5-degree grid used by the Earth stack, so
lunar cells can be rendered on the flatmap/globe and scored by
:mod:`src.compute.similarity` exactly like Earth cells.

Parcel layout
-------------
LOLA SLDEM2015 (lrolol_1xxx/sldem2015_128_60s_60n_000_360_float.img)
    PDS3 UNCOMPRESSED_FILE.  Lines=15360 (latitude -60..+60), LINE_SAMPLES=46080
    (longitude 0..360 E), SAMPLE_TYPE=PC_REAL, 32-bit float.  Elevation is the
    height above the 1737.4 km reference sphere, in KILOMETERS (SCALING_FACTOR=1,
    OFFSET=1737.4). Pixel size = 1/128 deg, i.e. 64 pixels per 0.5 deg cell.

Diviner DLRE (dgdr_st_avg_cyl_128_img.img)
    PDS3 UNCOMPRESSED_FILE.  Lines=20480 (latitude -80..+80), LINE_SAMPLES=46080,
    16-bit LSB integer. Surface temperature average in K, Ra = DN*0.02 + 0,
    offset 0, MISSING_CONSTANT=-32768.  A single global-mean product (no
    day/night pair).

Both parcels are re-projected onto the (360, 720) grid with a ``block_mean``
over 64x64 native pixels (one 0.5 degree cell), using the same convention as
the Earth terrain stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.acquire.build_predictor_stack import (
    CELL,
    NROW,
    NCOL,
    _block_mean,
    _fine_roughness,
    _slope_from_cell_field,
)
from src.acquire.download import REPO_ROOT, fetch, offline_mode, utc_now_iso

REPO_ROOT = Path(REPO_ROOT)
RAW_DIR = REPO_ROOT / "cache" / "raw"
DERIVED_DIR = REPO_ROOT / "cache" / "derived"
LUNAR_STACK_PATH = REPO_ROOT / "cache" / "lunar_predictors.zarr"

# ---------------------------------------------------------------------------
# Parcel constants (read from the .lbl files)
# ---------------------------------------------------------------------------
LOLA_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/sldem2015/global/float_img/sldem2015_128_60s_60n_000_360_float.img"
)
LOLA_LABEL_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/sldem2015/global/float_img/sldem2015_128_60s_60n_000_360_float.lbl"
)
LOLA_NAME = "sldem2015_128_60s_60n_000_360_float"

DIVINER_URL = (
    "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_gdr_l3/cylindrical/img/dgdr_st_avg_cyl_128_img.img"
)
DIVINER_LABEL_URL = (
    "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_gdr_l3/cylindrical/img/dgdr_st_avg_cyl_128_img.lbl"
)
DIVINER_NAME = "dgdr_st_avg_cyl_128_img"
DIVINER_SCALE = 0.02
DIVINER_OFFSET = 0.0
DIVINER_MISSING = -32768
DIVINER_LAT_MIN = -80.0
DIVINER_LAT_MAX = 80.0

# One 0.5 degree output cell is 64 x 64 native parcel pixels.
BLOCK = 64


# ---------------------------------------------------------------------------
# Parcel fetchers
# ---------------------------------------------------------------------------

def _downloaded(name: str) -> Path:
    """Resolve the local raw parcel, downloading on first use."""
    return fetch(name, offline=offline_mode(), timeout=1200).path


def _load_lola() -> np.ndarray:
    """Return the SLDEM2015 elevation grid (15360, 46080) in km.

    Raises :class:`BuildError` (not a vague reshape) if the parcel is still
    being downloaded / truncated, so the run fails clearly instead of
    silently on a partial file.
    """
    src = RAW_DIR / (LOLA_NAME + ".img")
    if not src.exists():
        src = _downloaded(LOLA_NAME + ".img")
    total = 15360 * 46080
    a = np.fromfile(src, dtype="<f4", count=total)
    if a.size < total:
        raise RuntimeError(
            f"SLDEM2015 parcel {src.name} is truncated: got {a.size} samples, "
            f"need {total} ({src.stat().st_size} of 2831155200 bytes). "
            "Re-run after the geosciences node finishes the transfer."
        )
    grid = a.reshape(15360, 46080)
    return np.where(np.isfinite(grid), grid, np.nan).astype(np.float32)


def _load_diviner() -> np.ndarray:
    """Return the Diviner cylinderothermal grid (20480, 46080) in K."""
    src = RAW_DIR / (DIVINER_NAME + ".img")
    if not src.exists():
        src = _downloaded(DIVINER_NAME + ".img")
    b = np.fromfile(src, dtype="<i2", count=20480 * 46080).reshape(20480, 46080)
    a = b.astype(np.float32) * DIVINER_SCALE + DIVINER_OFFSET
    a[b == DIVINER_MISSING] = np.nan
    return a


# ---------------------------------------------------------------------------
# Resampling onto the (360, 720) 0.5 deg Earth grid
# ---------------------------------------------------------------------------

def _lunar_dem_onto_grid() -> np.ndarray:
    """SLDEM2015 elevation -> (360, 720) km.

    The parcel and the output grid share the same 0.5 deg edges, so a 64x64
    block mean lands exactly on one output cell.  Parcel row r is at latitude
    ``60 - r/128`` deg (line 1 is the northernmost row), so output row ``i``
    (centre ``89.75 - 0.5 i``) corresponds to parcel block ``i - 60``; only
    ``|lat| <= 60`` deg is covered, so rows outside 60..299 are NaN.  Columns
    map one-to-one: parcel block ``j`` is at longitude ``0.5 j + 0.25`` deg E,
    the same as output column ``j``.
    """
    dem = _load_lola()
    blocks = _block_mean(dem, BLOCK)                 # (240, 720) km
    out = np.full((NROW, NCOL), np.nan, dtype=np.float32)
    start = 60                                       # first covered output row
    out[start : start + blocks.shape[0], :] = blocks
    return out


def _lunar_temperature_onto_grid() -> np.ndarray:
    """Diviner cylinderothermal -> (360, 720) K.

    Same alignment as the DEM: output row ``i`` (centre ``89.75 - 0.5 i``)
    corresponds to parcel block ``i - 20`` because the Diviner grid starts at
    +80 deg.  Only ``|lat| <= 80`` deg is covered, so rows outside 20..339 are
    NaN.  Columns map one-to-one.
    """
    dv = _load_diviner()
    blocks = _block_mean(dv, BLOCK)                  # (320, 720) K
    out = np.full((NROW, NCOL), np.nan, dtype=np.float32)
    start = 20                                       # first covered output row
    out[start : start + blocks.shape[0], :] = blocks
    return out


def stage_lunar(force: bool = False, offline: bool | None = None) -> dict[str, np.ndarray]:
    """Return the lunar predictor surface arrays on the (360, 720) grid."""
    keys = ("lunar_elevation", "lunar_slope", "lunar_roughness", "lunar_temperature")
    cached = {k: (DERIVED_DIR / f"{k}.npy").exists() for k in keys}
    if all(cached.values()) and not force:
        return {k: np.load(DERIVED_DIR / f"{k}.npy", allow_pickle=True) for k in keys}

    # Store elevation in metres (the parcel's native unit is km; the Earth
    # elevation criterion and the slope/roughness helpers all work in metres).
    elev = (_lunar_dem_onto_grid() * 1000.0).astype(np.float32)
    slope = _slope_from_cell_field(elev.astype(np.float64)).astype(np.float32)
    rough = _fine_roughness(elev.astype(np.float64), band=360).astype(np.float32)
    temp = _lunar_temperature_onto_grid().astype(np.float32)

    for k, a in zip(keys, (elev, slope, rough, temp)):
        np.save(DERIVED_DIR / f"{k}.npy", a)

    return {k: np.load(DERIVED_DIR / f"{k}.npy", allow_pickle=True) for k in keys}


def lunar_stack(force: bool = False, offline: bool | None = None) -> Path:
    """Write ``cache/lunar_predictors.zarr`` with the same group schema as the
    Earth stack; all arrays are (360, 720) float32, NaN outside valid terrain."""
    stack = stage_lunar(force, offline)
    import zarr

    root = zarr.open_group(str(LUNAR_STACK_PATH), mode="w", zarr_format=3)
    for k, v in stack.items():
        arr = root.create_array(k, shape=(NROW, NCOL), dtype="f4",
                                 chunks=(NROW, NCOL), fill_value=np.nan)
        arr[:] = v

    root.attrs["schema_version"] = "2.0"
    root.attrs["created_utc"] = utc_now_iso()
    root.attrs["sources"] = [
        {"dataset_id": "nasa_lola_sldem2015",
         "title": "SLDEM2015 lunar elevation model (LOLA + SELENE TC)",
         "url": LOLA_LABEL_URL,
         "used_for": "lunar_elevation"},
        {"dataset_id": "nasa_lro_diviner_cylinder_mean_temperature",
         "title": "Diviner global cylindrical surface-temperature mean",
         "url": DIVINER_LABEL_URL,
         "used_for": "lunar_temperature"},
    ]
    root.attrs["grid"] = {
        "crs": "EPSG:4326",
        "shape": [NROW, NCOL],
        "cell_degrees": CELL,
        "row0_centre_latitude": 89.75,
        "column0_centre_longitude": -179.75,
        "ordering": "row-major, north to south then west to east",
    }
    root.attrs["note"] = (
        "Lunar stack built from PDS3 parcels on the geosciences node. "
        "SLDEM covers latitude +/-60 deg; Diviner covers +/-80 deg. "
        "Neutron flux is intentionally excluded until a global LEND raster is "
        "ingested (LRO LEND reduced records are per-orbit only)."
    )
    return LUNAR_STACK_PATH


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m src.acquire.lunar [--force] [--offline]``.

    Builds ``cache/lunar_predictors.zarr`` from the raw PDS3 parcels cached
    under ``cache/raw``.  Re-running is cheap once the arrays are derived.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build the lunar predictor stack.")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    args = parser.parse_args(argv)

    offline = True if args.offline else None
    started = time.time()
    path = lunar_stack(force=args.force, offline=offline)
    print(f"[lunar] wrote {path} in {time.time() - started:.1f}s")
    for key, values in stage_lunar(force=False, offline=offline).items():
        finite = int(np.isfinite(values).sum())
        print(f"[lunar] {key}: {finite} valid cells, "
              f"{np.nanmin(values):.2f} .. {np.nanmax(values):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
