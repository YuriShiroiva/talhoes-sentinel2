"""Polygonization helpers for predicted field masks.

The functions in this module intentionally stay lightweight: they depend on
Rasterio/Shapely when vector output is requested, but the segmentation pipeline
can still write raster masks without using a GIS stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine

try:
    from shapely.geometry import mapping, shape
except ImportError:  # pragma: no cover - handled at runtime by caller.
    mapping = None
    shape = None


def _pixel_area(transform: Affine, crs: Any | None) -> float | None:
    """Return approximate pixel area in square meters when the CRS is projected."""

    if crs is None:
        return None

    try:
        if not crs.is_projected:
            return None
    except AttributeError:
        return None

    return abs(float(transform.a) * float(transform.e))


def polygonize_mask(
    mask: np.ndarray,
    transform: Affine,
    crs: Any | None = None,
    min_area_px: int = 1,
    simplify_tolerance: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert a binary mask into GeoJSON features and tabular attributes."""

    if shape is None or mapping is None:
        raise RuntimeError("Shapely is required for polygonization. Install with: pip install shapely")

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    pixel_area = _pixel_area(transform, crs)
    # area do pixel nas unidades do CRS (graus^2 ou m^2) — sempre disponivel pelo transform,
    # garante que o filtro de area funcione mesmo em CRS geografico (EPSG:4326).
    pixel_footprint = abs(float(transform.a) * float(transform.e)) or 1.0
    features: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for idx, (geom, value) in enumerate(shapes(binary, mask=binary.astype(bool), transform=transform), start=1):
        if int(value) != 1:
            continue

        polygon = shape(geom)
        area_px = float(polygon.area / pixel_footprint)

        if area_px < min_area_px:
            continue

        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

        area_m2 = float(polygon.area) if pixel_area else None

        props = {
            "candidate_id": idx,
            "area_px": area_px,
            "area_m2": area_m2,
            "area_ha": None if area_m2 is None else area_m2 / 10000.0,
        }

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(polygon),
            }
        )
        rows.append(props)

    return features, rows


def write_geojson(
    features: Iterable[Dict[str, Any]],
    output_path: Path,
    crs: Any | None = None,
) -> None:
    """Write features to a GeoJSON FeatureCollection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": list(features),
    }

    if crs is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": str(crs)},
        }

    output_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")


def read_raster_grid(path: Path) -> Tuple[Affine, Any | None, int, int]:
    """Read transform/CRS/shape from a raster-like file when Rasterio supports it."""

    with rasterio.open(path) as src:
        return src.transform, src.crs, src.height, src.width
