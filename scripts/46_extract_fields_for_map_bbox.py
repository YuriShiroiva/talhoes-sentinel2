from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import box, mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "fields_yuri.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sentinel2" / "map_selected_fields"


def fix_geometry(geom):
    if geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    if fixed.is_empty or not fixed.is_valid:
        return None
    return fixed


def geom_in_lonlat_range(geom) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def extract_fields(fields_csv: Path, bbox_lonlat: tuple[float, float, float, float], output_dir: Path) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    selection = box(min_lon, min_lat, max_lon, max_lat)

    output_dir.mkdir(parents=True, exist_ok=True)
    features = []
    rows = []
    stats = {
        "rows": 0,
        "invalid_geom": 0,
        "source_out_of_range": 0,
        "outside_bbox": 0,
        "kept": 0,
    }

    reader = pd.read_csv(
        fields_csv,
        chunksize=5000,
        dtype="string",
        keep_default_na=False,
    )
    for chunk in reader:
        for row in chunk.to_dict("records"):
            stats["rows"] += 1
            try:
                geom = fix_geometry(wkt.loads(str(row.get("geom", ""))))
            except Exception:
                geom = None

            if geom is None:
                stats["invalid_geom"] += 1
                continue
            if not geom_in_lonlat_range(geom):
                stats["source_out_of_range"] += 1
                continue
            if not geom.intersects(selection):
                stats["outside_bbox"] += 1
                continue

            clipped = geom.intersection(selection)
            if clipped.is_empty:
                stats["outside_bbox"] += 1
                continue

            stats["kept"] += 1
            props = {
                "id": row.get("id", ""),
                "field_id": row.get("field_id", ""),
                "organization_id": row.get("organization_id", ""),
                "field_name": row.get("field_name", ""),
                "type": row.get("type", ""),
            }
            features.append({"type": "Feature", "properties": props, "geometry": mapping(clipped)})
            rows.append({**props, "geom": clipped.wkt})

    geojson_path = output_dir / "selected_fields.geojson"
    csv_path = output_dir / "selected_fields.csv"
    summary_path = output_dir / "selection_summary.csv"

    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)

    print(f"Fields dentro da selecao: {stats['kept']}")
    print(f"GeoJSON: {geojson_path}")
    print(f"CSV: {csv_path}")
    print(f"Resumo: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai fields que intersectam uma bbox selecionada no mapa.")
    parser.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="BBox em EPSG:4326, igual ao que o mapa normalmente envia.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fields_csv.exists():
        raise FileNotFoundError(args.fields_csv)

    min_lon, min_lat, max_lon, max_lat = args.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("--bbox precisa estar na ordem: min_lon min_lat max_lon max_lat")

    extract_fields(args.fields_csv, (min_lon, min_lat, max_lon, max_lat), args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
