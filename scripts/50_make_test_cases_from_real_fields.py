from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, box, mapping
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "reais" / "fields_yuri.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fields" / "testes_simulados"

METERS_PER_DEG_LAT = 111_320.0
WANTED_PROP_COLUMNS = ["id", "field_id", "organization_id", "field_name", "type"]


def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    if fixed.is_empty or not fixed.is_valid:
        return None
    return fixed


def in_lonlat_range(geom) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def load_fields(
    fields_csv: Path,
    geom_column: str,
    chunksize: int,
    max_fields: int | None,
) -> tuple[list[Any], list[dict[str, str]], dict[str, int]]:
    available = pd.read_csv(fields_csv, nrows=0).columns.tolist()
    if geom_column not in available:
        raise ValueError(f"Coluna de geometria '{geom_column}' nao existe. Colunas: {available}")
    prop_columns = [c for c in WANTED_PROP_COLUMNS if c in available]
    usecols = list(dict.fromkeys([geom_column, *prop_columns]))

    geoms: list[Any] = []
    props: list[dict[str, str]] = []
    seen_geoms: set[bytes] = set()
    stats = {"rows": 0, "missing_geom": 0, "invalid_geom": 0, "out_of_range": 0, "duplicates": 0, "kept": 0}

    reader = pd.read_csv(
        fields_csv,
        usecols=usecols,
        chunksize=chunksize,
        dtype="string",
        keep_default_na=False,
    )
    for chunk in reader:
        for row in chunk.to_dict("records"):
            if max_fields is not None and stats["kept"] >= max_fields:
                return geoms, props, stats
            stats["rows"] += 1

            geom_text = str(row.get(geom_column, "") or "")
            if not geom_text:
                stats["missing_geom"] += 1
                continue
            try:
                geom = fix_geometry(wkt.loads(geom_text))
            except Exception:
                geom = None
            if geom is None:
                stats["invalid_geom"] += 1
                continue
            if not in_lonlat_range(geom):
                stats["out_of_range"] += 1
                continue

            geom_key = geom.wkb
            if geom_key in seen_geoms:
                stats["duplicates"] += 1
                continue
            seen_geoms.add(geom_key)

            geoms.append(geom)
            props.append({c: str(row.get(c, "") or "") for c in prop_columns})
            stats["kept"] += 1

    return geoms, props, stats


def cluster_by_proximity(
    centroids: np.ndarray,
    neighbor_dist_deg: float,
    max_per_case: int,
) -> list[list[int]]:
    """Agrupa fields vizinhos por proximidade (region growing limitado por contagem)."""
    n = len(centroids)
    points = [Point(float(x), float(y)) for x, y in centroids]
    tree = STRtree(points)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    for seed in range(n):
        if assigned[seed]:
            continue
        cluster = [seed]
        assigned[seed] = True
        frontier = [seed]
        while frontier and len(cluster) < max_per_case:
            cur = frontier.pop()
            cx, cy = float(centroids[cur][0]), float(centroids[cur][1])
            query_box = box(cx - neighbor_dist_deg, cy - neighbor_dist_deg,
                            cx + neighbor_dist_deg, cy + neighbor_dist_deg)
            for j in tree.query(query_box):
                j = int(j)
                if assigned[j]:
                    continue
                if points[cur].distance(points[j]) <= neighbor_dist_deg:
                    assigned[j] = True
                    cluster.append(j)
                    frontier.append(j)
                    if len(cluster) >= max_per_case:
                        break
        clusters.append(cluster)

    return clusters


def padded_bounds(
    tight_bounds: tuple[float, float, float, float],
    margin_frac: float,
    min_margin_m: float,
    min_span_m: float,
) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    """Expande a bbox justa dos talhoes para simular o enquadramento de um print."""
    minx, miny, maxx, maxy = tight_bounds
    lat0 = (miny + maxy) / 2.0
    m_per_deg_lon = max(METERS_PER_DEG_LAT * math.cos(math.radians(lat0)), 1.0)
    m_per_deg_lat = METERS_PER_DEG_LAT

    width_m = (maxx - minx) * m_per_deg_lon
    height_m = (maxy - miny) * m_per_deg_lat
    span_m = max(width_m, height_m)

    margin_m = max(margin_frac * span_m, min_margin_m)
    pad_x_m = margin_m
    pad_y_m = margin_m
    if width_m + 2 * pad_x_m < min_span_m:
        pad_x_m = (min_span_m - width_m) / 2.0
    if height_m + 2 * pad_y_m < min_span_m:
        pad_y_m = (min_span_m - height_m) / 2.0

    pad_x_deg = pad_x_m / m_per_deg_lon
    pad_y_deg = pad_y_m / m_per_deg_lat
    padded = (minx - pad_x_deg, miny - pad_y_deg, maxx + pad_x_deg, maxy + pad_y_deg)

    info = {
        "margin_m": round(margin_m, 1),
        "pad_x_m": round(pad_x_m, 1),
        "pad_y_m": round(pad_y_m, 1),
        "print_width_m": round((padded[2] - padded[0]) * m_per_deg_lon, 1),
        "print_height_m": round((padded[3] - padded[1]) * m_per_deg_lat, 1),
        "fields_width_m": round(width_m, 1),
        "fields_height_m": round(height_m, 1),
    }
    return padded, info


def rectangle_ring(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    min_lon, min_lat, max_lon, max_lat = bounds
    return [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]


def build_area_feature(case_name: str, bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"kind": "area", "role": "input_area", "name": case_name},
        "geometry": {"type": "Polygon", "coordinates": [rectangle_ring(bounds)]},
    }


def build_answers(
    cluster: list[int],
    geoms: list[Any],
    props: list[dict[str, str]],
    case_name: str,
    created_at: str,
) -> dict[str, Any]:
    features = []
    for index, field_idx in enumerate(cluster, start=1):
        properties: dict[str, Any] = {
            "kind": "talhao_resposta",
            "role": "expected_answer",
            "index": index,
            "name": f"{case_name}_t{index}",
            "created_at": created_at,
        }
        properties.update(props[field_idx])
        features.append(
            {"type": "Feature", "properties": properties, "geometry": mapping(geoms[field_idx])}
        )
    return {"type": "FeatureCollection", "features": features}


def write_case(
    case_dir: Path,
    case_name: str,
    cluster: list[int],
    geoms: list[Any],
    props: list[dict[str, str]],
    tight_bounds: tuple[float, float, float, float],
    padded: tuple[float, float, float, float],
    margin_info: dict[str, float],
    created_at: str,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    area = build_area_feature(case_name, padded)
    answers = build_answers(cluster, geoms, props, case_name, created_at)
    training_case = {
        "type": "field_boundary_manual_training_case",
        "schema_version": 1,
        "created_at": created_at,
        "area": area,
        "answers": answers,
        "metadata": {
            "case_name": case_name,
            "crs": "EPSG:4326",
            "bounds": list(padded),
            "fields_tight_bounds": list(tight_bounds),
            "field_count": len(cluster),
            "purpose": "simulated_print_for_testing",
            "source": "simulated_from_real_fields",
            "margin": margin_info,
        },
    }
    (case_dir / "area.geojson").write_text(json.dumps(area, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "answers.geojson").write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "training_case.json").write_text(
        json.dumps(training_case, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_cases(
    fields_csv: Path,
    output_root: Path,
    geom_column: str,
    chunksize: int,
    max_fields: int | None,
    neighbor_dist_m: float,
    max_per_case: int,
    min_fields_per_case: int,
    margin_frac: float,
    min_margin_m: float,
    min_span_m: float,
    limit_cases: int | None,
    case_prefix: str,
) -> None:
    print(f"Lendo fields de {fields_csv} ...")
    geoms, props, stats = load_fields(fields_csv, geom_column, chunksize, max_fields)
    print(f"Fields lidos: {stats['rows']} | unicos validos: {stats['kept']} | "
          f"duplicados: {stats['duplicates']} | invalidos: {stats['invalid_geom']} | "
          f"fora de range: {stats['out_of_range']}")
    if stats["kept"] == 0:
        raise RuntimeError("Nenhum field valido em lon/lat encontrado.")

    centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)
    lat0 = float(np.median(centroids[:, 1]))
    neighbor_dist_deg = neighbor_dist_m / METERS_PER_DEG_LAT
    print(f"Agrupando por proximidade (<= {neighbor_dist_m:.0f} m, max {max_per_case} talhoes/caso) ...")
    clusters = cluster_by_proximity(centroids, neighbor_dist_deg, max_per_case)

    kept_clusters = [c for c in clusters if len(c) >= min_fields_per_case]
    kept_clusters.sort(key=len, reverse=True)
    dropped_small = len(clusters) - len(kept_clusters)
    if limit_cases is not None:
        kept_clusters = kept_clusters[:limit_cases]

    print(f"Clusters formados: {len(clusters)} | com >= {min_fields_per_case} talhoes: "
          f"{len(clusters) - dropped_small} | descartados (poucos talhoes): {dropped_small}")
    print(f"Casos a gerar: {len(kept_clusters)}")

    output_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    index_rows: list[dict[str, Any]] = []
    all_areas: list[dict[str, Any]] = []

    for case_idx, cluster in enumerate(kept_clusters, start=1):
        sub = [geoms[i] for i in cluster]
        bounds = sub[0].bounds
        minx, miny, maxx, maxy = bounds
        for g in sub[1:]:
            bx = g.bounds
            minx, miny = min(minx, bx[0]), min(miny, bx[1])
            maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])
        tight_bounds = (minx, miny, maxx, maxy)
        padded, margin_info = padded_bounds(tight_bounds, margin_frac, min_margin_m, min_span_m)

        case_name = f"{case_prefix}_{case_idx:04d}_{len(cluster)}talhoes"
        case_dir = output_root / case_name
        write_case(case_dir, case_name, cluster, geoms, props, tight_bounds, padded, margin_info, created_at)

        all_areas.append(build_area_feature(case_name, padded))
        index_rows.append({
            "case_name": case_name,
            "field_count": len(cluster),
            "min_lon": padded[0], "min_lat": padded[1], "max_lon": padded[2], "max_lat": padded[3],
            "print_width_m": margin_info["print_width_m"],
            "print_height_m": margin_info["print_height_m"],
            "margin_m": margin_info["margin_m"],
            "bbox_arg": f"{padded[0]:.8f} {padded[1]:.8f} {padded[2]:.8f} {padded[3]:.8f}",
        })

    index_path = output_root / "cases_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["case_name"])
        writer.writeheader()
        writer.writerows(index_rows)

    areas_path = output_root / "all_areas.geojson"
    with areas_path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": all_areas}, f, ensure_ascii=False)

    print("\nConcluido.")
    print(f"Casos gerados: {len(index_rows)} em {output_root}")
    print(f"Indice: {index_path}")
    print(f"Areas (todas): {areas_path}")
    if index_rows:
        example = index_rows[0]
        print("\nExemplo para baixar Sentinel-2 desse print (script 47):")
        print(f"  python scripts/47_download_sentinel2_for_fields.py --bbox {example['bbox_arg']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera casos de teste (area + answers) a partir dos talhoes reais, "
        "com bordas maiores para simular o print do cliente no mapa."
    )
    parser.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geom-column", type=str, default="geom")
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--max-fields", type=int, default=None, help="Limita fields lidos (util para smoke).")
    parser.add_argument("--neighbor-dist-m", type=float, default=600.0,
                        help="Distancia maxima entre centroides para agrupar talhoes vizinhos.")
    parser.add_argument("--max-per-case", type=int, default=10, help="Maximo de talhoes por print/caso.")
    parser.add_argument("--min-fields-per-case", type=int, default=2,
                        help="Minimo de talhoes para o caso ser gerado.")
    parser.add_argument("--margin-frac", type=float, default=0.5,
                        help="Borda como fracao do lado da bbox dos talhoes (0.5 = ~50%% por lado).")
    parser.add_argument("--min-margin-m", type=float, default=150.0, help="Borda minima em metros.")
    parser.add_argument("--min-span-m", type=float, default=800.0,
                        help="Tamanho minimo do print em metros (por eixo).")
    parser.add_argument("--limit-cases", type=int, default=None, help="Gera no maximo N casos.")
    parser.add_argument("--case-prefix", type=str, default="sim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fields_csv.exists():
        raise FileNotFoundError(args.fields_csv)
    if args.min_fields_per_case < 1:
        raise ValueError("--min-fields-per-case precisa ser >= 1.")
    if args.max_per_case < args.min_fields_per_case:
        raise ValueError("--max-per-case precisa ser >= --min-fields-per-case.")

    generate_cases(
        fields_csv=args.fields_csv,
        output_root=args.output_root,
        geom_column=args.geom_column,
        chunksize=args.chunksize,
        max_fields=args.max_fields,
        neighbor_dist_m=args.neighbor_dist_m,
        max_per_case=args.max_per_case,
        min_fields_per_case=args.min_fields_per_case,
        margin_frac=args.margin_frac,
        min_margin_m=args.min_margin_m,
        min_span_m=args.min_span_m,
        limit_cases=args.limit_cases,
        case_prefix=args.case_prefix,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
