from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fields" / "testes_manuais"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "manual_case"


def rectangle_ring(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[list[float]]:
    return [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]


def area_feature(case_name: str, bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "kind": "area",
            "role": "input_area",
            "name": case_name,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [rectangle_ring(*bounds)],
        },
    }


def empty_answers() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def load_answers(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        raise ValueError("answers precisa ser GeoJSON FeatureCollection ou Feature.")

    normalized = []
    created_at = datetime.now().isoformat(timespec="seconds")
    for index, feature in enumerate(features, start=1):
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        properties = dict(feature.get("properties") or {})
        properties.setdefault("kind", "talhao_resposta")
        properties.setdefault("role", "expected_answer")
        properties.setdefault("index", index)
        properties.setdefault("created_at", created_at)
        normalized.append({"type": "Feature", "properties": properties, "geometry": geometry})

    return {"type": "FeatureCollection", "features": normalized}


def write_case(
    case_name: str,
    bounds: tuple[float, float, float, float],
    answers: dict[str, Any],
    output_root: Path,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = output_root / f"{stamp}_{safe_name(case_name)}"
    case_dir.mkdir(parents=True, exist_ok=False)

    area = area_feature(case_name, bounds)
    training_case = {
        "type": "field_boundary_manual_training_case",
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "area": area,
        "answers": answers,
        "metadata": {
            "case_name": case_name,
            "crs": "EPSG:4326",
            "bounds": list(bounds),
            "field_count": len(answers.get("features", [])),
            "purpose": "manual_ground_truth_for_training",
            "source": "manual_user_annotation",
        },
    }

    (case_dir / "area.geojson").write_text(json.dumps(area, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "answers.geojson").write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "training_case.json").write_text(
        json.dumps(training_case, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return case_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um caso REAL de treino manual: area selecionada + talhoes resposta desenhados pelo usuario."
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Area real selecionada em EPSG:4326.",
    )
    parser.add_argument(
        "--answers-geojson",
        type=Path,
        help="GeoJSON com talhoes reais desenhados manualmente. Se omitido, cria answers vazio.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_lon, min_lat, max_lon, max_lat = args.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("--bbox precisa estar na ordem min_lon min_lat max_lon max_lat")

    answers = load_answers(args.answers_geojson) if args.answers_geojson else empty_answers()
    case_dir = write_case(args.case_name, (min_lon, min_lat, max_lon, max_lat), answers, args.output_root)

    print(f"Caso real criado: {case_dir}")
    print(f"Talhoes resposta: {len(answers.get('features', []))}")
    print(f"- area: {case_dir / 'area.geojson'}")
    print(f"- answers: {case_dir / 'answers.geojson'}")
    print(f"- training_case: {case_dir / 'training_case.json'}")


if __name__ == "__main__":
    main()
