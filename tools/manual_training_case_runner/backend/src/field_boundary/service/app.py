"""Serviço HTTP de inferência (FastAPI) — backend do web app.

Endpoints:
- GET  /health            status + modelo carregado
- GET  /model-info        metadados completos do(s) checkpoint(s)
- GET  /models            modelos do registry local (cli model register)
- POST /predict           imagem enviada (multipart) -> GeoJSON (síncrono)
- POST /predict-area      bounds -> provider de satélite -> job assíncrono
- GET  /jobs/{job_id}     status/resultado do job (pending|running|completed|failed)
- GET  /exports/{job_id}  download do resultado (geojson | shp | gpkg)

Configuração por variáveis de ambiente (ver .env.example):
  FB_PROVIDER (mock|google), GOOGLE_MAPS_API_KEY, FB_MAX_AREA_KM2,
  FB_EXPORT_DIR, FB_MONITOR_LOG.

Subir: python -m field_boundary.cli serve --checkpoint best.pt
Notas: processamento de /predict-area roda em ThreadPoolExecutor local;
jobs vivem em memória (somem em restart). Requer a extra [service].
"""

import dataclasses
import io
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from field_boundary import __version__
from field_boundary.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_AREA_KM2 = 50.0


def _settings() -> dict:
    return {
        "provider": os.environ.get("FB_PROVIDER", "mock"),
        "max_area_km2": float(os.environ.get("FB_MAX_AREA_KM2", DEFAULT_MAX_AREA_KM2)),
        "export_dir": Path(os.environ.get("FB_EXPORT_DIR", tempfile.gettempdir())) / "fb_exports",
        "manual_test_dir": Path(os.environ.get("FB_MANUAL_TEST_DIR", r"C:\tcc_sentinel2\fields\testes_manuais")),
        "monitor_log": os.environ.get("FB_MONITOR_LOG"),
        "registry_dir": os.environ.get("FB_REGISTRY_DIR", "models"),
    }


def create_app(predictor, model_info=None, pipeline_config=None, settings=None, provider_factory=None):
    """Cria o app FastAPI. `predictor`/`provider_factory` injetáveis para teste."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    from field_boundary.export.writers import records_to_feature_collection
    from field_boundary.monitoring import append_stats, compute_scene_stats
    from field_boundary.pipeline import PipelineConfig, prediction_to_polygons
    from field_boundary.postprocessing.raster import RasterPostprocessConfig
    from field_boundary.postprocessing.vector import VectorPostprocessConfig
    from field_boundary.satellite import bounds_area_km2, get_provider, validate_bounds
    from field_boundary.service.jobs import JobStore
    from field_boundary.service.schemas import JobInfo, ManualTestRequest, PredictAreaRequest

    app = FastAPI(title="field-boundary-dl", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    cfg = {**_settings(), **(settings or {})}
    default_pipeline = pipeline_config or PipelineConfig()
    info = model_info or {}
    jobs = JobStore(max_workers=1)
    provider_factory = provider_factory or get_provider

    # ------------------------------------------------------------- helpers

    def _pipeline_config(req: PredictAreaRequest) -> PipelineConfig:
        return PipelineConfig(
            field_threshold=req.field_threshold,
            boundary_threshold=req.boundary_threshold,
            raster=RasterPostprocessConfig(min_object_area=req.min_object_area),
            vector=VectorPostprocessConfig(simplify_tolerance=req.simplify_tolerance),
        )

    def _result_payload(result, output, params: dict, provider_warnings: list[str], source_meta: dict) -> dict:
        stats = compute_scene_stats(result, output)
        warnings = list(provider_warnings)
        if output.crs is None:
            warnings.append("Saída em coordenadas de pixel (sem georreferenciamento).")
        return {
            "fields": records_to_feature_collection(output.fields, output.crs),
            "boundaries": records_to_feature_collection(output.boundaries, output.crs),
            "metadata": {
                "crs": output.crs or "pixel",
                "model": info.get("name", "desconhecido"),
                "package_version": __version__,
                "checkpoint_configs": info.get("configs", []),
                "parameters": params,
                "source": source_meta,
            },
            "quality": stats,
            "warnings": warnings,
        }

    # ------------------------------------------------------------ endpoints

    @app.get("/health")
    def health():
        return {"status": "ok", "package_version": __version__,
                "model": info.get("name", "desconhecido"), "provider_default": cfg["provider"]}

    @app.get("/model-info")
    def model_information():
        return info

    @app.get("/models")
    def models():
        from field_boundary.models.registry import list_models

        return {"models": list_models(cfg["registry_dir"]), "loaded": info.get("name")}

    @app.post("/manual-tests")
    def save_manual_test(req: ManualTestRequest):
        """Salva um caso manual: area.geojson + answers.geojson + training_case.json."""
        area = req.area
        answers = req.answers
        features = answers.get("features", []) if isinstance(answers, dict) else []
        if not isinstance(features, list) or not features:
            raise HTTPException(422, "Salve ao menos um talhao manual antes de gravar o caso.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_name = req.case_name or req.metadata.get("case_name") or "manual_test"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_name)).strip("_") or "manual_test"
        case_dir = Path(cfg["manual_test_dir"]) / f"{stamp}_{safe_name}"
        case_dir.mkdir(parents=True, exist_ok=False)

        payload = {
            "type": "field_boundary_manual_training_case",
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "area": area,
            "answers": answers,
            "metadata": {
                **req.metadata,
                "case_name": raw_name,
                "field_count": len(features),
                "crs": req.metadata.get("crs", "EPSG:4326"),
            },
        }

        (case_dir / "area.geojson").write_text(json.dumps(area, indent=2, ensure_ascii=False), encoding="utf-8")
        (case_dir / "answers.geojson").write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
        (case_dir / "training_case.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "status": "saved",
            "case_dir": str(case_dir),
            "files": {
                "area": str(case_dir / "area.geojson"),
                "answers": str(case_dir / "answers.geojson"),
                "training_case": str(case_dir / "training_case.json"),
            },
            "field_count": len(features),
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        min_lon: float | None = Form(None),
        min_lat: float | None = Form(None),
        max_lon: float | None = Form(None),
        max_lat: float | None = Form(None),
        mode: str = Form("model"),
        field_threshold: float = Form(0.55),
        boundary_threshold: float = Form(0.5),
        min_object_area: int = Form(800),
        simplify_tolerance: float = Form(0.00004),
    ):
        bounds_values = (min_lon, min_lat, max_lon, max_lat)
        bounds = None
        if any(v is not None for v in bounds_values):
            if any(v is None for v in bounds_values):
                raise HTTPException(422, "bounds incompletos: informe min_lon, min_lat, max_lon e max_lat")
            bounds = bounds_values

        suffix = Path(image.filename or "upload.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = Path(tmp.name)
        try:
            if mode in {"screenshot_rgb", "rgb_heuristic", "print"}:
                if bounds is None:
                    raise ValueError("Para print/screenshot, informe bounds para georreferenciar a saída.")
                from PIL import Image

                from field_boundary.geospatial.crs import affine_from_bounds
                from field_boundary.inference.rgb_heuristic import predict_rgb_heuristic
                from field_boundary.inference.screenshot import crop_likely_map_region
                from field_boundary.pipeline import PipelineConfig
                from field_boundary.postprocessing.raster import RasterPostprocessConfig
                from field_boundary.postprocessing.vector import VectorPostprocessConfig

                raw_img = np.asarray(Image.open(tmp_path).convert("RGB"))
                cropped, crop_meta = crop_likely_map_region(raw_img)
                transform = affine_from_bounds(bounds, cropped.shape[1], cropped.shape[0])
                result = predict_rgb_heuristic(
                    cropped,
                    transform=transform,
                    crs="EPSG:4326",
                    source=f"screenshot-rgb:{image.filename}",
                )
                pipeline_cfg = PipelineConfig(
                    field_threshold=field_threshold,
                    boundary_threshold=boundary_threshold,
                    split_into_instances=False,
                    raster=RasterPostprocessConfig(min_object_area=min_object_area),
                    vector=VectorPostprocessConfig(simplify_tolerance=simplify_tolerance),
                )
                output = prediction_to_polygons(result, pipeline_cfg)
                return _result_payload(
                    result,
                    output,
                    params={
                        "bounds": bounds,
                        "mode": mode,
                        "field_threshold": field_threshold,
                        "boundary_threshold": boundary_threshold,
                        "min_object_area": min_object_area,
                        "simplify_tolerance": simplify_tolerance,
                    },
                    provider_warnings=[
                        "Modo print/screenshot: usa imagem RGB enviada e bounds informados; resultado é aproximação.",
                        "Segmentação RGB de teste: não substitui um modelo treinado com prints anotados.",
                    ],
                    source_meta={"upload": image.filename, **crop_meta},
                )

            result = predictor.predict_image(tmp_path, bounds=bounds)
            output = prediction_to_polygons(result, default_pipeline)
            return _result_payload(
                result,
                output,
                params={"bounds": bounds, "mode": mode},
                provider_warnings=[],
                source_meta={"upload": image.filename},
            )
        except Exception as exc:
            logger.exception("Falha na inferência do serviço")
            raise HTTPException(500, f"Falha na inferência: {type(exc).__name__}: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/predict-area")
    def predict_area(req: PredictAreaRequest):
        bounds = tuple(req.bounds)
        error = validate_bounds(bounds)
        if error:
            raise HTTPException(422, error)
        area_km2 = bounds_area_km2(bounds)
        if area_km2 > cfg["max_area_km2"]:
            raise HTTPException(
                422,
                f"Área de {area_km2:.1f} km² excede o limite de {cfg['max_area_km2']:.1f} km² "
                "(FB_MAX_AREA_KM2). Selecione uma área menor.",
            )
        provider_name = req.provider or cfg["provider"]
        use_rgb_heuristic = provider_name in {"esri_rgb", "esri-rgb", "esri_heuristic"}
        fetch_provider_name = "esri" if use_rgb_heuristic else provider_name
        try:
            provider = provider_factory(fetch_provider_name)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        def work(job) -> dict:
            sat = provider.fetch(bounds, max_dim=req.max_dim)
            if sat.data.size == 0 or sat.data.shape[0] < 16 or sat.data.shape[1] < 16:
                raise ValueError(f"Imagem inválida/vazia do provider ({sat.data.shape}).")
            if use_rgb_heuristic:
                from field_boundary.inference.rgb_heuristic import predict_rgb_heuristic

                result = predict_rgb_heuristic(
                    sat.data,
                    transform=sat.transform,
                    crs=sat.crs,
                    source=f"{sat.provider}-rgb-heuristic:{job.job_id}",
                )
                sat.warnings.append(
                    "Modo heurístico RGB: usa imagem real Esri sem chave, mas não é deep learning treinado."
                )
            else:
                raw = predictor.predict_array(sat.data)
                result = dataclasses.replace(raw, transform=sat.transform, crs=sat.crs, source=f"{sat.provider}:{job.job_id}")
            pipeline_cfg = _pipeline_config(req)
            if use_rgb_heuristic:
                if pipeline_cfg.field_threshold < 0.60:
                    sat.warnings.append(
                        "Modo RGB aplicou limiar minimo 0.60 para evitar selecao excessiva de estradas, solo claro e talhoes vizinhos."
                    )
                pipeline_cfg = dataclasses.replace(
                    pipeline_cfg,
                    field_threshold=max(pipeline_cfg.field_threshold, 0.60),
                    split_into_instances=True,
                )
            output = prediction_to_polygons(result, pipeline_cfg)
            payload = _result_payload(
                result, output,
                params=req.model_dump(),
                provider_warnings=sat.warnings,
                source_meta={"provider": sat.provider, "bounds": list(sat.bounds), **sat.metadata},
            )
            # Exportações em disco para /exports/{job_id}
            from field_boundary.pipeline import export_output

            job.export_dir = Path(cfg["export_dir"]) / job.job_id
            export_output(output, job.export_dir, formats=("geojson",))
            payload["exports"] = {"geojson": True}
            job._output = output  # guarda para exportações sob demanda (shp/gpkg)
            if req.monitor_log and cfg["monitor_log"]:
                append_stats(compute_scene_stats(result, output), Path(cfg["monitor_log"]))
            return payload

        job = jobs.submit(work)
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/jobs/{job_id}", response_model=JobInfo)
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job {job_id} não encontrado (jobs vivem em memória até o restart).")
        return JobInfo(job_id=job.job_id, status=job.status, created_at=job.created_at,
                       elapsed_seconds=job.elapsed_seconds, error=job.error,
                       result=job.result if job.status == "completed" else None)

    @app.get("/exports/{job_id}")
    def export(job_id: str, format: str = "geojson", layer: str = "fields"):
        from field_boundary.export.writers import write_vector_file

        job = jobs.get(job_id)
        if job is None or job.status != "completed" or job.export_dir is None:
            raise HTTPException(404, f"Job {job_id} não encontrado ou não concluído.")
        if layer not in ("fields", "boundaries"):
            raise HTTPException(422, "layer deve ser fields ou boundaries")
        if format not in ("geojson", "shp", "gpkg"):
            raise HTTPException(422, "format deve ser geojson, shp ou gpkg")

        if format == "geojson":
            path = job.export_dir / f"{layer}.geojson"
            if not path.exists():
                raise HTTPException(404, f"Sem polígonos de {layer} neste job.")
            return Response(path.read_bytes(), media_type="application/geo+json",
                            headers={"Content-Disposition": f'attachment; filename="{layer}.geojson"'})

        # shp/gpkg: gera sob demanda (requer geopandas) e entrega (shp = zip dos sidecars)
        output = getattr(job, "_output", None)
        records = output.fields if layer == "fields" else output.boundaries
        if not records:
            raise HTTPException(404, f"Sem polígonos de {layer} neste job.")
        try:
            target = job.export_dir / f"{layer}.{format}"
            if not target.exists():
                write_vector_file(records, target, output.crs)
        except ImportError as exc:
            raise HTTPException(501, "Exportação shp/gpkg requer geopandas (extra [raster]).") from exc

        if format == "gpkg":
            return Response(target.read_bytes(), media_type="application/geopackage+sqlite3",
                            headers={"Content-Disposition": f'attachment; filename="{layer}.gpkg"'})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for sidecar in job.export_dir.glob(f"{layer}.*"):
                if sidecar.suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    z.write(sidecar, sidecar.name)
        return Response(buffer.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{layer}_shp.zip"'})

    return app


def build_app_from_checkpoints(checkpoint_paths, inference_config=None):
    """Carrega checkpoints e devolve o app pronto (usado pela CLI serve)."""
    from field_boundary.inference.engine import FieldBoundaryPredictor
    from field_boundary.models.checkpoint import load_payload, read_checkpoint_metadata

    predictor = FieldBoundaryPredictor.from_checkpoints(checkpoint_paths, inference_config)
    metas = [read_checkpoint_metadata(load_payload(Path(p))) for p in checkpoint_paths]
    info = {
        "name": metas[0].get("experiment_name", Path(checkpoint_paths[0]).stem) if metas else "desconhecido",
        "checkpoints": [str(p) for p in checkpoint_paths],
        "configs": metas,
    }
    return create_app(predictor, model_info=info)
