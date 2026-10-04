"""Schemas Pydantic do serviço HTTP."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictAreaRequest(BaseModel):
    """Requisição de inferência por área: o backend busca a imagem no provider."""

    bounds: list[float] = Field(..., min_length=4, max_length=4,
                                description="(min_lon, min_lat, max_lon, max_lat) em EPSG:4326")
    provider: str | None = Field(None, description="mock | google (padrão: FB_PROVIDER do ambiente)")
    max_dim: int = Field(1024, ge=64, le=4096, description="Maior dimensão da imagem buscada (px)")
    field_threshold: float = Field(0.5, ge=0.05, le=0.95)
    boundary_threshold: float = Field(0.5, ge=0.05, le=0.95)
    min_object_area: int = Field(64, ge=0, description="Área mínima de talhão (px)")
    simplify_tolerance: float = Field(0.0, ge=0.0, description="Simplificação (unidades do CRS)")
    monitor_log: bool = Field(False, description="Acrescenta a cena ao log de monitoramento do serviço")


class ManualTestRequest(BaseModel):
    """Caso manual: area selecionada + respostas esperadas desenhadas pelo usuario."""

    case_name: str | None = Field(None, description="Nome opcional do caso manual")
    area: dict = Field(..., description="Feature GeoJSON da area selecionada")
    answers: dict = Field(..., description="FeatureCollection GeoJSON dos talhoes manuais")
    metadata: dict = Field(default_factory=dict)


class JobInfo(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    elapsed_seconds: float | None = None
    error: str | None = None
    result: dict | None = None  # presente quando completed


class ModelEntry(BaseModel):
    name: str
    version: int
    registered_at: str
    best_score: float | None = None
    sha256: str
    config: dict
    metrics: dict
