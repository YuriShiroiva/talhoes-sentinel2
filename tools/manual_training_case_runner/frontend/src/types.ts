/** Tipos compartilhados do app (espelham os schemas do backend). */

export type Bounds = [number, number, number, number]; // min_lon, min_lat, max_lon, max_lat
export type LonLat = [number, number];

export interface ManualField {
  id: string;
  name: string;
  coordinates: LonLat[];
  created_at: string;
}

export interface ManualTrainingCase {
  case_name: string;
  area: {
    type: "Feature";
    properties: Record<string, unknown>;
    geometry: { type: "Polygon"; coordinates: LonLat[][] };
  };
  answers: FeatureCollection;
  metadata: Record<string, unknown>;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: { type: string; coordinates: unknown };
    properties: Record<string, unknown>;
  }>;
  crs_note?: string;
}

export interface QualityStats {
  n_fields: number;
  field_area_ratio: number;
  mean_field_prob: number;
  median_field_area: number;
  total_field_area: number;
}

export interface PredictResult {
  fields: FeatureCollection;
  boundaries: FeatureCollection;
  metadata: {
    crs: string;
    model: string;
    package_version: string;
    parameters: Record<string, unknown>;
    source: Record<string, unknown>;
  };
  quality: QualityStats;
  warnings: string[];
}

export interface JobInfo {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  elapsed_seconds: number | null;
  error: string | null;
  result: PredictResult | null;
}

export interface InferenceParams {
  provider: string;
  max_dim: number;
  field_threshold: number;
  boundary_threshold: number;
  min_object_area: number;
  simplify_tolerance: number;
  monitor_log: boolean;
}

export const DEFAULT_PARAMS: InferenceParams = {
  provider: "esri_rgb",
  max_dim: 1024,
  field_threshold: 0.6,
  boundary_threshold: 0.45,
  min_object_area: 1000,
  simplify_tolerance: 0.00004,
  monitor_log: false,
};
