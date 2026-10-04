/** Cliente tipado da API do backend (FastAPI). */

import type { Bounds, InferenceParams, JobInfo, ManualTrainingCase, PredictResult } from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      /* corpo não-JSON: mantém o status */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; model: string; provider_default: string }>("/health"),
  models: () => request<{ models: Array<{ name: string; version: number }>; loaded: string }>("/models"),
  predictArea: (bounds: Bounds, params: InferenceParams) =>
    request<{ job_id: string; status: string }>("/predict-area", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bounds, ...params }),
    }),
  predictScreenshot: (file: File, bounds: Bounds, params: InferenceParams) => {
    const body = new FormData();
    body.append("image", file);
    body.append("min_lon", String(bounds[0]));
    body.append("min_lat", String(bounds[1]));
    body.append("max_lon", String(bounds[2]));
    body.append("max_lat", String(bounds[3]));
    body.append("mode", "screenshot_rgb");
    body.append("field_threshold", String(params.field_threshold));
    body.append("boundary_threshold", String(params.boundary_threshold));
    body.append("min_object_area", String(params.min_object_area));
    body.append("simplify_tolerance", String(params.simplify_tolerance));
    return request<PredictResult>("/predict", { method: "POST", body });
  },
  jobStatus: (jobId: string) => request<JobInfo>(`/jobs/${jobId}`),
  saveManualTest: (payload: ManualTrainingCase) =>
    request<{ status: string; case_dir: string; field_count: number; files: Record<string, string> }>("/manual-tests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  exportUrl: (jobId: string, format: string, layer: string) =>
    `${BASE}/exports/${jobId}?format=${format}&layer=${layer}`,
};
