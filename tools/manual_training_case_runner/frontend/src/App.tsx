/** Aplicação principal: mapa + painéis, orquestra o fluxo de inferência. */

import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import ConfigPanel from "./components/ConfigPanel";
import LogPanel from "./components/LogPanel";
import { searchLocation } from "./api/geocode";
import MapView from "./components/MapView";
import ResultsPanel from "./components/ResultsPanel";
import { useInference } from "./hooks/useInference";
import { DEFAULT_PARAMS, type Bounds, type InferenceParams, type LonLat, type ManualField, type ManualTrainingCase } from "./types";

const MANUAL_TEST_STORAGE_KEY = "field-boundary-manual-test-v1";

export default function App() {
  const [drawing, setDrawing] = useState(false);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const [viewportBounds, setViewportBounds] = useState<Bounds | null>(null);
  const [params, setParams] = useState<InferenceParams>(DEFAULT_PARAMS);
  const [models, setModels] = useState<Array<{ name: string; version: number }>>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [showFields, setShowFields] = useState(true);
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [selectedFeature, setSelectedFeature] = useState<Record<string, unknown> | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [flyTo, setFlyTo] = useState<[number, number] | null>(null);
  const [searching, setSearching] = useState(false);
  const [manualDrawing, setManualDrawing] = useState(false);
  const [manualDraft, setManualDraft] = useState<LonLat[]>([]);
  const [manualFields, setManualFields] = useState<ManualField[]>([]);
  const [manualCaseName, setManualCaseName] = useState("");
  const [manualFieldName, setManualFieldName] = useState("");

  const addLog = useCallback(
    (message: string) => setLogs((prev) => [...prev, `${new Date().toLocaleTimeString()} ${message}`]),
    [],
  );
  const { status, result, error, jobId, elapsed, run, reset } = useInference(addLog);

  useEffect(() => {
    api.health()
      .then((h) => addLog(`Backend ok: modelo ${h.model}, provider padrão ${h.provider_default}`))
      .catch(() => addLog("AVISO: backend indisponível — verifique se o serviço está rodando."));
    api.models().then((m) => setModels(m.models)).catch(() => undefined);
  }, [addLog]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(MANUAL_TEST_STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as { bounds?: Bounds; fields?: ManualField[]; caseName?: string };
      if (saved.bounds) setBounds(saved.bounds);
      if (Array.isArray(saved.fields)) setManualFields(saved.fields);
      if (saved.caseName) setManualCaseName(saved.caseName);
    } catch {
      addLog("Anotações locais ignoradas: arquivo salvo inválido.");
    }
  }, [addLog]);

  useEffect(() => {
    localStorage.setItem(
      MANUAL_TEST_STORAGE_KEY,
      JSON.stringify({ bounds, fields: manualFields, caseName: manualCaseName }),
    );
  }, [bounds, manualFields, manualCaseName]);

  const onBoundsSelected = useCallback(
    (selected: Bounds) => {
      setBounds(selected);
      setDrawing(false);
      addLog(`Área selecionada: [${selected.map((v) => v.toFixed(4)).join(", ")}]`);
    },
    [addLog],
  );

  const useVisibleMapArea = useCallback(() => {
    if (!viewportBounds) return;
    setBounds(viewportBounds);
    setDrawing(false);
    addLog(`Área visível selecionada: [${viewportBounds.map((v) => v.toFixed(5)).join(", ")}]`);
  }, [addLog, viewportBounds]);

  const clearSelection = () => {
    setBounds(null);
    setSelectedFeature(null);
    setManualDraft([]);
    setManualDrawing(false);
    reset();
  };

  const finishManualField = useCallback(() => {
    if (manualDraft.length < 3) return;
    const nextIndex = manualFields.length + 1;
    const field: ManualField = {
      id: crypto.randomUUID(),
      name: manualFieldName.trim() || `Talhão ${nextIndex}`,
      coordinates: manualDraft,
      created_at: new Date().toISOString(),
    };
    setManualFields((prev) => [...prev, field]);
    setManualDraft([]);
    setManualFieldName("");
    addLog(`Talhão manual salvo: ${field.name} (${field.coordinates.length} vértices)`);
  }, [addLog, manualDraft, manualFieldName, manualFields.length]);

  const undoManualPoint = () => setManualDraft((prev) => prev.slice(0, -1));

  const removeManualField = (id: string) => {
    setManualFields((prev) => prev.filter((field) => field.id !== id));
  };

  const clearManualTest = () => {
    setManualDraft([]);
    setManualFields([]);
    localStorage.removeItem(MANUAL_TEST_STORAGE_KEY);
    addLog("Anotações manuais removidas.");
  };

  const manualTrainingCase = (): ManualTrainingCase | null => {
    if (!bounds) return null;
    const caseName = manualCaseName.trim() || `teste_manual_${new Date().toISOString().slice(0, 10)}`;
    const area: ManualTrainingCase["area"] = {
      type: "Feature",
      properties: {
        kind: "area",
        name: caseName,
        role: "input_area",
      },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [bounds[0], bounds[1]],
          [bounds[2], bounds[1]],
          [bounds[2], bounds[3]],
          [bounds[0], bounds[3]],
          [bounds[0], bounds[1]],
        ]],
      },
    };

    const answers = {
      type: "FeatureCollection" as const,
      features: manualFields.map((field, index) => {
        const ring = [...field.coordinates, field.coordinates[0]];
        return {
          type: "Feature" as const,
          properties: {
            kind: "talhao_resposta",
            role: "expected_answer",
            id: field.id,
            name: field.name,
            index: index + 1,
            created_at: field.created_at,
          },
          geometry: { type: "Polygon", coordinates: [ring] },
        };
      }),
    };

    return {
      case_name: caseName,
      area,
      answers,
      metadata: {
        schema_version: 1,
        created_at: new Date().toISOString(),
        crs: "EPSG:4326",
        bounds,
        field_count: manualFields.length,
        purpose: "manual_ground_truth_for_training",
      },
    };
  };

  const combinedManualGeoJSON = () => {
    const trainingCase = manualTrainingCase();
    if (!trainingCase) return null;
    return {
      type: "FeatureCollection",
      name: trainingCase.case_name,
      metadata: trainingCase.metadata,
      features: [
        trainingCase.area,
        ...trainingCase.answers.features.map((feature) => ({
          ...feature,
          properties: {
            ...feature.properties,
            case_name: trainingCase.case_name,
          },
        })),
      ],
    };
  };

  const saveManualTest = async () => {
    const payload = manualTrainingCase();
    if (!payload || manualFields.length === 0) return;
    const saved = await api.saveManualTest(payload);
    addLog(`Caso salvo: ${saved.case_dir} (${saved.field_count} talhões resposta)`);
  };

  const downloadManualTest = () => {
    const payload = manualTrainingCase();
    if (!payload) return;
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.href = url;
    link.download = `${payload.case_name || "manual-fields-test"}-${stamp}.training_case.json`;
    link.click();
    URL.revokeObjectURL(url);
    addLog(`Caso de treino baixado com ${manualFields.length} talhões resposta.`);
  };

  const downloadManualGeoJSON = () => {
    const payload = combinedManualGeoJSON();
    if (!payload) return;
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.href = url;
    link.download = `manual-fields-test-${stamp}.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  /*const manualTestGeoJSON = () => {
    const features = [];
    for (const field of manualFields) {
      const ring = [...field.coordinates, field.coordinates[0]];
      features.push({
        type: "Feature",
        properties: {
          kind: "manual_field",
          id: field.id,
          name: field.name,
          created_at: field.created_at,
        },
        geometry: { type: "Polygon", coordinates: [ring] },
      });
    }

    return {
      type: "FeatureCollection",
      name: "manual_field_boundary_test",
      metadata: {
        created_at: new Date().toISOString(),
        crs: "EPSG:4326",
        bounds,
        field_count: manualFields.length,
      },
      features,
    };
  };*/

  const busy = status === "submitting" || status === "running";

  return (
    <div className="layout">
      <aside className="sidebar">
        <header>
          <h1>Field Boundary DL</h1>
          <p className="subtitle">Talhões e bordaduras por IA</p>
        </header>

        <section className="panel">
          <h3>Área</h3>
          <form
            className="search"
            onSubmit={async (e) => {
              e.preventDefault();
              if (!searchQuery.trim()) return;
              setSearching(true);
              try {
                const found = await searchLocation(searchQuery);
                if (found) {
                  setFlyTo(found);
                  addLog(`Localização encontrada: ${searchQuery} -> ${found[0].toFixed(4)}, ${found[1].toFixed(4)}`);
                } else {
                  addLog(`Localização não encontrada: ${searchQuery}`);
                }
              } catch {
                addLog("Busca de localidade indisponível (Nominatim).");
              } finally {
                setSearching(false);
              }
            }}
          >
            <input
              type="search"
              data-testid="search-input"
              placeholder="Buscar cidade/localidade…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button type="submit" data-testid="search-button" disabled={searching || !searchQuery.trim()}>
              {searching ? "…" : "Ir"}
            </button>
          </form>
          <button type="button" onClick={() => setDrawing(!drawing)} disabled={busy}>
            {drawing ? "Desenhando… (2 cliques no mapa)" : "✏ Desenhar área"}
          </button>
          <button type="button" onClick={useVisibleMapArea} disabled={!viewportBounds || busy}>
            Usar área visível do mapa
          </button>
          <button type="button" onClick={clearSelection} disabled={!bounds || busy}>
            Limpar seleção
          </button>
          {bounds && (
            <p className="hint" data-testid="bounds-display">
              Coordenadas EPSG:4326 [min_lon, min_lat, max_lon, max_lat]: [{bounds.map((b) => b.toFixed(5)).join(", ")}]
            </p>
          )}
        </section>

        <ConfigPanel params={params} models={models} onChange={setParams} />

        <section className="panel">
          <h3>Teste manual</h3>
          <input
            type="text"
            placeholder="Nome da área/caso"
            value={manualCaseName}
            onChange={(e) => setManualCaseName(e.target.value)}
          />
          <button type="button" onClick={() => setManualDrawing((value) => !value)} disabled={!bounds}>
            {manualDrawing ? "Desenhando talhão..." : "Desenhar talhão manual"}
          </button>
          <input
            type="text"
            placeholder="Nome do talhão"
            value={manualFieldName}
            onChange={(e) => setManualFieldName(e.target.value)}
          />
          <div className="button-row">
            <button type="button" onClick={undoManualPoint} disabled={manualDraft.length === 0}>
              Desfazer ponto
            </button>
            <button type="button" onClick={finishManualField} disabled={manualDraft.length < 3}>
              Fechar talhão
            </button>
          </div>
          <div className="button-row">
            <button type="button" onClick={saveManualTest} disabled={!bounds || manualFields.length === 0}>
              Salvar treino
            </button>
            <button type="button" onClick={clearManualTest} disabled={manualFields.length === 0 && manualDraft.length === 0}>
              Limpar teste
            </button>
          </div>
          <div className="button-row">
            <button type="button" onClick={downloadManualTest} disabled={!bounds || manualFields.length === 0}>
              Baixar JSON
            </button>
            <button type="button" onClick={downloadManualGeoJSON} disabled={!bounds || manualFields.length === 0}>
              Baixar GeoJSON
            </button>
          </div>
          <p className="hint">
            Salva separado: area.geojson = area selecionada; answers.geojson = talhões resposta para treinar a IA.
          </p>
          <ul className="annotation-list">
            {manualFields.map((field, index) => (
              <li key={field.id}>
                <span>{index + 1}. {field.name}</span>
                <button type="button" className="link" onClick={() => removeManualField(field.id)}>remover</button>
              </li>
            ))}
          </ul>
        </section>

        <button
          type="button"
          className="primary"
          data-testid="run-button"
          disabled={!bounds || busy}
          onClick={() => bounds && run(bounds, params)}
        >
          {busy ? "Processando…" : params.provider === "mock" ? "Gerar demonstração sintética" : "Gerar talhões nesta área"}
        </button>

        {status === "running" && <p className="status" data-testid="status">Inferindo… aguarde.</p>}
        {error && <p className="error" data-testid="error">Erro: {error}</p>}

        <ResultsPanel
          result={result}
          jobId={jobId}
          elapsed={elapsed}
          showFields={showFields}
          showBoundaries={showBoundaries}
          selectedFeature={selectedFeature}
          onToggleFields={setShowFields}
          onToggleBoundaries={setShowBoundaries}
        />
        <LogPanel logs={logs} />
      </aside>

      <main className="map-container">
        <MapView
          drawing={drawing}
          bounds={bounds}
          result={result}
          showFields={showFields}
          showBoundaries={showBoundaries}
          flyTo={flyTo}
          manualDrawing={manualDrawing}
          manualDraft={manualDraft}
          manualFields={manualFields}
          onBoundsSelected={onBoundsSelected}
          onViewportBoundsChange={setViewportBounds}
          onFeatureClick={setSelectedFeature}
          onManualDraftChange={setManualDraft}
          onManualFieldFinish={finishManualField}
        />
      </main>
    </div>
  );
}
