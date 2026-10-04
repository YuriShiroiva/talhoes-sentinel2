/**
 * Mapa Leaflet: camada satélite (tile configurável), desenho de retângulo
 * por dois cliques, busca por localidade (Nominatim) e camadas de resultado.
 *
 * A camada de satélite do MAPA é só visualização (tile XYZ configurável via
 * VITE_SATELLITE_TILE_URL; padrão Esri World Imagery, com atribuição). A
 * imagem usada na INFERÊNCIA vem do backend via provider oficial — o mapa
 * nunca é raspado.
 */

import type { LatLngBoundsExpression, LeafletMouseEvent } from "leaflet";
import { useCallback, useMemo, useState } from "react";
import { useEffect } from "react";
import { CircleMarker, GeoJSON, MapContainer, Polygon, Polyline, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import type { Bounds, LonLat, ManualField, PredictResult } from "../types";

const SATELLITE_URL =
  import.meta.env.VITE_SATELLITE_TILE_URL ??
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const SATELLITE_ATTRIBUTION =
  import.meta.env.VITE_SATELLITE_ATTRIBUTION ??
  "Imagens &copy; Esri — fonte de visualização; verifique os termos de uso";

function FlyTo({ target }: { target: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (target) map.flyTo(target, 14);
  }, [map, target]);
  return null;
}

interface Props {
  drawing: boolean;
  bounds: Bounds | null;
  result: PredictResult | null;
  showFields: boolean;
  showBoundaries: boolean;
  flyTo: [number, number] | null;
  manualDrawing: boolean;
  manualDraft: LonLat[];
  manualFields: ManualField[];
  onBoundsSelected: (bounds: Bounds) => void;
  onViewportBoundsChange: (bounds: Bounds) => void;
  onFeatureClick: (properties: Record<string, unknown>) => void;
  onManualDraftChange: (points: LonLat[] | ((prev: LonLat[]) => LonLat[])) => void;
  onManualFieldFinish: () => void;
}

function ViewportBoundsReporter({ onChange }: { onChange: (bounds: Bounds) => void }) {
  const map = useMap();

  const report = useCallback(() => {
    const b = map.getBounds();
    onChange([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
  }, [map, onChange]);

  useEffect(() => {
    report();
  }, [report]);

  useMapEvents({
    moveend: report,
    zoomend: report,
  });

  return null;
}

function DrawRectangle({ drawing, onBoundsSelected }: Pick<Props, "drawing" | "onBoundsSelected">) {
  const [corner, setCorner] = useState<[number, number] | null>(null);
  const [preview, setPreview] = useState<[number, number] | null>(null);

  useMapEvents({
    click(event: LeafletMouseEvent) {
      if (!drawing) return;
      const point: [number, number] = [event.latlng.lng, event.latlng.lat];
      if (!corner) {
        setCorner(point);
      } else {
        const sel: Bounds = [
          Math.min(corner[0], point[0]),
          Math.min(corner[1], point[1]),
          Math.max(corner[0], point[0]),
          Math.max(corner[1], point[1]),
        ];
        setCorner(null);
        setPreview(null);
        onBoundsSelected(sel);
      }
    },
    mousemove(event: LeafletMouseEvent) {
      if (drawing && corner) setPreview([event.latlng.lng, event.latlng.lat]);
    },
  });

  if (corner && preview) {
    const rect: LatLngBoundsExpression = [
      [Math.min(corner[1], preview[1]), Math.min(corner[0], preview[0])],
      [Math.max(corner[1], preview[1]), Math.max(corner[0], preview[0])],
    ];
    return <Rectangle bounds={rect} pathOptions={{ color: "#f59e0b", dashArray: "6" }} />;
  }
  return null;
}

function DrawManualField({
  enabled,
  draft,
  onDraftChange,
  onFinish,
}: {
  enabled: boolean;
  draft: LonLat[];
  onDraftChange: Props["onManualDraftChange"];
  onFinish: () => void;
}) {
  const [hover, setHover] = useState<LonLat | null>(null);

  useMapEvents({
    click(event: LeafletMouseEvent) {
      if (!enabled) return;
      onDraftChange((prev) => [...prev, [event.latlng.lng, event.latlng.lat]]);
    },
    dblclick() {
      if (!enabled || draft.length < 3) return;
      onFinish();
    },
    mousemove(event: LeafletMouseEvent) {
      if (!enabled || draft.length === 0) return;
      setHover([event.latlng.lng, event.latlng.lat]);
    },
  });

  const draftLatLngs = draft.map((point) => [point[1], point[0]] as [number, number]);
  const previewLatLngs = hover ? [...draftLatLngs, [hover[1], hover[0]] as [number, number]] : draftLatLngs;

  return (
    <>
      {previewLatLngs.length >= 2 && (
        <Polyline positions={previewLatLngs} pathOptions={{ color: "#2563eb", weight: 3, dashArray: "5 6" }} />
      )}
      {draftLatLngs.length >= 3 && (
        <Polygon positions={draftLatLngs} pathOptions={{ color: "#2563eb", fillColor: "#60a5fa", fillOpacity: 0.18 }} />
      )}
      {draftLatLngs.map((position, index) => (
        <CircleMarker
          key={`${position[0]}-${position[1]}-${index}`}
          center={position}
          radius={5}
          pathOptions={{ color: "#1d4ed8", fillColor: "#bfdbfe", fillOpacity: 0.9, weight: 2 }}
        />
      ))}
    </>
  );
}

export default function MapView({
  drawing,
  bounds,
  result,
  showFields,
  showBoundaries,
  flyTo,
  manualDrawing,
  manualDraft,
  manualFields,
  onBoundsSelected,
  onViewportBoundsChange,
  onFeatureClick,
  onManualDraftChange,
  onManualFieldFinish,
}: Props) {
  const selectionRect: LatLngBoundsExpression | null = useMemo(
    () => (bounds ? [[bounds[1], bounds[0]], [bounds[3], bounds[2]]] : null),
    [bounds],
  );

  const onEachField = useCallback(
    (feature: { properties?: Record<string, unknown> }, layer: { on: (e: string, fn: () => void) => void }) => {
      layer.on("click", () => onFeatureClick(feature.properties ?? {}));
    },
    [onFeatureClick],
  );

  return (
    <MapContainer center={[-24.955, -53.405]} zoom={13} style={{ height: "100%", width: "100%" }} doubleClickZoom={false}>
      <TileLayer url={SATELLITE_URL} attribution={SATELLITE_ATTRIBUTION} maxZoom={19} />
      <FlyTo target={flyTo} />
      <ViewportBoundsReporter onChange={onViewportBoundsChange} />
      <DrawRectangle drawing={drawing} onBoundsSelected={onBoundsSelected} />
      <DrawManualField
        enabled={manualDrawing && Boolean(bounds)}
        draft={manualDraft}
        onDraftChange={onManualDraftChange}
        onFinish={onManualFieldFinish}
      />
      {selectionRect && <Rectangle bounds={selectionRect} pathOptions={{ color: "#f59e0b", weight: 2 }} />}
      {manualFields.map((field, index) => (
        <Polygon
          key={field.id}
          positions={field.coordinates.map((point) => [point[1], point[0]] as [number, number])}
          pathOptions={{ color: "#2563eb", weight: 2, fillColor: "#3b82f6", fillOpacity: 0.22 }}
          eventHandlers={{
            click: () => onFeatureClick({ kind: "manual_field", id: field.id, name: field.name, vertices: field.coordinates.length, index: index + 1 }),
          }}
        />
      ))}
      {result && showFields && (
        <GeoJSON
          key={`fields-${result.metadata.source["provider"]}-${result.quality.n_fields}`}
          data={result.fields as never}
          style={(feature) => ({
            color: "#16a34a",
            weight: 2,
            // destaca geometrias problemáticas: área minúscula
            fillColor: (feature?.properties?.area_ha as number) < 0.1 ? "#dc2626" : "#22c55e",
            fillOpacity: 0.25,
          })}
          onEachFeature={onEachField}
        />
      )}
      {result && showBoundaries && (
        <GeoJSON
          key={`bounds-${result.quality.n_fields}`}
          data={result.boundaries as never}
          style={{ color: "#dc2626", weight: 1.5, fillOpacity: 0.05 }}
        />
      )}
    </MapContainer>
  );
}
