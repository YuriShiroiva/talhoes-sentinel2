from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "reais" / "fields_yuri.csv"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "prepared" / "rgb_fields_yuri"


def _load_builder():
    path = PROJECT_ROOT / "scripts" / "51_build_rgb_dataset_from_fields.py"
    spec = importlib.util.spec_from_file_location("rgb_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_field_boundary(geoms, transform, h: int, w: int, width_px: int) -> np.ndarray:
    """Borda = uniao das arestas de CADA talhao (marca limites entre talhoes colados)."""
    boundary = np.zeros((h, w), dtype=np.uint8)
    struct = ndimage.generate_binary_structure(2, 2)
    for g in geoms:
        single = rasterize([(g, 1)], out_shape=(h, w), fill=0, transform=transform,
                           all_touched=False, dtype="uint8").astype(bool)
        if not single.any():
            continue
        eroded = ndimage.binary_erosion(single, structure=struct, iterations=max(1, width_px))
        boundary |= (single & ~eroded)
    return boundary.astype(np.uint8)


def per_field_energy(geoms, transform, h: int, w: int) -> np.ndarray:
    """Energia: distancia-ao-limite normalizada POR talhao (1 no centro, 0 na borda). uint8 0..255.

    Alvo de regressao para separacao tipo Deep Watershed: picos = centros -> 1 semente por talhao.
    """
    energy = np.zeros((h, w), dtype=np.float32)
    for g in geoms:
        single = rasterize([(g, 1)], out_shape=(h, w), fill=0, transform=transform,
                           all_touched=False, dtype="uint8").astype(bool)
        if not single.any():
            continue
        dist = ndimage.distance_transform_edt(single).astype(np.float32)
        m = float(dist.max())
        if m > 0:
            energy = np.maximum(energy, dist / m)  # normaliza por talhao
    return (np.clip(energy, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_preview(path: Path, img_path: Path, area: np.ndarray, boundary: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(img_path) as s:
        rgb = np.transpose(s.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
    ov = rgb.copy()
    ov[area.astype(bool)] = (0.5 * ov[area.astype(bool)] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    ov[boundary.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(np.concatenate([rgb, ov], axis=1)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="(Re)gera mascaras de 2 bandas (area + borda) alinhadas as imagens ja baixadas."
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Raiz com images/{train,val}.")
    ap.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    ap.add_argument("--boundary-width-px", type=int, default=2)
    ap.add_argument("--energy", action="store_true",
                    help="Banda 2 = energia (distancia normalizada por talhao) em vez de borda binaria.")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    builder = _load_builder()
    print(f"Lendo e deduplicando fields de {args.fields_csv} ...")
    geoms = builder.load_fields(args.fields_csv, args.chunksize, None)
    print(f"Fields unicos: {len(geoms)}")
    tree = STRtree(geoms)

    total = 0
    for split in ["train", "val"]:
        img_dir = args.root / "images" / split
        mask_dir = args.root / "masks" / split
        if not img_dir.exists():
            continue
        mask_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(img_dir.glob("*.tif"))
        print(f"[{split}] {len(images)} imagens")
        for i, img_path in enumerate(images):
            with rasterio.open(img_path) as src:
                transform = src.transform
                h, w = src.height, src.width
                crs = src.crs
                bnds = src.bounds
            clip = box(bnds.left, bnds.bottom, bnds.right, bnds.top)
            members = [geoms[int(j)] for j in tree.query(clip) if geoms[int(j)].intersects(clip)]
            clipped = [g.intersection(clip) for g in members]
            clipped = [g for g in clipped if not g.is_empty]

            area = rasterize([(g, 1) for g in clipped], out_shape=(h, w), fill=0,
                             transform=transform, all_touched=False, dtype="uint8") if clipped \
                else np.zeros((h, w), dtype=np.uint8)
            if args.energy:
                band2 = per_field_energy(clipped, transform, h, w)
            else:
                band2 = per_field_boundary(clipped, transform, h, w, args.boundary_width_px)

            mask = np.stack([area, band2], axis=0).astype(np.uint8)
            mask_path = mask_dir / f"{img_path.stem.replace('_ortho_', '_ortholabel_')}.tif"
            with rasterio.open(
                mask_path, "w", driver="GTiff", height=h, width=w, count=2, dtype="uint8",
                crs=crs, transform=transform, compress="deflate", nodata=0,
            ) as dst:
                dst.write(mask)
            if args.preview and i < 6:
                save_preview(args.root / "previews_boundary" / f"{img_path.stem}.png", img_path, area, (band2 > 0).astype(np.uint8))
            total += 1
        print(f"[{split}] mascaras 2-bandas geradas.")

    print(f"\nPronto: {total} mascaras (banda1=area, banda2=borda) em {args.root}/masks")
    print("Treinar (boundary-aware): python scripts/30_train_orthos_boundary_aware_multitask.py "
          f"--root {args.root} --output-dir outputs/rgb_fields/boundary_v1 "
          "--architecture unet --encoder-name resnet34 --batch-size 2 --num-workers 4 --epochs 30 --use-tversky-area")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
