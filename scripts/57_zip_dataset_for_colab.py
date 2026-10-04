from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "prepared" / "rgb_fields_yuri_v3e"


def main() -> None:
    ap = argparse.ArgumentParser(description="Empacota o dataset (images+masks) num .zip para subir ao Colab/Drive.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "rgb_dataset_v3e.zip")
    args = ap.parse_args()
    if not (args.root / "images").exists():
        raise FileNotFoundError(f"Nao achei {args.root}/images. Rode 51/53 antes.")

    files = sorted(p for p in args.root.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".tif", ".tiff", ".csv", ".json"})
    print(f"Compactando {len(files)} arquivos de {args.root} ...")
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, arcname=str(f.relative_to(args.root.parent)))
    size_mb = args.out.stat().st_size / 1e6
    print(f"Pronto: {args.out} ({size_mb:.0f} MB)")
    print("Suba esse .zip para o Google Drive e aponte o notebook do Colab para ele.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
