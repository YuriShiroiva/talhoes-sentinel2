from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.field_boundary.data.dataset import find_image_mask_pairs, list_raster_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona arquivos do AI4Boundaries")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--max-files", type=int, default=80)
    args = parser.parse_args()

    files = list_raster_files(args.data_root)
    print(f"Total de rasters encontrados: {len(files)}")
    print("\nPrimeiros arquivos:")
    for p in files[: args.max_files]:
        print(" -", p.relative_to(args.data_root))

    pairs = find_image_mask_pairs(args.data_root)
    print(f"\nPares imagem/máscara encontrados automaticamente: {len(pairs)}")
    for image_path, mask_path in pairs[:10]:
        print("IMG:", image_path.relative_to(args.data_root))
        print("MSK:", mask_path.relative_to(args.data_root))
        print("---")

    if not pairs:
        print("\nNenhum par encontrado. Me mande a saída deste script para ajustarmos o loader ao formato exato do download.")


if __name__ == "__main__":
    main()
