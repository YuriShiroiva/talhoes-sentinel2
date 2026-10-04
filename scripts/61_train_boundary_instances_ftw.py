"""Fase 2 — separacao de INSTANCIAS via cabeca de borda.

Treina um modelo 3-classes (interior / borda / fundo) no FTW-Brasil, partindo do
modelo de campo ja fine-tunado (best_ftw_finetuned.pt). A borda predita separa
talhoes vizinhos: instancias = componentes conexos do interior + preenchimento por
vizinho-mais-proximo (Voronoi via scipy, sem skimage).

Metrica de selecao = objeto-F1@IoU0.5 no val (o que realmente importa).
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import torch  # antes de rasterio (DLL no Windows)
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S59 = _load("s59", PROJECT_ROOT / "scripts/59_finetune_on_ftw.py")   # build_stack (band_time+percentil)
S60 = _load("s60", PROJECT_ROOT / "scripts/60_object_level_metrics_ftw.py")  # object_metrics, instances_from_label

# semantic_3class: 1=interior(campo), 2=borda, 3=nao-campo. CE indices: 0,1,2. nodata(0)->ignore.
REMAP = {1: 0, 2: 1, 3: 2}
IGNORE = 255


def instances_from_probs(prob3, min_area, field_thr=0.5):
    """prob3 (3,H,W) softmax [interior,borda,fundo] -> rotulos de instancia.
    Interior conexo = sementes (a borda quebra a conexao); preenche o campo por vizinho."""
    p_int, p_bnd, p_bg = prob3
    field = p_bg < field_thr
    interior = np.argmax(prob3, 0) == 0
    # sementes = interior, removendo pequenos
    seeds, n = ndimage.label(interior, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(field.shape, np.int32), 0
    sizes = np.bincount(seeds.ravel())
    small = np.where(sizes < max(10, min_area // 3))[0]
    for s in small:
        if s != 0:
            seeds[seeds == s] = 0
    if seeds.max() == 0:
        return np.zeros(field.shape, np.int32), 0
    # preenche cada pixel de campo com a semente mais proxima (Voronoi)
    idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)[1]
    grown = seeds[tuple(idx)]
    grown[~field] = 0
    # compacta e remove instancias pequenas
    out = np.zeros_like(grown)
    new = 1
    for i in [v for v in np.unique(grown) if v > 0]:
        m = grown == i
        if int(m.sum()) >= min_area:
            out[m] = new
            new += 1
    return out.astype(np.int32), new - 1


class FTW3ClassDataset(Dataset):
    def __init__(self, zip_path, root, ids, augment=False):
        self.zip_path = str(zip_path)
        self.root = root
        self.ids = ids
        self.augment = augment
        self._z = None

    @property
    def z(self):
        if self._z is None:
            self._z = zipfile.ZipFile(self.zip_path)
        return self._z

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = self.ids[i]
        with rasterio.open(self.root / "s2_images/window_a" / f"{a}.tif") as s: wa = s.read()
        with rasterio.open(self.root / "s2_images/window_b" / f"{a}.tif") as s: wb = s.read()
        with rasterio.open(io.BytesIO(self.z.read(f"label_masks/semantic_3class/{a}.tif"))) as s:
            m3 = s.read(1)
        x = S59.build_stack(wa, wb)  # (30,H,W) band_time + percentil
        tgt = np.full(m3.shape, IGNORE, dtype=np.int64)
        for k, v in REMAP.items():
            tgt[m3 == k] = v
        x = torch.from_numpy(x).float()
        tgt = torch.from_numpy(tgt)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [2]); tgt = torch.flip(tgt, [1])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [1]); tgt = torch.flip(tgt, [0])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, [1, 2]); tgt = torch.rot90(tgt, k, [0, 1])
        return {"x": x, "tgt": tgt, "id": a}


def gt_instances(z, a, min_area):
    with rasterio.open(io.BytesIO(z.read(f"label_masks/instance/{a}.tif"))) as s:
        inst = s.read(1)
    return S60.instances_from_label(inst, min_area)[0]


@torch.no_grad()
def eval_object_f1(model, ids, root, z, device, min_area):
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, sum_iou=0.0, n_gt=0, n_pred=0, under=0, over=0)
    for a in ids:
        with rasterio.open(root / "s2_images/window_a" / f"{a}.tif") as s: wa = s.read()
        with rasterio.open(root / "s2_images/window_b" / f"{a}.tif") as s: wb = s.read()
        x = torch.from_numpy(S59.build_stack(wa, wb)).unsqueeze(0).float().to(device)
        prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        pred_lab, _ = instances_from_probs(prob, min_area)
        gt_lab = gt_instances(z, a, min_area)
        m = S60.object_metrics(pred_lab, gt_lab)
        for k in agg:
            agg[k] += m[k]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pq = agg["sum_iou"] / (tp + 0.5 * fp + 0.5 * fn + 1e-9)
    return {"obj_f1": f1, "precision": prec, "recall": rec, "pq": pq,
            "n_gt": agg["n_gt"], "n_pred": agg["n_pred"], "under": agg["under"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/ftw/brazil")
    ap.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/ftw/brazil.zip")
    ap.add_argument("--init", type=Path,
                    default=PROJECT_ROOT / "outputs/sentinel2/ftw_brazil_finetune/best_ftw_finetuned.pt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/ftw_boundary_instances")
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--bnd-weight", type=float, default=4.0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = zipfile.ZipFile(args.zip)
    df = pd.read_parquet(io.BytesIO(z.read("chips_brazil.parquet")))
    splits = {s: df[df["split"] == s]["aoi_id"].tolist() for s in ["train", "val", "test"]}
    print({k: len(v) for k, v in splits.items()})

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=30, classes=3).to(device)
    # inicializa encoder+decoder do modelo de campo ja fine-tunado (descarta a cabeca de 1 classe)
    ck = torch.load(args.init, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck)
    state = {k: v for k, v in state.items() if "segmentation_head" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"init de {args.init.name}: faltando {len(missing)} (cabeca nova), inesperado {len(unexpected)}")

    tl = DataLoader(FTW3ClassDataset(args.zip, args.root, splits["train"], augment=True),
                    batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    weight = torch.tensor([1.0, args.bnd_weight, 1.0], device=device)
    ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

    base = eval_object_f1(model, splits["val"], args.root, z, device, args.min_area)
    print(f"val objeto-F1 ANTES (cabeca aleatoria): {base['obj_f1']:.3f}")
    best = -1.0; hist = []
    from tqdm import tqdm
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for b in tqdm(tl, desc=f"ep {ep}", leave=False):
            x = b["x"].to(device); y = b["tgt"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); n += 1
        sched.step()
        vm = eval_object_f1(model, splits["val"], args.root, z, device, args.min_area)
        hist.append({"epoch": ep, "train_loss": tot / n, **{f"val_{k}": v for k, v in vm.items()}})
        pd.DataFrame(hist).to_csv(args.output_dir / "history.csv", index=False)
        print(f"ep {ep:02d}/{args.epochs} loss={tot/n:.3f} val_obj_f1={vm['obj_f1']:.3f} "
              f"val_pq={vm['pq']:.3f} prec={vm['precision']:.3f} rec={vm['recall']:.3f}")
        if vm["obj_f1"] > best:
            best = vm["obj_f1"]
            torch.save({"model_state_dict": model.state_dict(),
                        "config": {"architecture": "unet", "encoder_name": "resnet34",
                                   "in_channels": 30, "classes": 3},
                        "val_obj_f1": best}, args.output_dir / "best_boundary_3class.pt")

    ck2 = torch.load(args.output_dir / "best_boundary_3class.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck2["model_state_dict"])
    tm = eval_object_f1(model, splits["test"], args.root, z, device, args.min_area)
    summary = {"dataset": "FTW-brazil (boundary 3-class)", "val_obj_f1_antes": base["obj_f1"],
               "best_val_obj_f1": best, "test": tm, "epochs": args.epochs, "lr": args.lr,
               "bnd_weight": args.bnd_weight}
    (args.output_dir / "summary_boundary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== TEST objeto-F1 (FTW-Brasil) com separacao de instancias ===")
    for k in ["obj_f1", "precision", "recall", "pq"]:
        print(f"  {k:10s} {tm[k]:.3f}")
    print(f"Salvo: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
