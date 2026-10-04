from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Compatibilidade: checkpoints salvos no Linux podem conter pathlib.PosixPath.
# No Windows, isso quebra o torch.load. Este patch converte PosixPath para WindowsPath.
import sys
import pathlib
if sys.platform.startswith("win"):
    pathlib.PosixPath = pathlib.WindowsPath
from scipy import ndimage
from tqdm import tqdm


def load_s23(script_path: Path):
    spec = importlib.util.spec_from_file_location('s23_eval_module', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Não consegui importar {script_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_name(x: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(str(x)).stem)


def apply_tta(x: torch.Tensor, t: str) -> torch.Tensor:
    if t == 'id': return x
    if t == 'h': return torch.flip(x, dims=[3])
    if t == 'v': return torch.flip(x, dims=[2])
    if t == 'hv': return torch.flip(x, dims=[2, 3])
    raise ValueError(t)


def tta_list(mode: str):
    return ['id'] if mode == 'none' else ['id', 'h', 'v', 'hv']


def remove_small(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    st = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    labels, n = ndimage.label(mask.astype(bool), structure=st)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def load_model(s23, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    arch = cfg.get('architecture', 'unet')
    enc = cfg.get('encoder_name', 'resnet34')
    inc = int(cfg.get('in_channels', 30))
    print(f'Carregando {ckpt_path} | arch={arch} encoder={enc} in_channels={inc}')
    model = s23.build_model(architecture=arch, encoder_name=enc, encoder_weights=None, in_channels=inc)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(image: torch.Tensor, models, weights, device, threshold, min_area, connectivity, tta_mode):
    image = image.to(device)
    total = None
    modes = tta_list(tta_mode)
    for t in modes:
        xt = apply_tta(image, t)
        pt = None
        for w, model in zip(weights, models):
            p = torch.sigmoid(model(xt))
            pt = w * p if pt is None else pt + w * p
        pt = apply_tta(pt, t)
        total = pt if total is None else total + pt
    prob = (total / len(modes))[0, 0].detach().float().cpu().numpy()
    raw = (prob >= threshold).astype(np.uint8)
    pred = remove_small(raw, min_area, connectivity)
    return prob, pred


def stack_to_rgb(img: torch.Tensor, bands):
    arr = img.detach().cpu().numpy().astype(np.float32)  # C,H,W
    c, h, w = arr.shape
    # Script 23 empilha por banda e depois por tempo: B2_t0..B2_t5, B3_t0.. etc.
    if c >= len(bands) * 2 and all(b in bands for b in ['B2','B3','B4']):
        ts = c // len(bands)
        bmap = {b:i for i,b in enumerate(bands)}
        def med(b):
            i = bmap[b]
            return np.median(arr[i*ts:(i+1)*ts], axis=0)
        rgb = np.stack([med('B4'), med('B3'), med('B2')], axis=-1)
    elif c >= 3:
        rgb = np.transpose(arr[:3], (1,2,0))
    else:
        rgb = np.repeat(np.transpose(arr[:1], (1,2,0)), 3, axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for k in range(3):
        band = np.nan_to_num(rgb[..., k], nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(band, [2,98])
        out[..., k] = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def comp_rgb(target, pred):
    target = target.astype(bool); pred = pred.astype(bool)
    out = np.ones((*target.shape, 3), dtype=np.float32)
    out[target & pred] = [0.05, 0.65, 0.15]      # acerto
    out[(~target) & pred] = [0.90, 0.10, 0.05]   # falso positivo
    out[target & (~pred)] = [0.10, 0.20, 0.90]   # falso negativo
    return out


def metrics(target, pred):
    target = target.astype(bool); pred = pred.astype(bool)
    tp = int((target & pred).sum()); fp = int(((~target) & pred).sum()); fn = int((target & (~pred)).sum())
    eps=1e-7
    dice=(2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    return dict(dice=float(dice), iou=float(iou), target_ratio=float(target.mean()), pred_ratio=float(pred.mean()), tp=tp, fp=fp, fn=fn)


def save_panel(rgb, target, pred, prob, row, out_path):
    m = metrics(target, pred)
    group = str(row.get('selection_group', row.get('overview_group', '')))
    fig, ax = plt.subplots(1,5,figsize=(18,4.2))
    ax[0].imshow(rgb); ax[0].set_title('Sentinel-2 RGB')
    ax[1].imshow(target, cmap='gray', vmin=0, vmax=1); ax[1].set_title('Máscara real')
    ax[2].imshow(pred, cmap='gray', vmin=0, vmax=1); ax[2].set_title('Máscara criada')
    ax[3].imshow(comp_rgb(target, pred)); ax[3].set_title('Comparação\nverde acerto | vermelho extra | azul faltou', fontsize=9)
    ax[4].imshow(prob, cmap='viridis', vmin=0, vmax=1); ax[4].set_title('Probabilidade')
    for a in ax: a.axis('off')
    fig.suptitle(f"{safe_name(row['image_path'])} | {group}\nDice={m['dice']:.3f} | IoU={m['iou']:.3f} | target={m['target_ratio']:.3f} | pred={m['pred_ratio']:.3f}", fontsize=12, y=1.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close(fig)
    return m


def contact_sheet(paths, out_path, cols=2):
    if not paths:
        return
    import matplotlib.image as mpimg
    rows = math.ceil(len(paths)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*9.5, rows*2.8))
    axes = np.asarray(axes).reshape(-1)
    for a,p in zip(axes, paths):
        a.imshow(mpimg.imread(p)); a.set_title(p.stem, fontsize=7); a.axis('off')
    for a in axes[len(paths):]: a.axis('off')
    fig.tight_layout(); fig.savefig(out_path, dpi=160, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Gera PNGs real vs predito do Sentinel-2 usando o loader do script 23 que já funciona.')
    ap.add_argument('--s23-script', type=Path, default=Path('scripts/23_evaluate_sentinel2_postprocess.py'))
    ap.add_argument('--selected-csv', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--bands', type=str, default='B2,B3,B4,B8,NDVI')
    ap.add_argument('--time-mode', type=str, default='stack')
    ap.add_argument('--time-index', type=int, default=-1)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--positive-values', type=str, default='')
    ap.add_argument('--threshold', type=float, default=0.49)
    ap.add_argument('--min-area', type=int, default=90)
    ap.add_argument('--connectivity', type=int, default=8)
    ap.add_argument('--tta-mode', type=str, default='flips', choices=['none','flips'])
    ap.add_argument('--max-cases', type=int, default=20)
    ap.add_argument('--groups', type=str, default='')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.output_dir/'panels'; masks_dir = args.output_dir/'masks'
    panels_dir.mkdir(parents=True, exist_ok=True); masks_dir.mkdir(parents=True, exist_ok=True)

    s23 = load_s23(args.s23_script)
    df = pd.read_csv(args.selected_csv)
    if args.groups.strip():
        groups = [g.strip() for g in args.groups.split(',') if g.strip()]
        df = df[df['selection_group'].isin(groups)].copy()
    df = df.head(args.max_cases).copy()

    bands = [b.strip() for b in args.bands.split(',') if b.strip()]
    weights = np.array([float(x.strip()) for x in args.weights.split(',') if x.strip()], dtype=np.float32)
    weights = (weights/weights.sum()).tolist()
    if len(weights) != len(args.checkpoints):
        raise ValueError('Número de pesos diferente do número de checkpoints')
    pos = None if not args.positive_values.strip() else [int(x.strip()) for x in args.positive_values.split(',') if x.strip()]
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    models = [load_model(s23, ck, device) for ck in args.checkpoints]

    panel_paths=[]; rows=[]
    for idx,row in tqdm(df.iterrows(), total=len(df), desc='Gerando PNGs'):
        try:
            image_path = Path(str(row['image_path']))
            mask_path = Path(str(row['mask_path']))
            ds = s23.Sentinel2Dataset(pairs=[(image_path, mask_path)], bands=bands, time_mode=args.time_mode, time_index=args.time_index, image_size=args.image_size, positive_values=pos)
            sample = ds[0]
            image_tensor = sample['image']
            target = sample['mask'].squeeze().detach().cpu().numpy().astype(np.uint8)
            prob, pred = predict(image_tensor.unsqueeze(0), models, weights, device, args.threshold, args.min_area, args.connectivity, args.tta_mode)
            rgb = stack_to_rgb(image_tensor, bands)
            stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"{idx+1:03d}_{row.get('selection_group','case')}_{safe_name(row['image_path'])}")
            panel = panels_dir/f'{stem}.png'
            met = save_panel(rgb, target, pred, prob, row, panel)
            plt.imsave(masks_dir/f'{stem}_real.png', target, cmap='gray', vmin=0, vmax=1)
            plt.imsave(masks_dir/f'{stem}_pred.png', pred, cmap='gray', vmin=0, vmax=1)
            plt.imsave(masks_dir/f'{stem}_comparison.png', comp_rgb(target, pred))
            panel_paths.append(panel)
            rec = row.to_dict(); rec.update(met); rec['panel_path']=str(panel); rows.append(rec)
        except Exception as e:
            print(f'[ERRO] {row.get("image_path")}: {e}')
    pd.DataFrame(rows).to_csv(args.output_dir/'generated_png_index.csv', index=False)
    contact_sheet(panel_paths, args.output_dir/'boss_contact_sheet.png', cols=2)
    print('\nFinalizado.')
    print('PNGs gerados:', len(panel_paths))
    print('Painéis:', panels_dir)
    print('Máscaras:', masks_dir)
    print('Contact sheet:', args.output_dir/'boss_contact_sheet.png')
    print('Índice:', args.output_dir/'generated_png_index.csv')

if __name__ == '__main__':
    main()
