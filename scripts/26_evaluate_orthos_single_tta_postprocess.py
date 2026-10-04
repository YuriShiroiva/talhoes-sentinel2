from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp


def img_to_mask(img_path: Path, root: Path, split: str) -> Path:
    return root / 'masks' / split / img_path.name.replace('_ortho_', '_ortholabel_')


def build_pairs(root: Path, split: str):
    img_dir = root / 'images' / split
    out = []
    for img in sorted(img_dir.glob('*.tif')):
        mask = img_to_mask(img, root, split)
        if img.exists() and mask.exists():
            out.append((img, mask))
    return out


def size_class(r: float) -> str:
    if r == 0: return 'empty'
    if r < 0.01: return 'very_small_<1%'
    if r < 0.05: return 'small_1_5%'
    if r < 0.25: return 'medium_5_25%'
    if r < 0.75: return 'large_25_75%'
    return 'very_large_>75%'


class OrthoDS(Dataset):
    def __init__(self, pairs, image_size=512, mask_band=1, mask_mode='band', normalization='imagenet'):
        self.pairs = pairs
        self.image_size = image_size
        self.mask_band = mask_band
        self.mask_mode = mask_mode
        self.normalization = normalization

    def __len__(self): return len(self.pairs)

    def _read_img(self, p):
        with rasterio.open(p) as src:
            x = src.read([1, 2, 3]).astype(np.float32)
        if x.max() > 2: x /= 255.0
        if self.normalization == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            x = (x - mean) / std
        return x.astype(np.float32)

    def _read_mask(self, p):
        with rasterio.open(p) as src:
            if self.mask_mode == 'band':
                y = (src.read(self.mask_band) > 0).astype(np.float32)
            else:
                y = (src.read() > 0).any(axis=0).astype(np.float32)
        return y[None].astype(np.float32)

    def _resize(self, x, y):
        if x.shape[-2:] == (self.image_size, self.image_size): return x, y
        x = F.interpolate(x[None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
        y = F.interpolate(y[None], size=(self.image_size, self.image_size), mode='nearest')[0]
        return x, y

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        x = torch.from_numpy(self._read_img(ip)).float()
        y = torch.from_numpy(self._read_mask(mp)).float()
        x, y = self._resize(x, y)
        return {'image': x, 'mask': y, 'image_path': str(ip), 'mask_path': str(mp)}


def build_model(arch, enc, in_ch):
    arch = arch.lower()
    if arch == 'unet': return smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=1)
    if arch == 'unetplusplus': return smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=1)
    if arch == 'deeplabv3plus': return smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=1)
    if arch == 'fpn': return smp.FPN(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=1)
    if arch == 'pan': return smp.PAN(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=1)
    raise ValueError(f'Arquitetura invalida: {arch}')


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get('config', {})
    m = build_model(cfg.get('architecture', 'unet'), cfg.get('encoder_name', 'resnet34'), int(cfg.get('in_channels', 3))).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, cfg


def tta_names(mode):
    if mode == 'none': return ['id']
    if mode == 'flips': return ['id', 'h', 'v', 'hv']
    if mode == 'd4': return ['id', 'h', 'v', 'hv', 'r90', 'r180', 'r270']
    raise ValueError(mode)


def tr(x, t):
    if t == 'id': return x
    if t == 'h': return torch.flip(x, [3])
    if t == 'v': return torch.flip(x, [2])
    if t == 'hv': return torch.flip(x, [2, 3])
    if t == 'r90': return torch.rot90(x, 1, [2, 3])
    if t == 'r180': return torch.rot90(x, 2, [2, 3])
    if t == 'r270': return torch.rot90(x, 3, [2, 3])
    raise ValueError(t)


def inv(x, t):
    if t == 'id': return x
    if t == 'h': return torch.flip(x, [3])
    if t == 'v': return torch.flip(x, [2])
    if t == 'hv': return torch.flip(x, [2, 3])
    if t == 'r90': return torch.rot90(x, 3, [2, 3])
    if t == 'r180': return torch.rot90(x, 2, [2, 3])
    if t == 'r270': return torch.rot90(x, 1, [2, 3])
    raise ValueError(t)


def remove_small(mask, min_area, conn=8):
    if min_area <= 0: return mask.astype(np.uint8)
    st = np.ones((3, 3), dtype=np.uint8) if conn == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    lab, n = ndimage.label(mask.astype(bool), structure=st)
    if n == 0: return mask.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab].astype(np.uint8)


def metrics(pred, target, eps=1e-7):
    p = pred.reshape(pred.shape[0], -1).astype(np.float32)
    y = target.reshape(target.shape[0], -1).astype(np.float32)
    tp = (p*y).sum(1); fp = (p*(1-y)).sum(1); fn = ((1-p)*y).sum(1); tn = ((1-p)*(1-y)).sum(1)
    dice = (2*tp+eps)/(2*tp+fp+fn+eps)
    iou = (tp+eps)/(tp+fp+fn+eps)
    prec = (tp+eps)/(tp+fp+eps)
    rec = (tp+eps)/(tp+fn+eps)
    f1 = (2*prec*rec+eps)/(prec+rec+eps)
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'pred_pixels': p.sum(1), 'target_pixels': y.sum(1), 'total_pixels': np.full(len(dice), y.shape[1], dtype=np.float32)}


def summarize(a):
    a = np.asarray(a, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std()), 'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


@torch.no_grad()
def evaluate(model, loader, device, threshold, amp, dtype, ttas, min_areas, conn):
    rows_by_area = {a: [] for a in min_areas}
    for b in tqdm(loader, desc='Avaliando TTA/min_area'):
        x = b['image'].to(device, non_blocking=True)
        y = b['mask'].to(device, non_blocking=True)
        ps = None
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=amp):
            for t in ttas:
                p = torch.sigmoid(model(tr(x, t)))
                p = inv(p, t)
                ps = p if ps is None else ps + p
            prob = ps / len(ttas)
        raw = (prob >= threshold).detach().cpu().numpy().astype(np.uint8)
        yy = y.detach().cpu().numpy().astype(np.uint8)
        for area in min_areas:
            pp = np.zeros_like(raw)
            for i in range(raw.shape[0]): pp[i, 0] = remove_small(raw[i, 0], area, conn)
            mt = metrics(pp, yy)
            for i in range(raw.shape[0]):
                tratio = float(mt['target_pixels'][i] / mt['total_pixels'][i])
                rows_by_area[area].append({'image_path': b['image_path'][i], 'mask_path': b['mask_path'][i], 'dice': float(mt['dice'][i]), 'iou': float(mt['iou'][i]), 'precision': float(mt['precision'][i]), 'recall': float(mt['recall'][i]), 'f1': float(mt['f1'][i]), 'tp': float(mt['tp'][i]), 'tn': float(mt['tn'][i]), 'fp': float(mt['fp'][i]), 'fn': float(mt['fn'][i]), 'pred_pixels': float(mt['pred_pixels'][i]), 'target_pixels': float(mt['target_pixels'][i]), 'total_pixels': float(mt['total_pixels'][i]), 'target_ratio': tratio, 'pred_ratio': float(mt['pred_pixels'][i] / mt['total_pixels'][i]), 'mask_size_class': size_class(tratio)})
    return {k: pd.DataFrame(v) for k, v in rows_by_area.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/raw/ai4boundaries/orthos'))
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--tta-mode', default='flips', choices=['none', 'flips', 'd4'])
    ap.add_argument('--min-area-values', default='0,50,100,200,500,1000,2000,5000')
    ap.add_argument('--connectivity', type=int, default=8, choices=[4, 8])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--amp-dtype', default='bf16', choices=['bf16', 'fp16'])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.checkpoint, device)
    ds = OrthoDS(build_pairs(args.root, args.split), image_size=int(cfg.get('image_size', 512)), mask_band=int(cfg.get('mask_band', 1)), mask_mode=cfg.get('mask_mode', 'band'), normalization=cfg.get('normalization', 'imagenet'))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    min_areas = [int(x.strip()) for x in args.min_area_values.split(',') if x.strip()]
    ttas = tta_names(args.tta_mode)
    print('Configuração:', {'checkpoint': str(args.checkpoint), 'threshold': args.threshold, 'tta': ttas, 'min_area_values': min_areas, 'connectivity': args.connectivity, 'pairs': len(ds)})
    dfs = evaluate(model, loader, device, args.threshold, args.amp, dtype, ttas, min_areas, args.connectivity)
    rank_rows = []
    for area, df in dfs.items():
        run_dir = args.output_dir / f'min_area_{area}'
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / 'per_sample_metrics.csv', index=False)
        summary = {'split': args.split, 'checkpoint': str(args.checkpoint), 'threshold': args.threshold, 'tta_mode': args.tta_mode, 'tta_list': ttas, 'min_area': area, 'connectivity': args.connectivity, 'num_samples': int(len(df))}
        for m in ['dice', 'iou', 'precision', 'recall', 'f1']: summary[m] = summarize(df[m].values)
        (run_dir / 'summary_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        by = df.groupby('mask_size_class').agg(count=('dice','size'), dice_mean=('dice','mean'), dice_median=('dice','median'), iou_mean=('iou','mean'), precision_mean=('precision','mean'), recall_mean=('recall','mean'), f1_mean=('f1','mean'), pred_pixels_mean=('pred_pixels','mean'), target_pixels_mean=('target_pixels','mean'), target_ratio_mean=('target_ratio','mean'), pred_ratio_mean=('pred_ratio','mean')).reset_index()
        by.to_csv(run_dir / 'summary_by_mask_size.csv', index=False)
        rank_rows.append({'min_area': area, 'dice': summary['dice']['mean'], 'iou': summary['iou']['mean'], 'precision': summary['precision']['mean'], 'recall': summary['recall']['mean'], 'f1': summary['f1']['mean'], 'run_dir': str(run_dir)})
    rank = pd.DataFrame(rank_rows).sort_values(['dice', 'f1', 'iou'], ascending=False)
    rank.to_csv(args.output_dir / 'postprocess_ranking.csv', index=False)
    print('\nRanking pós-processamento:')
    print(rank.to_string(index=False))
    print('\nArquivos salvos em:', args.output_dir)


if __name__ == '__main__': main()
