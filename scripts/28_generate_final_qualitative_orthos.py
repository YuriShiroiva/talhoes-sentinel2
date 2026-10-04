from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def import_ensemble_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Não encontrei o script base: {script_path}")
    spec = importlib.util.spec_from_file_location('ensemble27', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_weights(text: str, n: int):
    weights = [float(x.strip()) for x in text.split(',') if x.strip()]
    if len(weights) != n:
        raise ValueError(f'Esperado {n} pesos, recebido: {weights}')
    total = sum(weights)
    if total <= 0:
        raise ValueError('A soma dos pesos precisa ser positiva.')
    return [w / total for w in weights]


def read_rgb(image_path: str, image_size: int) -> np.ndarray:
    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    if rgb.max() > 2.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    if rgb.shape[-2:] != (image_size, image_size):
        import torch.nn.functional as F
        t = torch.from_numpy(rgb)[None]
        t = F.interpolate(t, size=(image_size, image_size), mode='bilinear', align_corners=False)[0]
        rgb = t.numpy()
    return np.moveaxis(rgb, 0, -1)


def metric_single(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    p = pred.reshape(-1).astype(np.float32)
    y = target.reshape(-1).astype(np.float32)
    tp = float((p * y).sum())
    fp = float((p * (1 - y)).sum())
    fn = float(((1 - p) * y).sum())
    tn = float(((1 - p) * (1 - y)).sum())
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    total = float(p.size)
    pred_pixels = float(p.sum())
    target_pixels = float(y.sum())
    return {
        'dice': float(dice), 'iou': float(iou), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'pred_pixels': pred_pixels, 'target_pixels': target_pixels, 'total_pixels': total,
        'target_ratio': target_pixels / total, 'pred_ratio': pred_pixels / total,
    }


@torch.no_grad()
def predict_batch(mod, models, weights, images, threshold, amp, amp_dtype, tta_list, min_area, connectivity):
    probs_sum = None
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        for tta in tta_list:
            x_t = mod.apply_transform(images, tta)
            probs_t = None
            for weight, model in zip(weights, models):
                p = torch.sigmoid(model(x_t))
                probs_t = weight * p if probs_t is None else probs_t + weight * p
            probs_t = mod.invert_transform(probs_t, tta)
            probs_sum = probs_t if probs_sum is None else probs_sum + probs_t
        probs = probs_sum / float(len(tta_list))

    probs_np = probs.detach().cpu().numpy().astype(np.float32)
    raw = (probs_np >= threshold).astype(np.uint8)
    processed = np.zeros_like(raw, dtype=np.uint8)
    for i in range(raw.shape[0]):
        processed[i, 0] = mod.remove_small_components(raw[i, 0], min_area=min_area, connectivity=connectivity)
    return probs_np, processed


def error_rgb(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(bool)
    target = target.astype(bool)
    out = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.float32)
    tp = pred & target
    fp = pred & (~target)
    fn = (~pred) & target
    out[tp] = [1.0, 1.0, 1.0]
    out[fp] = [1.0, 0.0, 0.0]
    out[fn] = [0.0, 0.35, 1.0]
    return out


def save_panel(out_path: Path, rgb, target, pred, prob, row, group_name: str):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=180)
    axes[0].imshow(rgb)
    axes[0].set_title('Imagem RGB')
    axes[1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Máscara real')
    axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predição')
    axes[3].imshow(prob, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Probabilidade')
    axes[4].imshow(error_rgb(pred, target))
    axes[4].set_title('Erro: TP branco | FP vermelho | FN azul')
    for ax in axes:
        ax.axis('off')
    title = (
        f"{group_name} | Dice={row['dice']:.4f} | IoU={row['iou']:.4f} | "
        f"P={row['precision']:.4f} | R={row['recall']:.4f} | {row['mask_size_class']}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def clean_name(path: str):
    stem = Path(path).stem
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in stem)


def choose_groups(df: pd.DataFrame, n: int, seed: int):
    rng = random.Random(seed)
    groups = {}
    groups['best'] = df.sort_values(['dice', 'iou'], ascending=False).head(n)
    groups['worst'] = df.sort_values(['dice', 'iou'], ascending=True).head(n)
    non_empty = df[df['target_pixels'] > 0]
    groups['best_non_empty'] = non_empty.sort_values(['dice', 'iou'], ascending=False).head(n)
    groups['worst_non_empty'] = non_empty.sort_values(['dice', 'iou'], ascending=True).head(n)
    small = df[df['mask_size_class'].isin(['very_small_<1%', 'small_1_5%'])]
    if len(small):
        groups['small_fields'] = small.sort_values(['dice', 'iou'], ascending=False).head(n)
    empty = df[df['mask_size_class'] == 'empty']
    if len(empty):
        groups['empty_cases'] = empty.sort_values(['pred_pixels', 'dice'], ascending=[False, True]).head(n)
    medium = df[df['mask_size_class'] == 'medium_5_25%']
    if len(medium):
        idx = rng.sample(list(medium.index), k=min(n, len(medium)))
        groups['random_medium'] = medium.loc[idx]
    return groups


def main():
    parser = argparse.ArgumentParser(description='Gera figuras qualitativas finais para o ensemble Ortho O23.')
    parser.add_argument('--root', type=Path, default=Path('data/raw/ai4boundaries/orthos'))
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/final_report_assets/o23_qualitative'))
    parser.add_argument('--script-27', type=Path, default=Path('scripts/27_evaluate_orthos_ensemble_tta_postprocess.py'))
    parser.add_argument('--checkpoints', nargs='+', type=Path, default=[
        Path('outputs/orthos/test_o02_unetplusplus_r50_ortho_1m_512/best_ortho_model.pt'),
        Path('outputs/orthos/test_o05_deeplabv3plus_r50_ortho_1m_512/best_ortho_model.pt'),
        Path('outputs/orthos/test_o01_unet_r34_ortho_1m_512/best_ortho_model.pt'),
        Path('outputs/orthos/test_o18_deeplabv3plus_r50_ortho_1m_512_seed123/best_ortho_model.pt'),
    ])
    parser.add_argument('--weights', default='0.532,0.203,0.08,0.185')
    parser.add_argument('--threshold', type=float, default=0.6315)
    parser.add_argument('--min-area', type=int, default=1020)
    parser.add_argument('--connectivity', type=int, default=8, choices=[4, 8])
    parser.add_argument('--tta-mode', default='flips', choices=['none', 'flips', 'd4'])
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--mask-band', type=int, default=1)
    parser.add_argument('--mask-mode', default='band', choices=['band', 'any'])
    parser.add_argument('--normalization', default='imagenet', choices=['imagenet', 'none'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp-dtype', default='bf16', choices=['bf16', 'fp16'])
    parser.add_argument('--n-per-group', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mod = import_ensemble_module(args.script_27)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for ckpt in args.checkpoints:
        model, _ = mod.load_model(ckpt, device)
        model.eval()
        models.append(model)

    weights = parse_weights(args.weights, len(models))
    tta_list = mod.get_tta_list(args.tta_mode)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16

    pairs = mod.build_pairs(args.root, args.split)
    dataset = mod.OrthoDataset(
        pairs,
        image_size=args.image_size,
        mask_band=args.mask_band,
        mask_mode=args.mask_mode,
        normalization=args.normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    config = vars(args).copy()
    config['checkpoints'] = [str(x) for x in args.checkpoints]
    config['weights_normalized'] = weights
    config['tta_list'] = tta_list
    config['n_samples'] = len(dataset)
    (args.output_dir / 'qualitative_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

    rows = []
    print('Primeira passada: recomputando métricas...')
    for batch in tqdm(loader):
        images = batch['image'].to(device, non_blocking=True)
        probs, preds = predict_batch(
            mod, models, weights, images,
            threshold=args.threshold,
            amp=args.amp,
            amp_dtype=amp_dtype,
            tta_list=tta_list,
            min_area=args.min_area,
            connectivity=args.connectivity,
        )
        masks = batch['mask'].numpy().astype(np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i, 0]
            target = masks[i, 0]
            metric = metric_single(pred, target)
            row = {
                'image_path': batch['image_path'][i],
                'mask_path': batch['mask_path'][i],
                **metric,
            }
            row['mask_size_class'] = mod.size_class_from_ratio(row['target_ratio'])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / 'o23_per_sample_metrics_recomputed.csv', index=False)

    summary = {}
    for m in ['dice', 'iou', 'precision', 'recall', 'f1']:
        summary[m] = {
            'mean': float(df[m].mean()),
            'std': float(df[m].std(ddof=0)),
            'median': float(df[m].median()),
            'min': float(df[m].min()),
            'max': float(df[m].max()),
        }
    summary['num_samples'] = int(len(df))
    (args.output_dir / 'o23_summary_recomputed.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    by_size = df.groupby('mask_size_class').agg(
        count=('dice', 'size'),
        dice_mean=('dice', 'mean'),
        dice_median=('dice', 'median'),
        iou_mean=('iou', 'mean'),
        precision_mean=('precision', 'mean'),
        recall_mean=('recall', 'mean'),
        f1_mean=('f1', 'mean'),
        pred_pixels_mean=('pred_pixels', 'mean'),
        target_pixels_mean=('target_pixels', 'mean'),
        target_ratio_mean=('target_ratio', 'mean'),
        pred_ratio_mean=('pred_ratio', 'mean'),
    ).reset_index()
    by_size.to_csv(args.output_dir / 'o23_summary_by_mask_size_recomputed.csv', index=False)

    groups = choose_groups(df, n=args.n_per_group, seed=args.seed)
    selected_paths = set()
    for group_df in groups.values():
        selected_paths.update(group_df['image_path'].tolist())

    selected_pairs = [(Path(img), Path(mask)) for img, mask in pairs if str(img) in selected_paths]
    selected_ds = mod.OrthoDataset(
        selected_pairs,
        image_size=args.image_size,
        mask_band=args.mask_band,
        mask_mode=args.mask_mode,
        normalization=args.normalization,
    )
    selected_loader = DataLoader(selected_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    payload = {}
    print('Segunda passada: gerando predições para amostras selecionadas...')
    for batch in tqdm(selected_loader):
        images = batch['image'].to(device, non_blocking=True)
        probs, preds = predict_batch(
            mod, models, weights, images,
            threshold=args.threshold,
            amp=args.amp,
            amp_dtype=amp_dtype,
            tta_list=tta_list,
            min_area=args.min_area,
            connectivity=args.connectivity,
        )
        img_path = batch['image_path'][0]
        mask = batch['mask'].numpy().astype(np.uint8)[0, 0]
        payload[img_path] = {
            'rgb': read_rgb(img_path, args.image_size),
            'target': mask,
            'pred': preds[0, 0],
            'prob': probs[0, 0],
        }

    for group_name, group_df in groups.items():
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_df.to_csv(group_dir / f'{group_name}_selected_samples.csv', index=False)
        for idx, (_, row) in enumerate(group_df.iterrows(), start=1):
            img_path = row['image_path']
            p = payload.get(img_path)
            if p is None:
                continue
            filename = f"{idx:02d}_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}_{clean_name(img_path)}.png"
            save_panel(group_dir / filename, p['rgb'], p['target'], p['pred'], p['prob'], row, group_name)

    print('\nResumo recomposto:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('\nPor tamanho de máscara:')
    print(by_size.to_string(index=False))
    print('\nArquivos salvos em:', args.output_dir)


if __name__ == '__main__':
    main()
