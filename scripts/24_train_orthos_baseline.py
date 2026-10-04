from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def image_to_mask_path(image_path: Path, root: Path, split: str) -> Path:
    return root / 'masks' / split / image_path.name.replace('_ortho_', '_ortholabel_')


def build_pairs(root: Path, split: str):
    img_dir = root / 'images' / split
    pairs=[]
    for img in sorted(img_dir.glob('*.tif')):
        mask = image_to_mask_path(img, root, split)
        if mask.exists(): pairs.append((img, mask))
    return pairs


def size_class(r: float) -> str:
    if r == 0: return 'empty'
    if r < 0.01: return 'very_small_<1%'
    if r < 0.05: return 'small_1_5%'
    if r < 0.25: return 'medium_5_25%'
    if r < 0.75: return 'large_25_75%'
    return 'very_large_>75%'


class OrthoDataset(Dataset):
    def __init__(self, pairs, image_size=512, mask_band=1, mask_mode='band', augment=False, normalization='imagenet'):
        self.pairs=pairs; self.image_size=image_size; self.mask_band=mask_band; self.mask_mode=mask_mode
        self.augment=augment; self.normalization=normalization

    def __len__(self): return len(self.pairs)

    def read_image(self, path: Path):
        with rasterio.open(path) as src:
            x = src.read([1,2,3]).astype(np.float32)
        if x.max() > 2: x /= 255.0
        if self.normalization == 'imagenet':
            mean=np.array([0.485,0.456,0.406], dtype=np.float32)[:,None,None]
            std=np.array([0.229,0.224,0.225], dtype=np.float32)[:,None,None]
            x=(x-mean)/std
        elif self.normalization != 'none':
            raise ValueError(self.normalization)
        return x.astype(np.float32)

    def read_mask(self, path: Path):
        with rasterio.open(path) as src:
            if self.mask_mode == 'band':
                m = src.read(self.mask_band)
                m = (m > 0).astype(np.float32)
            elif self.mask_mode == 'any':
                arr = src.read()
                m = (arr > 0).any(axis=0).astype(np.float32)
            else:
                raise ValueError(self.mask_mode)
        return m[None].astype(np.float32)

    def resize(self, x, y):
        if x.shape[-2:] == (self.image_size, self.image_size): return x,y
        x = F.interpolate(x[None], size=(self.image_size,self.image_size), mode='bilinear', align_corners=False)[0]
        y = F.interpolate(y[None], size=(self.image_size,self.image_size), mode='nearest')[0]
        return x,y

    def aug(self, x, y):
        if random.random()<0.5: x=torch.flip(x,[2]); y=torch.flip(y,[2])
        if random.random()<0.5: x=torch.flip(x,[1]); y=torch.flip(y,[1])
        k=random.randint(0,3)
        if k: x=torch.rot90(x,k,[1,2]); y=torch.rot90(y,k,[1,2])
        return x,y

    def __getitem__(self, idx):
        img, mask = self.pairs[idx]
        x=torch.from_numpy(self.read_image(img)).float()
        y=torch.from_numpy(self.read_mask(mask)).float()
        x,y=self.resize(x,y)
        if self.augment: x,y=self.aug(x,y)
        r=float(y.mean().item())
        return {'image':x, 'mask':y, 'image_path':str(img), 'mask_path':str(mask), 'positive_ratio':r, 'size_class':size_class(r)}


def build_model(architecture, encoder_name, encoder_weights, in_channels=3):
    if encoder_weights and str(encoder_weights).lower() == 'none': encoder_weights=None
    a=architecture.lower()
    kwargs=dict(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    if a=='unet': return smp.Unet(**kwargs)
    if a=='unetplusplus': return smp.UnetPlusPlus(**kwargs)
    if a=='deeplabv3plus': return smp.DeepLabV3Plus(**kwargs)
    if a=='fpn': return smp.FPN(**kwargs)
    if a=='pan': return smp.PAN(**kwargs)
    raise ValueError(f'Arquitetura inválida: {architecture}')


def dice_loss(logits, targets, eps=1e-7):
    p=torch.sigmoid(logits).flatten(1); y=targets.flatten(1)
    inter=(p*y).sum(1); denom=p.sum(1)+y.sum(1)
    return 1 - ((2*inter+eps)/(denom+eps)).mean()


def loss_fn(logits, masks): return F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)


def batch_metrics(logits, masks, thr=0.5, eps=1e-7):
    p=(torch.sigmoid(logits)>=thr).float().flatten(1); y=masks.flatten(1)
    tp=(p*y).sum(1); fp=(p*(1-y)).sum(1); fn=((1-p)*y).sum(1)
    dice=(2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    prec=(tp+eps)/(tp+fp+eps); rec=(tp+eps)/(tp+fn+eps); f1=(2*prec*rec+eps)/(prec+rec+eps)
    return {k: float(v.mean().detach().cpu()) for k,v in dict(dice=dice,iou=iou,precision=prec,recall=rec,f1=f1).items()}


def run_epoch(model, loader, device, optimizer, amp, amp_dtype, scaler, train, threshold):
    model.train(train)
    total=0.0; sums={k:0.0 for k in ['dice','iou','precision','recall','f1']}; n=0
    for b in tqdm(loader, desc='Treino' if train else 'Validação', leave=False):
        x=b['image'].to(device, non_blocking=True); y=b['mask'].to(device, non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            logits=model(x); loss=loss_fn(logits,y)
        if train:
            if amp and scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        met=batch_metrics(logits.detach(), y, threshold)
        total += float(loss.detach().cpu())
        for k in sums: sums[k] += met[k]
        n += 1
    out={'loss': total/max(n,1)}; out.update({k:v/max(n,1) for k,v in sums.items()}); return out


def summarize_masks(pairs, mask_band, mask_mode):
    rows=[]
    for _, mp in tqdm(pairs, desc='Resumo máscaras', leave=False):
        with rasterio.open(mp) as src:
            m = (src.read(mask_band)>0) if mask_mode=='band' else (src.read()>0).any(axis=0)
        r=float(m.mean()); rows.append({'mask_path':str(mp),'positive_ratio':r,'size_class':size_class(r)})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/raw/ai4boundaries/orthos'))
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--architecture', default='unet', choices=['unet','unetplusplus','deeplabv3plus','fpn','pan'])
    ap.add_argument('--encoder-name', default='resnet34')
    ap.add_argument('--encoder-weights', default='imagenet')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--mask-band', type=int, default=1)
    ap.add_argument('--mask-mode', default='band', choices=['band','any'])
    ap.add_argument('--normalization', default='imagenet', choices=['imagenet','none'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--amp-dtype', default='bf16', choices=['bf16','fp16'])
    args=ap.parse_args()
    import sys as _sys
    if _sys.platform.startswith("win") and args.num_workers > 0:
        print(f"[aviso] Windows: num_workers={args.num_workers} pode travar (deadlock). Usando num_workers=0.")
        args.num_workers = 0
    set_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    train_pairs=build_pairs(args.root,'train'); val_pairs=build_pairs(args.root,'val')
    print('Configuração:', vars(args)); print('Train pairs:',len(train_pairs),'Val pairs:',len(val_pairs))
    tr_sum=summarize_masks(train_pairs,args.mask_band,args.mask_mode); va_sum=summarize_masks(val_pairs,args.mask_band,args.mask_mode)
    tr_sum.to_csv(args.output_dir/'train_mask_summary.csv', index=False); va_sum.to_csv(args.output_dir/'val_mask_summary.csv', index=False)
    print('\nDistribuição train:'); print(tr_sum['size_class'].value_counts().to_string())
    train_ds=OrthoDataset(train_pairs,args.image_size,args.mask_band,args.mask_mode,True,args.normalization)
    val_ds=OrthoDataset(val_pairs,args.image_size,args.mask_band,args.mask_mode,False,args.normalization)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(args.architecture,args.encoder_name,args.encoder_weights,3).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    amp_dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda', enabled=args.amp and args.amp_dtype=='fp16')
    config=vars(args).copy(); config.update({'in_channels':3,'task':'orthos','bands':['R','G','B']})
    (args.output_dir/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    hist=[]; best=-1.0
    for ep in range(1,args.epochs+1):
        tr=run_epoch(model,train_loader,device,opt,args.amp,amp_dtype,scaler,True,args.threshold)
        va=run_epoch(model,val_loader,device,None,args.amp,amp_dtype,None,False,args.threshold)
        row={'epoch':ep, **{f'train_{k}':v for k,v in tr.items()}, **{f'val_{k}':v for k,v in va.items()}}
        hist.append(row); pd.DataFrame(hist).to_csv(args.output_dir/'training_history.csv',index=False)
        print(f"Epoch {ep:03d}/{args.epochs:03d} | train_loss={tr['loss']:.4f} | train_dice={tr['dice']:.4f} | val_loss={va['loss']:.4f} | val_dice={va['dice']:.4f} | val_iou={va['iou']:.4f} | val_f1={va['f1']:.4f}")
        if va['dice'] > best:
            best=va['dice']; ck=args.output_dir/'best_ortho_model.pt'
            torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'val_metrics':va,'config':config,'best_val_dice':best}, ck)
            print(f'Novo melhor modelo: {ck} | val_dice={best:.4f}')
    torch.save({'model_state_dict':model.state_dict(),'config':config,'best_val_dice':best}, args.output_dir/'last_ortho_model.pt')
    print(f'Treino finalizado. Melhor val_dice={best:.4f}'); print('Arquivos salvos em:', args.output_dir)

if __name__=='__main__': main()
