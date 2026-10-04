from __future__ import annotations

import argparse, json
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
    def __init__(self, pairs, image_size=512, mask_band=1, mask_mode='band', normalization='imagenet'):
        self.pairs=pairs; self.image_size=image_size; self.mask_band=mask_band; self.mask_mode=mask_mode; self.normalization=normalization
    def __len__(self): return len(self.pairs)
    def read_image(self, path):
        with rasterio.open(path) as src: x=src.read([1,2,3]).astype(np.float32)
        if x.max()>2: x/=255.0
        if self.normalization=='imagenet':
            mean=np.array([0.485,0.456,0.406],dtype=np.float32)[:,None,None]
            std=np.array([0.229,0.224,0.225],dtype=np.float32)[:,None,None]
            x=(x-mean)/std
        elif self.normalization!='none': raise ValueError(self.normalization)
        return x.astype(np.float32)
    def read_mask(self, path):
        with rasterio.open(path) as src:
            if self.mask_mode=='band': m=(src.read(self.mask_band)>0).astype(np.float32)
            elif self.mask_mode=='any': m=(src.read()>0).any(axis=0).astype(np.float32)
            else: raise ValueError(self.mask_mode)
        return m[None].astype(np.float32)
    def resize(self, x,y):
        if x.shape[-2:] == (self.image_size,self.image_size): return x,y
        x=F.interpolate(x[None],size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)[0]
        y=F.interpolate(y[None],size=(self.image_size,self.image_size),mode='nearest')[0]
        return x,y
    def __getitem__(self,idx):
        img,mask=self.pairs[idx]
        x=torch.from_numpy(self.read_image(img)).float(); y=torch.from_numpy(self.read_mask(mask)).float()
        x,y=self.resize(x,y)
        return {'image':x,'mask':y,'image_path':str(img),'mask_path':str(mask)}


def build_model(architecture, encoder_name, encoder_weights=None, in_channels=3):
    if encoder_weights and str(encoder_weights).lower()=='none': encoder_weights=None
    a=architecture.lower(); kwargs=dict(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    if a=='unet': return smp.Unet(**kwargs)
    if a=='unetplusplus': return smp.UnetPlusPlus(**kwargs)
    if a=='deeplabv3plus': return smp.DeepLabV3Plus(**kwargs)
    if a=='fpn': return smp.FPN(**kwargs)
    if a=='pan': return smp.PAN(**kwargs)
    raise ValueError(f'Arquitetura inválida: {architecture}')


def load_model(checkpoint_path: Path, device):
    ck=torch.load(checkpoint_path,map_location=device,weights_only=False); cfg=ck.get('config',{})
    m=build_model(cfg.get('architecture','unet'), cfg.get('encoder_name','resnet34'), None, int(cfg.get('in_channels',3))).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval(); return m,cfg


def metrics(logits, masks, thr=0.5, eps=1e-7):
    p=(torch.sigmoid(logits)>=thr).float().flatten(1); y=masks.flatten(1)
    tp=(p*y).sum(1); fp=(p*(1-y)).sum(1); fn=((1-p)*y).sum(1); tn=((1-p)*(1-y)).sum(1)
    dice=(2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    prec=(tp+eps)/(tp+fp+eps); rec=(tp+eps)/(tp+fn+eps); f1=(2*prec*rec+eps)/(prec+rec+eps)
    out=dict(dice=dice,iou=iou,precision=prec,recall=rec,f1=f1,tp=tp,tn=tn,fp=fp,fn=fn,pred_pixels=p.sum(1),target_pixels=y.sum(1))
    return {k:v.detach().cpu().numpy() for k,v in out.items()} | {'total_pixels':np.full(len(dice),y.shape[1],dtype=np.float32)}


def summarize(vals):
    vals=np.asarray(vals,dtype=np.float64)
    return {'mean':float(vals.mean()),'std':float(vals.std()),'median':float(np.median(vals)),'min':float(vals.min()),'max':float(vals.max())}


@torch.no_grad()
def evaluate(model, loader, device, threshold, amp, amp_dtype):
    rows=[]; model.eval()
    for b in tqdm(loader,desc='Avaliando'):
        x=b['image'].to(device,non_blocking=True); y=b['mask'].to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=amp): logits=model(x)
        met=metrics(logits,y,threshold)
        for i in range(x.shape[0]):
            ratio=float(met['target_pixels'][i]/met['total_pixels'][i])
            row={'image_path':b['image_path'][i],'mask_path':b['mask_path'][i],
                 **{k:float(met[k][i]) for k in ['dice','iou','precision','recall','f1','tp','tn','fp','fn','pred_pixels','target_pixels']},
                 'total_pixels':float(met['total_pixels'][i]),'target_ratio':ratio,
                 'pred_ratio':float(met['pred_pixels'][i]/met['total_pixels'][i]),'mask_size_class':size_class(ratio)}
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('data/raw/ai4boundaries/orthos'))
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split',default='test',choices=['train','val','test'])
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--threshold',type=float,default=0.5)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--num-workers',type=int,default=8)
    ap.add_argument('--amp',action='store_true')
    ap.add_argument('--amp-dtype',default='bf16',choices=['bf16','fp16'])
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,cfg=load_model(args.checkpoint,device)
    image_size=int(cfg.get('image_size',512)); mask_band=int(cfg.get('mask_band',1)); mask_mode=cfg.get('mask_mode','band'); normalization=cfg.get('normalization','imagenet')
    pairs=build_pairs(args.root,args.split)
    ds=OrthoDataset(pairs,image_size,mask_band,mask_mode,normalization)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    amp_dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    print('Configuração carregada:'); print({'checkpoint':str(args.checkpoint),'split':args.split,'architecture':cfg.get('architecture'),'encoder':cfg.get('encoder_name'),'image_size':image_size,'mask_mode':mask_mode,'mask_band':mask_band,'threshold':args.threshold,'pairs':len(pairs)})
    df=evaluate(model,loader,device,args.threshold,args.amp,amp_dtype)
    df.to_csv(args.output_dir/'per_sample_metrics.csv',index=False)
    summary={'split':args.split,'checkpoint':str(args.checkpoint),'threshold':args.threshold,'num_samples':int(len(df))}
    for k in ['dice','iou','precision','recall','f1']: summary[k]=summarize(df[k].values)
    (args.output_dir/'summary_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    by=df.groupby('mask_size_class').agg(count=('dice','size'),dice_mean=('dice','mean'),dice_median=('dice','median'),iou_mean=('iou','mean'),precision_mean=('precision','mean'),recall_mean=('recall','mean'),f1_mean=('f1','mean'),pred_pixels_mean=('pred_pixels','mean'),target_pixels_mean=('target_pixels','mean'),target_ratio_mean=('target_ratio','mean'),pred_ratio_mean=('pred_ratio','mean')).reset_index()
    by.to_csv(args.output_dir/'summary_by_mask_size.csv',index=False)
    print('\nResumo final:')
    for k in ['dice','iou','precision','recall','f1']: print(k, summary[k])
    print('\nPor tamanho de máscara:'); print(by.to_string(index=False))
    print('\nArquivos salvos em:',args.output_dir)

if __name__=='__main__': main()
