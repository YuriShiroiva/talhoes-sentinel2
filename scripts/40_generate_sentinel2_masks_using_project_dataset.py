from __future__ import annotations
import argparse, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from field_boundary.data.dataset import AI4BoundariesSegmentationDataset
import segmentation_models_pytorch as smp


def build_model(architecture, encoder_name, encoder_weights=None, in_channels=3, classes=1):
    arch = str(architecture).lower().replace("_", "").replace("-", "")

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if arch in {"unetplusplus", "unet++"}:
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if arch in {"deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    raise ValueError(f"Arquitetura não suportada no script visual: {architecture}")


def norm_path(x):
    return str(x).replace('\\\\','/')


def safe(x):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',Path(str(x)).stem)


def guess_mask_path(image_path: Path) -> Path:
    s = norm_path(image_path).replace('/images/','/masks/')
    return Path(s).with_suffix('.tif')


def infer_arch(sd):
    ks=list(sd.keys())
    if any(k.startswith('decoder.aspp') or k.startswith('decoder.block1') for k in ks):
        return 'deeplabv3plus'
    if any(k.startswith('decoder.blocks') for k in ks):
        return 'unet'
    return None


def load_model(ckpt_path: Path, device):
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    cfg=ckpt.get('config',{})
    sd=ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    arch=cfg.get('architecture') or infer_arch(sd) or 'unet'
    enc=cfg.get('encoder_name','resnet34')
    inc=int(cfg.get('in_channels',30))
    print(f'Carregando {ckpt_path} | arch={arch} encoder={enc} in_channels={inc}')
    model=build_model(architecture=arch, encoder_name=enc, encoder_weights=None, in_channels=inc, classes=1).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def remove_small(mask, min_area, connectivity=8):
    if min_area<=0: return mask.astype(np.uint8)
    st=np.ones((3,3),dtype=np.uint8) if connectivity==8 else np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    lab,n=ndimage.label(mask.astype(bool),structure=st)
    if n==0: return mask.astype(np.uint8)
    sizes=np.bincount(lab.ravel())
    keep=sizes>=min_area
    keep[0]=False
    return keep[lab].astype(np.uint8)


def flip(x,m):
    if m=='id': return x
    if m=='h': return torch.flip(x,dims=[3])
    if m=='v': return torch.flip(x,dims=[2])
    if m=='hv': return torch.flip(x,dims=[2,3])
    raise ValueError(m)


@torch.no_grad()
def predict(img, models, weights, threshold, min_area, connectivity, tta):
    modes=['id'] if tta=='none' else ['id','h','v','hv']
    acc=None
    for m in modes:
        xt=flip(img,m)
        pm=None
        for model,w in zip(models,weights):
            y=model(xt)
            if y.shape[1]>1: y=y[:,0:1]
            p=torch.sigmoid(y)
            pm=w*p if pm is None else pm+w*p
        pm=flip(pm,m)
        acc=pm if acc is None else acc+pm
    prob=(acc/len(modes))[0,0].float().cpu().numpy()
    pred=remove_small((prob>=threshold).astype(np.uint8),min_area,connectivity)
    return prob,pred


def stretch(a):
    a=np.nan_to_num(a.astype(np.float32),nan=0,posinf=0,neginf=0)
    out=np.zeros_like(a)
    for c in range(3):
        lo,hi=np.percentile(a[...,c],[2,98])
        out[...,c]=np.clip((a[...,c]-lo)/(hi-lo+1e-6),0,1)
    return out


def make_rgb(x, stack_order='time_band'):
    arr=x.detach().cpu().numpy()
    c,h,w=arr.shape
    if c>=30 and stack_order=='time_band':
        rs,gs,bs=[],[],[]
        for t in range(6):
            base=t*5
            bs.append(arr[base+0]); gs.append(arr[base+1]); rs.append(arr[base+2])
        rgb=np.stack([np.median(rs,0),np.median(gs,0),np.median(bs,0)],axis=-1)
    elif c>=30:
        b=np.median(arr[0:6],0); g=np.median(arr[6:12],0); r=np.median(arr[12:18],0)
        rgb=np.stack([r,g,b],axis=-1)
    elif c>=3:
        rgb=np.transpose(arr[:3],(1,2,0))
    else:
        rgb=np.repeat(arr[0][...,None],3,axis=-1)
    return stretch(rgb)


def comp_rgb(target,pred):
    out=np.ones((*target.shape,3),dtype=np.float32)
    tp=(target==1)&(pred==1); fp=(target==0)&(pred==1); fn=(target==1)&(pred==0)
    out[tp]=[0.05,0.65,0.15]
    out[fp]=[0.9,0.1,0.05]
    out[fn]=[0.1,0.2,0.9]
    return out


def metrics(target,pred):
    t=target.astype(bool); p=pred.astype(bool)
    tp=(t&p).sum(); fp=((~t)&p).sum(); fn=(t&(~p)).sum(); eps=1e-7
    dice=(2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    return float(dice),float(iou),float(t.mean()),float(p.mean())


def panel(row,rgb,target,pred,prob,out):
    dice,iou,tr,pr=metrics(target,pred)
    fig,ax=plt.subplots(1,5,figsize=(18,4.3))
    ax[0].imshow(rgb); ax[0].set_title('Sentinel-2 RGB')
    ax[1].imshow(target,cmap='gray',vmin=0,vmax=1); ax[1].set_title('Máscara real')
    ax[2].imshow(pred,cmap='gray',vmin=0,vmax=1); ax[2].set_title('Máscara criada')
    ax[3].imshow(comp_rgb(target,pred)); ax[3].set_title('Comparação\nverde=acerto vermelho=extra azul=faltou')
    ax[4].imshow(prob,cmap='viridis',vmin=0,vmax=1); ax[4].set_title('Probabilidade')
    for a in ax: a.axis('off')
    group=row.get('selection_group',row.get('overview_group','case'))
    fig.suptitle(f"{safe(row['image_path'])} | {group}\nDice={dice:.3f} IoU={iou:.3f} target_ratio={tr:.3f} pred_ratio={pr:.3f}",fontsize=12,y=1.03)
    fig.tight_layout(); fig.savefig(out,dpi=180,bbox_inches='tight'); plt.close(fig)
    return {'dice_recomputed':dice,'iou_recomputed':iou,'target_ratio_recomputed':tr,'pred_ratio_recomputed':pr}


def contact_sheet(paths,out,cols=2):
    if not paths: return
    import matplotlib.image as mpimg
    imgs=[mpimg.imread(p) for p in paths]
    rows=math.ceil(len(imgs)/cols)
    fig,ax=plt.subplots(rows,cols,figsize=(cols*9.5,rows*2.8)); ax=np.asarray(ax).reshape(-1)
    for a,img,p in zip(ax,imgs,paths):
        a.imshow(img); a.set_title(p.stem,fontsize=7); a.axis('off')
    for a in ax[len(imgs):]: a.axis('off')
    fig.tight_layout(); fig.savefig(out,dpi=160,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selected-csv',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--image-size',type=int,default=256)
    ap.add_argument('--in-channels',type=int,default=30)
    ap.add_argument('--threshold',type=float,default=0.49)
    ap.add_argument('--min-area',type=int,default=90)
    ap.add_argument('--connectivity',type=int,default=8)
    ap.add_argument('--tta-mode',choices=['none','flips'],default='flips')
    ap.add_argument('--max-cases',type=int,default=20)
    ap.add_argument('--groups',default='')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--stack-order',choices=['time_band','band_time'],default='time_band')
    args=ap.parse_args()

    args.output_dir.mkdir(parents=True,exist_ok=True)
    panels=args.output_dir/'mask_comparison_panels'; panels.mkdir(exist_ok=True)
    masks=args.output_dir/'binary_masks'; masks.mkdir(exist_ok=True)

    df=pd.read_csv(args.selected_csv)
    if args.groups.strip():
        keep=[g.strip() for g in args.groups.split(',') if g.strip()]
        df=df[df['selection_group'].isin(keep)].copy()
    df=df.head(args.max_cases).copy()

    weights=np.array([float(x) for x in args.weights.split(',')],dtype=np.float32)
    weights=(weights/weights.sum()).tolist()
    device=torch.device(args.device if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    models=[load_model(p,device) for p in args.checkpoints]

    rows=[]; panel_paths=[]
    for idx,row in tqdm(df.iterrows(),total=len(df),desc='Gerando painéis'):
        img_path=Path(norm_path(row['image_path']))
        mask_path=Path(norm_path(row['mask_path'])) if 'mask_path' in row and pd.notna(row['mask_path']) else guess_mask_path(img_path)
        try:
            ds=AI4BoundariesSegmentationDataset(pairs=[(img_path,mask_path)],image_size=args.image_size,in_channels=args.in_channels,augment=False)
            sample=ds[0]
            img=sample['image'].unsqueeze(0).to(device)
            target=np.squeeze(sample['mask'].detach().cpu().numpy()); target=(target>0).astype(np.uint8)
            prob,pred=predict(img,models,weights,args.threshold,args.min_area,args.connectivity,args.tta_mode)
            rgb=make_rgb(sample['image'],args.stack_order)
            name=re.sub(r'[^A-Za-z0-9_.-]+','_',f"{idx+1:03d}_{row.get('selection_group','case')}_{safe(row['image_path'])}")
            ppath=panels/f'{name}.png'
            rec=row.to_dict(); rec.update(panel(row,rgb,target,pred,prob,ppath)); rec['panel_path']=str(ppath)
            plt.imsave(masks/f'{name}_real_mask.png',target,cmap='gray',vmin=0,vmax=1)
            plt.imsave(masks/f'{name}_pred_mask.png',pred,cmap='gray',vmin=0,vmax=1)
            plt.imsave(masks/f'{name}_comparison.png',comp_rgb(target,pred))
            rows.append(rec); panel_paths.append(ppath)
        except Exception as e:
            print(f'[ERRO] {img_path}: {e}')
    pd.DataFrame(rows).to_csv(args.output_dir/'generated_mask_comparison_index.csv',index=False)
    contact_sheet(panel_paths,args.output_dir/'boss_mask_comparison_contact_sheet.png')
    print('\nFinalizado.')
    print('Painéis gerados:',len(panel_paths))
    print('Painéis:',panels)
    print('Máscaras:',masks)
    print('Prancha geral:',args.output_dir/'boss_mask_comparison_contact_sheet.png')

if __name__=='__main__':
    main()
