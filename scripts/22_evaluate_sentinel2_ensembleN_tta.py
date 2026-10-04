
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, rasterio, torch, torch.nn as nn, torch.nn.functional as F, xarray as xr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

def norm(a):
    a=a.astype(np.float32); v=a[np.isfinite(a)]
    if v.size==0: return np.zeros_like(a,dtype=np.float32)
    lo,hi=np.percentile(v,[2,98])
    if np.isclose(lo,hi): return np.zeros_like(a,dtype=np.float32)
    a=np.clip(a,lo,hi); a=(a-lo)/(hi-lo)
    return np.nan_to_num(a,nan=0,posinf=1,neginf=0).astype(np.float32)

def pairs(root, split):
    root=Path(root); im=root/"images"/split; ma=root/"masks"/split; meta=root/"metadata"/f"{split}.csv"
    names=pd.read_csv(meta)["image"].astype(str).tolist() if meta.exists() else [p.name for p in sorted(im.glob("*.nc"))]
    return [(im/n, ma/n.replace(".nc",".tif")) for n in names if (im/n).exists() and (ma/n.replace(".nc",".tif")).exists()]

class DS(Dataset):
    def __init__(self, ps, bands, time_mode, image_size=256):
        self.ps=ps; self.bands=bands; self.time_mode=time_mode; self.image_size=image_size
    def __len__(self): return len(self.ps)
    def read_x(self,p):
        ds=xr.open_dataset(p); ch=[]
        try:
            for b in self.bands:
                d=ds[b]
                if "time" in d.dims and self.time_mode=="stack":
                    vals=d.values
                    for t in range(vals.shape[0]): ch.append(norm(vals[t]))
                elif "time" in d.dims and self.time_mode=="median":
                    ch.append(norm(d.median(dim="time",skipna=True).values))
                else:
                    ch.append(norm(d.values))
        finally:
            ds.close()
        return np.stack(ch,0).astype(np.float32)
    def read_y(self,p):
        with rasterio.open(p) as src: m=src.read(1)
        return (m>0).astype(np.float32)[None]
    def __getitem__(self,i):
        xp,yp=self.ps[i]
        x=torch.from_numpy(self.read_x(xp)).float(); y=torch.from_numpy(self.read_y(yp)).float()
        if x.shape[-2:]!=(self.image_size,self.image_size):
            x=F.interpolate(x[None],size=(self.image_size,self.image_size),mode="bilinear",align_corners=False)[0]
            y=F.interpolate(y[None],size=(self.image_size,self.image_size),mode="nearest")[0]
        return {"image":x,"mask":y,"image_path":str(xp),"mask_path":str(yp)}


def build_model(architecture: str, encoder_name: str, encoder_weights=None, in_channels=None) -> nn.Module:
    """
    Função compatível com scripts que chamam:
    - build_model(architecture, encoder_name, in_channels)
    - build_model(architecture, encoder_name, encoder_weights, in_channels)
    """

    # Compatibilidade com chamadas antigas:
    # build_model(architecture, encoder_name, in_channels)
    if in_channels is None:
        in_channels = encoder_weights
        encoder_weights = None

    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None

    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "pan":
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    raise ValueError(f"Arquitetura inválida: {architecture}")


def load_model(p, device):
    ck=torch.load(p,map_location=device,weights_only=False); c=ck.get("config",{})
    m=build_model(c.get("architecture","unet"), c.get("encoder_name","resnet34"), int(c.get("in_channels",30))).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m,c

def parse_weights(s,n):
    w=[float(x) for x in s.split(",") if x.strip()]
    if len(w)!=n: raise ValueError(f"weights precisa ter {n} valores")
    sm=sum(w)
    return [x/sm for x in w]

def tr(x,name):
    if name=="id": return x
    if name=="h": return torch.flip(x,[3])
    if name=="v": return torch.flip(x,[2])
    if name=="hv": return torch.flip(x,[2,3])
    if name=="r90": return torch.rot90(x,1,[2,3])
    if name=="r180": return torch.rot90(x,2,[2,3])
    if name=="r270": return torch.rot90(x,3,[2,3])
    raise ValueError(name)

def inv(y,name):
    if name=="id": return y
    if name=="h": return torch.flip(y,[3])
    if name=="v": return torch.flip(y,[2])
    if name=="hv": return torch.flip(y,[2,3])
    if name=="r90": return torch.rot90(y,3,[2,3])
    if name=="r180": return torch.rot90(y,2,[2,3])
    if name=="r270": return torch.rot90(y,1,[2,3])
    raise ValueError(name)

def ttas(mode):
    return {"none":["id"],"flips":["id","h","v","hv"],"rot":["id","r90","r180","r270"],"d4":["id","h","v","hv","r90","r180","r270"]}[mode]

def mclass(r):
    if r==0: return "empty"
    if r<.01: return "very_small_<1%"
    if r<.05: return "small_1_5%"
    if r<.25: return "medium_5_25%"
    if r<.75: return "large_25_75%"
    return "very_large_>75%"

def metrics(prob,mask,thr,eps=1e-7):
    pred=(prob>=thr).float(); p=pred.flatten(1); y=mask.flatten(1)
    tp=(p*y).sum(1); fp=(p*(1-y)).sum(1); fn=((1-p)*y).sum(1); tn=((1-p)*(1-y)).sum(1)
    dice=(2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    prec=(tp+eps)/(tp+fp+eps); rec=(tp+eps)/(tp+fn+eps); f1=(2*prec*rec+eps)/(prec+rec+eps)
    return {k:v.float().cpu().numpy() for k,v in dict(dice=dice,iou=iou,precision=prec,recall=rec,f1=f1,tp=tp,tn=tn,fp=fp,fn=fn,pred_pixels=p.sum(1),target_pixels=y.sum(1)).items()} | {"total_pixels":np.full(len(dice),y.shape[1],dtype=np.float32)}

def summarize(a):
    a=np.asarray(a,dtype=float); return {"mean":float(a.mean()),"std":float(a.std()),"median":float(np.median(a)),"min":float(a.min()),"max":float(a.max())}

@torch.no_grad()
def evaluate(models,weights,loader,device,thr,amp,dtype,tta):
    rows=[]
    for b in tqdm(loader,desc=f"TTA {tta}"):
        x=b["image"].to(device,non_blocking=True); y=b["mask"].to(device,non_blocking=True)
        ps=None
        with torch.autocast(device_type="cuda",dtype=dtype,enabled=amp):
            for t in tta:
                xt=tr(x,t); pt=None
                for w,m in zip(weights,models):
                    pr=torch.sigmoid(m(xt)); pt=w*pr if pt is None else pt+w*pr
                pt=inv(pt,t); ps=pt if ps is None else ps+pt
            prob=ps/len(tta)
        mm=metrics(prob,y,thr)
        for i in range(x.shape[0]):
            r=float(mm["target_pixels"][i]/mm["total_pixels"][i])
            rows.append({"image_path":b["image_path"][i],"mask_path":b["mask_path"][i],
                         **{k:float(mm[k][i]) for k in ["dice","iou","precision","recall","f1","tp","tn","fp","fn","pred_pixels","target_pixels"]},
                         "total_pixels":float(mm["total_pixels"][i]),"target_ratio":r,
                         "pred_ratio":float(mm["pred_pixels"][i]/mm["total_pixels"][i]),"mask_size_class":mclass(r)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("data/raw/ai4boundaries/sentinel2"))
    ap.add_argument("--checkpoints",nargs="+",type=Path,required=True)
    ap.add_argument("--weights",required=True); ap.add_argument("--split",default="test")
    ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--threshold",type=float,default=.52)
    ap.add_argument("--tta-mode",default="flips",choices=["none","flips","rot","d4"])
    ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--num-workers",type=int,default=8)
    ap.add_argument("--amp",action="store_true"); ap.add_argument("--amp-dtype",default="bf16",choices=["bf16","fp16"])
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models=[]; cfgs=[]
    for p in a.checkpoints:
        m,c=load_model(p,device); models.append(m); cfgs.append(c)
    weights=parse_weights(a.weights,len(models))
    bands=cfgs[0].get("bands",["B2","B3","B4","B8","NDVI"]); tm=cfgs[0].get("time_mode","stack")
    size=int(cfgs[0].get("image_size",256))
    ds=DS(pairs(a.root,a.split),bands,tm,size); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    dtype=torch.bfloat16 if a.amp_dtype=="bf16" else torch.float16; tta=ttas(a.tta_mode)
    print("TTA config:", {"weights":weights,"threshold":a.threshold,"tta":tta,"n":len(ds)})
    df=evaluate(models,weights,loader,device,a.threshold,a.amp,dtype,tta)
    df.to_csv(a.output_dir/"per_sample_metrics.csv",index=False)
    summary={"threshold":a.threshold,"weights":weights,"tta_mode":a.tta_mode,"tta_list":tta,"num_samples":int(len(df))}
    for k in ["dice","iou","precision","recall","f1"]: summary[k]=summarize(df[k].values)
    (a.output_dir/"summary_metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    by=df.groupby("mask_size_class").agg(count=("dice","size"),dice_mean=("dice","mean"),dice_median=("dice","median"),iou_mean=("iou","mean"),precision_mean=("precision","mean"),recall_mean=("recall","mean"),f1_mean=("f1","mean"),pred_pixels_mean=("pred_pixels","mean"),target_pixels_mean=("target_pixels","mean"),target_ratio_mean=("target_ratio","mean"),pred_ratio_mean=("pred_ratio","mean")).reset_index()
    by.to_csv(a.output_dir/"summary_by_mask_size.csv",index=False)
    print("\nResumo final:")
    for k in ["dice","iou","precision","recall","f1"]: print(k,summary[k])
    print("\nPor tamanho de máscara:"); print(by.to_string(index=False))
    print("\nArquivos salvos em:",a.output_dir)
if __name__=="__main__": main()
