# Pipeline RGB para a tese (modelo treinado do zero)

## Por que RGB e não Sentinel-2

Os modelos Sentinel-2 do projeto exigem 5 bandas (B2,B3,B4,B8,NDVI) × 6 datas = `in_channels=30`.
A plataforma só fornece imagem RGB (Esri World Imagery). Como não existia checkpoint RGB, foi
treinado um modelo do zero usando exatamente a RGB que a plataforma usa, com os talhões reais
(`fields/reais/fields_yuri.csv`) como gabarito.

## Etapas

### 1. Construir o dataset RGB + máscara

```bash
python scripts/51_build_rgb_dataset_from_fields.py \
  --output-root data/prepared/rgb_fields_yuri \
  --max-tiles 600 --margin-frac 0.2 --max-dim 640 --val-fraction 0.15
```

- Agrupa talhões vizinhos, baixa o mosaico RGB Esri da área (tiles XYZ, sem chave) e rasteriza
  os polígonos como máscara binária alinhada à imagem.
- Saída no formato do script 24: `images/{train,val}/*.tif` + `masks/{train,val}/*.tif`.
- `--preview` salva PNG `RGB | overlay` por tile para conferir alinhamento.
- Deduplica geometrias (o CSV tem ~15,6% de talhões repetidos).

### 2. Treinar o modelo RGB

```bash
python scripts/24_train_orthos_baseline.py \
  --root data/prepared/rgb_fields_yuri \
  --output-dir outputs/rgb_fields/train_v1 \
  --image-size 512 --batch-size 2 --num-workers 4 --epochs 30
```

IMPORTANTE: na GTX 1650 treinar em **fp32** (sem `--amp`). fp16 gera NaN e bf16 não é suportado.
Salva `best_ortho_model.pt` (melhor val_dice).

### 3. Predizer talhões e gerar painéis da tese

```bash
python scripts/52_predict_rgb_fields_and_panels.py \
  --checkpoint outputs/rgb_fields/train_v1/best_ortho_model.pt \
  --cases-root fields/testes_simulados \
  --output-dir outputs/rgb_fields/predictions_v1
```

Aceita `--bbox`, `--case-dir` (uma pasta com `area.geojson` + `answers.geojson`) ou `--cases-root`.
Gera por área: `vectors/<nome>_pred_fields.geojson` (talhões preditos) e
`panels/<nome>_panel.png` (RGB | probabilidade | preditos | gabarito, com Dice/IoU).

## Resultados v1

| Conjunto | Dice | IoU |
|---|---|---|
| Validação holdout (86 tiles, métrica limpa) | 0.76 | 0.68 |
| Casos "print" — 19 áreas sem overlap com treino | 0.62 | 0.47 |
| Casos "print" — todos os 30 | 0.60 | 0.44 |

O subconjunto limpo pontua igual ao geral → o modelo generaliza (sem vantagem de vazamento).

## v2 — separação por instâncias (talhões individuais)

O modelo v1 é semântico e funde talhões colados. O v2 prediz **área + borda** e separa instâncias.

```bash
# 1. máscaras 2 bandas (área + borda) reaproveitando as imagens já baixadas
python scripts/53_make_boundary_masks_for_images.py --root data/prepared/rgb_fields_yuri --preview

# 2. treino boundary-aware (fp32, mesmo backbone do baseline)
python scripts/30_train_orthos_boundary_aware_multitask.py \
  --root data/prepared/rgb_fields_yuri --output-dir outputs/rgb_fields/boundary_v1 \
  --architecture unet --encoder-name resnet34 --batch-size 2 --num-workers 4 --epochs 30 --use-tversky-area

# 3a. avaliação JUSTA in-distribution (tiles val, gabarito completo) + painéis
python scripts/54_predict_rgb_instances_and_panels.py \
  --checkpoint outputs/rgb_fields/boundary_v1/best_boundary_model.pt \
  --images-root data/prepared/rgb_fields_yuri --split val --tta --min-seed-dist 20 \
  --output-dir outputs/rgb_fields/instances_val

# 3b. predição em uma área qualquer (print do cliente)
python scripts/54_predict_rgb_instances_and_panels.py \
  --checkpoint outputs/rgb_fields/boundary_v1/best_boundary_model.pt \
  --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT --tta --min-seed-dist 20
```

Resultados v2: `val_area_dice` 0.79–0.81 no treino; avaliação justa (val, gabarito completo) area Dice ~0.75
(mediana ~0.88; tiles cheios 0.95+), com talhões separados em instâncias.
Painéis: `outputs/rgb_fields/instances_val/` (figura `figura_tese_instancias.png`).

> ⚠️ NÃO usar `fields/testes_simulados` como benchmark quantitativo: o `answers` só tem os talhões do
> cluster, mas a imagem contém outros talhões reais que o modelo acerta e que viram "falso positivo".
> Use os tiles de validação (gabarito completo gerado pelo script 53).

## Treinar num GPU forte (Colab) — para subir o recall

A GTX 1650 limita a fp32/batch 2/resnet34. Para backbone maior + bf16 + batch maior:

```bash
python scripts/57_zip_dataset_for_colab.py          # gera rgb_dataset_v3e.zip
# (opcional) mais dados: 51 --fixed-zoom 16 --max-tiles 2500 + 53 --energy, depois reempacote
```

Suba o `.zip` ao Google Drive e abra `scripts/colab/train_energy_colab.ipynb` no Colab (Runtime GPU).
O notebook detecta bf16/fp16 automaticamente, treina area+energia (Unet++/resnet50, batch 16) e salva o
checkpoint no Drive. Traga `best_energy_model.pt` de volta e avalie com `scripts/55 --energy`.

## Limitações e próximos passos

- Modelo é **semântico** (talhão vs não-talhão): talhões vizinhos colados se fundem num polígono.
  Para separar talhões individuais, treinar com saída de **borda/instância** (ver scripts 30/31,
  boundary-aware multitask) ou aplicar watershed na probabilidade.
- Dataset focado em regiões com talhões; poucos negativos. Adicionar tiles sem talhão reduz falsos positivos.
- Esri World Imagery: uso acadêmico/teste; respeitar os termos do provedor.
