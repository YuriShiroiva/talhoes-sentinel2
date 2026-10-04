# Revisão sênior do deep learning — projeto field boundary

Revisão técnica do pipeline (Sentinel-2 e RGB/orthos), com achados priorizados por impacto × esforço.
Foco: o que mais move a agulha para a tese e para uma eventual produção.

> Convenção de prioridade: **P0** = crítico/feito agora · **P1** = alto impacto, baixo esforço ·
> **P2** = alto impacto, esforço médio · **P3** = pesquisa/futuro.

---

## 1. Dados

**1.1 [P1] Resolução (GSD) inconsistente no dataset RGB.**
Os tiles foram baixados com zoom adaptativo (12–18 no dataset atual), então um "talhão" tem
tamanho em pixels muito diferente de um tile para outro. CNN de segmentação aprende em escala de
pixel — escala inconsistente é ruído. *Correção:* fixar a resolução de chão (ex.: sempre zoom 16 ≈
2,4 m/px) recortando janelas de pixel fixas, em vez de deixar o zoom variar com a bbox.

**1.2 [P1] Quase só exemplos positivos (poucos negativos reais).**
Os tiles são centrados em talhões; só ~25/600 são quase vazios. O modelo tende a "ver talhão em todo
lugar" e gerar falsos positivos em estrada/mata/cidade no print do cliente. *Correção:* amostrar
tiles negativos (bbox longe de qualquer talhão) — ex.: 15–25% do dataset sem talhão.

**1.3 [P1] Split treino/val é por tile aleatório → autocorrelação espacial.**
Tiles vizinhos podem cair em train e val. Para métrica defensável na tese, usar **split espacial**
(separar uma faixa de longitude/latitude inteira para validação). Já constatamos que 11/30 casos de
teste tocavam tiles de treino — reportar isso e preferir áreas sem overlap.

**1.4 [P2] Descasamento temporal imagem × rótulo (ruído de label).**
Os polígonos foram desenhados numa data; o Esri World Imagery é mosaico de datas variadas. Um talhão
pode aparecer arado/diferente na imagem atual. É ruído de rótulo real — vale citar como limitação e,
se possível, filtrar tiles com baixa concordância visual.

**1.5 [ok] Duplicatas tratadas.** `fields_yuri.csv` tinha ~15,6% de geometrias duplicadas; o pipeline
deduplica por WKB. Mantém contagens e métricas corretas.

---

## 2. Rótulos / alvos

**2.1 [P0 — feito] Segmentação semântica funde talhões colados.**
O modelo v1 (área binária) une talhões que se tocam num polígono só. *Corrigido* com o modelo
boundary-aware (área + borda) + watershed para separar instâncias (scripts 30 / 53 / 54).

**2.2 [P1] Largura da borda em pixels, não em metros.**
A borda é gerada com largura fixa em pixels; com GSD variável isso vira largura variável em metros.
Amarrar a largura da borda a metros (depende de 1.1).

**2.3 [P3] Alvo de borda binária é frágil.**
Bordas finas são difíceis (dice baixo). Alvos mais ricos — mapa de distância à borda / energia de
watershed (estilo HoVer-Net) — dão separação mais limpa. Trabalho futuro.

---

## 3. Modelo e treino

**3.1 [restrição de hardware] GTX 1650 obriga fp32.**
fp16 gera NaN (sem tensor cores) e bf16 não é suportado (Turing). Treinamos em fp32, batch 2, 512px.
*Para escalar:* rodar num GPU Ampere+ (Colab/A100) com bf16 → batch maior, resnet50/Unet++, mais épocas.

**3.2 [P1] Batch 2 desestabiliza BatchNorm.**
Estatísticas de BN com batch 2 são ruidosas. *Correções (baratas):* (a) **gradient accumulation**
(acumular 4–8 passos → batch efetivo 8–16); (b) trocar BN→GroupNorm no decoder. Grad-accum é o ganho fácil.

**3.3 [P1] Sem agregação no inferência RGB.**
O track Sentinel-2 usa ensemble + TTA (bom). No RGB falta. *Correção:* TTA por flips (h/v/hv) na
inferência — ganho barato e estável (adicionado no script 54 via `--tta`).

**3.4 [P2] Sem scheduler/early-stopping no script 24.**
O script 30 já tem CosineAnnealingLR (bom). Levar scheduler + early stopping para o 24 também.

**3.5 [ok] AdamW, seeds, grad-clip, checkpoint do melhor por score** — boas práticas presentes.

---

## 4. Loss e desbalanceamento

**4.1 [ok] Tversky (β>α) + pos_weight dinâmico na borda.** Boas escolhas para objetos pequenos e
borda esparsa. `losses.py` já tem Tversky/FocalTversky.

**4.2 [P2] Loss ponderada por separação.**
Para reforçar a divisão entre talhões colados, ponderar mais os pixels de borda *compartilhada*
(weight map estilo U-Net original) ou usar Lovász (otimiza IoU direto).

---

## 5. Avaliação / métricas

**5.1 [P1] ✅ Métricas de OBJETO (F1@IoU≥0.5) implementadas — revelaram o que o pixel-Dice esconde.**
`scripts/55_object_level_metrics.py` casa instâncias preditas↔gabarito por IoU (guloso) e reporta
precision/recall/F1@0.5. **Resultado-chave:** pixel-Dice ~0.79 mas **F1 de objeto só ~0.22–0.27** — o
pixel-Dice superestima muito a qualidade real de delineamento talhão-a-talhão. Comparação (val completo):
| método | dados | F1@0.5 | F1@0.5 holdout | precisão | recall |
|---|---|---|---|---|---|
| borda + watershed | 875 | 0.217 | 0.213 | 0.234 | 0.202 |
| energia (Deep Watershed, §2.3) Unet | 875 | 0.272 | 0.278 | 0.324 | 0.234 |
| energia Unet++ | 875 | 0.278 | 0.292 | 0.297 | 0.262 |
| **energia Unet++ + 3× dados** | **3125** | 0.284 | **0.298** | 0.325 | 0.253 |
O **método** (borda→energia, +25%) e a **arquitetura** (Unet→Unet++) moveram a agulha. Já **3× mais dados**
deu só +0.006 (empate) — a precisão subiu mas o **recall ficou ~0.28**. Conclusão: com resnet34, **dados
sozinhos dão retorno decrescente**; o gargalo é capacidade do modelo + dificuldade da tarefa (talhões
pequenos, rótulo incompleto). Próximo lever: **backbone maior** (resnet50/efficientnet) e/ou rótulos completos.

**5.2 [P1] ✅ Holdout espacial implementado.** O script 55 marca tiles val a ≥2 km de qualquer tile de
treino. Os números de holdout (~0.28) batem com o val completo (~0.27) → sem vantagem de vazamento, generaliza.

**5.3 [P2] Threshold fixo em 0,5.** Varrer o threshold no val e reportar curva PR / melhor F1.

---

## 6. Inferência e pós-processamento

**6.1 [ok→portar] Pós-processamento bom no track Sentinel-2** (remoção de componentes pequenos,
polygonize, simplify). Espelhado no RGB (scripts 52/54).

**6.2 [P2] Janela deslizante para prints grandes.**
Hoje a imagem inteira é reduzida a 512px → talhões pequenos somem em prints grandes. Inferir por
janelas com sobreposição e costurar (blend) preserva detalhe.

**6.3 [P3] Geo aproximado (4326 linear sobre tiles Mercator).**
Alinhamento imagem/máscara é exato (mesma transform), mas a georreferência é aproximada. Para
precisão métrica, reprojetar os tiles para UTM.

---

## 7. Engenharia / reprodutibilidade

- **Ambiente único:** o `.venv` não tem shapely e tem torch CPU; o `python` do sistema tem tudo +
  CUDA. Fixar um ambiente e um `requirements` para reprodutibilidade.
- **Proveniência dos dados:** imagens Esri World Imagery são uso acadêmico/teste; documentar e não
  redistribuir as imagens (só os scripts que as geram).
- **Curvas de treino:** já salvas em CSV; gerar um plot rápido por treino ajuda a defesa.

---

## Plano de ação priorizado

| Prioridade | Ação | Esforço | Impacto |
|---|---|---|---|
| **P0** ✅ | Boundary-aware + watershed (instâncias) | feito | Resolve a fusão de talhões |
| **P1** ✅ | GSD fixo no dataset (1.1) | feito | dataset v3 zoom único 16 |
| **P1** ✅ | Tiles negativos (1.2) | feito | **falso positivo 48%→26%** (v3) |
| **P1** ✅ | TTA na inferência (3.3) | feito | script 54 `--tta` |
| **P1** ✅ | Métricas de objeto + holdout espacial (5.1/5.2) | feito | script 55; **F1 obj 0.22→0.27** |
| **P1** ✅ | Janela deslizante (6.2) | feito | script 54 `--fixed-zoom` |
| **P2** ✅ | Alvo de energia / Deep Watershed (2.3) | feito | scripts 53 `--energy`/56; +25% F1 obj |
| **P1** ✅ | Aug fotométrica + multi-escala (RGB) | feito | script 56 |
| **P1** | Gradient accumulation (3.2) | baixo | Médio |
| **P1** | Métricas de objeto + holdout espacial (5.1/5.2) | médio | Alto (credibilidade da tese) |
| **P2** | GPU bf16 + resnet50/Unet++ + 2k tiles | médio | Alto |
| **P2** | Janela deslizante (6.2); threshold tunado (5.3) | médio | Médio |
| **P3** | Alvo de energia/distância (2.3); loss de separação (4.2) | alto | Alto (estado da arte) |

**Resumo:** a base está sólida (arquiteturas SMP, losses adequadas, ensemble/TTA no Sentinel-2,
pós-processamento). Os maiores ganhos rápidos para a tese: **GSD fixo**, **negativos**, **TTA**,
**métricas de objeto em holdout espacial**. O salto de qualidade maior vem de treinar num GPU com
bf16 (batch maior, backbone maior, mais dados).
