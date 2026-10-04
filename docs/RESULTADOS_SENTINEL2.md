# Resultados — detecção de talhões com Sentinel-2

Pronto para a tese. Avaliação do ensemble operacional **test_058** no conjunto de **teste (1.139 tiles)** com máscaras reais.

## 1. Abordagem

- Entrada **Sentinel-2 multi-temporal**: bandas B2, B3, B4, B8 + NDVI × 6 datas = **30 canais**
  (vê a evolução temporal da cultura, não só um instante).
- **Ensemble de 6 modelos** (U-Net / DeepLabV3+ / U-Net++ com encoders ResNet-34/50), pesos
  `[0.22, 0.12, 0.21, 0.13, 0.13, 0.19]`, **TTA por flips** e pós-processamento (remoção de
  componentes < 90 px), limiar 0.49.
- Pipeline: `scripts/44_predict_sentinel2_fields_end_to_end.py`. Config: `outputs/sentinel2/final_sentinel2_operational_v1_config.json`.

## 2. Métricas gerais (teste, 1.139 tiles)

| Métrica | Média | Mediana |
|---|---|---|
| **Dice** | **0.769** | **0.868** |
| IoU | 0.676 | 0.766 |
| Precisão | 0.791 | 0.862 |
| Recall | 0.819 | 0.920 |
| F1 | 0.782 | 0.872 |

Dice mediano **0.87** — resultado forte, em linha com a literatura de field boundary (pixel IoU/Dice, padrão da AI4Boundaries).

## 3. Desempenho por tamanho de talhão

| Classe (fração do tile) | n | Dice médio | IoU médio |
|---|---|---|---|
| Muito grande (>75%) | 143 | **0.928** | 0.876 |
| Grande (25–75%) | 578 | **0.835** | 0.739 |
| Médio (5–25%) | 270 | **0.695** | 0.572 |
| Pequeno (1–5%) | 75 | 0.490 | 0.369 |
| Muito pequeno (<1%) | 29 | 0.177 | 0.134 |

O modelo é **forte nos talhões grandes e médios** (os que mais importam operacionalmente) e perde
desempenho só nos **muito pequenos** (<5% do tile) — o caso difícil universal (poucos pixels, borda
ambígua). Figura: `outputs/sentinel2/figura_tese_sentinel2_por_tamanho.png`.

## 4. Por que Sentinel-2 funciona (e RGB de um instante não)

A informação **multi-temporal** (6 datas) deixa o modelo aprender o padrão de crescimento/manejo da
cultura, separando talhão de não-talhão de forma robusta — o que uma RGB de um único instante (cor/
textura) não consegue. Por isso o Sentinel-2 generaliza e é a base recomendada do projeto.

## 5. Limitações (honestas)

- **Talhões muito pequenos** (<5% do tile): Dice baixo. Mitigável com tiles de maior resolução ou
  treino com mais exemplos pequenos.
- **Falsos positivos em tiles vazios**: ~13/44 tiles sem talhão tiveram predição espúria (limiar/gate
  pode reduzir).
- Métrica é **pixel-Dice** (padrão da área); uma avaliação por objeto (F1@IoU≥0.5) complementaria.

## 5b. Generalização cross-dataset — Fields of The World (FTW), Brasil

Avaliação no benchmark **Fields of The World – Brasil** (split oficial: 1289 train / 130 val / **188 test**;
Cerrado/café/eucalipto, só 2 janelas temporais). Toda a avaliação usa a **mesma preprocessing do treino**
(ordem `band_time` + percentil 2–98 por canal).

**Transferência zero-shot** (sem re-treino), no split de teste:

| Modelo | Dice | Mediana |
|---|---|---|
| Modelo único (test_025) | 0.22 | 0.12 |
| Ensemble (6 modelos) | 0.06 | 0.00 |

**Leitura:** zero-shot é ruim. O modelo treinado no domínio europeu (AI4Boundaries, 6 datas) **não reconhece
os grandes talhões brasileiros** — o FTW só tem 2 janelas temporais (replicadas), então o sinal temporal é
parcial. O ensemble cai ainda mais porque os 6 modelos divergem fora do domínio e a média fica abaixo do
limiar (zero-shot, single model é a comparação justa para o fine-tuning abaixo).

**Conclusão (importante e honesta):** delineamento de talhão é **região/sensor-específico**. Um modelo
treinado num domínio cai muito em transferência zero-shot para outro domínio + outro regime temporal —
resultado esperado e alinhado à literatura (o próprio paper do FTW mostra quedas em transferência).
Figuras: `outputs/sentinel2/figura_tese_ftw_generalizacao.png` e `figura_tese_ftw_qualitativo.png`
(o modelo perde os talhões grandes). Avaliação: `scripts/58_validate_on_ftw.py`.

### Fine-tuning no FTW-Brasil (a contraparte positiva)

Usando o **split oficial do FTW** (1289 train / 130 val / 188 test), fizemos **fine-tuning** do modelo
pré-treinado no train do FTW-Brasil (`scripts/59_finetune_on_ftw.py`, 18 épocas, lr 5e-5) e avaliamos no test:

| Estágio | Dice (test FTW-Brasil) | Recall |
|---|---|---|
| Zero-shot (sem adaptar) | 0.25 | baixo |
| **Fine-tuned (dado local)** | **0.72** | **0.90** |

**+0.47 de Dice.** O modelo passa a reconhecer os grandes talhões brasileiros.
Test final: Dice **0.716**, IoU **0.671**, precisão 0.723, recall **0.903**, F1 **0.803** (188 tiles).

**Narrativa completa (forte pra tese):** o modelo (a) **funciona** no domínio de casa (Dice 0.87 mediano),
(b) **não transfere de graça** para um domínio muito diferente (zero-shot ~0.25), mas (c) **adapta muito bem
com pouco dado local** (fine-tuned 0.72). Ou seja, field boundary é region-specific, e a solução prática é
fine-tuning regional — não treinar do zero. Figura: `outputs/sentinel2/figura_tese_ftw_historia_completa.png`.
Resultados: `outputs/sentinel2/ftw_brazil_finetune/summary_finetune.json`.

> Nota metodológica: toda a avaliação FTW (zero-shot e fine-tuning) usa a **mesma preprocessing do treino**
> (ordem `band_time` + percentil 2–98 por canal). Uma versão anterior usava preprocessing diferente
> (`time_band` + ÷10000), que subestimava o zero-shot (≈0.16); corrigido, o fine-tuning dá praticamente o
> mesmo Dice (0.72), confirmando robustez do resultado.

## 5c. Detecção de talhões a nível de OBJETO (instâncias)

Dice/IoU são métricas de **pixel** (área). Para "detectar talhões" de fato é preciso avaliar **instâncias**
(cada talhão como um objeto). Usando o gabarito de instância do FTW (`label_masks/instance`, 653 talhões nos
188 tiles de teste), medimos **F1@IoU≥0.5** e **Panoptic Quality (PQ)**.

**O problema (baseline semântico):** o modelo de campo prediz campo-vs-fundo, então talhões vizinhos se
fundem num polígono só — **sub-segmentação em 118 de 188 tiles**. O objeto-F1 fica fraco apesar do
pixel-Dice alto (0.72): SQ 0.84 (forma boa quando casa) mas RQ 0.46 (reconhece poucos).

**A solução:** modelo **3-classes (interior / borda / fundo)** (`scripts/61_train_boundary_instances_ftw.py`),
partindo do modelo de campo já fine-tunado (encoder+decoder transferidos, só a cabeça é nova). A borda
predita **separa** talhões vizinhos: instâncias = componentes conexos do interior + preenchimento por
vizinho-mais-próximo (Voronoi, sem dependência de skimage). Pós-processamento calibrado no val
(`scripts/62_tune_eval_instances.py`: `seed_min=250, field_thr=0.5, min_area=120`).

| Métrica (test, 188 tiles) | Baseline semântico | **Borda + separação** | Ganho |
|---|---|---|---|
| **Objeto F1@IoU≥0.5** | 0.455 | **0.594** | **+0.14 (+30%)** |
| Panoptic Quality (PQ) | 0.381 | **0.512** | +0.13 |
| Precisão (objeto) | 0.450 | 0.565 | +0.12 |
| Recall (objeto) | 0.459 | 0.626 | +0.17 |

**Leitura:** a cabeça de borda eleva o F1 de objeto em **30%**, com precisão **e** recall subindo juntos.
A plataforma passa a entregar **talhões individuais separados**, não um blob fundido. Figura:
`outputs/sentinel2/figura_tese_instancias.png`. Resultados: `outputs/sentinel2/ftw_object_metrics/`.

> Nota honesta: na FTW-Brasil os talhões são grandes e pouco adjacentes, então o ganho é moderado; em
> regiões de parcelas pequenas e coladas (ex.: AI4Boundaries/Europa, onde medimos **46 talhões → 16 blocos**
> no baseline, razão 0.35) o ganho da separação tende a ser ainda maior.

## 5d. Limitações e trabalhos futuros

Esta seção consolida, de forma honesta, os limites observados do método e aponta caminhos de evolução
diretamente sustentados pelos resultados.

**Limitações observadas**

1. **Subdetecção de solo exposto / pousio.** O modelo aprende a identificar talhões pela assinatura
   fenológica da cultura (vegetação ao longo do tempo). Em campos sem cultura ativa (solo exposto, pós-colheita,
   pousio), o modelo atribui, com alta confiança, a classe *fundo* — verificado no mapa de probabilidade, em que
   essas áreas aparecem com P(campo) ≈ 0,2–0,3. Não é uma questão de limiar: elevar o limiar apenas faz talhões
   vizinhos "incharem" sobre a área seca, sem criar a instância correta (não há semente de interior ali). A causa
   está no domínio de treino e no fato de o *fine-tuning* no FTW dispor de apenas 2 janelas temporais (replicadas),
   o que limita a modelagem temporal.
2. **Talhões muito pequenos** (< 5 % do tile): Dice baixo (0,18–0,49). Limitação universal de poucos pixels e
   borda ambígua, agravada pela resolução de 10 m do Sentinel-2.
3. **Dependência de região, sensor e estação.** A transferência *zero-shot* entre domínios cai fortemente
   (Dice ≈ 0,25 no FTW-Brasil); o desempenho só se recupera com *fine-tuning* regional (0,72). Ou seja, não há um
   modelo único e universal — o delineamento é específico de região/regime de cultivo.
4. **Cobertura de nuvem.** Mitigada pela composição por mediana com máscara de nuvem (SCL), mas áreas com nuvem
   persistente em toda a janela temporal ainda degradam a entrada.
5. **Métrica vs. objeto.** O Dice de pixel (0,87) mede área; a detecção de talhões individuais (objeto-F1 0,59)
   é mais exigente e foi tratada com a cabeça de borda, mas ainda há sub/super-segmentação residual.

**Trabalhos futuros**

1. **Re-treino multiestação com diversidade de cobertura.** Incluir explicitamente exemplos de solo exposto,
   pousio e diferentes culturas, e usar **janelas temporais reais** (não replicadas) — endereça diretamente a
   limitação (1) e fortalece a modelagem fenológica.
2. **Fusão com Sentinel-1 (SAR).** O radar penetra nuvem e responde à estrutura/umidade do solo, ajudando a
   delinear talhões em solo exposto e em dias nublados — complementar ao ótico.
3. **Fine-tuning regional contínuo via aprendizado ativo.** A própria plataforma já coleta pares
   *área + talhões-resposta* (anotação manual). Esses rótulos locais permitem re-fine-tunar por região/safra —
   estratégia validada pelo salto 0,25 → 0,72 deste trabalho.
4. **Backbones fundacionais de sensoriamento remoto** (p.ex. Prithvi, SatMAE, Clay) como codificador, buscando
   melhor generalização cross-domínio antes do *fine-tuning*.
5. **Pós-processamento geométrico orientado a objeto** (ajuste de círculos para pivôs centrais, *snapping* a
   bordas) para contornos ainda mais fiéis à geometria das parcelas.

## 6. Artefatos

- Métricas: `outputs/sentinel2/test_058.../min_area_90/summary_metrics.json`, `summary_by_mask_size.csv`, `per_sample_metrics.csv`
- Figura quantitativa (Dice/IoU por tamanho): `outputs/sentinel2/figura_tese_sentinel2_por_tamanho.png`
- **Figura qualitativa (RGB | gabarito | predito):** `outputs/sentinel2/figura_tese_sentinel2_qualitativo.png`
- Painéis adicionais: `outputs/sentinel2/end_to_end_fields/panels/*.png` e `outputs/sentinel2/s2_png_real_vs_pred_final/`
- **Cross-dataset FTW (zero-shot)**: `outputs/sentinel2/figura_tese_ftw_generalizacao.png`, `figura_tese_ftw_qualitativo.png`, métricas em `outputs/sentinel2/ftw_brazil_eval/`
- **Fine-tuning FTW-Brasil**: `figura_tese_ftw_historia_completa.png` (3 barras: casa 0.77 / zero-shot 0.16 / fine-tuned 0.72) e `figura_tese_ftw_antes_depois.png` (painel qualitativo: zero-shot em branco vs fine-tuned acertando). Checkpoint `outputs/sentinel2/ftw_brazil_finetune/best_ftw_finetuned.pt`, métricas em `summary_finetune.json`. Treino: `scripts/59_finetune_on_ftw.py`
- Config operacional: `outputs/sentinel2/final_sentinel2_operational_v1_config.json`
- **Separação de instâncias (objeto)**: `figura_tese_instancias.png`; modelo 3-classes em `outputs/sentinel2/ftw_boundary_instances/best_boundary_3class.pt`; métricas em `outputs/sentinel2/ftw_object_metrics/`. Scripts: `60` (métrica de objeto), `61` (treino borda), `62` (calibração+figura).
- **Demo end-to-end (semântico)**: `figura_tese_end_to_end_demo.png`. Rodar em qualquer pasta de `.nc`/`.tif`:
  `python scripts/44_predict_sentinel2_fields_end_to_end.py --input <pasta> --output-dir outputs/sentinel2/demo_end_to_end --tta-mode flips`
  → GeoJSON de talhões + CSV com área (ha) + painéis. Amostras prontas em `sample_sentinel2_for_local_test/images/`.
- **Pipeline de INSTÂNCIAS (talhões separados) — produção**: o `scripts/44` aceita `--instance-checkpoint`, que troca o
  ensemble semântico pelo modelo 3-classes e entrega **1 polígono por talhão** (separa vizinhos):
  `python scripts/44_predict_sentinel2_fields_end_to_end.py --input <pasta> --instance-checkpoint outputs/sentinel2/ftw_boundary_instances/best_boundary_3class.pt --min-area 120 --tta-mode flips`
  → GeoJSON com `candidate_id` por talhão + `area_px`/`area_ha`. Figura: `figura_tese_pipeline_instancias.png`. Robusto a CRS
  geográfico (EPSG:4326) e projetado (área em pixels sempre; em ha quando projetado).
