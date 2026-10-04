# Resultados — detecção de talhões em imagem RGB

Documento-resumo pronto para a tese: problema, método, protocolo de avaliação, resultados e conclusões.

## 1. Problema

A plataforma web entrega ao usuário imagem **RGB** (basemap Esri World Imagery, 3 canais), mas os modelos
de detecção de talhões disponíveis foram treinados em **Sentinel-2** (5+ bandas). Não há modelo que aceite
RGB diretamente. Objetivo: **treinar um modelo que receba a RGB da plataforma e gere os talhões preditos**,
usando como gabarito os talhões reais cadastrados (`fields_yuri.csv`, 44.522 polígonos únicos após remover
~15,6% de duplicatas).

## 2. Método (pipeline)

1. **Dataset RGB+rótulo** (scripts 51/53): para cada região, baixa a imagem Esri (mesma fonte da plataforma)
   em **resolução fixa** (zoom 16, ~2,4 m/px) e rasteriza os polígonos como rótulo. Inclui ~20% de **tiles
   negativos** (sem talhão) para reduzir falsos positivos.
2. **Modelo área + energia (Deep Watershed)** (script 56): U-Net/Unet++ (encoder ResNet, ImageNet) com duas
   saídas — *área* do talhão (classificação) e *energia* = distância normalizada ao limite (regressão).
   A energia tem pico no centro de cada talhão.
3. **Separação por instâncias** (script 54): os **picos de energia** viram sementes (uma por talhão) e crescem
   por *watershed* preenchendo a área → talhões individuais, mesmo colados. TTA por flips e janela deslizante
   na resolução nativa para inferência em prints grandes.
4. **Aumento de dados específico para RGB**: flips/rotações + **fotométrico** (brilho/contraste/gamma/canal,
   robustez à variação do Esri) + **multi-escala**.

## 3. Protocolo de avaliação (o ponto crítico)

- **Métrica de objeto** `F1@IoU≥0.5` (script 55): casa cada talhão predito ao gabarito por IoU; reporta
  precisão, recall e F1. Mede delineamento **talhão-a-talhão**, não só acerto de pixel.
- **Holdout espacial**: só conta tiles de validação a **≥ 2 km** de qualquer tile de treino — elimina
  vazamento por autocorrelação espacial. Todos os modelos compartilham o mesmo conjunto de teste.

> **Por que isso importa:** o **pixel-Dice** do modelo é alto (~0,79–0,81 in-distribution), o que *parece*
> excelente. Mas a métrica de objeto revela que o delineamento individual é bem mais difícil. Reportar só
> pixel-Dice **superestima** a qualidade — a tese usa a métrica de objeto.

## 4. Resultados

Métrica de objeto **F1@IoU≥0.5 no holdout espacial** (mesmo conjunto de teste para todos):

| # | Modelo | Dados (tiles) | Precisão | Recall | **F1@0.5** |
|---|---|---|---|---|---|
| 1 | Borda binária + watershed | 875 | 0,216 | 0,210 | **0,213** |
| 2 | Energia (Deep Watershed), U-Net | 875 | 0,305 | 0,255 | **0,278** |
| 3 | Energia, Unet++ | 875 | 0,292 | 0,292 | **0,292** |
| 4 | Energia, Unet++ + 3× dados | 3.125 | 0,315 | 0,284 | **0,298** |

(Pixel-Dice in-distribution dos modelos de energia: ~0,79–0,81 — mostrado apenas para contraste.)

## 5. Análise e conclusões

- **O método foi o que mais moveu a agulha:** trocar a borda binária pela **energia (Deep Watershed)** subiu
  o F1 de objeto de **0,213 → 0,278** (+31%), sobretudo cortando super-segmentação.
- **A arquitetura ajudou:** Unet++ subiu para **0,292**.
- **Mais dados deu retorno decrescente:** triplicar os dados (875 → 3.125 tiles, de 17% para 50% de cobertura
  dos talhões) rendeu apenas **+0,006** (0,292 → 0,298), com a precisão subindo mas o recall estável (~0,28).
  Um experimento adicional com backbone maior (ResNet-50) **não melhorou** o score de treino — confirmando que
  o backbone também não é o limite.
- **Conclusão (testada com evidência):** o gargalo **não é quantidade de dados nem tamanho do backbone**, e
  sim a **dificuldade intrínseca da tarefa** em RGB nessa escala — talhões pequenos perto do limite de
  resolução, talhões colados e **rótulo incompleto** (`fields_yuri` é um registro parcial: parte dos "falsos
  positivos" são talhões reais não cadastrados).

## 6. Figuras (em `outputs/rgb_fields/`)

- `instances_v4/figura_tese_energia_v4.png` — painéis RGB | prob. área | energia | talhões (instâncias) | gabarito.
- `instances_colab/figura_tese_colab.png` — idem, modelo Unet++.
- `instances_prints/figura_cliente_print.png` — demonstração "cliente tira print → talhões preditos".
- `predictions_v1/figura_tese_contact_sheet.png` — baseline semântico (v1).

## 7. Limitações e trabalhos futuros

- **Recall (~0,28) é o limite atual**: o modelo perde/funde talhões pequenos.
- **Rótulo incompleto** penaliza a métrica (talhões reais não cadastrados contam como erro). Rótulos completos
  por região elevariam tanto o recall medido quanto o treino.
- **Restrições de hardware**: GTX 1650 obriga fp32 (fp16 dá NaN, bf16 não suportado) e batch pequeno; Colab/
  Kaggle grátis limitam horas de GPU. Treino em GPU forte (bf16, batch maior) permitiria backbones maiores.
- **Caminhos promissores**: rótulos completos por AOI; alvos de instância mais ricos; e validar contra a
  qualidade que o cliente percebe (não só IoU≥0,5).

---
*Métricas reproduzíveis com `scripts/55_object_level_metrics.py` (ver `docs/PIPELINE_RGB_TESE.md`).
Revisão técnica completa do deep learning em `docs/REVISAO_DEEP_LEARNING.md`.*
