# Roteiro de defesa — Detecção de talhões com Sentinel-2

Documento único para a defesa: narrativa, números (todos verificados nos JSONs do projeto),
mapa das figuras, perguntas prováveis da banca com respostas, e como demonstrar com segurança.

---

## 1. Frase de elevador (a tese em 2 linhas)

> Um pipeline de aprendizado profundo que detecta **talhões agrícolas individuais** a partir de
> imagens **Sentinel-2 multitemporais**, com Dice de pixel **0,87 (mediano)** no domínio de treino,
> transferência regional via **fine-tuning** (0,25 → **0,72** no Brasil) e **separação de instâncias**
> que eleva o F1 de objeto em **+30%** — entregue em uma **plataforma web funcional**.

---

## 2. Roteiro (≈ 7 min)

1. **Problema (30s):** delinear talhões manualmente é caro e lento; precisamos automatizar a partir de
   satélite gratuito (Sentinel-2, 10 m).
2. **Por que não RGB de um instante (45s):** cor/textura de uma única data são dependentes da cena e não
   generalizam (testei e descartei). A informação **multitemporal** (6 datas) captura a fenologia da
   cultura — é o que separa talhão de não-talhão de forma robusta.
3. **Método (1,5 min):** entrada de **30 canais** (B2,B3,B4,B8,NDVI × 6 datas), **ensemble** de 6 redes
   (U-Net/DeepLabV3+/U-Net++ com ResNet-34/50), TTA por flips, pós-processamento. Pré-processamento:
   ordem *band_time* + normalização por percentil 2–98 por canal.
4. **Resultado no domínio (1 min):** Dice mediano **0,87** (teste, 1.139 tiles); forte em talhões
   grandes/médios, cai só nos muito pequenos (<5% do tile). [Fig. por tamanho + qualitativa]
5. **Generalização (1,5 min):** zero-shot no benchmark **Fields of The World – Brasil** cai para ~**0,25**
   (domínio/sensor diferente). **Fine-tuning** com dado local recupera para **0,72** (+0,47). Conclusão:
   delineamento é **region-specific**; a solução prática é fine-tuning regional, não treinar do zero.
   [Fig. história completa + antes/depois]
6. **De pixel para objeto (1 min):** Dice mede área; "detectar talhões" exige **instâncias**. O baseline
   semântico funde vizinhos (sub-segmentação em 118/188 tiles). Um modelo **3-classes (interior/borda/
   fundo)** separa os talhões: objeto-F1 **0,455 → 0,594 (+30%)**, PQ **0,381 → 0,512**. [Fig. instâncias]
7. **Entrega (30s):** tudo integrado em uma **plataforma web** — o usuário desenha a área no mapa e recebe
   os talhões separados (GeoJSON + área em ha). [Fig. pipeline na plataforma]

---

## 3. Números canônicos (verificados — pode citar com segurança)

| Avaliação | Métrica | Valor |
|---|---|---|
| **Domínio de treino** (AI4Boundaries, teste, 1.139 tiles) | Dice médio / mediano | **0,77 / 0,87** |
| | IoU / Precisão / Recall / F1 (médios) | 0,68 / 0,79 / 0,82 / 0,78 |
| **FTW-Brasil zero-shot** (test, modelo único) | Dice | **~0,25** |
| **FTW-Brasil fine-tuned** (test, 188 tiles) | Dice / IoU / Prec / Recall / F1 | **0,72 / 0,67 / 0,72 / 0,90 / 0,80** |
| **Objeto — baseline semântico** (FTW test, 653 talhões) | F1@IoU≥0,5 / PQ | 0,455 / 0,381 |
| **Objeto — borda+separação** (FTW test) | F1@IoU≥0,5 / PQ / Prec / Recall | **0,594 / 0,512 / 0,565 / 0,626** |

Fontes: `outputs/sentinel2/test_058.../min_area_90/summary_metrics.json`,
`ftw_brazil_finetune/summary_finetune.json`, `ftw_object_metrics/summary_*.json`.

---

## 4. Mapa das figuras (o que cada uma prova)

| Figura | Prova |
|---|---|
| `figura_tese_sentinel2_por_tamanho.png` | Desempenho por tamanho de talhão (forte em grandes/médios) |
| `figura_tese_sentinel2_qualitativo.png` | RGB \| gabarito \| predito — boa aderência (domínio de treino) |
| `figura_tese_ftw_historia_completa.png` | **Casa 0,77 → zero-shot 0,25 → fine-tuned 0,72** (a narrativa) |
| `figura_tese_ftw_antes_depois.png` | Zero-shot quase vazio × fine-tuned recupera o talhão |
| `figura_tese_ftw_generalizacao.png` / `_qualitativo.png` | Queda em transferência zero-shot |
| `figura_tese_instancias.png` | Baseline funde vizinhos × borda separa instâncias |
| `figura_tese_end_to_end_demo.png` | Pipeline: imagem → probabilidade → talhões (vs gabarito) |
| `figura_tese_pipeline_instancias.png` | A plataforma entregando talhões separados (1 polígono/talhão) |

Legendas prontas (ABNT) em `docs/LEGENDAS_FIGURAS.md`. Resultados detalhados em `docs/RESULTADOS_SENTINEL2.md`.

---

## 5. Decisões de projeto (defensáveis)

- **RGB descartado:** segmentação de uma única imagem RGB é dependente da cena (cor/textura) e não
  generaliza; métodos clássicos de borda variam demais entre cenas. Multitemporal resolve.
- **Ensemble + TTA:** ganho de robustez sobre modelo único no domínio; **mas** no zero-shot fora do
  domínio o ensemble piora (modelos divergem) — por isso a comparação de fine-tuning usa **modelo único**.
- **Pré-processamento idêntico treino/inferência** (band_time + percentil): descobri em verificação que
  uma ordem/normalização errada zerava a predição — corrigido e blindado. Mostra rigor de engenharia.
- **Hardware modesto (GTX 1650):** treino em fp32 (fp16 dá NaN), `num_workers=0` no Windows — escolhas
  conscientes de engenharia, não acaso.

---

## 6. Perguntas prováveis da banca + respostas

**P: Dice de pixel 0,87 não é "detectar talhões" — é só área.**
R: Correto, e por isso medimos também **objeto-F1** e **Panoptic Quality**. O baseline semântico funde
vizinhos (F1 de objeto 0,455). A cabeça de borda 3-classes separa as instâncias e sobe para **0,594
(+30%)**. A tese vai de "segmentar área" para "detectar talhões individuais".

**P: 0,72 no Brasil é baixo comparado a 0,87 em casa.**
R: São domínios diferentes (Europa vs Cerrado, 6 datas vs 2 do FTW). O ponto não é o número absoluto e
sim a **mensagem**: transferência zero-shot cai (0,25), mas **fine-tuning com pouco dado local recupera**
para 0,72 (+0,47). É o resultado esperado e alinhado ao próprio paper do FTW.

**P: Então o modelo não generaliza?**
R: Delineamento de talhão é **região/sensor-específico** — isso é conhecido na literatura. A contribuição
prática é mostrar que **fine-tuning regional barato** (poucas épocas, um split local) resolve, em vez de
treinar do zero.

**P: Como garantem que não é overfitting / vazamento?**
R: Usamos o **split oficial do FTW** (train/val/test separados). Calibração do pós-processamento **no val**,
métricas finais **no test** (188 tiles nunca vistos). Nada do test entra em treino ou calibração.

**P: Por que não comparou com [método X / SAM / etc.]?**
R: O foco foi um pipeline **operacional e reprodutível** com dado gratuito e hardware modesto, validado em
benchmark público (FTW). Comparação com modelos fundacionais é trabalho futuro (documentado).

**P: Talhões pequenos vão mal.**
R: Sim, <5% do tile (Dice 0,18–0,49) — poucos pixels e borda ambígua, limite universal. Mitigável com
maior resolução ou mais exemplos pequenos. Os talhões grandes/médios (que importam operacionalmente)
vão bem (0,70–0,93).

**P: Qual a contribuição original?**
R: (1) Pipeline end-to-end Sentinel-2 → talhões, reprodutível e em plataforma web funcional; (2) estudo
honesto de **transferência + fine-tuning regional** num benchmark público; (3) **separação de instâncias**
(objeto) integrada à entrega, não só segmentação semântica.

**P: Por que ÷10000 não, e percentil sim?**
R: O treino usa normalização por **percentil 2–98 por canal** (adaptativa por imagem). Verificamos que a
preprocessing precisa ser idêntica na inferência, senão a predição degrada — está blindado no código.

---

## 7. Demonstração ao vivo — com plano B

**Recomendado (seguro):** mostrar as figuras prontas (`figura_tese_pipeline_instancias.png`,
`figura_tese_instancias.png`, `figura_tese_ftw_historia_completa.png`). Provam tudo sem depender de rede.

**Área testada que FUNCIONA bem (use na demo ao vivo):** região de **Rio Verde – GO** (soja, talhões
grandes retangulares). Busque "Rio Verde" no campo de busca, dê zoom e desenhe ~3–4 km. Verificado:
**18–82 talhões** detectados. Evite áreas de **curva de nível/terraços com talhões estreitos** (como a do
teste que deu 0) — são o caso mais difícil; o pipeline agora auto-afrouxa, mas o resultado é menos limpo.

> Importante: se já tinha o backend rodando, **reinicie-o** para pegar a correção de robustez (auto-ajuste
> quando não acha núcleos grandes + `seed_min` padrão menor). Áreas que antes davam 0 passam a recuperar
> talhões e mostram o aviso "Ajuste automático…".

**Demo ao vivo (se a internet estiver boa):**
1. Backend: `set FB_S2_INSTANCE_CKPT=C:\tcc_sentinel2\outputs\sentinel2\ftw_boundary_instances\best_boundary_3class.pt`
   depois `python -m field_boundary.cli serve --checkpoint <qualquer.pt>` (o S2 usa o env, não o `--checkpoint`).
2. Frontend: `npm run dev`. O provider já abre em **Sentinel-2 (talhões separados)**.
3. Desenhar uma área de **fazenda brasileira de ~3–4 km de lado** (áreas pequenas <2 km podem dar 0 talhões).
4. A barra de progresso mostra as etapas; leva **~30–60 s** (baixa 6 datas de satélite).

**Plano B se demorar/falhar:** já tenho a saída salva em `outputs/sentinel2/demo_instancias_ftw/` (GeoJSON +
painéis) — mostro o resultado pronto. Mensagem: "a busca de satélite é online; o resultado é este".

---

## 8. Checklist pré-defesa

- [ ] Slides com as figuras embutidas (não depender de abrir arquivos na hora).
- [ ] `docs/RESULTADOS_SENTINEL2.md` e este roteiro impressos/abertos.
- [ ] Se for demonstrar ao vivo: testar o `serve` + `npm run dev` **uma vez antes**, na mesma rede.
- [ ] Plano B (figuras + GeoJSON salvo) à mão.
- [ ] Decorar os 6 números-chave: **0,87 / 0,25 / 0,72 / +0,47 / 0,455→0,594 / +30%**.
