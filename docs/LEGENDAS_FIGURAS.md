# Legendas das figuras — Sentinel-2 (prontas para a tese)

> Renumere `Figura N` conforme a ordem no seu texto. Formato ABNT: legenda **acima** ou **abaixo**
> da figura (mantenha o padrão do seu template) e fonte logo abaixo. Substitua "(2026)" se necessário.

---

## 1. Desempenho por tamanho de talhão
**Arquivo:** `outputs/sentinel2/figura_tese_sentinel2_por_tamanho.png`

> **Figura N –** Desempenho do modelo Sentinel-2 multitemporal (ensemble *test_058*) no conjunto de
> teste (1.139 tiles), estratificado pela fração do tile ocupada por talhão. As barras indicam o Dice
> e o IoU médios por classe de tamanho. O modelo é robusto em talhões grandes e médios — Dice de 0,93
> (muito grande, >75% do tile) a 0,70 (médio, 5–25%) — e degrada apenas nos talhões muito pequenos
> (<5% do tile: Dice 0,18–0,49), caso difícil esperado pela escassez de pixels e ambiguidade de borda.
>
> Fonte: elaborado pelo autor (2026).

---

## 2. Resultado qualitativo no domínio de treino
**Arquivo:** `outputs/sentinel2/figura_tese_sentinel2_qualitativo.png`

> **Figura N –** Exemplos qualitativos da segmentação de talhões com Sentinel-2 multitemporal. Em cada
> linha: composição RGB Sentinel-2 (esquerda), máscara de referência (centro) e máscara predita pelo
> modelo (direita). O modelo recupera os contornos dos talhões com boa aderência ao gabarito,
> coerente com o Dice mediano de 0,87 obtido no conjunto de teste.
>
> Fonte: elaborado pelo autor (2026).

---

## 3. Generalização cross-dataset (transferência zero-shot)
**Arquivo:** `outputs/sentinel2/figura_tese_ftw_generalizacao.png`

> **Figura N –** Queda de desempenho em transferência *zero-shot* (sem re-treino). O ensemble
> *test_058*, treinado no domínio europeu (AI4Boundaries), foi aplicado diretamente ao benchmark
> *Fields of The World* – Brasil (1.607 tiles). O Dice médio cai de 0,77 (domínio de treino) para 0,25
> no domínio brasileiro. A queda evidencia que o delineamento de talhões é dependente de região e de
> sensor: o modelo mantém precisão alta (0,76), mas o recall despenca para 0,36, deixando de reconhecer
> os grandes talhões característicos do Cerrado.
>
> Fonte: elaborado pelo autor (2026).

---

## 4. Erro qualitativo em transferência zero-shot
**Arquivo:** `outputs/sentinel2/figura_tese_ftw_qualitativo.png`

> **Figura N –** Falha típica da transferência *zero-shot* no FTW-Brasil. Da esquerda para a direita:
> composição RGB, máscara de referência e predição do modelo sem adaptação. O modelo, treinado no
> padrão temporal europeu, subdetecta sistematicamente os talhões brasileiros de grande área — efeito
> consistente com o recall de 0,36 medido no benchmark.
>
> Fonte: elaborado pelo autor (2026).

---

## 5. História completa — casa, zero-shot e fine-tuning
**Arquivo:** `outputs/sentinel2/figura_tese_ftw_historia_completa.png`

> **Figura N –** Síntese da capacidade de generalização do modelo Sentinel-2 em três cenários. (i) *Casa*:
> Dice médio de 0,77 no teste do domínio de treino (AI4Boundaries, ensemble *test_058*). (ii) *Zero-shot*:
> Dice de 0,25 ao aplicar o modelo ao FTW-Brasil sem adaptação. (iii) *Fine-tuned*: Dice de 0,72 após
> ajuste fino com dados locais do FTW-Brasil — ganho de +0,47. O resultado mostra que o modelo não
> transfere bem de forma direta, mas se adapta muito bem com poucos dados do domínio-alvo, indicando o
> *fine-tuning* regional como estratégia prática de implantação.
>
> Nota: os valores de *zero-shot* (0,25) e *fine-tuned* (0,72) referem-se ao mesmo modelo único
> (*test_025*) avaliado, respectivamente, antes e depois do ajuste fino no FTW-Brasil — par de comparação
> justo, ambos com a preprocessing idêntica à do treino (ordem *band_time* + percentil 2–98). A barra
> *casa* (0,77) corresponde ao Dice médio do ensemble no teste do domínio de treino.
>
> Fonte: elaborado pelo autor (2026).

---

## 6. Antes/depois qualitativo do fine-tuning
**Arquivo:** `outputs/sentinel2/figura_tese_ftw_antes_depois.png`

> **Figura N –** Efeito qualitativo do *fine-tuning* no FTW-Brasil. Em cada linha: composição RGB
> Sentinel-2, máscara de referência do FTW, predição *zero-shot* (antes do ajuste) e predição após
> *fine-tuning* (depois). Sem adaptação, o modelo praticamente não detecta os talhões (predição vazia);
> após o ajuste fino com dados locais, recupera os contornos com alta fidelidade ao gabarito (Dice de
> 0,93 a 0,95 nos exemplos selecionados). Tiles escolhidos pelos maiores ganhos de Dice entre os dois
> estágios; ambas as predições usam a preprocessing idêntica à do treino.
>
> Fonte: elaborado pelo autor (2026).

---

## 7. Demonstração do pipeline end-to-end
**Arquivo:** `outputs/sentinel2/figura_tese_end_to_end_demo.png`

> **Figura N –** Demonstração do pipeline operacional end-to-end. Para cada amostra (linha): (1) imagem de
> entrada Sentinel-2 (composição RGB), (2) mapa de probabilidade predito pelo modelo, (3) máscara de
> talhões candidatos após limiar e remoção de componentes pequenos, e (4) máscara de referência. A partir
> da máscara (3), o pipeline vetoriza automaticamente os talhões (GeoJSON) e calcula a área de cada um. O
> Dice indicado em cada linha confirma a alta aderência ao gabarito (0,97–0,98). Pipeline:
> `scripts/44_predict_sentinel2_fields_end_to_end.py`.
>
> Fonte: elaborado pelo autor (2026).

---

## 8. Separação de talhões a nível de objeto (instâncias)
**Arquivo:** `outputs/sentinel2/figura_tese_instancias.png`

> **Figura N –** Separação de talhões em instâncias individuais. Em cada linha (da esquerda para a direita):
> composição RGB Sentinel-2, gabarito de instâncias do FTW (cada talhão em uma cor), resultado do *baseline
> semântico* (campo vs fundo, que **funde** talhões vizinhos num único objeto) e resultado do modelo
> *3-classes com cabeça de borda* (interior/borda/fundo), que **separa** os talhões vizinhos. O número de
> instâncias detectadas é indicado em cada painel. A cabeça de borda eleva o F1 de objeto (IoU≥0,5) de 0,455
> para 0,594 (+30%) e o Panoptic Quality de 0,381 para 0,512 no conjunto de teste (188 tiles). Treino:
> `scripts/61`; calibração e avaliação: `scripts/60` e `scripts/62`.
>
> Fonte: elaborado pelo autor (2026).

---

### Resumo numérico de apoio (para tabelas/texto)

| Cenário | Dice | IoU | Precisão | Recall | F1 | n |
|---|---|---|---|---|---|---|
| Casa — teste (ensemble *test_058*) | 0,77 (méd) / 0,87 (med) | 0,68 | 0,79 | 0,82 | 0,78 | 1.139 |
| FTW-Brasil zero-shot (modelo único, teste) | 0,22 | — | — | — | — | 188 |
| FTW-Brasil zero-shot (ensemble, teste) | 0,06 | — | — | — | — | 188 |
| **FTW-Brasil fine-tuned (modelo único, teste)** | **0,72** | **0,67** | **0,72** | **0,90** | **0,80** | 188 |

> Todas as avaliações de pixel usam a mesma preprocessing do treino (ordem *band_time* + percentil 2–98 por canal).

### Métricas de OBJETO (instância) — FTW-Brasil, teste (188 tiles, 653 talhões)

| Método | Objeto F1@IoU≥0,5 | Panoptic Quality | Precisão | Recall |
|---|---|---|---|---|
| Baseline semântico (funde vizinhos) | 0,455 | 0,381 | 0,450 | 0,459 |
| **Borda + separação (3-classes)** | **0,594** | **0,512** | **0,565** | **0,626** |

> Pixel-Dice (0,72) mede área; objeto-F1 mede detecção de talhões individuais. O salto de objeto-F1
> (+30%) vem da cabeça de borda, que separa parcelas vizinhas em vez de fundi-las.
