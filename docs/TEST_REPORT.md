# Relatório de teste dos scripts do TCC

## Resultado geral

Os scripts enviados foram testados em modo seguro, sem modificar repositório real e sem inicializar Git no projeto do usuário.

## Testes executados

| Item | Teste | Resultado |
|---|---|---|
| `compute_bootstrap_ci.py` | `python -m py_compile` | OK |
| `baseline_ndvi_threshold.py` | `python -m py_compile` | OK |
| `setup_git_retroactive.sh` | `bash -n` | OK |
| Dependências principais | import de `numpy`, `pandas`, `rasterio`, `xarray`, `scipy`, `tqdm` | OK no ambiente de teste |
| `compute_bootstrap_ci.py` | Execução com CSV sintético | OK; gerou `bootstrap_ci.csv`, `bootstrap_ci_table.csv`, `bootstrap_ci.json` |
| `baseline_ndvi_threshold.py` | Execução com mini dataset Sentinel-2 sintético | OK; gerou `val_ranking.csv`, `test_per_sample.csv`, `final_summary.json` |
| `make_tcc_tables_figures.py` | Execução usando saídas sintéticas | OK; gerou tabelas `.csv/.md` e figuras `.png` |
| `setup_git_retroactive_safe.sh` | `bash -n` | OK |

## Observações importantes

1. O `setup_git_retroactive.sh` original é sintaticamente válido, mas pode quebrar em um projeto real se algum arquivo listado não existir, pois usa `set -e` e muitos `git add` diretos.
2. Foi criada uma versão mais segura chamada `setup_git_retroactive_safe.sh`, que usa `git add --ignore-unmatch`, evita mexer na configuração global do Git e não tenta recriar histórico se já houver commits.
3. O `baseline_ndvi_threshold.py` funcionou com a estrutura esperada:
   - `images/val/*.nc`
   - `images/test/*.nc`
   - `masks/val/*.tif`
   - `masks/test/*.tif`
   - opcionalmente `metadata/val.csv` e `metadata/test.csv` com coluna `image`.
4. O `compute_bootstrap_ci.py` exige um CSV por amostra contendo, idealmente, as colunas: `dice`, `iou`, `precision`, `recall`, `f1`, `target_ratio`.
5. Para gerar materiais do TCC, foi adicionado o script `make_tcc_tables_figures.py`, que monta tabelas e figuras a partir dos outputs de bootstrap, ranking e métricas por amostra.

## Comandos principais para rodar no projeto real

```bash
python scripts/baseline_ndvi_threshold.py \
  --root data/raw/ai4boundaries/sentinel2 \
  --output-dir outputs/sentinel2/baseline_ndvi \
  --fill-holes \
  --score-mode balanced
```

```bash
python scripts/compute_bootstrap_ci.py \
  --input-csv outputs/sentinel2/baseline_ndvi/test_per_sample.csv \
  --output-dir outputs/sentinel2/baseline_ndvi_bootstrap_ci
```

```bash
python scripts/make_tcc_tables_figures.py \
  --bootstrap "NDVI baseline=outputs/sentinel2/baseline_ndvi_bootstrap_ci/bootstrap_ci.csv" \
  --per-sample "NDVI baseline=outputs/sentinel2/baseline_ndvi/test_per_sample.csv" \
  --ranking "NDVI baseline=outputs/sentinel2/baseline_ndvi/val_ranking.csv" \
  --final-json "NDVI baseline=outputs/sentinel2/baseline_ndvi/final_summary.json" \
  --output-dir outputs/tcc_assets
```
