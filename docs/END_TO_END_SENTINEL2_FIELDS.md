# Pipeline end-to-end Sentinel-2 para talhoes candidatos

Este projeto agora tem um ponto de entrada operacional para usar imagem Sentinel-2 gratuita/local e gerar talhoes candidatos sem precisar de mascara real.

## Entrada aceita

- Arquivo `.nc` com variaveis `B2`, `B3`, `B4`, `B8` e opcionalmente `NDVI`.
- Arquivo `.tif/.tiff` multibanda.
- Pasta contendo arquivos `.nc/.tif`.
- CSV com coluna `image_path`.

O formato mais compativel com o treino atual e o das amostras do projeto:

```text
time=6, y=256, x=256
B2, B3, B4, B8, NDVI
CRS em spatial_ref
coordenadas x/y
```

## Comando recomendado

```bash
python scripts/44_predict_sentinel2_fields_end_to_end.py \
  --input sample_sentinel2_for_local_test/images \
  --output-dir outputs/sentinel2/end_to_end_fields \
  --threshold 0.49 \
  --min-area 90 \
  --tta-mode flips
```

Por padrao o script usa o ensemble operacional `test_058`, com os seis checkpoints ja existentes em `outputs/sentinel2`.

## Saidas

Para cada imagem o pipeline gera:

- `rasters/*_probability.tif`: probabilidade pixel a pixel.
- `rasters/*_candidate_mask.tif`: mascara binaria dos talhoes candidatos.
- `vectors/*_candidate_fields.geojson`: poligonos dos talhoes candidatos.
- `vectors/*_candidate_fields.csv`: atributos por poligono.
- `panels/*_panel.png`: painel de inspecao visual.
- `end_to_end_index.csv`: indice geral das imagens processadas.
- `all_candidate_fields.csv`: todos os talhoes candidatos em tabela unica.

## Observacao sobre imagem gratuita

O script nao faz download automatico. Ele espera a cena Sentinel-2 ja baixada/preparada como `.nc` ou `.tif`. Isso evita acoplar o TCC a credenciais de Copernicus, Sentinel Hub ou Google Earth Engine. A etapa operacional fica clara: baixar/preparar Sentinel-2 gratuito, rodar o script e inspecionar os talhoes candidatos gerados.
