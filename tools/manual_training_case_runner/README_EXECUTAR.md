# Pacote para rodar teste manual de area + talhoes

Este ZIP contem os arquivos alterados/criados para salvar casos reais de treino manual.

## O que ele salva

Cada caso fica em:

```text
C:\tcc_sentinel2\fields\testes_manuais\<data>_<nome_do_caso>\
```

Com:

```text
area.geojson
answers.geojson
training_case.json
```

`area.geojson` e a area selecionada.  
`answers.geojson` sao os talhoes reais desenhados manualmente.

## Rodar backend

```powershell
.\run_backend.ps1
```

Backend:

```text
http://127.0.0.1:8000
```

## Rodar frontend

```powershell
.\run_frontend.ps1
```

Frontend:

```text
http://127.0.0.1:5173
```

## Criar caso por codigo

Cria apenas area real, sem inventar talhoes:

```powershell
.\create_empty_case_example.ps1
```

Ou use direto:

```powershell
python C:\tcc_sentinel2\scripts\49_create_manual_training_case.py --case-name minha_area_real --bbox -46.57 -20.67 -46.54 -20.64
```

Com talhoes reais ja desenhados em GeoJSON:

```powershell
python C:\tcc_sentinel2\scripts\49_create_manual_training_case.py --case-name minha_area_real --bbox -46.57 -20.67 -46.54 -20.64 --answers-geojson C:\caminho\talhoes_desenhados.geojson
```

## Aplicar arquivos do pacote

Os arquivos dentro de `frontend/` e `backend/` sao copias dos arquivos alterados no projeto:

```text
C:\Users\Pikadasgalaxias\Documents\field_boundary_dl_v061\field-boundary-dl
```

No seu ambiente atual eles ja foram aplicados. O ZIP serve para backup/transporte.

