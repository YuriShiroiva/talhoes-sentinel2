# Como criar testes manuais para treinar a IA

Este fluxo cria um caso de treino com duas partes separadas:

- `area`: a area que o cliente/usuario selecionou no mapa.
- `answers`: os talhoes corretos desenhados manualmente para essa area.

Isso e o formato correto para treinar ou validar a IA: a imagem vem da area selecionada, e os talhoes manuais sao a resposta esperada.

## 1. Subir backend e frontend

Backend:

```powershell
cd C:\Users\Pikadasgalaxias\Documents\field_boundary_dl_v061\field-boundary-dl
set PYTHONPATH=C:\Users\Pikadasgalaxias\Documents\field_boundary_dl_v061\field-boundary-dl\src
set FB_PROVIDER=esri
set FB_MAX_AREA_KM2=50
set FB_MANUAL_TEST_DIR=C:\tcc_sentinel2\fields\testes_manuais
python -m field_boundary.cli serve --checkpoint C:\tcc_sentinel2\outputs\sentinel2\train_smoke_fields_yuri_10m_bbox\best_sentinel2_unet.pt --host 127.0.0.1 --port 8000 --in-channels 5
```

Frontend:

```powershell
cd C:\Users\Pikadasgalaxias\Documents\field_boundary_dl_v061\field-boundary-dl\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Abrir:

```text
http://127.0.0.1:5173
```

## 2. Criar um caso manual

1. Busque a localidade ou navegue no mapa.
2. Clique em `Desenhar area`.
3. Clique duas vezes no mapa para definir o retangulo da area.
4. Em `Teste manual`, preencha `Nome da area/caso`.
5. Clique em `Desenhar talhao manual`.
6. Clique no mapa para marcar os vertices do talhao.
7. Clique em `Fechar talhao`.
8. Repita para todos os talhoes daquela area.
9. Clique em `Salvar treino`.

## 3. Onde os dados ficam salvos

Cada caso fica em:

```text
C:\tcc_sentinel2\fields\testes_manuais\<data>_<nome_do_caso>\
```

Arquivos gerados:

```text
area.geojson
answers.geojson
training_case.json
```

Significado:

- `area.geojson`: area selecionada no mapa. Esta e a entrada.
- `answers.geojson`: talhoes manuais desenhados. Esta e a resposta esperada.
- `training_case.json`: pacote completo com area, answers e metadados.

## 4. Separacao dos dados

Talhoes reais prontos:

```text
C:\tcc_sentinel2\fields\reais\fields_yuri.csv
```

Testes manuais criados no frontend:

```text
C:\tcc_sentinel2\fields\testes_manuais
```

Nao misture os dois sem controle. Os testes manuais devem ser usados primeiro para validar pequenas areas e depois podem entrar no pipeline de treino.

## 5. Validacao rapida

Backend:

```text
http://127.0.0.1:8000/health
```

Frontend via proxy:

```text
http://127.0.0.1:5173/api/health
```

Se os dois retornarem `status: ok`, o app esta pronto para salvar novos casos.

