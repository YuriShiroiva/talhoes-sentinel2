$Project = "C:\Users\Pikadasgalaxias\Documents\field_boundary_dl_v061\field-boundary-dl"
$Checkpoint = "C:\tcc_sentinel2\outputs\sentinel2\train_smoke_fields_yuri_10m_bbox\best_sentinel2_unet.pt"

$env:PYTHONPATH = "$Project\src"
$env:FB_PROVIDER = "esri"
$env:FB_MAX_AREA_KM2 = "50"
$env:FB_MANUAL_TEST_DIR = "C:\tcc_sentinel2\fields\testes_manuais"

Set-Location $Project
python -m field_boundary.cli serve --checkpoint $Checkpoint --host 127.0.0.1 --port 8000 --in-channels 5

