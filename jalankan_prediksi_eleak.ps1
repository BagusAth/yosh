# =============================================================================
# Prediksi per citra + analisis bootstrap E-LEAK (RQ1): sekitar 5 menit GPU + 5-10 menit CPU, tanpa training
#   cd D:\UNDIP\SKRIPSI
#   powershell -ExecutionPolicy Bypass -File .\jalankan_prediksi_eleak.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$log = "logs_v2\prediksi_eleak_$(Get-Date -Format yyyyMMdd_HHmm).log"
Start-Transcript -Path $log | Out-Null
$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"
$out = "$root/stage_v2_eleak_sim/per_image"

& $py predict_eleak_per_image.py --weights "$root/stage_v2_eleak_sim/exp_0_baseline/weights/best.pt" `
    --source "$root/eleak_sim_rq1/sim_val_all.txt" --out-preds "$out/preds_sim_on_simval.csv" --out-gt "$out/gt_simval.csv"
if ($LASTEXITCODE -ne 0) { Write-Host "GAGAL: prediksi model split acak" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

& $py predict_eleak_per_image.py --weights "$root/stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt" `
    --source "$root/cassava-v7-runtime-mirror/valid/images" --out-preds "$out/preds_group_on_cleanval.csv" --out-gt "$out/gt_cleanval.csv"
if ($LASTEXITCODE -ne 0) { Write-Host "GAGAL: prediksi model split per grup" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

& $py analyze_eleak_per_image.py prep --root $root
if ($LASTEXITCODE -ne 0) { Write-Host "GAGAL: analisis prep" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
foreach ($sd in 0, 1, 2, 3) {
    & $py analyze_eleak_per_image.py boot --root $root --iters 250 --seed $sd
    if ($LASTEXITCODE -ne 0) { Write-Host "GAGAL: bootstrap seed $sd" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
}
& $py analyze_eleak_per_image.py report --root $root

Write-Host "SELESAI. Laporan: $out/analysis/ELEAK_PER_IMAGE_REPORT.md" -ForegroundColor Green
Stop-Transcript | Out-Null
