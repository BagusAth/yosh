# =============================================================================
# Diagnostik E-LEAK (RQ1): sekitar 3 menit GPU, hanya evaluasi (tanpa training)
#   cd D:\UNDIP\SKRIPSI
#   powershell -ExecutionPolicy Bypass -File .\jalankan_diagnosa_eleak.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$log = "logs_v2\diagnosa_eleak_$(Get-Date -Format yyyyMMdd_HHmm).log"
Start-Transcript -Path $log | Out-Null
$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"
$out = "$root/stage_v2_eleak_sim/eval_diag"
& $py eval_eleak_subsets.py --tag sim --weights "$root/stage_v2_eleak_sim/exp_0_baseline/weights/best.pt" --out $out --sets "diag_A=eleak_sim_rq1/data_diag_A.yaml" "diag_B=eleak_sim_rq1/data_diag_B.yaml"
& $py eval_eleak_subsets.py --tag group --weights "$root/stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt" --out $out --sets "diag_A=eleak_sim_rq1/data_diag_A.yaml"
Write-Host "SELESAI. Hasil: $out/eleak_subset_metrics.csv" -ForegroundColor Green
Stop-Transcript | Out-Null
