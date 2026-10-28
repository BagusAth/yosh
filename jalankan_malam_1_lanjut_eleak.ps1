# =============================================================================
# Malam 1 (lanjutan): hanya langkah E-LEAK simulasi (RQ1) dari jalankan_malam_1.ps1.
# Uji regresi, manifest v2, dan EXP-0 seed 123 & 2026 sudah selesai pada run 28-10-2026.
# Jalankan dari PowerShell:
#   cd D:\UNDIP\SKRIPSI
#   powershell -ExecutionPolicy Bypass -File .\jalankan_malam_1_lanjut_eleak.ps1
# Perkiraan: sekitar 55 menit GPU. Berhenti otomatis bila ada langkah yang gagal.
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$log = "logs_v2\malam_1_lanjut_eleak_$(Get-Date -Format yyyyMMdd_HHmm).log"
Start-Transcript -Path $log | Out-Null
$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"

function Invoke-Step([string]$desc, [string[]]$argList) {
    Write-Host ""
    Write-Host "=== $desc ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    & $py @argList
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GAGAL: $desc (exit $LASTEXITCODE). Langkah berikutnya tidak dijalankan." -ForegroundColor Red
        Stop-Transcript | Out-Null
        exit $LASTEXITCODE
    }
}

# --- GPU: E-LEAK simulasi (RQ1 bagian A) ---------------------------------------
Invoke-Step "E-LEAK: latih baseline pada split acak per citra (GPU)" @(
    "stage_8_4_run_ablation_experiments.py", "--exp", "0", "--seed", "42",
    "--data-yaml", "$root/eleak_sim_rq1/data_eleak_sim.yaml",
    "--pretrained-weights", "$root/yolo11n.pt",
    "--output-dir", "$root/stage_v2_eleak_sim")

$evalOut = "$root/stage_v2_eleak_sim/eval_subsets"
Invoke-Step "E-LEAK: evaluasi model split acak" @(
    "eval_eleak_subsets.py", "--tag", "sim",
    "--weights", "$root/stage_v2_eleak_sim/exp_0_baseline/weights/best.pt",
    "--out", $evalOut, "--sets",
    "val_all=eleak_sim_rq1/data_eleak_sim.yaml",
    "val_leaked=eleak_sim_rq1/data_eleak_sim_val_leaked.yaml",
    "val_twinval=eleak_sim_rq1/data_eleak_sim_val_twinval.yaml",
    "val_single=eleak_sim_rq1/data_eleak_sim_val_single.yaml")
Invoke-Step "E-LEAK: evaluasi model split per grup (EXP-0 seed 42)" @(
    "eval_eleak_subsets.py", "--tag", "group",
    "--weights", "$root/stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt",
    "--out", $evalOut, "--sets",
    "val_all=cassava-v7-runtime-mirror/data.yaml",
    "val_twin=eleak_sim_rq1/data_clean_val_twin.yaml")
Invoke-Step "E-LEAK: laporan selisih" @("eval_eleak_subsets.py", "--report", "--out", $evalOut)

Write-Host ""
Write-Host "SELESAI. Log: $log" -ForegroundColor Green
Stop-Transcript | Out-Null
