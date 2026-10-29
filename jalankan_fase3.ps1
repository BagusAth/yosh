# =============================================================================
# Fase 3 (Pakem v1.3.1): ablasi multi-seed EXP-0R/1/2/3 + prediksi per citra validasi bersih
#
#   Cek siap (tanpa training, 15-25 menit; termasuk prediksi per citra EXP-0 tiga seed):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123,2026 -CekSaja
#   Malam A (seed 42 dan 123, 8 run, sekitar 6,5-7 jam):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123 -Nama A
#   Malam B (seed 2026, 4 run, sekitar 3,5 jam):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 2026 -Nama B
#
# - Aman dijalankan ulang: run yang sudah selesai dan cocok checksum-nya dilewati (--skip-if-done),
#   prediksi yang sudah ada tidak diulang.
# - Run yang gagal TIDAK menghentikan run berikutnya; ringkasan di akhir menunjukkan mana yang perlu diulang.
# - Laptop dicegah tidur selama skrip berjalan (layar boleh mati). Tutup laptop tetap bisa membuatnya tidur.
# - Test set tidak disentuh.
# =============================================================================
param(
    [string]$Seeds = "42,123",
    [string]$Nama = "A",
    [switch]$CekSaja,
    [switch]$TanpaCommit
)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$tag = if ($CekSaja) { "cek" } else { "malam_$Nama" }
$stamp = Get-Date -Format yyyyMMdd_HHmm
$log = "logs_v2\fase3_${tag}_$stamp.log"
Start-Transcript -Path $log | Out-Null

$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"
$data = "$root/cassava-v7-runtime-mirror/data.yaml"
$manifest = "$root/stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv"
$w0 = "$root/yolo11n.pt"
$valImages = "$root/cassava-v7-runtime-mirror/valid/images"
$predDir = "$root/stage_v2_ablation/per_image"
$expNames = @{ 0 = "exp_0_baseline"; 4 = "exp_0r_uniform_replacement"; 1 = "exp_1_adaptive_sampling"; 2 = "exp_2_targeted_aug"; 3 = "exp_3_full_proposed" }
$expLabel = @{ 0 = "EXP-0"; 4 = "EXP-0R"; 1 = "EXP-1"; 2 = "EXP-2"; 3 = "EXP-3" }
$urutan = @(4, 1, 2, 3)
$hasil = New-Object System.Collections.ArrayList
# -Seeds diterima sebagai teks ("42,123") karena powershell -File tidak mengurai array
$SeedList = @(("$Seeds" -split "[,\s]+") | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
Write-Host "Seed: $($SeedList -join ', ')  |  Mode: $tag"

# Cegah laptop tidur selama skrip berjalan (tidak mengubah pengaturan Windows; berlaku hanya selama proses ini hidup)
try {
    Add-Type -Name Power -Namespace Skripsi -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [Skripsi.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { Write-Host "Catatan: gagal mencegah sleep otomatis; pastikan mode tidur dimatikan." -ForegroundColor Yellow }

function Add-Hasil([string]$tahap, [string]$run, [string]$status, [double]$menit, [string]$map = "") {
    [void]$hasil.Add([pscustomobject]@{ Tahap = $tahap; Run = $run; Status = $status; Menit = $menit; "mAP50-95" = $map })
}
function Get-OutDir([int]$s) { return "$root/stage_v2_ablation/seed_$s" }
function Get-Weights([int]$s, [int]$e) {
    if ($e -eq 0 -and $s -eq 42) { return "$root/stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt" }
    return "$(Get-OutDir $s)/$($expNames[$e])/weights/best.pt"
}
function Get-Map([int]$s, [int]$e) {
    $dir = if ($e -eq 0 -and $s -eq 42) { "$root/stage_8_ablation_clean_20261020/exp_0_baseline" } else { "$(Get-OutDir $s)/$($expNames[$e])" }
    $f = "$dir/metrics_summary.json"
    if (-not (Test-Path $f)) { return "" }
    try { $j = Get-Content $f -Raw | ConvertFrom-Json; return ("{0:N2}%" -f ($j.metrics_validation.mAP50_95 * 100)) } catch { return "" }
}
function Get-TrainArgs([int]$s, [int]$e) {
    return @("stage_8_4_run_ablation_experiments.py", "--exp", "$e", "--seed", "$s",
        "--data-yaml", $data, "--manifest-path", $manifest,
        "--pretrained-weights", $w0, "--output-dir", (Get-OutDir $s), "--skip-if-done")
}
function Invoke-Predict([int]$s, [int]$e) {
    $run = "$($expLabel[$e]) seed $s"
    $out = "$predDir/preds_seed${s}_$($expNames[$e]).csv"
    if (Test-Path $out) { Add-Hasil "prediksi" $run "sudah ada" 0; return }
    New-Item -ItemType Directory -Force -Path $predDir | Out-Null
    $old = "$root/stage_v2_eleak_sim/per_image/preds_group_on_cleanval.csv"
    if ($e -eq 0 -and $s -eq 42 -and (Test-Path $old)) {
        Copy-Item $old $out; Add-Hasil "prediksi" $run "disalin dari E-LEAK" 0; return
    }
    $w = Get-Weights $s $e
    if (-not (Test-Path $w)) { Add-Hasil "prediksi" $run "dilewati: bobot tidak ada" 0; return }
    Write-Host ""
    Write-Host "=== Prediksi per citra $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    & $py predict_eleak_per_image.py --weights $w --source $valImages --out-preds "$out.tmp" --out-gt "$predDir/gt_cleanval.csv"
    $code = $LASTEXITCODE
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0 -and (Test-Path "$out.tmp")) { Move-Item -Force "$out.tmp" $out; Add-Hasil "prediksi" $run "OK" $menit }
    else { Add-Hasil "prediksi" $run "GAGAL (exit $code)" $menit }
}
function Show-Ringkasan {
    Write-Host ""
    Write-Host "=================== RINGKASAN ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===================" -ForegroundColor Cyan
    $hasil | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    $hasil | Export-Csv -NoTypeInformation -Encoding UTF8 -Path "logs_v2\fase3_${tag}_${stamp}_ringkasan.csv"
    $gagal = @($hasil | Where-Object { $_.Status -like "GAGAL*" })
    if ($gagal.Count -gt 0) {
        Write-Host "$($gagal.Count) langkah GAGAL. Jalankan ulang file .bat yang sama; yang sudah selesai akan dilewati." -ForegroundColor Red
        Write-Host "Kirim log ini bila gagal lagi: $log" -ForegroundColor Red
    } else {
        Write-Host "Semua langkah selesai tanpa kegagalan. Log: $log" -ForegroundColor Green
    }
}

# --- 0. Commit skrip baru (sekali; diabaikan bila git tidak tersedia) ---------
if (-not $TanpaCommit) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $files = @("predict_eleak_per_image.py", "analyze_eleak_per_image.py", "analyze_eleak_did.py",
            "jalankan_prediksi_eleak.ps1", "jalankan_prediksi_eleak.bat", "jalankan_fase3.ps1", "cek_siap_fase3.bat",
            "jalankan_malam_A.bat", "jalankan_malam_B.bat", "build_label_audit.py", "cek_struktur_kelas.py",
            "audit_label_150/README_AUDIT.md", "MALAM_INI.md") | Where-Object { Test-Path $_ }
        git add -- $files 2>$null
        git diff --cached --quiet -- $files
        if ($LASTEXITCODE -ne 0) {
            git commit -m "Fase 2 selesai, persiapan Fase 3 dan audit label" `
                -m "Uji efek kembaran E-LEAK (selisih-dalam-selisih), runner ablasi multi-seed dengan prediksi per citra, pembangun lembar audit label 150 citra, cek struktur kelas." `
                -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_015oXrwUQZdycPnPqBtm95qT" -- $files
            if ($LASTEXITCODE -eq 0) { Write-Host "Skrip baru di-commit ke branch $(git branch --show-current)." -ForegroundColor Green }
            else { Write-Host "Commit gagal (tidak menghentikan proses)." -ForegroundColor Yellow }
        }
    } else { Write-Host "git tidak ditemukan di PATH; commit dilewati." -ForegroundColor Yellow }
}

# --- 1. Preflight -------------------------------------------------------------
Write-Host ""
Write-Host "=== Preflight: GPU dan dry-run konfigurasi ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA tidak tersedia'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { Add-Hasil "preflight" "GPU" "GAGAL (CUDA tidak tersedia)" 0 } else { Add-Hasil "preflight" "GPU" "OK" 0 }
foreach ($e in $urutan) {
    $s = $SeedList[0]
    $t0 = Get-Date
    & $py @((Get-TrainArgs $s $e) + @("--dry-run"))
    $code = $LASTEXITCODE
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) { Add-Hasil "preflight" "$($expLabel[$e]) (dry-run)" "GAGAL (exit $code)" $menit }
    else { Add-Hasil "preflight" "$($expLabel[$e]) (dry-run)" "OK" $menit }
}
$preGagal = @($hasil | Where-Object { $_.Tahap -eq "preflight" -and $_.Status -like "GAGAL*" })
if ($preGagal.Count -gt 0) {
    Write-Host "Preflight gagal; training TIDAK dimulai." -ForegroundColor Red
    Show-Ringkasan; Stop-Transcript | Out-Null; exit 1
}
if ($CekSaja) {
    # Uji jalur prediksi sekaligus membuat prediksi EXP-0 yang dibutuhkan Fase 5 (sekitar 3-5 menit per model)
    foreach ($s in $SeedList) { Invoke-Predict $s 0 }
    Write-Host "Preflight lolos. Siap untuk malam A dan B." -ForegroundColor Green
    Show-Ringkasan; Stop-Transcript | Out-Null; exit 0
}

# --- 2. Training + prediksi per citra -----------------------------------------
foreach ($s in $SeedList) {
    Invoke-Predict $s 0
    foreach ($e in $urutan) {
        $run = "$($expLabel[$e]) seed $s"
        Write-Host ""
        Write-Host "=== Training $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        & $py @(Get-TrainArgs $s $e)
        $code = $LASTEXITCODE
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -ne 0) { Add-Hasil "training" $run "GAGAL (exit $code)" $menit; continue }
        Add-Hasil "training" $run "OK" $menit (Get-Map $s $e)
        Invoke-Predict $s $e
    }
}

try { [Skripsi.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch {}
Show-Ringkasan
Stop-Transcript | Out-Null
