# =============================================================================
# Fase 3 (Pakem v1.3.1): ablasi multi-seed EXP-0R/1/2/3 + prediksi per citra validasi bersih
#
#   Cek siap (tanpa training, 15-25 menit; termasuk prediksi per citra EXP-0 tiga seed):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123,2026 -CekSaja
#   Malam A (seed 42 dan 123, 8 run, sekitar 7-7,5 jam; -UlangELeak = prediksi ulang E-LEAK dengan NMS seperti val):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123 -Nama A -UlangELeak
#   Malam B (seed 2026, 4 run, sekitar 3 jam):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 2026 -Nama B
#   Malam B lengkap (30 Okt 2026; sekitar 6 jam): uji baji hitam + 4 run seed 2026 + CONV
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 2026 -Nama B -UjiBaji -Conv
#
# -UjiBaji : uji jalan pintas baji hitam (uji_baji_hitam.py), inferensi saja pada validasi bersih dengan
#            EXP-0 tiga seed, sekitar 10-15 menit. Dijalankan sebelum training. Keluaran: stage_v2_baji\
# -Conv    : Fase 4, uji konvergensi EXP-0 dan EXP-1, seed 42, 60 epoch (resep lain sama), sekitar 3 jam.
#            Dijalankan sesudah training Fase 3. Keluaran: stage_v2_conv60\
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
    [switch]$TanpaCommit,
    [switch]$UlangELeak,
    [switch]$UjiBaji,
    [switch]$Conv
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
# Fase 4 (CONV): resep sama dengan Fase 3 kecuali jumlah epoch
$convEpochs = 60
$convSeed = 42
$convExps = @(0, 1)
$convRoot = "$root/stage_v2_conv60"
$bajiDir = "$root/stage_v2_baji"
$hasil = New-Object System.Collections.ArrayList
# Keluaran Python dialirkan lewat PowerShell agar ikut tercatat di log (transcript)
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Invoke-Py([string[]]$argList) {
    & $py @argList 2>&1 | ForEach-Object { Write-Host "$_" }
    return $LASTEXITCODE
}
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
function Get-ConvArgs([int]$e) {
    return @("stage_8_4_run_ablation_experiments.py", "--exp", "$e", "--seed", "$convSeed", "--epochs", "$convEpochs",
        "--data-yaml", $data, "--manifest-path", $manifest,
        "--pretrained-weights", $w0, "--output-dir", "$convRoot/seed_$convSeed", "--skip-if-done")
}
function Get-MapDir([string]$dir) {
    $f = "$dir/metrics_summary.json"
    if (-not (Test-Path $f)) { return "" }
    try { $j = Get-Content $f -Raw | ConvertFrom-Json; return ("{0:N2}%" -f ($j.metrics_validation.mAP50_95 * 100)) } catch { return "" }
}
function Invoke-PredictConv([int]$e) {
    $run = "$($expLabel[$e]) seed $convSeed ${convEpochs}ep"
    $pdir = "$convRoot/per_image"
    $out = "$pdir/preds_seed${convSeed}_$($expNames[$e])_e${convEpochs}.csv"
    if (Test-Path $out) { Add-Hasil "prediksi" $run "sudah ada" 0; return }
    New-Item -ItemType Directory -Force -Path $pdir | Out-Null
    $w = "$convRoot/seed_$convSeed/$($expNames[$e])/weights/best.pt"
    if (-not (Test-Path $w)) { Add-Hasil "prediksi" $run "dilewati: bobot tidak ada" 0; return }
    Write-Host ""
    Write-Host "=== Prediksi per citra $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py @("predict_eleak_per_image.py", "--weights", $w, "--source", $valImages, "--out-preds", "$out.tmp", "--out-gt", "$pdir/gt_cleanval.csv")
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0 -and (Test-Path "$out.tmp")) { Move-Item -Force "$out.tmp" $out; Add-Hasil "prediksi" $run "OK" $menit }
    else { Add-Hasil "prediksi" $run "GAGAL (exit $code)" $menit }
}
function Invoke-Predict([int]$s, [int]$e) {
    $run = "$($expLabel[$e]) seed $s"
    $out = "$predDir/preds_seed${s}_$($expNames[$e]).csv"
    if (Test-Path $out) { Add-Hasil "prediksi" $run "sudah ada" 0; return }
    New-Item -ItemType Directory -Force -Path $predDir | Out-Null
    $w = Get-Weights $s $e
    if (-not (Test-Path $w)) { Add-Hasil "prediksi" $run "dilewati: bobot tidak ada" 0; return }
    Write-Host ""
    Write-Host "=== Prediksi per citra $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py @("predict_eleak_per_image.py", "--weights", $w, "--source", $valImages, "--out-preds", "$out.tmp", "--out-gt", "$predDir/gt_cleanval.csv")
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
            "audit_label_150/README_AUDIT.md", "MALAM_INI.md", "uji_baji_hitam.py", "analyze_ablation.py") | Where-Object { Test-Path $_ }
        git add -- $files 2>&1 | Out-Null
        git diff --cached --quiet -- $files
        if ($LASTEXITCODE -ne 0) {
            git commit -m "Malam B: uji jalan pintas baji hitam dan uji konvergensi 60 epoch" `
                -m "uji_baji_hitam.py: baji hitam vs baji warna rata pada validasi bersih, inferensi saja dengan EXP-0 tiga seed. jalankan_fase3.ps1: opsi -UjiBaji dan -Conv (EXP-0 dan EXP-1, seed 42, 60 epoch). MALAM_INI.md: runbook malam B." `
                -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_015oXrwVC1rxV8W8friYcgig" -- $files 2>&1 | ForEach-Object { Write-Host "$_" }
            if ($LASTEXITCODE -eq 0) { Write-Host "Skrip baru di-commit ke branch $(git branch --show-current)." -ForegroundColor Green }
            else { Write-Host "Commit gagal (tidak menghentikan proses)." -ForegroundColor Yellow }
        }
    } else { Write-Host "git tidak ditemukan di PATH; commit dilewati." -ForegroundColor Yellow }
}

# --- 1. Preflight -------------------------------------------------------------
Write-Host ""
Write-Host "=== Preflight: GPU dan dry-run konfigurasi ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
$code = Invoke-Py @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA tidak tersedia'; print('GPU:', torch.cuda.get_device_name(0))")
if ($code -ne 0) { Add-Hasil "preflight" "GPU" "GAGAL (CUDA tidak tersedia)" 0 } else { Add-Hasil "preflight" "GPU" "OK" 0 }
foreach ($e in $urutan) {
    $s = $SeedList[0]
    $t0 = Get-Date
    $code = Invoke-Py ((Get-TrainArgs $s $e) + @("--dry-run"))
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) { Add-Hasil "preflight" "$($expLabel[$e]) (dry-run)" "GAGAL (exit $code)" $menit }
    else { Add-Hasil "preflight" "$($expLabel[$e]) (dry-run)" "OK" $menit }
}
if ($Conv) {
    foreach ($e in $convExps) {
        $t0 = Get-Date
        $code = Invoke-Py ((Get-ConvArgs $e) + @("--dry-run"))
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        $nm = "$($expLabel[$e]) ${convEpochs}ep (dry-run)"
        if ($code -ne 0) { Add-Hasil "preflight" $nm "GAGAL (exit $code)" $menit } else { Add-Hasil "preflight" $nm "OK" $menit }
    }
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

# --- 2a. Opsional: prediksi ulang E-LEAK dengan NMS seperti val (RQ1) ----------
if ($UlangELeak) {
    $ev2 = "$root/stage_v2_eleak_sim/per_image_v2"
    New-Item -ItemType Directory -Force -Path $ev2 | Out-Null
    if (Test-Path "$ev2/preds_sim_on_simval.csv") { Add-Hasil "E-LEAK" "model split acak" "sudah ada" 0 }
    else {
        Write-Host ""
        Write-Host "=== E-LEAK: prediksi ulang model split acak ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        $code = Invoke-Py @("predict_eleak_per_image.py", "--weights", "$root/stage_v2_eleak_sim/exp_0_baseline/weights/best.pt",
            "--source", "$root/eleak_sim_rq1/sim_val_all.txt", "--out-preds", "$ev2/preds_sim_on_simval.csv.tmp", "--out-gt", "$ev2/gt_simval.csv")
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -eq 0 -and (Test-Path "$ev2/preds_sim_on_simval.csv.tmp")) {
            Move-Item -Force "$ev2/preds_sim_on_simval.csv.tmp" "$ev2/preds_sim_on_simval.csv"; Add-Hasil "E-LEAK" "model split acak" "OK" $menit
        } else { Add-Hasil "E-LEAK" "model split acak" "GAGAL (exit $code)" $menit }
    }
    Invoke-Predict 42 0
    $g42 = "$predDir/preds_seed42_exp_0_baseline.csv"
    if (Test-Path $g42) {
        Copy-Item -Force $g42 "$ev2/preds_group_on_cleanval.csv"
        Copy-Item -Force "$predDir/gt_cleanval.csv" "$ev2/gt_cleanval.csv"
        Add-Hasil "E-LEAK" "model split per grup (EXP-0 seed 42)" "disalin" 0
    }
}

# --- 2b. Opsional: uji jalan pintas baji hitam (inferensi saja, beberapa menit) ---
if ($UjiBaji) {
    if (Test-Path "$bajiDir/SELESAI.txt") { Add-Hasil "uji baji" "EXP-0 tiga seed" "sudah ada" 0 }
    else {
        Write-Host ""
        Write-Host "=== Uji jalan pintas baji hitam ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        $code = Invoke-Py @("uji_baji_hitam.py", "--out-dir", $bajiDir)
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -eq 0 -and (Test-Path "$bajiDir/SELESAI.txt")) { Add-Hasil "uji baji" "EXP-0 tiga seed" "OK" $menit }
        else { Add-Hasil "uji baji" "EXP-0 tiga seed" "GAGAL (exit $code)" $menit }
    }
}

# --- 2. Training + prediksi per citra -----------------------------------------
foreach ($s in $SeedList) {
    Invoke-Predict $s 0
    foreach ($e in $urutan) {
        $run = "$($expLabel[$e]) seed $s"
        Write-Host ""
        Write-Host "=== Training $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        $code = Invoke-Py (Get-TrainArgs $s $e)
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -ne 0) { Add-Hasil "training" $run "GAGAL (exit $code)" $menit; continue }
        Add-Hasil "training" $run "OK" $menit (Get-Map $s $e)
        Invoke-Predict $s $e
    }
}

# --- 3. Opsional: Fase 4 CONV (60 epoch, seed 42, EXP-0 lalu EXP-1) -----------
if ($Conv) {
    foreach ($e in $convExps) {
        $run = "$($expLabel[$e]) seed $convSeed ${convEpochs}ep"
        Write-Host ""
        Write-Host "=== Training CONV $run ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        $code = Invoke-Py (Get-ConvArgs $e)
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -ne 0) { Add-Hasil "CONV" $run "GAGAL (exit $code)" $menit; continue }
        Add-Hasil "CONV" $run "OK" $menit (Get-MapDir "$convRoot/seed_$convSeed/$($expNames[$e])")
        Invoke-PredictConv $e
    }
}

try { [Skripsi.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch {}
Show-Ringkasan
Stop-Transcript | Out-Null
