# =============================================================================
# Fase 6: evaluasi test set SEKALI (17 model) sesuai PRA_REGISTRASI_FASE6.md
#
#   Cek prasyarat saja (tidak menyentuh GPU maupun test set):
#       powershell -ExecutionPolicy Bypass -File .\jalankan_fase6.ps1 -CekSaja
#   Jalankan (hanya setelah pra-registrasi berstatus DISETUJUI):
#       klik dua kali jalankan_fase6.bat
#
# Pengaman sekali jalan:
#   - Menolak berjalan bila stage_v2_fase6\SELESAI.txt sudah ada.
#   - Menolak berjalan bila PRA_REGISTRASI_FASE6.md tidak memuat baris "Status: DISETUJUI".
#   - Bila terputus di tengah, menjalankan ulang hanya melengkapi model yang belum punya prediksi
#     (tidak ada model yang dinilai dua kali; tidak ada pemilihan ulang).
# Urutan: prediksi per citra 17 model -> val() pendamping 17 model -> analisis CPU -> SELESAI.txt
# Laptop dicegah tidur selama skrip berjalan.
# =============================================================================
param([switch]$CekSaja)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmm
$tag = if ($CekSaja) { "cek" } else { "jalan" }
$log = "logs_v2\fase6_${tag}_$stamp.log"
Start-Transcript -Path $log | Out-Null

$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"
$data = "$root/cassava-v7-standardized/data.yaml"
$testImages = "$root/cassava-v7-standardized/test/images"
$testLabels = "$root/cassava-v7-standardized/test/labels"
$f6 = "$root/stage_v2_fase6"
$predDir = "$f6/per_image"
$convDir = "$f6/per_image_conv"
$ana = "stage_v2_fase6/analysis"
$expNames = @{ 0 = "exp_0_baseline"; 4 = "exp_0r_uniform_replacement"; 1 = "exp_1_adaptive_sampling"; 2 = "exp_2_targeted_aug"; 3 = "exp_3_full_proposed" }
$expLabel = @{ 0 = "EXP-0"; 4 = "EXP-0R"; 1 = "EXP-1"; 2 = "EXP-2"; 3 = "EXP-3" }
$seeds = @(42, 123, 2026)
$urutan = @(0, 4, 1, 2, 3)
$hasil = New-Object System.Collections.ArrayList
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Invoke-Py([string[]]$argList) {
    & $py @argList 2>&1 | ForEach-Object { Write-Host "$_" }
    return $LASTEXITCODE
}
function Add-Hasil([string]$tahap, [string]$run, [string]$status, [double]$menit) {
    [void]$hasil.Add([pscustomobject]@{ Tahap = $tahap; Run = $run; Status = $status; Menit = $menit })
}
function Get-Weights([int]$s, [int]$e) {
    if ($e -eq 0 -and $s -eq 42) { return "$root/stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt" }
    return "$root/stage_v2_ablation/seed_$s/$($expNames[$e])/weights/best.pt"
}
function Get-Sha([string]$p) { if (Test-Path $p) { return (Get-FileHash -Algorithm SHA256 -Path $p).Hash.Substring(0, 16) } else { return "TIDAK ADA" } }

# Daftar 17 model: 15 ablasi + 2 CONV
$models = New-Object System.Collections.ArrayList
foreach ($s in $seeds) { foreach ($e in $urutan) {
    [void]$models.Add([pscustomobject]@{ Key = "$s|$($expLabel[$e])"; W = (Get-Weights $s $e); Out = "$predDir/preds_seed${s}_$($expNames[$e]).csv"; Gt = "$predDir/gt_test.csv" })
} }
foreach ($e in @(0, 1)) {
    [void]$models.Add([pscustomobject]@{ Key = "conv|$($expLabel[$e]) 60ep"; W = "$root/stage_v2_conv60/seed_42/$($expNames[$e])/weights/best.pt"; Out = "$convDir/preds_seed42_$($expNames[$e])_e60.csv"; Gt = "$convDir/gt_test.csv" })
}

# ------------------------------------------------------------------ pengaman
$ok = $true
Write-Host "=== Cek prasyarat Fase 6 ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===" -ForegroundColor Cyan
if (Test-Path "$f6/SELESAI.txt") { Write-Host "DITOLAK: Fase 6 sudah selesai (stage_v2_fase6\SELESAI.txt ada). Test set hanya dijalankan sekali." -ForegroundColor Red; $ok = $false }
if (-not (Test-Path "PRA_REGISTRASI_FASE6.md")) { Write-Host "DITOLAK: PRA_REGISTRASI_FASE6.md tidak ada." -ForegroundColor Red; $ok = $false }
elseif (-not (Select-String -Path "PRA_REGISTRASI_FASE6.md" -Pattern "^Status: DISETUJUI" -Quiet)) { Write-Host "DITOLAK: pra-registrasi belum berstatus DISETUJUI." -ForegroundColor Red; $ok = $false }
$nImg = @(Get-ChildItem $testImages -File).Count
$nLbl = @(Get-ChildItem $testLabels -File -Filter *.txt).Count
Write-Host "Test set: $nImg citra, $nLbl label (harus 1282)"
if ($nImg -ne 1282 -or $nLbl -ne 1282) { Write-Host "DITOLAK: jumlah citra/label test tidak 1282." -ForegroundColor Red; $ok = $false }
foreach ($m in $models) { if (-not (Test-Path $m.W)) { Write-Host "DITOLAK: bobot tidak ada: $($m.W)" -ForegroundColor Red; $ok = $false } }
foreach ($f in @("predict_eleak_per_image.py", "val_test_fase6.py", "analyze_ablation.py", "analyze_conv.py", "analyze_ukuran.py", "analyze_error.py", "analyze_fase6.py")) {
    if (-not (Test-Path $f)) { Write-Host "DITOLAK: skrip tidak ada: $f" -ForegroundColor Red; $ok = $false }
}
if (-not (Test-Path $py)) { Write-Host "DITOLAK: python .venv-cuda tidak ditemukan" -ForegroundColor Red; $ok = $false }
Write-Host "Model: $($models.Count) (harus 17)"
if ($models.Count -ne 17) { $ok = $false }
if (-not $ok) { Write-Host "Prasyarat TIDAK terpenuhi. Tidak ada yang dijalankan." -ForegroundColor Red; Stop-Transcript | Out-Null; return }
Write-Host "Prasyarat terpenuhi." -ForegroundColor Green
if ($CekSaja) { Write-Host "Mode cek: berhenti di sini, test set tidak disentuh."; Stop-Transcript | Out-Null; return }

try {
    Add-Type -Name Power -Namespace Skripsi6 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [Skripsi6.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { Write-Host "Catatan: gagal mencegah sleep otomatis." -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path $predDir, $convDir, "$f6/analysis" | Out-Null
if (-not (Test-Path "$f6/MULAI.txt")) {
    $git = ""
    try { $git = (& git rev-parse HEAD 2>$null) } catch {}
    $info = @("mulai: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "git HEAD: $git",
        "PRA_REGISTRASI_FASE6.md: $(Get-Sha 'PRA_REGISTRASI_FASE6.md')",
        "predict_eleak_per_image.py: $(Get-Sha 'predict_eleak_per_image.py')", "val_test_fase6.py: $(Get-Sha 'val_test_fase6.py')",
        "analyze_ablation.py: $(Get-Sha 'analyze_ablation.py')", "analyze_conv.py: $(Get-Sha 'analyze_conv.py')",
        "analyze_ukuran.py: $(Get-Sha 'analyze_ukuran.py')", "analyze_error.py: $(Get-Sha 'analyze_error.py')", "analyze_fase6.py: $(Get-Sha 'analyze_fase6.py')")
    foreach ($m in $models) { $info += "bobot $($m.Key): $(Get-Sha $m.W)" }
    $info | Set-Content -Encoding ASCII "$f6/MULAI.txt"
} else { Write-Host "MULAI.txt sudah ada: melanjutkan run yang terputus (model yang sudah punya prediksi dilewati)." -ForegroundColor Yellow }

# ------------------------------------------------------------------ 1. prediksi per citra
foreach ($m in $models) {
    if (Test-Path $m.Out) { Add-Hasil "prediksi" $m.Key "sudah ada" 0; continue }
    Write-Host ""
    Write-Host "=== Prediksi per citra test: $($m.Key) ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py @("predict_eleak_per_image.py", "--weights", $m.W, "--source", $testImages, "--out-preds", "$($m.Out).tmp", "--out-gt", $m.Gt)
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0 -and (Test-Path "$($m.Out).tmp")) { Move-Item -Force "$($m.Out).tmp" $m.Out; Add-Hasil "prediksi" $m.Key "OK" $menit }
    else { Add-Hasil "prediksi" $m.Key "GAGAL (exit $code)" $menit }
}

# ------------------------------------------------------------------ 2. val() pendamping
$valArgs = @("val_test_fase6.py", "--data", $data, "--out", "$f6/val_test.json", "--models")
foreach ($m in $models) { $valArgs += "$($m.Key)=$($m.W)" }
$t0 = Get-Date
$code = Invoke-Py $valArgs
Add-Hasil "val() pendamping" "17 model" $(if ($code -eq 0) { "OK" } else { "GAGAL (exit $code)" }) ([math]::Round(((Get-Date) - $t0).TotalMinutes, 1))

# ------------------------------------------------------------------ 3. analisis (CPU)
$gagalPred = @($hasil | Where-Object { $_.Tahap -eq "prediksi" -and $_.Status -like "GAGAL*" }).Count
if ($gagalPred -gt 0) {
    Write-Host "Ada $gagalPred prediksi gagal; analisis tidak dijalankan. Jalankan ulang .bat untuk melengkapi." -ForegroundColor Red
} else {
    $langkah = @(
        @("analyze_ablation.py", "prep", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--gt-name", "gt_test.csv", "--val-json", "stage_v2_fase6/val_test.json", "--nama-data", "citra test", "--judul", "Fase 6 test set"),
        @("analyze_ablation.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--seed", "0", "--iters", "250"),
        @("analyze_ablation.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--seed", "1", "--iters", "250"),
        @("analyze_ablation.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--seed", "2", "--iters", "250"),
        @("analyze_ablation.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--seed", "3", "--iters", "250"),
        @("analyze_ablation.py", "report", "--per-image-dir", "stage_v2_fase6/per_image", "--out", $ana, "--nama-data", "citra test", "--judul", "Fase 6 test set"),
        @("analyze_conv.py", "--out", "$ana/conv", "--gt", "stage_v2_fase6/per_image/gt_test.csv", "--pred-30-dir", "stage_v2_fase6/per_image", "--pred-60-dir", "stage_v2_fase6/per_image_conv", "--nama-data", "citra test"),
        @("analyze_ukuran.py", "prep", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran"),
        @("analyze_ukuran.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran", "--seed", "0", "--iters", "250"),
        @("analyze_ukuran.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran", "--seed", "1", "--iters", "250"),
        @("analyze_ukuran.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran", "--seed", "2", "--iters", "250"),
        @("analyze_ukuran.py", "boot", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran", "--seed", "3", "--iters", "250"),
        @("analyze_ukuran.py", "report", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/ukuran", "--nama-data", "citra test"),
        @("analyze_error.py", "prep", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/error"),
        @("analyze_error.py", "report", "--per-image-dir", "stage_v2_fase6/per_image", "--gt-name", "gt_test.csv", "--out", "$ana/error", "--nama-data", "citra test"),
        @("analyze_fase6.py")
    )
    foreach ($l in $langkah) {
        $t0 = Get-Date
        Write-Host ""
        Write-Host "=== Analisis: $($l -join ' ') ===" -ForegroundColor Cyan
        $code = Invoke-Py $l
        Add-Hasil "analisis" "$($l[0]) $($l[1])" $(if ($code -eq 0) { "OK" } else { "GAGAL (exit $code)" }) ([math]::Round(((Get-Date) - $t0).TotalMinutes, 1))
    }
}

# ------------------------------------------------------------------ ringkasan
Write-Host ""
Write-Host "=================== RINGKASAN FASE 6 ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===================" -ForegroundColor Cyan
$hasil | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$hasil | Export-Csv -NoTypeInformation -Encoding UTF8 -Path "logs_v2\fase6_${tag}_${stamp}_ringkasan.csv"
$gagal = @($hasil | Where-Object { $_.Status -like "GAGAL*" }).Count
if ($gagal -eq 0) {
    @("selesai: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "log: $log") | Set-Content -Encoding ASCII "$f6/SELESAI.txt"
    Write-Host "Fase 6 selesai tanpa kegagalan. Hasil: stage_v2_fase6\analysis\FASE6_RINGKASAN.md" -ForegroundColor Green
    Write-Host "Kirim ke Claude: 'Fase 6 selesai. Log: $log'"
} else {
    Write-Host "Ada $gagal langkah gagal. SELESAI.txt belum ditulis; jalankan ulang .bat untuk melengkapi." -ForegroundColor Red
}
Stop-Transcript | Out-Null
