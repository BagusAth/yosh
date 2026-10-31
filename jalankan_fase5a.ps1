# =============================================================================
# Fase 5 A (Pakem v1.4, 31 Okt 2026): B1 uji baji EXP-1/EXP-2 + B2 cek pycocotools
#
#   powershell -ExecutionPolicy Bypass -File .\jalankan_fase5a.ps1
#
# B1: uji_baji_hitam.py pada EXP-1 dan EXP-2 (seed 42, 123, 2026), inferensi saja, sekitar 10-15 menit GPU.
#     Keluaran: stage_v2_baji_exp12\ (dilewati bila SELESAI.txt sudah ada).
# B2: pasang pycocotools ke .venv-cuda bila belum ada, lalu cek_coco_ukuran.py pada EXP-0, EXP-1, EXP-2 seed 42
#     (CPU, sekitar 5 menit). Keluaran: stage_v2_ablation\analysis\coco_cek\
# - Langkah yang gagal tidak menghentikan langkah berikutnya; ringkasan di akhir.
# - Laptop dicegah tidur selama skrip berjalan. Test set tidak disentuh.
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmm
$log = "logs_v2\fase5a_$stamp.log"
Start-Transcript -Path $log | Out-Null

$py = ".\.venv-cuda\Scripts\python.exe"
$root = "D:/UNDIP/SKRIPSI"
$hasil = New-Object System.Collections.ArrayList
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Invoke-Py([string[]]$argList) {
    & $py @argList 2>&1 | ForEach-Object { Write-Host "$_" }
    return $LASTEXITCODE
}
function Add-Hasil([string]$tahap, [string]$status, [double]$menit) {
    [void]$hasil.Add([pscustomobject]@{ Tahap = $tahap; Status = $status; Menit = $menit })
}
try {
    Add-Type -Name Power -Namespace Skripsi5 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [Skripsi5.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { Write-Host "Catatan: gagal mencegah sleep otomatis; pastikan mode tidur dimatikan." -ForegroundColor Yellow }

# --- Preflight GPU ------------------------------------------------------------
Write-Host "=== Preflight GPU ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
$code = Invoke-Py @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA tidak tersedia'; print('GPU:', torch.cuda.get_device_name(0))")
if ($code -ne 0) { Add-Hasil "preflight GPU" "GAGAL (CUDA tidak tersedia)" 0 } else { Add-Hasil "preflight GPU" "OK" 0 }

# --- B1: uji baji EXP-1 dan EXP-2 --------------------------------------------
$bajiDir = "$root/stage_v2_baji_exp12"
if (Test-Path "$bajiDir/SELESAI.txt") { Add-Hasil "B1 uji baji EXP-1/EXP-2" "sudah ada" 0 }
else {
    $models = @()
    foreach ($s in @(42, 123, 2026)) {
        $models += "exp1_s$s=$root/stage_v2_ablation/seed_$s/exp_1_adaptive_sampling/weights/best.pt"
        $models += "exp2_s$s=$root/stage_v2_ablation/seed_$s/exp_2_targeted_aug/weights/best.pt"
    }
    Write-Host ""
    Write-Host "=== B1: uji baji EXP-1 dan EXP-2, tiga seed ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py (@("uji_baji_hitam.py", "--out-dir", $bajiDir, "--models") + $models)
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0 -and (Test-Path "$bajiDir/SELESAI.txt")) { Add-Hasil "B1 uji baji EXP-1/EXP-2" "OK" $menit }
    else { Add-Hasil "B1 uji baji EXP-1/EXP-2" "GAGAL (exit $code)" $menit }
}

# --- B2: pycocotools ------------------------------------------------------------
Write-Host ""
Write-Host "=== B2: pycocotools ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
$code = Invoke-Py @("-c", "import pycocotools; print('pycocotools sudah terpasang')")
if ($code -ne 0) {
    Write-Host "Memasang pycocotools ke .venv-cuda ..."
    $code = Invoke-Py @("-m", "pip", "install", "pycocotools")
    if ($code -ne 0) { Add-Hasil "B2 pasang pycocotools" "GAGAL (exit $code)" 0 } else { Add-Hasil "B2 pasang pycocotools" "OK" 0 }
} else { Add-Hasil "B2 pasang pycocotools" "sudah ada" 0 }
$t0 = Get-Date
$code = Invoke-Py @("cek_coco_ukuran.py")
$menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
if ($code -eq 0) { Add-Hasil "B2 cek pycocotools (3 model)" "OK" $menit } else { Add-Hasil "B2 cek pycocotools (3 model)" "GAGAL (exit $code)" $menit }

# --- Ringkasan ------------------------------------------------------------------
try { [Skripsi5.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch {}
Write-Host ""
Write-Host "=================== RINGKASAN ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===================" -ForegroundColor Cyan
$hasil | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$hasil | Export-Csv -NoTypeInformation -Encoding UTF8 -Path "logs_v2\fase5a_${stamp}_ringkasan.csv"
$gagal = @($hasil | Where-Object { $_.Status -like "GAGAL*" })
if ($gagal.Count -gt 0) { Write-Host "$($gagal.Count) langkah GAGAL. Kirim log ini: $log" -ForegroundColor Red }
else { Write-Host "Semua langkah selesai tanpa kegagalan. Log: $log" -ForegroundColor Green }
Stop-Transcript | Out-Null
