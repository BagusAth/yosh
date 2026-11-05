# =============================================================================
# Cek implementasi TIDE terhadap kode tidecv asli (analisis pasca-hoc butir 6, 5 Nov 2026)
#
#   klik dua kali jalankan_cek_tidecv.bat
#
# Tanpa GPU dan tanpa test set. Langkah:
#   1. Bila tidecv belum terpasang di .venv-cuda, minta izin lalu pasang dengan --no-deps
#      (tidak mengubah paket lain; appdirs/seaborn diganti shim bila tidak ada).
#   2. analyze_tide_iou.py prep  (sekitar 1-2 menit; dilewati bila cache sudah ada)
#   3. analyze_tide_iou.py cek-tidecv  (15 model, perkiraan 10-30 menit CPU)
# Keluaran: stage_v2_ablation\analysis\tide_iou\cek_tidecv.json. Beri tahu Claude setelah selesai.
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmm
$log = "logs_v2\cek_tidecv_$stamp.log"
Start-Transcript -Path $log | Out-Null

$py = ".\.venv-cuda\Scripts\python.exe"
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

# --- 1. tidecv ------------------------------------------------------------------
Write-Host "=== Cek paket tidecv ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
$code = Invoke-Py @("-c", "import importlib.util as u, sys; sys.exit(0 if u.find_spec('tidecv') else 1)")
if ($code -ne 0) {
    Write-Host "tidecv belum terpasang di .venv-cuda." -ForegroundColor Yellow
    $jawab = Read-Host "Pasang tidecv 1.0.1 dengan pip --no-deps (tidak mengubah paket lain)? ketik y lalu Enter"
    if ($jawab -eq "y") {
        $code = Invoke-Py @("-m", "pip", "install", "--no-deps", "tidecv==1.0.1")
        if ($code -eq 0) { Add-Hasil "pasang tidecv" "OK" 0 } else { Add-Hasil "pasang tidecv" "GAGAL (exit $code)" 0 }
    } else { Add-Hasil "pasang tidecv" "GAGAL (dibatalkan pengguna)" 0 }
} else { Add-Hasil "pasang tidecv" "sudah ada" 0 }

$gagal0 = @($hasil | Where-Object { $_.Status -like "GAGAL*" })
if ($gagal0.Count -eq 0) {
    # --- 2. prep ------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== analyze_tide_iou.py prep ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py @("analyze_tide_iou.py", "prep")
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0) { Add-Hasil "prep" "OK" $menit } else { Add-Hasil "prep" "GAGAL (exit $code)" $menit }

    # --- 3. cek-tidecv ----------------------------------------------------------------
    if ($code -eq 0) {
        Write-Host ""
        Write-Host "=== analyze_tide_iou.py cek-tidecv: 15 model ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
        $t0 = Get-Date
        $code = Invoke-Py @("analyze_tide_iou.py", "cek-tidecv")
        $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -eq 0) { Add-Hasil "cek-tidecv" "OK" $menit } else { Add-Hasil "cek-tidecv" "GAGAL (exit $code)" $menit }
    }
}

# --- Ringkasan ------------------------------------------------------------------
Write-Host ""
Write-Host "=================== RINGKASAN ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===================" -ForegroundColor Cyan
$hasil | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$gagal = @($hasil | Where-Object { $_.Status -like "GAGAL*" })
if ($gagal.Count -gt 0) { Write-Host "$($gagal.Count) langkah GAGAL. Kirim log ini: $log" -ForegroundColor Red }
else { Write-Host "Selesai tanpa kegagalan. Log: $log. Beri tahu Claude untuk memeriksa cek_tidecv.json." -ForegroundColor Green }
Stop-Transcript | Out-Null
return
