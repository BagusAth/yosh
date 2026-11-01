# =============================================================================
# Uji pinggiran bergaris (Pakem v1.4.1, 31 Okt 2026 malam)
#
#   klik dua kali jalankan_uji_garis.bat
#
# uji_pinggiran_bergaris.py pada EXP-0, EXP-1, EXP-2 (seed 42, 123, 2026): 1.183 citra validasi tanpa baji hitam
# dan tanpa garis, 3 varian (asli, garis, rata). Inferensi saja, sekitar 10-15 menit GPU.
# Keluaran: stage_v2_garis\ (dilewati bila SELESAI.txt sudah ada). Test set tidak disentuh.
# Laptop dicegah tidur selama skrip berjalan.
# =============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs_v2" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmm
$log = "logs_v2\uji_garis_$stamp.log"
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
try {
    Add-Type -Name Power -Namespace SkripsiGaris -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [SkripsiGaris.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { Write-Host "Catatan: gagal mencegah sleep otomatis; pastikan mode tidur dimatikan." -ForegroundColor Yellow }

# --- Preflight GPU ------------------------------------------------------------
Write-Host "=== Preflight GPU ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
$code = Invoke-Py @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA tidak tersedia'; print('GPU:', torch.cuda.get_device_name(0))")
if ($code -ne 0) { Add-Hasil "preflight GPU" "GAGAL (CUDA tidak tersedia)" 0 } else { Add-Hasil "preflight GPU" "OK" 0 }

# --- Uji pinggiran bergaris -----------------------------------------------------
if (Test-Path "stage_v2_garis\SELESAI.txt") { Add-Hasil "uji pinggiran bergaris" "sudah ada" 0 }
else {
    Write-Host ""
    Write-Host "=== Uji pinggiran bergaris: 9 model x 3 varian ($(Get-Date -Format HH:mm)) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $code = Invoke-Py @("uji_pinggiran_bergaris.py")
    $menit = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0 -and (Test-Path "stage_v2_garis\SELESAI.txt")) { Add-Hasil "uji pinggiran bergaris" "OK" $menit }
    else { Add-Hasil "uji pinggiran bergaris" "GAGAL (exit $code)" $menit }
}

# --- Ringkasan ------------------------------------------------------------------
try { [SkripsiGaris.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch {}
Write-Host ""
Write-Host "=================== RINGKASAN ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) ===================" -ForegroundColor Cyan
$hasil | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$hasil | Export-Csv -NoTypeInformation -Encoding UTF8 -Path "logs_v2\uji_garis_${stamp}_ringkasan.csv"
$gagal = @($hasil | Where-Object { $_.Status -like "GAGAL*" })
if ($gagal.Count -gt 0) { Write-Host "$($gagal.Count) langkah GAGAL. Kirim log ini: $log" -ForegroundColor Red }
else { Write-Host "Selesai tanpa kegagalan. Log: $log. Beri tahu Claude untuk analisis." -ForegroundColor Green }
Stop-Transcript | Out-Null
