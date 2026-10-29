@echo off
cd /d "%~dp0"
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0jalankan_fase3.ps1" -Seeds 42,123,2026 -CekSaja
