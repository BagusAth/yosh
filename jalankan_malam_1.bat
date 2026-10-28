@echo off
REM Launcher Malam 1 (Pakem v1.3): menjalankan jalankan_malam_1.ps1 di jendela PowerShell yang tetap terbuka.
cd /d "%~dp0"
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0jalankan_malam_1.ps1"
