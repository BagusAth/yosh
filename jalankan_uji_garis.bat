@echo off
cd /d "%~dp0"
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0jalankan_uji_garis.ps1"
