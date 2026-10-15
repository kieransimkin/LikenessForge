@echo off
setlocal
cd /d "%~dp0"
echo LikenessForge: bootstrapping with uv and starting the GUI...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-gui.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
endlocal & exit /b %EXIT_CODE%
