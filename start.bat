@echo off
title Claude Voice Server
chcp 65001 >nul

echo.
echo  ========================================
echo       Claude Voice Interface
echo       XTTS v2 (all modes)
echo  ========================================
echo.

cd /d "%~dp0"
set PYTHON=C:\Python311\python.exe

if not exist "%PYTHON%" (
    echo  [ERROR] Python 3.11 not found at %PYTHON%
    pause
    exit /b 1
)

if not exist "server.py" (
    echo  [ERROR] server.py not found.
    pause
    exit /b 1
)

%PYTHON% diagnose.py
if errorlevel 1 (
    echo  See startup.log for details.
    pause
    exit /b 1
)

if not exist "voices\olena.wav" (
    echo  [!] Running setup_voices.py...
    %PYTHON% setup_voices.py 2>&1
)
if not exist "voices\board\cso.wav" (
    echo  [!] Running setup_board_voices.py...
    %PYTHON% setup_board_voices.py 2>&1
)

:: --- ngrok (centralized) ---
echo.
echo  [ngrok] Checking tunnels...
powershell -NoProfile -Command "try { Invoke-WebRequest 'http://localhost:4040/api/tunnels' -UseBasicParsing >$null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo  [ngrok] Starting all tunnels...
    start /B ngrok start --all --config "E:\project\NRok\ngrok.yml" > nul 2>&1
    timeout /t 4 /nobreak > nul
)
powershell -NoProfile -Command "$tunnels = (Invoke-WebRequest 'http://localhost:4040/api/tunnels' -UseBasicParsing).Content | ConvertFrom-Json | Select-Object -ExpandProperty tunnels; $map = @{ 'market-analyzer' = 'Market Analyzer'; 'cli-helper' = 'CLI Helper'; 'reader' = 'Reader' }; Write-Host ''; Write-Host '  ============================================' -ForegroundColor DarkGray; Write-Host '   NGROK TUNNELS' -ForegroundColor White; Write-Host '  ============================================' -ForegroundColor DarkGray; foreach ($t in $tunnels) { $label = $map[$t.name]; if (-not $label) { $label = $t.name }; Write-Host ('  {0,-18} {1}' -f ($label + ':'), $t.public_url) -ForegroundColor Cyan }; Write-Host '  ============================================' -ForegroundColor DarkGray; Write-Host ''"

:: --- Telegram Bridge (background) ---
echo.
echo  [Telegram] Starting bridge...
start "Telegram Bridge" /min cmd /c "C:\Python311\python.exe telegram_bridge.py"
echo  [Telegram] Bridge started in background.

echo.
echo  ----------------------------------------
echo   Starting server...
echo  ----------------------------------------
echo.

start "" /b cmd /c "timeout /t 15 >nul && start http://localhost:8000"

set PYTHONIOENCODING=utf-8
%PYTHON% server.py

echo.
echo  [!] Server stopped.
pause
