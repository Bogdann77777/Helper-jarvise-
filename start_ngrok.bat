@echo off
title ngrok - External Access Tunnel
chcp 65001 >nul

echo.
echo  ========================================
echo       ngrok - External Access Tunnel
echo       Port: 8000
echo  ========================================
echo.
echo  Keep this window open during your presentation.
echo  The public URL will appear below (look for "Forwarding").
echo.
echo  TIP: Use the https:// URL, not http://
echo.

:: Check ngrok in PATH
where ngrok >nul 2>&1
if not errorlevel 1 (
    echo  [OK] ngrok found in PATH.
    ngrok http 8000
    goto :end
)

:: Check ngrok.exe in same folder as this bat
if exist "%~dp0ngrok.exe" (
    echo  [OK] ngrok.exe found in project folder.
    "%~dp0ngrok.exe" http 8000
    goto :end
)

:: Not found
echo  [ERROR] ngrok not found!
echo.
echo  Steps to set it up:
echo.
echo  1. Go to: https://ngrok.com/download
echo  2. Download ngrok for Windows
echo  3. Place ngrok.exe in this folder:
echo     %~dp0
echo  4. Run this script again.
echo.
echo  (Free account gives you a random public URL each time)
echo.

:end
pause
