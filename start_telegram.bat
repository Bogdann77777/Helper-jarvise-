@echo off
REM Запуск Telegram Bridge — принимает файлы и пересылает в CLI Helper
REM Установи токен ниже или в переменную окружения TELEGRAM_BOT_TOKEN

set TELEGRAM_BOT_TOKEN=ВСТАВЬ_ТОКЕН_СЮДА
set CLI_SERVER_URL=http://localhost:8000

echo Starting Telegram Bridge...
echo Bot token: %TELEGRAM_BOT_TOKEN:~0,10%...
echo Server: %CLI_SERVER_URL%
echo.

cd /d "%~dp0"
python telegram_bridge.py

pause
