#!/usr/bin/env bash
# =============================================================
#  CLI Helper — ЗАПУСК (Linux)
#  Эквивалент start.bat для Fedora/Ubuntu
#  Запускать: bash start.sh
# =============================================================
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo ""
echo " ========================================"
echo "      Claude Voice Interface"
echo "      CLI Helper (Linux)"
echo " ========================================"
echo ""

# ---- Активация venv ----
if [ -f ".venv/bin/activate" ]; then
    echo " [OK] Активирую .venv..."
    source .venv/bin/activate
else
    echo " [!] .venv не найден — сначала запусти: bash setup.sh"
    exit 1
fi

# ---- Проверка server.py ----
if [ ! -f "server.py" ]; then
    echo " [ERROR] server.py не найден!"
    exit 1
fi

# ---- Диагностика ----
python diagnose.py
if [ $? -ne 0 ]; then
    echo " [!] diagnose.py упал, смотри startup.log"
    exit 1
fi

# ---- Голоса ----
if [ ! -f "voices/olena.wav" ]; then
    echo " [!] Генерирую голоса (setup_voices.py)..."
    python setup_voices.py
fi
if [ ! -f "voices/board/cso.wav" ]; then
    echo " [!] Генерирую board-голоса (setup_board_voices.py)..."
    python setup_board_voices.py
fi

# ---- Локальный адрес ----
LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
echo ""
echo " ============================================"
echo "  ЛОКАЛЬНЫЙ ДОСТУП"
echo " ============================================"
echo "  Локально:   http://localhost:8000"
[ -n "$LOCAL_IP" ] && echo "  По сети:    http://$LOCAL_IP:8000"
echo " ============================================"

# ---- ngrok ----
echo ""
if ! command -v ngrok &>/dev/null; then
    echo " [ngrok] ngrok не установлен — туннели недоступны."
    echo "         Установи: sudo dnf install ngrok"
else
    echo " [ngrok] Проверяю туннели..."
    if ! curl -sf http://localhost:4040/api/tunnels > /dev/null 2>&1; then
        echo " [ngrok] Запускаю туннели..."
        NGROK_CONFIG="$HOME/.config/ngrok/ngrok.yml"
        if [ -f "$NGROK_CONFIG" ]; then
            nohup ngrok start --all --config "$NGROK_CONFIG" > /tmp/ngrok.log 2>&1 &
        else
            nohup ngrok http 8000 > /tmp/ngrok.log 2>&1 &
        fi
        sleep 4
    fi

    # Показываем туннели
    TUNNELS=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null)
    if [ -n "$TUNNELS" ]; then
        echo ""
        echo " ============================================"
        echo "  NGROK TUNNELS"
        echo " ============================================"
        echo "$TUNNELS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data.get('tunnels', []):
    name = t.get('name', '')
    url = t.get('public_url', '')
    labels = {'market-analyzer': 'Market Analyzer', 'cli-helper': 'CLI Helper', 'reader': 'Reader'}
    label = labels.get(name, name)
    print(f'  {label:<18} {url}')
"
        echo " ============================================"
    fi
fi

# ---- Telegram Bridge (фоновый процесс) ----
echo ""
echo " [Telegram] Запускаю bridge..."
nohup python telegram_bridge.py > /tmp/telegram_bridge.log 2>&1 &
TG_PID=$!
echo " [Telegram] Bridge запущен (PID: $TG_PID)"

# ---- Автооткрытие браузера (через 15 сек) ----
(sleep 15 && xdg-open http://localhost:8000 2>/dev/null || true) &

# ---- Запуск сервера ----
echo ""
echo " ----------------------------------------"
echo "   Starting server..."
echo " ----------------------------------------"
echo ""

export PYTHONIOENCODING=utf-8
export COQUI_TOS_AGREED=1
python server.py

echo ""
echo " [!] Сервер остановлен."
