#!/usr/bin/env bash
# =============================================================
#  CLI Helper — ОДНОРАЗОВАЯ УСТАНОВКА (Fedora Linux + NVIDIA)
#  Запускать: bash setup.sh
#  Повторный запуск: безопасен, не сломает уже установленное
# =============================================================
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON=${PYTHON:-python3.12}

echo ""
echo " ========================================"
echo "      CLI Helper — Setup (Fedora)"
echo " ========================================"
echo ""

# ---- 1. Системные пакеты ----
echo "[1/6] Системные пакеты (dnf)..."
sudo dnf install -y \
    python3.12 python3.12-devel \
    portaudio portaudio-devel \
    ffmpeg \
    git curl wget \
    gcc gcc-c++ make \
    libsndfile \
    espeak-ng \
    nodejs npm 2>&1 | tail -5
echo " [OK] Системные пакеты установлены."

# ---- 2. Claude CLI (npm) ----
echo ""
echo "[2/6] Claude CLI..."
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
    echo " [OK] Claude CLI установлен."
else
    echo " [OK] Claude CLI уже установлен: $(claude --version 2>/dev/null || echo 'ok')"
fi

# ---- 3. Python venv ----
echo ""
echo "[3/6] Python venv..."
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo " [OK] .venv создан."
else
    echo " [OK] .venv уже существует."
fi

source .venv/bin/activate

# ---- 4. PyTorch с CUDA (cu128) ----
echo ""
echo "[4/6] PyTorch cu128..."
if ! python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    pip install --quiet torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    echo " [OK] PyTorch cu128 установлен."
else
    echo " [OK] PyTorch с CUDA уже работает: $(python -c 'import torch; print(torch.__version__)')"
fi

# ---- 5. Pip зависимости ----
echo ""
echo "[5/6] Pip зависимости..."
pip install --quiet -r requirements.txt
echo " [OK] requirements.txt установлен."

# ---- 6. .env файл ----
echo ""
echo "[6/6] .env конфиг..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# CLI Helper — переменные окружения
# Заполни токены перед запуском!

TELEGRAM_BOT_TOKEN=ВСТАВЬ_ТОКЕН_СЮДА
OPENROUTER_API_KEY=ВСТАВЬ_КЛЮЧ_СЮДА
PERPLEXITY_API_KEY=
VASTAI_API_KEY=03c0d90ebb40cbfaee919af20a82aaeb39f79990ebdaf05473b9d40cf194a018
EOF
    echo " [OK] .env создан. Заполни токены в .env!"
else
    echo " [OK] .env уже существует."
fi

# ---- ngrok ----
echo ""
if ! command -v ngrok &>/dev/null; then
    echo "[!] ngrok не установлен."
    echo "    Скачай: https://ngrok.com/download"
    echo "    Или: curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/yum.repos.d/ngrok.repo"
    echo "    sudo dnf install ngrok"
else
    echo "[OK] ngrok найден."
fi

# ---- SSH ключ для VAST.ai ----
echo ""
if [ ! -f "$HOME/.ssh/vast_ai" ]; then
    echo "[!] SSH ключ для VAST.ai не найден: ~/.ssh/vast_ai"
    echo "    Скопируй свой ключ: cp /path/to/vast_ai ~/.ssh/vast_ai && chmod 600 ~/.ssh/vast_ai"
else
    echo "[OK] SSH ключ VAST.ai найден."
fi

echo ""
echo " ========================================"
echo "   Установка завершена!"
echo "   Запускай: bash start.sh"
echo " ========================================"
echo ""
