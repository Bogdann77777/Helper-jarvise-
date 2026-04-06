@echo off
echo =============================================
echo  Inference Server (PC2 - RTX 3070 Ti 8GB)
echo  STT: Whisper large-v3 (GPU)
echo  TTS: Qwen3TTS 1.7B (GPU)
echo  Port: 8010
echo =============================================
echo.

cd /d "E:\project\ceo 2.0\cli helper"
call .venv\Scripts\activate

set INFERENCE_HOST=0.0.0.0
set INFERENCE_PORT=8010
set CUDA_VISIBLE_DEVICES=0

python inference_server/server.py
pause
