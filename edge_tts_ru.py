"""
edge_tts_ru.py — edge-tts Russian TTS для CLI Helper.

Использует Microsoft Azure TTS через edge-tts (бесплатно, облако).
Голоса: Svetlana (женский), Dmitry (мужской).
Сохраняет MP3 в static/audio/, возвращает URL.
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path

import edge_tts

logger = logging.getLogger(__name__)

_HERE      = Path(__file__).parent
_AUDIO_DIR = _HERE / "static" / "audio"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Доступные русские голоса
EDGE_VOICES_RU = {
    "svetlana": "ru-RU-SvetlanaNeural",   # женский, нейтральный
    "dmitry":   "ru-RU-DmitryNeural",     # мужской, нейтральный
}

_current_voice: str = "svetlana"


def set_voice(name: str):
    global _current_voice
    if name in EDGE_VOICES_RU:
        _current_voice = name


async def synthesize(text: str, voice: str | None = None, rate: str = "-5%") -> dict:
    """
    Синтез речи через edge-tts.

    Args:
        text:  текст для озвучки
        voice: ключ голоса ("svetlana" / "dmitry") или None — берёт текущий
        rate:  скорость речи (+10%, -10%, ...)

    Returns:
        {"ok": True, "url": "/static/audio/xxx.mp3", "file": "..."}
    """
    voice_id = EDGE_VOICES_RU.get(voice or _current_voice, EDGE_VOICES_RU["svetlana"])
    filename  = f"edge_{uuid.uuid4().hex[:10]}.mp3"
    filepath  = _AUDIO_DIR / filename
    url       = f"/static/audio/{filename}"

    try:
        communicate = edge_tts.Communicate(text, voice_id, rate=rate)
        await communicate.save(str(filepath))
        logger.info(f"[edge-tts] ✓ {voice_id} | {len(text)}ч → {filename}")
        return {"ok": True, "url": url, "file": str(filepath)}
    except Exception as e:
        logger.error(f"[edge-tts] ✗ {e}")
        return {"ok": False, "error": str(e), "url": ""}
