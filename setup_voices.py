# setup_voices.py — генерация стандартных русских голосов для XTTS v2
#
# Использует Microsoft Edge TTS (бесплатно, не требует API ключа).
# Запусти один раз: python setup_voices.py
#
# После выполнения в папке voices/ появятся:
#   svetlana.wav  — женский голос (мягкий)
#   dariya.wav    — женский голос (нейтральный)
#   dmitry.wav    — мужской голос

import asyncio
import os
import sys

try:
    import edge_tts
except ImportError:
    print("Устанавливаю edge-tts...")
    os.system(f"{sys.executable} -m pip install edge-tts")
    import edge_tts

try:
    import torchaudio
    import torch
except ImportError:
    print("Устанавливаю torch/torchaudio...")
    os.system(f"{sys.executable} -m pip install torch torchaudio")
    import torchaudio
    import torch

VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Текст для генерации эталонного образца голоса
# Достаточно длинный чтобы CosyVoice мог хорошо захватить характеристики
SAMPLE_TEXT = (
    "Добрый день! Меня зовут Клод. "
    "Я готов помочь вам с любыми вопросами. "
    "Спрашивайте — я постараюсь дать подробный и понятный ответ. "
    "Вместе мы разберёмся с любой задачей."
)

# Список голосов: (имя_файла, edge_tts_voice_id, описание)
DEFAULT_VOICES = [
    ("svetlana", "ru-RU-SvetlanaNeural",  "Женский, мягкий (Microsoft Svetlana)"),
    ("dariya",   "ru-RU-DariyaNeural",    "Женский, нейтральный (Microsoft Dariya)"),
    ("dmitry",   "ru-RU-DmitryNeural",    "Мужской (Microsoft Dmitry)"),
]


async def generate_voice(name: str, voice_id: str, description: str):
    out_mp3 = os.path.join(VOICES_DIR, f"{name}.mp3")
    out_wav = os.path.join(VOICES_DIR, f"{name}.wav")

    if os.path.exists(out_wav):
        print(f"  ✓ {name}.wav уже существует, пропускаю")
        return

    print(f"  Генерирую {name}.wav ({description})...")

    # Генерируем через Edge TTS → MP3
    communicate = edge_tts.Communicate(SAMPLE_TEXT, voice_id, rate="-5%")
    await communicate.save(out_mp3)

    # Конвертируем MP3 → WAV 22050Hz mono (XTTS v2 reference format)
    waveform, sr = torchaudio.load(out_mp3)
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    torchaudio.save(out_wav, waveform, 22050)
    os.remove(out_mp3)

    print(f"  ✓ {name}.wav создан")


async def main():
    print(f"\nГенерация стандартных русских голосов в: {VOICES_DIR}\n")
    for name, voice_id, description in DEFAULT_VOICES:
        await generate_voice(name, voice_id, description)

    print(f"\n✅ Готово! Голоса доступны:")
    for name, _, desc in DEFAULT_VOICES:
        wav = os.path.join(VOICES_DIR, f"{name}.wav")
        if os.path.exists(wav):
            print(f"   • {name:12s} — {desc}")

    print(f"\nДобавить свой голос: положи WAV файл (10-30 сек) в папку voices/")
    print(f"Он автоматически появится в выпадающем списке в интерфейсе.\n")


if __name__ == "__main__":
    asyncio.run(main())
