"""
transcribe_audio.py — транскрибация аудио через faster-whisper (GPU).

Использование как инструмент:
    from transcribe_audio import transcribe

    result = transcribe("path/to/audio.wav")
    print(result["text"])
    print(result["language"])
    print(result["duration_sec"])

CLI:
    python transcribe_audio.py audio.wav
    python transcribe_audio.py audio.wav --model large-v3
    python transcribe_audio.py audio.wav --language ru
"""

import sys
import time
from pathlib import Path

# Модели по размеру / скорости:
#   tiny   — ~1 ГБ VRAM, очень быстро, качество ок
#   base   — ~1 ГБ VRAM, быстро, хорошее качество
#   small  — ~2 ГБ VRAM, хорошо
#   medium — ~5 ГБ VRAM, отлично
#   large-v3 — ~10 ГБ VRAM, максимальное качество
DEFAULT_MODEL = "base"
DEVICE = "cuda"       # "cuda" для GPU, "cpu" для CPU
COMPUTE = "float16"   # float16 быстрее на NVIDIA


def transcribe(
    audio_path: str,
    model_size: str = DEFAULT_MODEL,
    language: str = None,
    device: str = DEVICE,
) -> dict:
    """
    Транскрибирует аудиофайл.

    Args:
        audio_path:  путь к WAV / MP3 / любому аудио
        model_size:  tiny / base / small / medium / large-v3
        language:    "ru" / "en" / None (авто-детект)
        device:      "cuda" или "cpu"

    Returns:
        {
            "text": str,           — полный транскрипт
            "language": str,       — определённый язык (ru, en, …)
            "language_prob": float,— уверенность в языке
            "duration_sec": float, — длина аудио в секундах
            "segments": list,      — список сегментов с таймкодами
            "model": str,
            "elapsed_sec": float,
        }
    """
    from faster_whisper import WhisperModel

    audio_path = str(Path(audio_path).resolve())

    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=COMPUTE)

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,   # фильтр тишины — точнее на фрагментах без речи
    )

    segments = []
    texts = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })
        texts.append(seg.text.strip())

    elapsed = round(time.time() - t0, 1)

    return {
        "text":          "\n".join(texts),
        "language":      info.language,
        "language_prob": round(info.language_probability, 3),
        "duration_sec":  round(info.duration, 1),
        "segments":      segments,
        "model":         model_size,
        "elapsed_sec":   elapsed,
    }


def transcribe_to_text(audio_path: str, model_size: str = DEFAULT_MODEL) -> str:
    """Быстрый вызов — возвращает просто строку с транскриптом."""
    result = transcribe(audio_path, model_size=model_size)
    return result["text"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Транскрибация аудио через Whisper")
    parser.add_argument("audio", help="Путь к аудиофайлу")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Размер модели (default: base)")
    parser.add_argument("--language", default=None,
                        help="Язык: ru / en / … (default: авто)")
    parser.add_argument("--segments", action="store_true",
                        help="Показать сегменты с таймкодами")
    args = parser.parse_args()

    print(f"Транскрибирую: {args.audio}")
    print(f"Модель: {args.model} | Язык: {args.language or 'авто'}")
    print("-" * 60)

    result = transcribe(args.audio, model_size=args.model, language=args.language)

    print(f"Язык: {result['language']} (уверенность: {result['language_prob']:.1%})")
    print(f"Длина: {result['duration_sec']}с | Время обработки: {result['elapsed_sec']}с")
    print()

    if args.segments:
        for seg in result["segments"]:
            print(f"  [{seg['start']:6.1f} → {seg['end']:6.1f}]  {seg['text']}")
    else:
        print(result["text"])
