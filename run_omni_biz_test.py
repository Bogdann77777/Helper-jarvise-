"""
run_omni_biz_test.py — OmniAvatar тест деловой стиль.
Извлекает кадры каждые 0.5 сек и проверяет качество.
Авто-отправка в Telegram.
"""

import sys, time, json, wave
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "E:/project/ceo 2.0/cli helper")
from omni_avatar_tool import generate_avatar_video
from tg_send import tg_msg, tg_video, tg_photo
from extract_video_frames import extract_frames

# ── Входные файлы ────────────────────────────────────────────────────────────
IMAGE = "E:/project/ceo 2.0/cli helper/static/uploads/46275f59_asian-business-woman-office-worker-260nw-2687073433.jpg"
AUDIO = "E:/project/ceo 2.0/cli helper/static/audio/tts_5301123b.wav"

# ── Деловой промпт (главный рычаг OmniAvatar) ────────────────────────────────
PROMPT_BUSINESS = (
    "A professional Asian businesswoman sitting at a white desk in a bright modern office. "
    "She is speaking directly to the camera with confident eye contact and natural composure. "
    "Subtle professional hand gestures on the desk, straight posture, business jacket. "
    "Smooth natural head movements while speaking, no exaggerated expressions. "
    "Steady camera, sharp focus, realistic high-quality video."
)

# ── Настройки ─────────────────────────────────────────────────────────────────
TEST_CONFIGS = [
    {
        "id": "OA_BIZ_1",
        "label": "Деловой: guidance=5.0, audio=4.5, overlap=1",
        "num_steps":     20,
        "guidance_scale": 5.0,   # строго следует промпту → контролируемый
        "audio_scale":    4.5,   # чёткий но не перебор
        "overlap_frame":  1,     # VALID: overlap % 4 == 1. допустимо: 1, 5, 9...
        "prompt":         PROMPT_BUSINESS,
    },
    {
        "id": "OA_BIZ_2",
        "label": "Деловой: guidance=6.0, audio=5.0, overlap=5",
        "num_steps":     20,
        "guidance_scale": 6.0,   # максимально следует промпту
        "audio_scale":    5.0,   # стандартный lipSync
        "overlap_frame":  5,     # VALID: 5 % 4 == 1. 5 кадров = 0.2с перекрытия → плавнее
        "prompt":         PROMPT_BUSINESS,
    },
]


def check_audio_duration(path):
    with wave.open(path) as f:
        return f.getnframes() / f.getframerate()


def run_test(cfg: dict, test_num: int, total: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  OmniAvatar {cfg['id']} ({test_num}/{total})")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    tg_msg(
        f"🎬 OmniAvatar {cfg['id']} ({test_num}/{total})\n\n"
        f"📝 {cfg['label']}\n"
        f"  guidance={cfg['guidance_scale']}, audio={cfg['audio_scale']}\n"
        f"  steps={cfg['num_steps']}, overlap={cfg['overlap_frame']}\n"
        f"⏳ ~19 мин"
    )

    t0 = time.time()
    result = generate_avatar_video(
        image_path      = IMAGE,
        audio_path      = AUDIO,
        prompt          = cfg["prompt"],
        output_name     = cfg["id"].lower() + "_" + datetime.now().strftime("%H%M%S"),
        num_steps       = cfg["num_steps"],
        guidance_scale  = cfg["guidance_scale"],
        audio_scale     = cfg["audio_scale"],
        max_tokens      = 15000,
        overlap_frame   = cfg["overlap_frame"],
    )
    elapsed = time.time() - t0
    mins, secs = int(elapsed//60), int(elapsed%60)

    entry = {
        "id":             cfg["id"],
        "label":          cfg["label"],
        "guidance_scale": cfg["guidance_scale"],
        "audio_scale":    cfg["audio_scale"],
        "num_steps":      cfg["num_steps"],
        "overlap_frame":  cfg["overlap_frame"],
        "duration_sec":   elapsed,
        "status":         "ok" if result["success"] else "error",
        "video_path":     result.get("video_path"),
        "error":          result.get("error"),
        "timestamp":      datetime.now().isoformat(),
    }

    if not result["success"]:
        print(f"  ❌ ОШИБКА: {result['error']}")
        tg_msg(f"❌ OmniAvatar {cfg['id']} ОШИБКА:\n{result['error']}")
        return entry

    video_path = result["video_path"]
    print(f"  ✅ Готово за {mins}м {secs}с → {video_path}")

    # ── Извлекаем кадры каждые 0.5 сек для качественной проверки ────────────
    frames = extract_frames(video_path, every_n_sec=0.5, max_frames=24)
    print(f"  Извлечено {len(frames)} кадров (каждые 0.5 сек)")

    # Отправляем видео в Telegram
    tg_msg(
        f"✅ OmniAvatar {cfg['id']} ГОТОВО!\n\n"
        f"⏱ {mins}м {secs}с\n"
        f"⚙️ guidance={cfg['guidance_scale']}, audio={cfg['audio_scale']}\n"
        f"📊 {len(frames)} кадров проверено\n"
        f"📁 {Path(video_path).name}"
    )
    tg_video(video_path, caption=f"OmniAvatar {cfg['id']} — {cfg['label']}")

    # Отправляем ключевые кадры (0с, 2с, 4с, 6с, 8с, 10с) для проверки качества
    key_frame_indices = [0, 4, 8, 12, 16, 20]  # каждые 2 сек из 24 кадров
    for i in key_frame_indices:
        if i < len(frames):
            sec = i * 0.5
            tg_photo(frames[i], caption=f"{cfg['id']} — {sec:.1f}s")
            time.sleep(0.5)  # не спамить Telegram

    return entry


def main():
    print("\n" + "="*60)
    print("  OmniAvatar — Деловой стиль")
    print("="*60)
    print(f"  Аудио: {check_audio_duration(AUDIO):.1f}s")
    print(f"  Тестов: {len(TEST_CONFIGS)}")

    log_path = Path("E:/project/ceo 2.0/cli helper/omni_biz_test_log.json")
    entries = []

    for i, cfg in enumerate(TEST_CONFIGS, 1):
        entry = run_test(cfg, i, len(TEST_CONFIGS))
        entries.append(entry)

        log_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if i < len(TEST_CONFIGS):
            print("  ⏸ Пауза 15с перед следующим тестом...")
            time.sleep(15)

    # ── Итог ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ИТОГ")
    print("="*60)
    ok = [e for e in entries if e["status"] == "ok"]
    print(f"  Успешно: {len(ok)}/{len(entries)}")
    for e in ok:
        dur = int(e["duration_sec"]//60)
        print(f"  {e['id']} | guid={e['guidance_scale']} audio={e['audio_scale']} | {dur}м")

    tg_msg(
        f"🏁 OmniAvatar деловой тест завершён\n\n"
        f"Успешно: {len(ok)}/{len(entries)}\n\n"
        f"Жду твою оценку видео!"
    )


if __name__ == "__main__":
    main()
