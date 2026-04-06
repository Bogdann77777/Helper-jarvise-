"""
vastai_generate_wedding.py — генерация 4 рекламных клипов свадебных причёсок
через VAST.ai serverless ComfyUI (Wan2.2 14B).

Workflow: 4 клипа × 5 сек, камера вокруг головы, чистый фон.
Результаты → Telegram. После завершения → уничтожить инстанс.

Запуск: python vastai_generate_wedding.py
"""

import sys
import time
import json
import copy
import requests
import tempfile
import os
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from tg_send import tg_msg, tg_video, tg_file

# ── Credentials ───────────────────────────────────────────────────────────────
try:
    from config import (
        VASTAI_API_KEY,
        S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
        S3_BUCKET_NAME, S3_ENDPOINT_URL, S3_REGION,
    )
except ImportError:
    VASTAI_API_KEY        = os.environ.get("VASTAI_API_KEY", "")
    S3_ACCESS_KEY_ID      = os.environ.get("S3_ACCESS_KEY_ID", "")
    S3_SECRET_ACCESS_KEY  = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    S3_BUCKET_NAME        = os.environ.get("S3_BUCKET_NAME", "multitalk-output-videos")
    S3_ENDPOINT_URL       = os.environ.get("S3_ENDPOINT_URL", "https://s3.eu-west-1.amazonaws.com")
    S3_REGION             = os.environ.get("S3_REGION", "eu-west-1")

ENDPOINT_NAME  = "265bt7dd"
ENDPOINT_ID    = 18187
INSTANCE_ID    = 33917364
VASTAI_API     = "https://console.vast.ai/api/v0"
ROUTE_URL      = "https://run.vast.ai/route/"

HEADERS = {"Authorization": f"Bearer {VASTAI_API_KEY}"}

# ── Wan2.2 base workflow ───────────────────────────────────────────────────────
BASE_WORKFLOW = {
    "37": {
        "inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"},
        "class_type": "UNETLoader"
    },
    "38": {
        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"},
        "class_type": "CLIPLoader"
    },
    "39": {
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        "class_type": "VAELoader"
    },
    "54": {
        "inputs": {"model": ["37", 0], "sampling": "v_prediction", "zsnr": False, "shift": 8.0},
        "class_type": "ModelSamplingSD3"
    },
    "55": {
        "inputs": {"model": ["56", 0], "sampling": "v_prediction", "zsnr": False, "shift": 8.0},
        "class_type": "ModelSamplingSD3"
    },
    "56": {
        "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"},
        "class_type": "UNETLoader"
    },
    "6": {
        "inputs": {"text": "", "clip": ["38", 0]},
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "clip": ["38", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "61": {
        "inputs": {"width": 832, "height": 480, "length": 81, "batch_size": 1},
        "class_type": "EmptyHunyuanLatentVideo"
    },
    "57": {
        "inputs": {
            "model": ["54", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["61", 0], "add_noise": "enable",
            "noise_seed": "__RANDOM_INT__",
            "control_after_generate": "randomize",
            "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable"
        },
        "class_type": "KSamplerAdvanced"
    },
    "58": {
        "inputs": {
            "model": ["55", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["57", 0], "add_noise": "disable",
            "noise_seed": 0, "control_after_generate": "fixed",
            "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 10, "end_at_step": 10000, "return_with_leftover_noise": "disable"
        },
        "class_type": "KSamplerAdvanced"
    },
    "8": {
        "inputs": {"samples": ["58", 0], "vae": ["39", 0]},
        "class_type": "VAEDecode"
    },
    "47": {
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "wedding_hair",
            "codec": "vp9", "fps": 16.0, "crf": 13
        },
        "class_type": "SaveWEBM"
    }
}

# ── 4 промпта (свадебные причёски) ────────────────────────────────────────────
CLIPS = [
    {
        "name": "Braided Crown",
        "prompt": (
            "Cinematic close-up portrait of a beautiful young woman with an elegant braided crown hairstyle, "
            "long dark hair intricately braided and woven into a royal halo crown pattern, adorned with small "
            "pearl and crystal pins sparkling softly. Camera slowly orbits 360 degrees around her head in smooth "
            "cinematic dolly motion. Pure white seamless studio background, soft diffused beauty lighting with "
            "subtle rim light, professional bridal advertisement quality, photorealistic, 8K, sharp focus, "
            "luxury hair salon commercial."
        ),
    },
    {
        "name": "French Twist",
        "prompt": (
            "Cinematic close-up portrait of a beautiful young woman with a sophisticated classic French twist "
            "updo hairstyle, silky dark brown hair elegantly swept up and twisted at the back, secured with "
            "delicate crystal and gold hairpins. Camera slowly orbits 360 degrees around her head in smooth "
            "cinematic motion. Soft warm cream seamless background, professional studio beauty lighting, "
            "luxury bridal advertisement quality, photorealistic, 8K, sharp details, elegant hair commercial."
        ),
    },
    {
        "name": "Romantic Waves",
        "prompt": (
            "Cinematic close-up portrait of a beautiful young woman with romantic long bridal wavy hairstyle, "
            "soft golden blonde waves cascading gracefully with delicate white floral pins and small roses "
            "woven throughout the hair. Camera slowly orbits 360 degrees around her in smooth gentle cinematic "
            "motion. Clean soft lavender-to-white gradient seamless background, warm diffused studio lighting, "
            "bridal beauty advertisement quality, photorealistic, 8K, dreamy elegant atmosphere."
        ),
    },
    {
        "name": "Sleek Low Chignon",
        "prompt": (
            "Cinematic close-up portrait of a beautiful young woman with a polished sleek low chignon bun "
            "hairstyle, jet black hair smoothly pulled back into a perfect low bun at the nape of the neck, "
            "secured with a delicate gold and pearl hairpiece that catches the light. Camera slowly orbits "
            "360 degrees around her head with smooth elegant cinematic motion. Neutral soft gray seamless "
            "background, professional ring beauty lighting, luxury upscale bridal advertisement quality, "
            "photorealistic, 8K, ultra sharp."
        ),
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def wait_for_worker(max_wait_min: int = 40) -> str | None:
    """Ждать пока воркер станет Ready, затем вернуть его URL."""
    deadline = time.time() + max_wait_min * 60
    attempt  = 0
    last_tg  = time.time()

    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.post(
                ROUTE_URL,
                headers={**HEADERS, "Accept": "application/json", "Content-Type": "application/json"},
                json={"endpoint": ENDPOINT_NAME, "cost": 100},
                timeout=30,
            )
            if r.ok:
                data = r.json()
                url  = data.get("url")
                if url:
                    print(f"[route] Worker ready: {url}")
                    return url
                status = data.get("status", "")
                print(f"[route] #{attempt} Not ready: {status}")
            else:
                print(f"[route] #{attempt} HTTP {r.status_code}")
        except Exception as e:
            print(f"[route] #{attempt} Exception: {e}")

        # Telegram-отчёт каждые 10 мин ожидания
        if time.time() - last_tg >= 600:
            elapsed_min = int((time.time() - (deadline - max_wait_min * 60)) / 60)
            tg_msg(f"⏳ Воркер загружается... {elapsed_min} мин\nОсталось ≤ {max_wait_min - elapsed_min} мин")
            last_tg = time.time()

        time.sleep(30)

    tg_msg("❌ Воркер не стал Ready за 40 мин — генерация отменена")
    return None


def get_route() -> str | None:
    """Получить URL воркера (один запрос, без ожидания)."""
    try:
        r = requests.post(
            ROUTE_URL,
            headers={**HEADERS, "Accept": "application/json", "Content-Type": "application/json"},
            json={"endpoint": ENDPOINT_NAME, "cost": 100},
            timeout=30,
        )
        if r.ok:
            return r.json().get("url")
        return None
    except Exception:
        return None


def submit_job(worker_url: str, prompt: str, clip_name: str) -> dict | None:
    """Отправить задачу на генерацию (sync)."""
    workflow = copy.deepcopy(BASE_WORKFLOW)
    workflow["6"]["inputs"]["text"] = prompt
    workflow["47"]["inputs"]["filename_prefix"] = f"wedding_{clip_name.lower().replace(' ', '_')}"

    payload = {
        "input": {
            "workflow_json": workflow,
            "s3": {
                "access_key_id":     S3_ACCESS_KEY_ID,
                "secret_access_key": S3_SECRET_ACCESS_KEY,
                "endpoint_url":      S3_ENDPOINT_URL,
                "bucket_name":       S3_BUCKET_NAME,
                "region":            S3_REGION,
            },
        }
    }

    generate_url = worker_url.rstrip("/") + "/generate/sync"
    print(f"[generate] POST {generate_url}")
    try:
        r = requests.post(generate_url, json=payload, timeout=1800)  # 30 min timeout
        if r.ok:
            return r.json()
        print(f"[generate] Error {r.status_code}: {r.text[:500]}")
        return None
    except Exception as e:
        print(f"[generate] Exception: {e}")
        return None


def download_video(url: str, path: Path) -> bool:
    """Скачать видео по presigned URL."""
    try:
        r = requests.get(url, timeout=120)
        if r.ok:
            path.write_bytes(r.content)
            return True
        return False
    except Exception as e:
        print(f"[download] Exception: {e}")
        return False


def destroy_instance():
    """Уничтожить инстанс после генерации."""
    try:
        r = requests.delete(
            f"{VASTAI_API}/instances/{INSTANCE_ID}/",
            headers=HEADERS,
            timeout=15,
        )
        return r.ok
    except Exception as e:
        print(f"[destroy] Exception: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_total = time.time()
    outputs_dir = _HERE / "outputs" / "wedding_hair"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tg_msg(
        "🎬 <b>Запуск генерации свадебных причёсок</b>\n"
        "4 клипа × 5 сек | Wan2.2 14B | RTX 5090\n"
        "Камера вокруг головы, чистый фон\n"
        "⏳ Жду воркер, потом генерирую..."
    )

    # Ждём воркер (до 40 мин)
    print("[main] Ждём воркер...")
    first_url = wait_for_worker(max_wait_min=40)
    if not first_url:
        return

    results = []
    failed  = []

    for idx, clip in enumerate(CLIPS, 1):
        clip_start = time.time()
        name = clip["name"]
        print(f"\n[{idx}/4] Генерация: {name}")

        # Отчёт начала
        tg_msg(f"🎥 <b>Клип {idx}/4: {name}</b>\nОтправляю задачу на генерацию...")

        # Получить route (воркер уже готов, просто берём URL)
        worker_url = first_url if idx == 1 else None
        if not worker_url:
            for attempt in range(5):
                worker_url = get_route()
                if worker_url:
                    break
                time.sleep(15)

        if not worker_url:
            msg = f"❌ <b>Клип {idx}/4: {name}</b> — не удалось получить worker URL"
            tg_msg(msg)
            failed.append(name)
            continue

        print(f"  Worker: {worker_url}")

        # Запустить генерацию (sync — блокирует до готовности)
        result = submit_job(worker_url, clip["prompt"], name)
        elapsed = int(time.time() - clip_start)

        if not result:
            msg = f"❌ <b>Клип {idx}/4: {name}</b> — ошибка генерации ({elapsed}с)"
            tg_msg(msg)
            failed.append(name)
            continue

        # Найти видео URL в результате
        output_files = result.get("output", {})
        video_urls = []
        for key, val in output_files.items():
            if isinstance(val, list):
                for item in val:
                    url = item.get("url") if isinstance(item, dict) else item
                    if isinstance(url, str) and url.startswith("http"):
                        video_urls.append(url)
            elif isinstance(val, str) and val.startswith("http"):
                video_urls.append(val)

        if not video_urls:
            # Результат пришёл, но структура другая — сохраним как JSON
            result_file = outputs_dir / f"{idx}_{name.replace(' ', '_')}_result.json"
            result_file.write_text(json.dumps(result, indent=2))
            tg_file(result_file, caption=f"Клип {idx} {name} — результат")
            results.append(name)
            print(f"  Результат сохранён в {result_file}")
            continue

        # Скачать и отправить видео
        for vidx, vurl in enumerate(video_urls):
            out_path = outputs_dir / f"{idx}_{name.replace(' ', '_')}.webm"
            ok = download_video(vurl, out_path)
            if ok:
                tg_video(out_path, caption=f"✅ Клип {idx}/4: {name} ({elapsed}с)")
                results.append(name)
                print(f"  Отправлено: {out_path}")
            else:
                tg_msg(f"⚠️ Клип {idx} скачан, но не отправлен. URL:\n{vurl}")

        # Прогресс-отчёт каждые 10 мин
        total_elapsed = int(time.time() - start_total)
        remaining_clips = 4 - idx
        if remaining_clips > 0:
            avg_per_clip = total_elapsed / idx
            eta_min = int(avg_per_clip * remaining_clips / 60)
            tg_msg(
                f"📊 Прогресс: {idx}/4 ({idx * 25}%)\n"
                f"Осталось: ~{eta_min} мин"
            )

    # ── Итог ─────────────────────────────────────────────────────────────────
    total_min = int((time.time() - start_total) / 60)
    summary = (
        f"🏁 <b>Генерация завершена</b> ({total_min} мин)\n"
        f"✅ Готово: {len(results)}/4\n"
    )
    if failed:
        summary += f"❌ Ошибки: {', '.join(failed)}\n"
    tg_msg(summary)

    # ── Уничтожить инстанс ────────────────────────────────────────────────────
    print("\n[destroy] Уничтожаю инстанс...")
    ok = destroy_instance()
    tg_msg(
        f"{'✅' if ok else '❌'} Инстанс {INSTANCE_ID} {'уничтожен — биллинг остановлен' if ok else 'не уничтожен — проверь вручную!'}"
    )
    print(f"[destroy] {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
