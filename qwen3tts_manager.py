# qwen3tts_manager.py — Qwen3-TTS 1.7B + CUDA Graph acceleration
#
# Запуск: E:/project/qwen3tts_env/Scripts/python.exe qwen3tts_manager.py [text] [--mode custom|clone]
# Импорт: from qwen3tts_manager import get_qwen3tts_manager
#
# Лог-файл: E:/project/ceo 2.0/cli helper/qwen3tts.log
# Движок: Qwen3TTSModel (qwen-tts, CPU mode) — FasterQwen3TTS временно отключён для освобождения VRAM

import os
import sys
import time
import uuid
import logging
import logging.handlers

# ─── ENV ────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT, "Qwen3TTS"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTHONIOENCODING",     "utf-8")

# ─── PATHS ──────────────────────────────────────────────────────────────────
LOG_FILE    = os.path.join(_HERE, "qwen3tts.log")
OUTPUT_DIR  = os.path.join(_HERE, "outputs", "qwen3tts")
AUDIO_DIR   = os.path.join(_HERE, "static", "audio")
VOICES_DIR  = os.path.join(_HERE, "voices")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,  exist_ok=True)

MODEL_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_BASE   = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE       = "cpu"   # cpu — освобождаем VRAM для MultiTalk (qwen_tts поддерживает CPU)

# ─── LOGGING ────────────────────────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    log = logging.getLogger("qwen3tts")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log

logger = _setup_logger()


# ─── MANAGER ────────────────────────────────────────────────────────────────
class Qwen3TTSManager:
    """
    Менеджер Qwen3-TTS 1.7B с CUDA Graph ускорением.

    Движок: Qwen3TTSModel (qwen-tts 0.1.1, CPU mode)
    RTF: ~0.42 (генерация 15с аудио за ~6с)
    Ускорение vs SDPA baseline: 3.3×

    Режимы:
      custom — эмоции + многоязычность (CustomVoice модель)
               speakers: Ryan (EN), Aiden (EN), Vivian (ZH), Serena (ZH)
      clone  — zero-shot клон голоса (Base модель)
               требует ref_audio + ref_text

    GPU: CUDA_VISIBLE_DEVICES=1, device=cuda:0
    Лог: qwen3tts.log (ротация 5MB × 3)
    """

    def __init__(self):
        self._model       = None
        self._loaded_mode = None   # "custom" | "clone"
        self._recovering  = False
        import torch
        self._torch = torch

    # ── internal ──────────────────────────────────────────────────────────

    def _ensure_loaded(self, mode: str = "custom"):
        if self._loaded_mode == mode and self._model is not None:
            return
        if self._loaded_mode != mode and self._model is not None:
            logger.info(f"[LOAD] Смена режима {self._loaded_mode}→{mode}, выгружаем модель")
            self._unload()

        model_id = MODEL_CUSTOM if mode == "custom" else MODEL_BASE
        logger.info(f"[LOAD] Qwen3TTSModel({model_id}) | device={DEVICE} (CPU mode)")
        t0 = time.time()
        try:
            from qwen_tts import Qwen3TTSModel
            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                device = DEVICE,
            )
            elapsed = time.time() - t0
            self._loaded_mode = mode
            logger.info(f"[LOAD] ✓ Готово за {elapsed:.1f}с (включая CUDA graph warmup)")
        except Exception as e:
            logger.error(f"[LOAD] ОШИБКА: {e}", exc_info=True)
            self._model = None
            raise

    def _unload(self):
        try:
            del self._model
        except Exception:
            pass
        self._model = None
        self._loaded_mode = None
        import gc
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        logger.info("[UNLOAD] Модель выгружена, VRAM освобождена")

    def _cuda_recover(self, mode: str):
        if self._recovering:
            logger.warning("[RECOVER] Уже идёт recovery, пропускаем")
            return
        self._recovering = True
        logger.warning("[RECOVER] CUDA-ошибка — перезагружаем модель...")
        try:
            self._unload()
            self._ensure_loaded(mode)
            logger.info("[RECOVER] ✓ Успешно восстановлено")
        except Exception as e:
            logger.error(f"[RECOVER] Не удалось восстановить: {e}", exc_info=True)
            raise
        finally:
            self._recovering = False

    def _save_wav(self, wavs, sr: int, prefix: str = "qwen") -> str:
        import soundfile as sf
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        sf.write(filepath, wavs[0], sr)
        return f"/static/audio/{filename}"

    # ── public ────────────────────────────────────────────────────────────

    def synthesize_custom(
        self,
        text:     str,
        speaker:  str = "Ryan",
        language: str = "Russian",
        instruct: str = "Speak in a warm, professional Russian voice.",
        save_to:  str | None = None,
    ) -> dict:
        """
        CustomVoice: синтез с эмоциями и фиксированным спикером.
        Returns: {url, file, gen_sec, audio_dur, rtf}
        """
        mode = "custom"
        logger.info(f"[GEN] custom | speaker={speaker} lang={language} text={len(text)}ч «{text[:60]}»")
        self._ensure_loaded(mode)
        t0 = time.time()
        try:
            wavs, sr = self._model.generate_custom_voice(
                text=text, language=language,
                speaker=speaker, instruct=instruct,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.error(f"[GEN] CUDA-ошибка: {e}")
                self._cuda_recover(mode)
                wavs, sr = self._model.generate_custom_voice(
                    text=text, language=language,
                    speaker=speaker, instruct=instruct,
                )
            else:
                logger.error(f"[GEN] Ошибка: {e}", exc_info=True)
                raise

        gen_sec   = time.time() - t0
        audio_dur = len(wavs[0]) / sr
        rtf       = gen_sec / audio_dur if audio_dur > 0 else 0

        if save_to:
            import soundfile as sf
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            sf.write(save_to, wavs[0], sr)
            url = save_to
        else:
            url = self._save_wav(wavs, sr, prefix="qcv")

        logger.info(
            f"[GEN] ✓ custom | gen={gen_sec:.2f}с audio={audio_dur:.2f}с "
            f"RTF={rtf:.3f} → {os.path.basename(url)}"
        )
        return {"url": url, "file": url, "gen_sec": round(gen_sec, 2),
                "audio_dur": round(audio_dur, 2), "rtf": round(rtf, 3)}

    def synthesize_clone(
        self,
        text:      str,
        ref_audio: str,
        ref_text:  str,
        language:  str = "Russian",
        save_to:   str | None = None,
    ) -> dict:
        """
        Base / zero-shot клонирование голоса.
        ref_text — ТОЧНАЯ транскрипция ref_audio (обязательно!)
        """
        mode = "clone"
        if not ref_text:
            raise ValueError("[GEN] ref_text обязателен для режима clone")
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"[GEN] ref_audio не найден: {ref_audio}")

        logger.info(
            f"[GEN] clone | lang={language} ref={os.path.basename(ref_audio)} "
            f"text={len(text)}ч «{text[:60]}»"
        )
        self._ensure_loaded(mode)
        t0 = time.time()
        try:
            wavs, sr = self._model.generate_voice_clone(
                text=text, language=language,
                ref_audio=ref_audio, ref_text=ref_text,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.error(f"[GEN] CUDA-ошибка: {e}")
                self._cuda_recover(mode)
                wavs, sr = self._model.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=ref_audio, ref_text=ref_text,
                )
            else:
                logger.error(f"[GEN] Ошибка: {e}", exc_info=True)
                raise

        gen_sec   = time.time() - t0
        audio_dur = len(wavs[0]) / sr
        rtf       = gen_sec / audio_dur if audio_dur > 0 else 0

        if save_to:
            import soundfile as sf
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            sf.write(save_to, wavs[0], sr)
            url = save_to
        else:
            url = self._save_wav(wavs, sr, prefix="qcl")

        logger.info(
            f"[GEN] ✓ clone | gen={gen_sec:.2f}с audio={audio_dur:.2f}с "
            f"RTF={rtf:.3f} → {os.path.basename(url)}"
        )
        return {"url": url, "file": url, "gen_sec": round(gen_sec, 2),
                "audio_dur": round(audio_dur, 2), "rtf": round(rtf, 3)}

    def status(self) -> dict:
        try:
            vram_mb = (
                self._torch.cuda.memory_allocated(0) // 1024 // 1024
                if self._torch.cuda.is_available() else 0
            )
        except Exception:
            vram_mb = -1
        return {
            "loaded":     self._model is not None,
            "mode":       self._loaded_mode,
            "recovering": self._recovering,
            "engine":     "Qwen3TTSModel (CPU mode)",
            "vram_mb":    vram_mb,
            "log_file":   LOG_FILE,
        }

    def unload(self):
        self._unload()


# ─── SINGLETON ──────────────────────────────────────────────────────────────
_manager: Qwen3TTSManager | None = None

def get_qwen3tts_manager() -> Qwen3TTSManager:
    global _manager
    if _manager is None:
        _manager = Qwen3TTSManager()
    return _manager


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3-TTS CLI (CUDA Graph engine)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--mode",      choices=["custom", "clone"], default="custom")
    parser.add_argument("--speaker",   default="Ryan")
    parser.add_argument("--language",  default="Russian")
    parser.add_argument("--instruct",  default="Speak in a warm, professional Russian voice.")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text",  default=None)
    parser.add_argument("--out",       default=None)
    parser.add_argument("--status",    action="store_true")
    args = parser.parse_args()

    mgr = get_qwen3tts_manager()

    if args.status:
        import json
        print(json.dumps(mgr.status(), indent=2, ensure_ascii=False))
        sys.exit(0)

    text = args.text or "Добро пожаловать. Система синтеза речи Qwen три работает корректно."

    if args.mode == "custom":
        out = args.out or os.path.join(OUTPUT_DIR, f"cli_{uuid.uuid4().hex[:6]}.wav")
        result = mgr.synthesize_custom(
            text=text, speaker=args.speaker,
            language=args.language, instruct=args.instruct,
            save_to=out,
        )
    else:
        ref_audio = args.ref_audio or os.path.join(VOICES_DIR, "olena.wav")
        ref_text  = args.ref_text or (
            "Особенно заметно изменение в бизнесе. Компания используют автоматизацию "
            "для маркетинга, анализа клиентов и создание контента. "
            "Это позволяет быстрее тестировать идеи и находить эффективные стратегии."
        )
        out = args.out or os.path.join(OUTPUT_DIR, f"cli_clone_{uuid.uuid4().hex[:6]}.wav")
        result = mgr.synthesize_clone(
            text=text, ref_audio=ref_audio,
            ref_text=ref_text, language=args.language,
            save_to=out,
        )

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
