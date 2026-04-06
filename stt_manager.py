# stt_manager.py — faster-whisper, GPU, потоковая обработка аудио

import io
import logging
import tempfile
import os

from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)


class STTManager:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Загружает Whisper модель. При первом запуске скачивает автоматически."""
        logger.info(f"Загрузка Whisper модели: {WHISPER_MODEL} на {WHISPER_DEVICE}")
        try:
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("Whisper STT загружен успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {e}")
            raise

    def _reload_model(self):
        """Перезагрузить модель после CUDA-ошибки."""
        import gc
        import torch
        logger.warning("STT: перезагрузка модели после CUDA-ошибки...")
        try:
            del self.model
            self.model = None
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._load_model()

    def full_unload(self):
        """Полностью выгрузить модель из памяти (RAM + VRAM) для MultiTalk."""
        import gc
        import torch
        try:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("STT: модель полностью выгружена из памяти")
        except Exception as e:
            logger.error(f"STT full_unload error: {e}")

    def offload_to_cpu(self):
        """Перенести модель на CPU (освобождает VRAM для MultiTalk)."""
        import gc
        import torch
        if self.model is None:
            return
        try:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            self.model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
            )
            logger.info("STT: модель переведена на CPU")
        except Exception as e:
            logger.error(f"STT offload_to_cpu error: {e}")

    def reload_to_gpu(self):
        """Вернуть модель на GPU после MultiTalk."""
        import gc
        import torch
        try:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._load_model()
        logger.info("STT: модель возвращена на GPU")

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Принимает аудио в байтах (WebM/WAV/MP3 из браузера).
        Возвращает расшифрованный текст.
        """
        if self.model is None:
            raise RuntimeError("Модель STT не загружена")

        # Validate audio before processing — mobile browsers sometimes send corrupt/empty webm
        if len(audio_bytes) < 3000:
            raise ValueError(f"Audio too short ({len(audio_bytes)} bytes) — hold mic longer")
        if "webm" in mime_type and audio_bytes[:4] != b'\x1a\x45\xdf\xa3':
            raise ValueError("Invalid WebM data from browser — recording may have been cut off")

        # Определяем расширение по mime-type
        ext_map = {
            "audio/webm": ".webm",
            "audio/wav": ".wav",
            "audio/ogg": ".ogg",
            "audio/mp4": ".mp4",
            "audio/mpeg": ".mp3",
        }
        ext = ext_map.get(mime_type, ".webm")

        # Сохраняем во временный файл (faster-whisper требует путь к файлу)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            logger.info(f"Транскрибирую аудио: {len(audio_bytes)} байт, формат={mime_type}")

            try:
                segments, info = self.model.transcribe(
                    tmp_path,
                    language=None,            # auto-detect (RU/UK/EN)
                    vad_filter=True,          # фильтрует тишину автоматически
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                    },
                    beam_size=5,
                    word_timestamps=False,
                )
            except Exception as e:
                err_str = str(e)
                if "CUDA" in err_str or "cuda" in err_str:
                    # CUDA context corruption — reload model and retry once
                    logger.warning(f"STT: CUDA ошибка, перезагружаю модель: {e}")
                    self._reload_model()
                    try:
                        segments, info = self.model.transcribe(
                            tmp_path,
                            language=None,
                            vad_filter=True,
                            vad_parameters={"min_silence_duration_ms": 500},
                            beam_size=5,
                            word_timestamps=False,
                        )
                    except Exception as e2:
                        raise RuntimeError(f"STT недоступен (CUDA ошибка, перезагрузка не помогла): {e2}") from e2
                else:
                    raise ValueError(f"Ошибка распознавания аудио: {e}") from e

            # Собираем текст из сегментов
            text = " ".join(segment.text.strip() for segment in segments).strip()

            logger.info(
                f"Распознано ({info.language}, {info.duration:.1f}s): "
                f"{text[:100]}{'...' if len(text) > 100 else ''}"
            )
            return text

        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def transcribe_stream(self, audio_chunks: list[bytes], mime_type: str = "audio/webm") -> str:
        """
        Принимает список аудио-кусков (накопленных во время записи),
        склеивает их и транскрибирует как единый файл.
        Вызывается когда детектирована 5-секундная тишина.
        """
        full_audio = b"".join(audio_chunks)
        return self.transcribe(full_audio, mime_type)


# Глобальный экземпляр
stt_manager: STTManager | None = None


def get_stt_manager() -> "STTManager | RemoteSTTProxy":
    """Returns RemoteSTTProxy when USE_REMOTE_AI=True, else local STTManager."""
    try:
        from config import USE_REMOTE_AI, STT_REMOTE_URL
    except ImportError:
        USE_REMOTE_AI = False
        STT_REMOTE_URL = ""

    if USE_REMOTE_AI:
        return RemoteSTTProxy(STT_REMOTE_URL)

    global stt_manager
    if stt_manager is None:
        stt_manager = STTManager()
    return stt_manager


class RemoteSTTProxy:
    """Drop-in replacement for STTManager that routes to remote HTTP server."""

    def __init__(self, base_url: str):
        self._url = base_url.rstrip("/")

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        import requests as _req
        ext_map = {
            "audio/webm": ".webm", "audio/wav": ".wav",
            "audio/ogg": ".ogg",  "audio/mp4": ".mp4", "audio/mpeg": ".mp3",
        }
        ext = ext_map.get(mime_type, ".webm")
        resp = _req.post(
            f"{self._url}/transcribe",
            files={"file": (f"audio{ext}", audio_bytes, mime_type)},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    def offload_to_cpu(self):
        pass   # model lives on remote machine

    def full_unload(self):
        pass

    def reload_to_gpu(self):
        pass

    def transcribe_stream(self, audio_chunks: list, mime_type: str = "audio/webm") -> str:
        combined = b"".join(audio_chunks)
        return self.transcribe(combined, mime_type)
