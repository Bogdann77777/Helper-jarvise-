# xtts_manager.py — XTTS v2 TTS for all modes (Board voices + general voices)

import glob as globmod
import os
import re
import uuid
import logging
from typing import BinaryIO, Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
#  torchaudio 2.10 removed backend system and hardcoded torchcodec (requires FFmpeg).
#  Monkey-patch torchaudio.load/save to use soundfile instead.
# ---------------------------------------------------------------------------

def _soundfile_load(
    uri: Union[BinaryIO, str, os.PathLike],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    stop = None if num_frames <= 0 else frame_offset + num_frames
    data, sample_rate = sf.read(uri, start=frame_offset, stop=stop,
                                dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data)
    if channels_first:
        waveform = waveform.T
    return waveform, sample_rate


def _soundfile_save(
    uri: Union[str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[Union[float, int]] = None,
) -> None:
    if src.dtype != torch.float32:
        src = src.float()
    if channels_first and src.ndim == 2:
        src = src.T
    sf.write(uri, src.cpu().numpy(), sample_rate)


torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save

# ---------------------------------------------------------------------------

from config import (
    XTTS_DEVICE,
    XTTS_VOICES_DIR,
    XTTS_VOICE_CSO,
    XTTS_VOICE_CFO,
    XTTS_VOICE_CTO,
    XTTS_VOICE_CEO,
    XTTS_LANGUAGE,
    XTTS_DEFAULT_VOICE,
    VOICES_DIR,
    AUDIO_DIR,
    AUDIO_MAX_FILES,
)

logger = logging.getLogger(__name__)


def _cleanup_audio_files() -> None:
    """Удаляет старые TTS файлы, оставляя только последние AUDIO_MAX_FILES."""
    files = sorted(
        globmod.glob(os.path.join(AUDIO_DIR, "tts_*.wav")),
        key=os.path.getmtime
    )
    while len(files) > AUDIO_MAX_FILES:
        try:
            os.remove(files.pop(0))
        except OSError:
            pass


# Board role → reference WAV path
ROLE_VOICE_MAP = {
    "cso": XTTS_VOICE_CSO,
    "cfo": XTTS_VOICE_CFO,
    "cto": XTTS_VOICE_CTO,
    "ceo": XTTS_VOICE_CEO,
}


def _detect_language(text: str) -> str:
    """Detect language from text. Cyrillic → 'ru', else 'en'."""
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return "ru" if cyrillic > len(text) * 0.3 else "en"


def _split_sentences(text: str, max_length: int = 350) -> list[str]:
    """Split text at sentence boundaries, respecting max_length."""
    if len(text) <= max_length:
        return [text]

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?;])\s+', text)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + 1 > max_length and current:
            chunks.append(current.strip())
            current = part
        else:
            current = (current + " " + part).strip() if current else part

    if current.strip():
        chunks.append(current.strip())

    # If no splits happened (no punctuation), hard-cut
    if not chunks:
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    return chunks


class XTTSManager:
    def __init__(self):
        self.model = None
        self._loaded = False
        self.current_voice = XTTS_DEFAULT_VOICE
        self._recovering = False
        import threading
        self._tts_lock = threading.Lock()  # serialize model calls (not thread-safe)

    def _ensure_loaded(self):
        """Lazy-load XTTS model on first use."""
        if self._loaded:
            return

        from TTS.api import TTS

        # PyTorch 2.6+ defaults weights_only=True, but TTS 0.22 doesn't pass weights_only=False.
        _original_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_load(*args, **kwargs)
        torch.load = _patched_load

        device = XTTS_DEVICE if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading XTTS v2 on device={device}...")

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self._loaded = True
        logger.info("XTTS v2 loaded successfully")

    def _cuda_recover(self):
        """Reload XTTS model after a CUDA error to restore a clean GPU state."""
        if self._recovering:
            return
        self._recovering = True
        logger.warning("XTTS: CUDA error detected — reloading model...")
        try:
            del self.model
            self.model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self._loaded = False
            self._ensure_loaded()
            logger.info("XTTS: CUDA recovery successful")
        except Exception as e:
            logger.error(f"XTTS: CUDA recovery failed: {e}")
            raise
        finally:
            self._recovering = False

    def full_unload(self):
        """Полностью выгрузить модель из памяти (RAM + VRAM) для MultiTalk."""
        try:
            if self.model is not None:
                self.model.to("cpu")
            del self.model
            self.model = None
            self._loaded = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("XTTS: модель полностью выгружена из памяти")
        except Exception as e:
            logger.error(f"XTTS full_unload error: {e}")

    def offload_to_cpu(self):
        """Перенести модель на CPU (освобождает VRAM для MultiTalk)."""
        if not self._loaded or self.model is None:
            return
        try:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("XTTS: модель переведена на CPU")
        except Exception as e:
            logger.error(f"XTTS offload_to_cpu error: {e}")

    def reload_to_gpu(self):
        """Вернуть модель на GPU после MultiTalk."""
        if not self._loaded or self.model is None:
            return
        try:
            device = XTTS_DEVICE if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            logger.info(f"XTTS: модель возвращена на {device}")
        except Exception as e:
            logger.error(f"XTTS reload_to_gpu error: {e}")

    def _tts_call(self, text: str, voice_path: str, lang: str) -> list:
        """Call model.tts() with automatic CUDA recovery on failure. Thread-safe via lock."""
        with self._tts_lock:
            if self.model is None:
                logger.warning("XTTS: model is None, attempting recovery before call...")
                self._cuda_recover()
            try:
                return self.model.tts(text=text, speaker_wav=voice_path, language=lang)
            except RuntimeError as e:
                if "CUDA" not in str(e) and "cuda" not in str(e):
                    raise
                self._cuda_recover()
                return self.model.tts(text=text, speaker_wav=voice_path, language=lang)

    # ------------------------------------------------------------------
    #  Voice management (general voices in voices/, board voices in voices/board/)
    # ------------------------------------------------------------------

    def get_voices(self) -> list[dict]:
        """Return available general voices (WAV files in voices/ top-level)."""
        wavs = globmod.glob(os.path.join(VOICES_DIR, "*.wav"))
        voices = []
        for wav in sorted(wavs):
            name = os.path.splitext(os.path.basename(wav))[0]
            voices.append({"name": name, "path": wav})

        if not voices:
            logger.warning(f"No voice WAVs found in: {VOICES_DIR}")

        return voices

    def get_voice_names(self) -> list[str]:
        return [v["name"] for v in self.get_voices()]

    def set_voice(self, voice_name: str):
        """Set active general voice by name (filename without .wav)."""
        available = self.get_voice_names()
        if voice_name not in available:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")
        self.current_voice = voice_name
        logger.info(f"Voice changed to: {voice_name}")

    def _get_voice_path(self, voice_name: str) -> str:
        """Return path to general voice WAV file."""
        path = os.path.join(VOICES_DIR, f"{voice_name}.wav")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Voice file not found: {path}\n"
                f"Add a WAV file (10-30 sec) to: {VOICES_DIR}"
            )
        return path

    # ------------------------------------------------------------------
    #  Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str, role: str | None = None, language: str | None = None) -> str:
        """
        Synthesize text using XTTS.
        - If role is a board role (cso/cfo/cto/ceo) -> use board voice from voices/board/
        - Otherwise use general voice from voices/
        Returns URL path: /static/audio/tts_XXXX.wav
        """
        self._ensure_loaded()

        # Determine voice path
        if role and role.lower() in ROLE_VOICE_MAP:
            voice_path = ROLE_VOICE_MAP[role.lower()]
            if not voice_path or not os.path.exists(voice_path):
                raise FileNotFoundError(
                    f"Voice file for role '{role}' not found at {voice_path}. "
                    f"Run: python setup_board_voices.py"
                )
        else:
            # General voice
            voice_name = role if role else self.current_voice
            voice_path = self._get_voice_path(voice_name)

        # Determine language
        if language and language != "auto":
            lang = language
        elif XTTS_LANGUAGE and XTTS_LANGUAGE != "auto":
            lang = XTTS_LANGUAGE
        else:
            lang = _detect_language(text)

        logger.info(f"XTTS synthesize: voice={role or self.current_voice}, lang={lang}, text={len(text)} chars")

        # Split long text into chunks
        chunks = _split_sentences(text)

        # Generate audio for each chunk, then concatenate
        audio_parts = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            logger.debug(f"XTTS chunk {i + 1}/{len(chunks)}: {chunk[:60]}...")

            wav = self._tts_call(chunk, voice_path, lang)

            # tts() returns list of floats → convert to tensor
            audio_parts.append(torch.tensor(wav).unsqueeze(0))

        if not audio_parts:
            raise ValueError("No audio generated")

        full_audio = torch.cat(audio_parts, dim=-1)

        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        # XTTS v2 outputs at 24kHz
        torchaudio.save(filepath, full_audio, 24000)
        _cleanup_audio_files()

        logger.info(f"XTTS file saved: {filename}")
        return f"/static/audio/{filename}"


    def synthesize_chunks(self, text: str, role: str | None = None, language: str | None = None):
        """
        Generator: yields (chunk_url, chunk_index, total_chunks) for each sentence chunk.
        Each chunk is a separate WAV file ready to play immediately.
        """
        self._ensure_loaded()

        # Determine voice path
        if role and role.lower() in ROLE_VOICE_MAP:
            voice_path = ROLE_VOICE_MAP[role.lower()]
            if not voice_path or not os.path.exists(voice_path):
                raise FileNotFoundError(f"Voice file for role '{role}' not found at {voice_path}")
        else:
            voice_name = role if role else self.current_voice
            voice_path = self._get_voice_path(voice_name)

        # Determine language
        if language and language != "auto":
            lang = language
        elif XTTS_LANGUAGE and XTTS_LANGUAGE != "auto":
            lang = XTTS_LANGUAGE
        else:
            lang = _detect_language(text)

        chunks = _split_sentences(text)
        total = len(chunks)
        logger.info(f"XTTS streaming: voice={role or self.current_voice}, lang={lang}, {total} chunks, {len(text)} chars")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            logger.debug(f"XTTS chunk {i + 1}/{total}: {chunk[:60]}...")

            wav = self._tts_call(chunk, voice_path, lang)

            audio_tensor = torch.tensor(wav).unsqueeze(0)
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            torchaudio.save(filepath, audio_tensor, 24000)

            url = f"/static/audio/{filename}"
            logger.info(f"XTTS chunk {i + 1}/{total} saved: {filename}")
            yield (url, i, total)

        # Cleanup once after all chunks — not inside loop (early chunks got deleted before served)
        _cleanup_audio_files()


# Singleton
_xtts_manager: XTTSManager | None = None


def get_xtts_manager() -> XTTSManager:
    global _xtts_manager
    if _xtts_manager is None:
        _xtts_manager = XTTSManager()
    return _xtts_manager
