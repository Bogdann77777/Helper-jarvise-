# qwen3tts_client.py — async client for qwen3tts_service.py subprocess
#
# Manages subprocess lifecycle, sends TTS requests via stdin/stdout JSON protocol.
# The subprocess loads the model once and stays alive between requests.

import asyncio
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT     = os.path.dirname(os.path.dirname(_HERE))  # .../project/
if sys.platform == "win32":
    QWEN_PYTHON = os.path.join(_PROJECT, "qwen3tts_env", "Scripts", "python.exe")
else:
    QWEN_PYTHON = os.path.join(_PROJECT, "qwen3tts_env_linux", "bin", "python")
QWEN_SERVICE = os.path.join(_HERE, "qwen3tts_service.py")

STARTUP_TIMEOUT  = 150  # seconds to wait for model to load
SYNTHESIS_TIMEOUT = 90  # seconds per synthesis request

DEFAULT_INSTRUCT = (
    "Speak with genuine emotion and natural intonation. "
    "Be warm, confident, and expressive. "
    "Emphasize key points, vary tone naturally, add feeling to the words."
)

QWEN_SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena"]


class Qwen3TTSClient:
    """
    Async client that manages qwen3tts_service.py subprocess.

    The subprocess (running in qwen3tts_env) loads the model once and stays
    alive, receiving TTS requests over stdin/stdout JSON lines protocol.
    """

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._lock  = asyncio.Lock()
        self._ready = False
        self._starting = False
        self._speaker = "Ryan"
        self._language = "Russian"
        self._instruct = DEFAULT_INSTRUCT

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        """Launch subprocess and wait for {"status": "ready"}."""
        if self._ready:
            return
        if self._starting:
            for _ in range(STARTUP_TIMEOUT * 2):
                await asyncio.sleep(0.5)
                if self._ready:
                    return
            raise RuntimeError("Qwen3-TTS startup timeout (waited for in-progress start)")

        self._starting = True
        try:
            if self._proc and self._proc.returncode is None:
                await self._kill()

            logger.info(f"[Qwen3TTS] Launching subprocess: {QWEN_PYTHON} {QWEN_SERVICE}")
            self._proc = await asyncio.create_subprocess_exec(
                QWEN_PYTHON, QWEN_SERVICE,
                stdin  = asyncio.subprocess.PIPE,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE,
                cwd    = _HERE,
            )
            logger.info(f"[Qwen3TTS] PID={self._proc.pid} — loading model (may take 30–120s)...")

            # Drain stderr in background so it doesn't buffer-fill and deadlock
            asyncio.create_task(self._drain_stderr())

            # Read lines until we get {"status":"ready"} — skip any stray log lines
            deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
            data = {}
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=max(remaining, 1),
                )
                raw = line.decode("utf-8").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    break  # got valid JSON
                except json.JSONDecodeError:
                    logger.debug(f"[Qwen3TTS] startup non-JSON: {raw[:80]}")
                    continue
            if data.get("status") == "ready":
                self._ready = True
                logger.info("[Qwen3TTS] Ready!")
            else:
                raise RuntimeError(f"[Qwen3TTS] Unexpected startup message: {data}")
        except Exception as e:
            logger.error(f"[Qwen3TTS] Start failed: {e}")
            self._ready = False
            raise
        finally:
            self._starting = False

    async def stop(self):
        """Gracefully stop the subprocess (unloads model from VRAM)."""
        self._ready = False
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}).encode() + b"\n")
                await self._proc.stdin.drain()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
                logger.info("[Qwen3TTS] Subprocess stopped gracefully")
            except Exception:
                await self._kill()

    async def _kill(self):
        try:
            self._proc.kill()
            await self._proc.wait()
        except Exception:
            pass
        self._ready = False

    async def _drain_stderr(self):
        """Read and log stderr in background to prevent buffer deadlocks."""
        try:
            async for line in self._proc.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[Qwen3TTS/stderr] {text}")
        except Exception:
            pass

    # ── speaker config ────────────────────────────────────────────────────

    def set_speaker(self, speaker: str):
        if speaker not in QWEN_SPEAKERS:
            raise ValueError(f"Unknown speaker: {speaker}. Available: {QWEN_SPEAKERS}")
        self._speaker = speaker
        logger.info(f"[Qwen3TTS] Speaker set to: {speaker}")

    def get_speakers(self) -> list[str]:
        return QWEN_SPEAKERS

    @property
    def current_speaker(self) -> str:
        return self._speaker

    # ── synthesis ─────────────────────────────────────────────────────────

    async def synthesize(
        self,
        text:     str,
        speaker:  str | None = None,
        language: str | None = None,
        instruct: str | None = None,
    ) -> dict:
        """
        Send TTS request to subprocess.
        Returns: {"ok": True, "url": "/static/audio/xxx.wav", "gen_sec": ..., "rtf": ...}
        On error:  {"ok": False, "error": "..."}
        """
        if not self._ready:
            try:
                await self.start()
            except Exception as e:
                return {"ok": False, "error": f"TTS not available: {e}"}

        async with self._lock:
            # Check if process died while we were waiting for lock
            if not self._ready or self._proc is None or self._proc.returncode is not None:
                logger.warning("[Qwen3TTS] Process died, attempting restart...")
                self._ready = False
                try:
                    await self.start()
                except Exception as e:
                    return {"ok": False, "error": f"TTS restart failed: {e}"}

            req_obj = {
                "text":     text,
                "speaker":  speaker  or self._speaker,
                "language": language or self._language,
                "instruct": instruct or self._instruct,
            }
            req_line = json.dumps(req_obj, ensure_ascii=False).encode("utf-8") + b"\n"

            try:
                self._proc.stdin.write(req_line)
                await self._proc.stdin.drain()
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=SYNTHESIS_TIMEOUT,
                )
                result = json.loads(raw.decode("utf-8").strip())
                if result.get("ok"):
                    logger.debug(f"[Qwen3TTS] gen={result.get('gen_sec')}s RTF={result.get('rtf')} → {result.get('url','?')}")
                else:
                    logger.warning(f"[Qwen3TTS] Synthesis error: {result.get('error')}")
                return result
            except asyncio.TimeoutError:
                logger.error(f"[Qwen3TTS] Synthesis timeout after {SYNTHESIS_TIMEOUT}s")
                self._ready = False
                return {"ok": False, "error": "synthesis timeout"}
            except Exception as e:
                logger.error(f"[Qwen3TTS] Synthesis exception: {e}")
                self._ready = False
                return {"ok": False, "error": str(e)}

    @property
    def is_ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.returncode is None


# ─── SINGLETON ────────────────────────────────────────────────────────────────
_client: Qwen3TTSClient | None = None


def get_qwen3tts_client() -> Qwen3TTSClient:
    global _client
    if _client is None:
        _client = Qwen3TTSClient()
    return _client
