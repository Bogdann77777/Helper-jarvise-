# board/openrouter_client.py — Async HTTP клиент для OpenRouter API

import asyncio
import logging
from typing import Optional

import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, BOARD_DIRECTOR_TIMEOUT

logger = logging.getLogger(__name__)

# Singleton instance
_client: Optional["OpenRouterClient"] = None


class OpenRouterClient:
    """Async клиент OpenRouter с retry и rate limit handling."""

    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0  # секунды: 1 → 2 → 4

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(BOARD_DIRECTOR_TIMEOUT, connect=10.0),
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Board of Directors",
                },
            )
        return self._http

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Отправляет запрос к OpenRouter и возвращает текст ответа.
        Retry с exponential backoff при 429/5xx.
        """
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY не задан. Установи переменную окружения."
            )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                client = await self._ensure_client()
                response = await client.post(OPENROUTER_BASE_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"OpenRouter [{model}]: {len(content)} символов, "
                        f"attempt={attempt + 1}"
                    )
                    return content

                if response.status_code == 429:
                    # Rate limit — ждём и пробуем снова
                    wait = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"OpenRouter 429 rate limit [{model}], "
                        f"retry in {wait}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"OpenRouter {response.status_code} [{model}], "
                        f"retry in {wait}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(wait)
                    continue

                # 4xx (не 429) — не retryable
                error_text = response.text[:500]
                raise RuntimeError(
                    f"OpenRouter ошибка {response.status_code} [{model}]: {error_text}"
                )

            except httpx.TimeoutException:
                last_error = f"Timeout [{model}] (attempt {attempt + 1})"
                logger.warning(last_error)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.BACKOFF_BASE * (2 ** attempt))
                continue

            except httpx.HTTPError as e:
                last_error = f"HTTP error [{model}]: {e}"
                logger.warning(last_error)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.BACKOFF_BASE * (2 ** attempt))
                continue

        raise RuntimeError(
            f"OpenRouter: все {self.MAX_RETRIES} попыток не удались [{model}]. "
            f"Последняя ошибка: {last_error}"
        )

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None


def get_openrouter_client() -> OpenRouterClient:
    """Получить singleton клиент OpenRouter."""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
