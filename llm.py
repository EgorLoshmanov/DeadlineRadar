"""Small GigaChat client with verified TLS and a shared request time budget."""

from __future__ import annotations

from http.client import HTTPException
import json
import math
from pathlib import Path
import ssl
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v1/chat/completions"
MAX_RESPONSE_BYTES = 65_536
TOKEN_REFRESH_MARGIN = 30.0
MAX_REQUEST_TIMEOUT = 2.5
_TIMEOUT_MESSAGE = "Время ожидания сервиса распознавания истекло."


class LLMError(ValueError):
    """A safe error category for diagnostics, without remote bodies or credentials."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _reject_constant(value: str) -> None:
    raise ValueError("Некорректный JSON в ответе сервиса.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Некорректный JSON в ответе сервиса.")
        result[key] = value
    return result


def _json_object(value: bytes | str) -> dict[str, Any]:
    try:
        result = json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Некорректный JSON в ответе сервиса.") from None
    if not isinstance(result, dict):
        raise ValueError("Ответ сервиса должен содержать JSON-объект.")
    return result


class GigaChatClient:
    """One request at a time; an occupied client immediately falls back locally.

    The optional external deadline includes time spent waiting for an executor
    worker. OAuth and generation always share the same monotonic deadline.
    Access tokens are cached only in memory and never written to a file.
    """

    def __init__(self, api_key: str, model: str = "GigaChat-2", timeout: float = 2.5,
                 api_url: str = CHAT_URL, ca_bundle: Path | None = None) -> None:
        if (not isinstance(api_key, str) or not api_key.isascii()
                or any(ord(character) < 32 or ord(character) == 127 for character in api_key)):
            raise ValueError("Некорректный ключ авторизации сервиса распознавания.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Укажите модель сервиса распознавания.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Тайм-аут должен быть положительным конечным числом.")
        endpoint = urlparse(api_url)
        if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password:
            raise ValueError("Адрес сервиса распознавания должен использовать HTTPS.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = min(float(timeout), MAX_REQUEST_TIMEOUT)
        self.api_url = api_url
        self.ca_bundle = Path(ca_bundle) if ca_bundle is not None else None
        self._request_lock = threading.Lock()
        self._access_token = ""
        self._expires_at = 0.0
        self._ssl_context: ssl.SSLContext | None = None

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(_TIMEOUT_MESSAGE)
        return remaining

    def _context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            context = ssl.create_default_context()
            if self.ca_bundle is not None:
                context.load_verify_locations(cafile=str(self.ca_bundle))
            self._ssl_context = context
        return self._ssl_context

    def _post(self, request: Request, deadline: float) -> dict[str, Any]:
        context = self._context()
        with urlopen(request, timeout=self._remaining(deadline), context=context) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        self._remaining(deadline)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Ответ сервиса распознавания превышает допустимый размер.")
        return _json_object(raw)

    def _token(self, deadline: float) -> str:
        if self._access_token and time.time() < self._expires_at - TOKEN_REFRESH_MARGIN:
            return self._access_token
        request = Request(OAUTH_URL, data=urlencode({"scope": "GIGACHAT_API_PERS"}).encode("ascii"),
                          method="POST", headers={
                              "Authorization": f"Basic {self.api_key}",
                              "RqUID": str(uuid4()),
                              "Content-Type": "application/x-www-form-urlencoded",
                              "Accept": "application/json",
                          })
        value = self._post(request, deadline)
        token, expires_at = value.get("access_token"), value.get("expires_at")
        if (not isinstance(token, str) or not token.strip() or len(token) > 8192
                or not token.isascii() or any(ord(character) < 32 or ord(character) == 127 for character in token)
                or type(expires_at) is not int or expires_at <= time.time() * 1000
                or expires_at > 253_402_300_799_999):
            raise ValueError("Некорректный ответ авторизации сервиса распознавания.")
        self._access_token = token
        self._expires_at = expires_at / 1000
        return token

    def request(self, system: str, user_content: str, *, deadline: float | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("Не указан ключ авторизации сервиса распознавания.")
        if not isinstance(system, str) or not isinstance(user_content, str):
            raise ValueError("Сервис распознавания принимает только текст.")
        if deadline is not None and (isinstance(deadline, bool)
                                     or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError("Некорректный срок ожидания сервиса распознавания.")
        request_deadline = time.monotonic() + self.timeout
        if deadline is not None:
            request_deadline = min(request_deadline, deadline)
        if not self._request_lock.acquire(blocking=False):
            raise LLMError("busy", "Сервис распознавания занят. Используются локальные правила.")
        try:
            self._remaining(request_deadline)
            token = self._token(request_deadline)
            # Never send a late generation request after OAuth or worker startup
            # has consumed the budget allocated to parsing this message.
            self._remaining(request_deadline)
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user_content}],
                "temperature": 0,
                "max_tokens": 512,
                "stream": False,
            }, ensure_ascii=False).encode("utf-8")
            request = Request(self.api_url, data=body, method="POST", headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            try:
                envelope = self._post(request, request_deadline)
            except HTTPError as exc:
                if exc.code == 401:
                    self._access_token = ""
                    self._expires_at = 0.0
                raise
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("Некорректный ответ модели сервиса распознавания.")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("Некорректный ответ модели сервиса распознавания.")
            return _json_object(message["content"])
        except HTTPError as exc:
            code = exc.code
            exc.close()
            reason = "auth" if code in {401, 403} else "quota" if code == 429 else "network"
            raise LLMError(reason, f"Сервис распознавания отклонил запрос (HTTP {code}).") from None
        except TimeoutError:
            raise TimeoutError(_TIMEOUT_MESSAGE) from None
        except URLError as exc:
            reason = "tls" if isinstance(exc.reason, ssl.SSLError) else "network"
            raise LLMError(reason, "Не удалось безопасно подключиться к сервису распознавания.") from None
        except (OSError, HTTPException) as exc:
            reason = "tls" if isinstance(exc, ssl.SSLError) else "network"
            raise LLMError(reason, "Не удалось безопасно подключиться к сервису распознавания.") from None
        finally:
            self._request_lock.release()
