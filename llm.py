"""Chat Completions clients with verified TLS and a bounded request budget."""

from __future__ import annotations

import asyncio
from http.client import HTTPException
import json
import math
from pathlib import Path
import ssl
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from uuid import uuid4


OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v1/chat/completions"
OPENAI_CHAT_URL = "https://deepcode.ci.nsu.ru/api/chat/completions"
OPENAI_DEFAULT_MODEL = "Qwen3.8-27B"
MAX_RESPONSE_BYTES = 65_536
TOKEN_REFRESH_MARGIN = 30.0
MAX_REQUEST_TIMEOUT = 5.0
_TIMEOUT_MESSAGE = "Время ожидания сервиса распознавания истекло."


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        # Authorization must never be copied to a redirect destination, even
        # within the same host. A changed endpoint requires explicit setup.
        return None


def urlopen(request: Request, timeout: float, context: ssl.SSLContext):
    """Use a request-local TLS opener without redirects or global side effects."""
    opener = build_opener(HTTPSHandler(context=context), _NoRedirect())
    return opener.open(request, timeout=timeout)


class LLMError(ValueError):
    """A safe error category for diagnostics, without remote bodies or credentials."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class RequestWorker:
    """Run one blocking request without making process shutdown wait for DNS.

    A timed-out request retains its slot until its thread finishes. Further
    requests fall back locally instead of creating more stuck worker threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def run(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        if not self._lock.acquire(blocking=False):
            raise LLMError("busy", "Сервис распознавания занят. Используются локальные правила.")
        future = loop.create_future()

        def finish(value: Any, error: BaseException | None) -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

        def execute() -> None:
            value, error = None, None
            try:
                value = callback(*args, **kwargs)
            except BaseException as exc:
                error = exc
            finally:
                self._lock.release()
            try:
                loop.call_soon_threadsafe(finish, value, error)
            except RuntimeError:
                # A cancelled request may finish after the bot's loop closes.
                pass

        try:
            threading.Thread(target=execute, name="deadline-llm", daemon=True).start()
        except BaseException:
            self._lock.release()
            raise
        return await future


def validate_api_url(value: str) -> None:
    """Reject malformed endpoints before configuration checks or HTTP requests."""
    message = "Адрес сервиса распознавания должен быть корректным HTTPS URL без учётных данных и пробелов."
    if not isinstance(value, str) or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(message)
    try:
        endpoint = urlparse(value)
        port = endpoint.port
        valid = (endpoint.scheme == "https" and bool(endpoint.hostname)
                 and endpoint.username is None and endpoint.password is None
                 and (port is None or 1 <= port <= 65535))
    except ValueError:
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)


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


class OpenAICompatibleClient:
    """One request at a time; an occupied client immediately falls back locally.

    The optional external deadline includes time spent starting a worker.
    A timed-out worker cannot overlap the next remote request.
    The API key is sent directly as a Bearer token, without an OAuth exchange.
    """

    def __init__(self, api_key: str, model: str = OPENAI_DEFAULT_MODEL, timeout: float = MAX_REQUEST_TIMEOUT,
                 api_url: str = OPENAI_CHAT_URL, ca_bundle: Path | None = None, *,
                 chat_template_kwargs: dict[str, bool] | None = None) -> None:
        if (not isinstance(api_key, str) or not api_key.isascii()
                or any(ord(character) < 32 or ord(character) == 127 for character in api_key)):
            raise ValueError("Некорректный ключ авторизации сервиса распознавания.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Укажите модель сервиса распознавания.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Тайм-аут должен быть положительным конечным числом.")
        validate_api_url(api_url)
        if chat_template_kwargs is not None and (
            not isinstance(chat_template_kwargs, dict)
            or set(chat_template_kwargs) - {"thinking", "enable_thinking"}
            or any(type(value) is not bool for value in chat_template_kwargs.values())
        ):
            raise ValueError("Параметры шаблона модели должны содержать только булевы thinking/enable_thinking.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = min(float(timeout), MAX_REQUEST_TIMEOUT)
        self.api_url = api_url
        self.ca_bundle = Path(ca_bundle) if ca_bundle is not None else None
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._request_lock = threading.Lock()
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
        return self.api_key

    def _invalidate_token(self) -> None:
        pass

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
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user_content}],
                "temperature": 0,
                "max_tokens": 512,
                "stream": False,
            }
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(self.api_url, data=body, method="POST", headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            try:
                envelope = self._post(request, request_deadline)
            except HTTPError as exc:
                if exc.code == 401:
                    self._invalidate_token()
                raise
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("Некорректный ответ модели сервиса распознавания.")
            if choices[0].get("finish_reason") not in (None, "stop"):
                raise ValueError("Модель не завершила ответ сервиса распознавания.")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("Некорректный ответ модели сервиса распознавания.")
            return _json_object(message["content"])
        except HTTPError as exc:
            code = exc.code
            exc.close()
            reason = ("auth" if code in {401, 403} else "quota" if code == 429
                      else "request" if code in {400, 404, 405, 413, 415, 422} else "network")
            raise LLMError(reason, f"Сервис распознавания отклонил запрос (HTTP {code}).") from None
        except TimeoutError:
            raise TimeoutError(_TIMEOUT_MESSAGE) from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(_TIMEOUT_MESSAGE) from None
            reason = "tls" if isinstance(exc.reason, ssl.SSLError) else "network"
            raise LLMError(reason, "Не удалось безопасно подключиться к сервису распознавания.") from None
        except (OSError, HTTPException) as exc:
            reason = "tls" if isinstance(exc, ssl.SSLError) else "network"
            raise LLMError(reason, "Не удалось безопасно подключиться к сервису распознавания.") from None
        finally:
            self._request_lock.release()


class GigaChatClient(OpenAICompatibleClient):
    """Chat Completions with a cached GigaChat OAuth token in the same budget."""

    def __init__(self, api_key: str, model: str = "GigaChat-2", timeout: float = MAX_REQUEST_TIMEOUT,
                 api_url: str = CHAT_URL, ca_bundle: Path | None = None) -> None:
        super().__init__(api_key, model, timeout, api_url, ca_bundle)
        self._access_token = ""
        self._expires_at = 0.0

    def _invalidate_token(self) -> None:
        self._access_token = ""
        self._expires_at = 0.0

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
