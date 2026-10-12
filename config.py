"""Application configuration; the offline core needs only the standard library."""
from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llm import MAX_REQUEST_TIMEOUT, OPENAI_CHAT_URL, OPENAI_DEFAULT_MODEL, validate_api_url

ROOT = Path(__file__).resolve().parent
MAX_LLM_TIMEOUT = MAX_REQUEST_TIMEOUT


def load_env(path: Path) -> None:
    """Load simple KEY=value settings without overwriting process environment."""
    if not path.exists():
        return
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"Некорректная строка {number} в .env")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Незакрытая кавычка в строке {number} .env")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def semester_start(now: datetime) -> datetime:
    """Fall semester starts September 1, spring starts February 1."""
    if now.month >= 9:
        return datetime(now.year, 9, 1)
    if now.month >= 2:
        return datetime(now.year, 2, 1)
    return datetime(now.year - 1, 9, 1)


@dataclass(frozen=True)
class Settings:
    bot_token: str = ""
    database_path: Path = ROOT / "deadlines.db"
    timezone: str = "Asia/Novosibirsk"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_api_url: str = "https://api.anthropic.com/v1/messages"
    llm_timeout: float = MAX_LLM_TIMEOUT
    scheduler_interval: int = 30
    log_level: str = "INFO"
    llm_provider: str = "anthropic"
    llm_ca_bundle: Path | None = None
    llm_chat_template_kwargs: dict[str, bool] | None = None

    @classmethod
    def from_env(cls, *, require_token: bool = False, env_file: Path | None = None) -> "Settings":
        load_env(env_file or ROOT / ".env")
        try:
            timeout = float(os.getenv("LLM_TIMEOUT", str(MAX_LLM_TIMEOUT)))
            interval = int(os.getenv("SCHEDULER_INTERVAL", "30"))
        except ValueError as exc:
            raise ValueError("LLM_TIMEOUT и SCHEDULER_INTERVAL должны быть числами") from exc
        if not math.isfinite(timeout) or not 0 < timeout <= MAX_LLM_TIMEOUT:
            raise ValueError(f"LLM_TIMEOUT должен быть больше 0 и не больше {MAX_LLM_TIMEOUT:g} секунд")
        # Configuration and transport share one bounded request budget.
        timeout = min(timeout, MAX_LLM_TIMEOUT)
        if not 5 <= interval <= 300:
            raise ValueError("SCHEDULER_INTERVAL должен быть от 5 до 300 секунд")
        timezone = os.getenv("APP_TIMEZONE", "Asia/Novosibirsk")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("APP_TIMEZONE должен содержать существующий IANA часовой пояс") from exc
        token = os.getenv("BOT_TOKEN", "").strip()
        if require_token and not token:
            raise ValueError("Укажите BOT_TOKEN в .env. Без токенов доступна команда python run_test.py")
        database = Path(os.getenv("DATABASE_PATH", "deadlines.db")).expanduser()
        if not database.is_absolute():
            database = ROOT / database
        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        provider_defaults = {
            "anthropic": ("", "https://api.anthropic.com/v1/messages"),
            "gigachat": ("GigaChat-2", "https://api.giga.chat/v1/chat/completions"),
            "openai": (OPENAI_DEFAULT_MODEL, OPENAI_CHAT_URL),
        }
        if provider not in provider_defaults:
            raise ValueError("LLM_PROVIDER должен быть anthropic, gigachat или openai")
        default_model, default_url = provider_defaults[provider]
        api_url = os.getenv("LLM_API_URL", default_url).strip()
        try:
            validate_api_url(api_url)
        except ValueError:
            raise ValueError("LLM_API_URL должен быть корректным HTTPS URL без учётных данных и пробелов") from None
        chat_template_value = os.getenv("LLM_CHAT_TEMPLATE_KWARGS", "").strip()
        chat_template_kwargs = None
        if chat_template_value:
            template_error = "LLM_CHAT_TEMPLATE_KWARGS должен быть JSON-объектом с ключами thinking/enable_thinking и логическими значениями"
            try:
                chat_template_kwargs = json.loads(chat_template_value)
            except (ValueError, RecursionError):
                raise ValueError(template_error) from None
            if not isinstance(chat_template_kwargs, dict) or any(
                key not in {"thinking", "enable_thinking"} or not isinstance(value, bool)
                for key, value in chat_template_kwargs.items()
            ):
                raise ValueError(template_error) from None
        ca_value = os.getenv("LLM_CA_BUNDLE", "").strip()
        ca_bundle = None
        if ca_value:
            ca_bundle = Path(ca_value).expanduser()
            if not ca_bundle.is_absolute():
                ca_bundle = ROOT / ca_bundle
            if not ca_bundle.is_file():
                raise ValueError("LLM_CA_BUNDLE должен указывать на существующий PEM-файл")
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Неизвестный LOG_LEVEL")
        return cls(
            bot_token=token, database_path=database, timezone=timezone,
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", default_model).strip(),
            llm_api_url=api_url, llm_timeout=timeout,
            scheduler_interval=interval, log_level=level,
            llm_provider=provider, llm_ca_bundle=ca_bundle,
            llm_chat_template_kwargs=chat_template_kwargs,
        )

    def now(self) -> datetime:
        # The database stores local wall-clock values in the configured common timezone.
        return datetime.now(ZoneInfo(self.timezone)).replace(tzinfo=None)
