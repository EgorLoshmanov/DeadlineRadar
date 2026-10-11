"""Run the bot, an offline configuration check, or an isolated model probe."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from contextlib import AsyncExitStack
from inspect import isawaitable

from config import Settings, semester_start
from db import Database
from notifications import DeliveryDisabled, DeliverySkipped, NotificationService, RetryDelivery
from models import parse_due_date
from parser import ParserEngine
from presentation import reminder_text
from publications import PublicationService
from services import DeadlineService

logger = logging.getLogger(__name__)


async def _close_safely(close) -> None:
    try:
        result = close()
        if isawaitable(result):
            await result
    except Exception as exc:
        logger.error("Resource cleanup failed: %s", type(exc).__name__)


async def _close_runtime_resources(bot, database=None, dispatcher=None) -> None:
    # ExitStack continues with the remaining callbacks even if one is cancelled.
    async with AsyncExitStack() as resources:
        if database is not None:
            resources.push_async_callback(_close_safely, database.close)
        resources.push_async_callback(_close_safely, bot.session.close)
        if dispatcher is not None:
            resources.push_async_callback(_close_safely, dispatcher.fsm.events_isolation.close)
            resources.push_async_callback(_close_safely, dispatcher.storage.close)


def create_parser(settings: Settings) -> ParserEngine:
    return ParserEngine(
        settings.llm_api_key, settings.llm_model, settings.llm_timeout, settings.llm_api_url,
        provider=getattr(settings, "llm_provider", "anthropic"),
        ca_bundle=getattr(settings, "llm_ca_bundle", None),
    )


async def check_llm(settings: Settings) -> int:
    """Probe the configured model with a public fixture, without Telegram or SQLite."""
    if not settings.llm_api_key or not settings.llm_model:
        print("LLM не подтверждён: заполните LLM_API_KEY и LLM_MODEL в локальном .env.")
        return 1
    parser = create_parser(settings)
    started = time.monotonic()
    result = await parser.parse("Лаба 3 по БД до пятницы в 18:00", now=settings.now())
    elapsed = time.monotonic() - started
    if result["parsing_method"] != "llm":
        reasons = {
            "timeout": "модель не ответила в пределах установленного времени",
            "auth": "сервис отклонил ключ или доступ аккаунта",
            "quota": "сервис отклонил запрос из-за квоты или ограничения частоты",
            "tls": "не удалось проверить сертификат сервера; проверьте LLM_CA_BUNDLE",
            "network": "не удалось получить ответ сервиса по сети",
            "response": "ответ модели не прошёл проверку JSON, полей или даты",
            "busy": "модель уже обрабатывает другой запрос",
        }
        reason = reasons.get(parser.last_remote_error, "не получен проверенный ответ модели")
        print(f"LLM не подтверждён: {reason}. Использован локальный разбор ({elapsed:.2f} с).")
        return 1
    print(f"Проверка LLM пройдена: {parser.provider}; корректный JSON; {elapsed:.2f} с.")
    return 0


async def run(settings: Settings) -> None:
    # Import network dependencies only in live mode; --check and run_test remain offline.
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter
    from aiogram.fsm.storage.memory import MemoryStorage, SimpleEventIsolation
    from aiogram.types import BotCommand
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from bot import create_router, inline, send_group_publication

    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode="HTML"))
    database = dispatcher = None
    try:
        database = Database(settings.database_path, clock=settings.now)
        parser = create_parser(settings)
        service = DeadlineService(database, parser, settings.now)
        notifications = NotificationService(database, settings.now)
        publications = PublicationService(database, settings.now)
        dispatcher = Dispatcher(storage=MemoryStorage(), events_isolation=SimpleEventIsolation())
        scheduler = AsyncIOScheduler(timezone=settings.timezone)
    except BaseException:
        await _close_runtime_resources(bot, database, dispatcher)
        raise
    notification_task = None
    active_deliveries: set[asyncio.Task] = set()
    stopping = False
    last_delivery: dict[int, float] = {}
    next_delivery = 0.0
    pacing_lock = asyncio.Lock()

    async def prepare_delivery(item: dict) -> None:
        nonlocal next_delivery
        async with pacing_lock:
            loop = asyncio.get_running_loop()
            # Another in-flight send may extend the shared flood wait while this
            # task sleeps, so recheck the limit after every wake-up.
            while True:
                delay = max(next_delivery, last_delivery.get(item["user_id"], 0.0) + 1.05) - loop.time()
                if delay <= 0:
                    break
                await asyncio.sleep(delay)
            current = database.get_deadline(item["user_id"], item["id"])
            user = database.get_user(item["user_id"])
            if not current or current["completed"] or not user or not user["delivery_enabled"]:
                raise DeliverySkipped
            if parse_due_date(current["due_date"]) <= settings.now():
                raise DeliverySkipped
            last_delivery[item["user_id"]] = loop.time()
            # Pacing finishes before the durable send gate, so queued work stays recoverable.
            next_delivery = loop.time() + 0.05

    def pause_deliveries(retry_after: int) -> None:
        nonlocal next_delivery
        next_delivery = max(next_delivery, asyncio.get_running_loop().time() + max(1, retry_after))

    async def send_publication(item: dict) -> None:
        try:
            await send_group_publication(bot, item)
        except RetryDelivery as exc:
            pause_deliveries(exc.retry_after)
            raise

    async def on_publication(deadline_id: int) -> None:
        if stopping:
            # The transaction already queued this publication. Startup will
            # recover it; no new background work may outlive resource cleanup.
            return
        task = asyncio.create_task(publications.tick(
            send_publication, prepare=prepare_delivery, deadline_id=deadline_id,
        ))
        active_deliveries.add(task)

        def finished(delivery: asyncio.Task) -> None:
            active_deliveries.discard(delivery)
            if not delivery.cancelled() and delivery.exception() is not None:
                logger.error("Publication task failed: %s", type(delivery.exception()).__name__)

        task.add_done_callback(finished)

    async def send_reminder(item: dict) -> None:
        try:
            await bot.send_message(item["user_id"], reminder_text(item), reply_markup=inline([
                [("✅ Я сдал", f"done:{item['id']}")],
            ]))
        except TelegramForbiddenError as exc:
            raise DeliveryDisabled from exc
        except TelegramRetryAfter as exc:
            pause_deliveries(exc.retry_after)
            raise RetryDelivery(exc.retry_after) from exc

    async def deliver_pending() -> None:
        task = asyncio.current_task()
        active_deliveries.add(task)
        try:
            await publications.tick(send_publication, prepare=prepare_delivery)
            await notifications.tick(send_reminder, prepare=prepare_delivery)
        finally:
            active_deliveries.discard(task)

    def cleanup() -> None:
        count = database.cleanup_completed_before(semester_start(settings.now()))
        if count:
            logger.info("Removed %s completed deadlines from previous semesters", count)

    try:
        me = await bot.get_me()
        dispatcher.include_router(create_router(database, service, settings, me.username, on_publication=on_publication))
        await bot.set_my_commands([
            BotCommand(command="deadlines", description="Активные дедлайны"),
            BotCommand(command="add", description="Добавить вручную"),
            BotCommand(command="group", description="Мои группы"),
            BotCommand(command="publish", description="Добавить для группы"),
            BotCommand(command="history", description="Сданные задания"),
            BotCommand(command="help", description="Помощь"),
            BotCommand(command="cancel", description="Отменить ввод"),
            BotCommand(command="settings", description="Настройки и данные"),
            BotCommand(command="plan", description="Учебный план и лимиты"),
        ])
        cleanup()
        scheduler.add_job(deliver_pending, "interval", seconds=settings.scheduler_interval,
                          id="reminders", max_instances=1, coalesce=True,
                          misfire_grace_time=60)
        scheduler.add_job(cleanup, "interval", hours=24, id="semester_cleanup", max_instances=1, coalesce=True)
        scheduler.start()
        notification_task = asyncio.create_task(deliver_pending())
        logger.info("DeadlineRadar started; timezone=%s; parser=%s", settings.timezone,
                    f"{parser.provider} with local fallback" if settings.llm_api_key and settings.llm_model else "local")
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types(),
                                       tasks_concurrency_limit=32, close_bot_session=False)
    finally:
        stopping = True
        try:
            if scheduler.running:
                await _close_safely(lambda: scheduler.shutdown(wait=False))
                # AsyncIOScheduler schedules shutdown on the event loop.
                await asyncio.sleep(0)
            pending = set(active_deliveries)
            # aiogram stops its polling tasks separately from update handlers.
            # Cancel and join those handlers before closing their SQLite/FSM.
            pending.update(getattr(dispatcher, "_handle_update_tasks", ()))
            if notification_task is not None:
                pending.add(notification_task)
            for task in pending:
                task.cancel()
            if pending:
                results = await asyncio.gather(*pending, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error("Reminder task failed: %s", type(result).__name__)
        finally:
            await _close_runtime_resources(bot, database, dispatcher)


def main() -> int:
    arguments = argparse.ArgumentParser(description="DeadlineRadar — Telegram-бот учебных дедлайнов")
    checks = arguments.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="Проверить конфигурацию и БД без сети и токена")
    checks.add_argument("--check-llm", action="store_true", help="Проверить модель без Telegram и изменения БД")
    args = arguments.parse_args()
    try:
        settings = Settings.from_env(require_token=not (args.check or args.check_llm))
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if args.check_llm:
            return asyncio.run(check_llm(settings))
        if args.check:
            with Database(settings.database_path, clock=settings.now):
                pass
            print(f"Конфигурация и БД доступны. Часовой пояс: {settings.timezone}.")
            print("Сеть и токены не проверялись. Офлайн-сценарий: python run_test.py")
            return 0
        asyncio.run(run(settings))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError:
        print("Установите зависимости: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except sqlite3.Error:
        print("Ошибка SQLite. Проверьте DATABASE_PATH, права на файл и свободное место.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Do not print request URLs, authorization headers, tokens or user message bodies.
        print(f"Не удалось запустить бота ({type(exc).__name__}). Проверьте BOT_TOKEN и подключение к сети.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
