"""Run the real scheduler and lifecycle with a simulated Telegram transport."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from db import Database
from main import _close_runtime_resources, run

try:
    import aiogram
    import apscheduler
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class FakeBot:
    def __init__(self, block_send: bool = False, lose_ack: bool = False):
        self.get_me = AsyncMock(return_value=SimpleNamespace(username="deadline_radar_test_bot"))
        self.set_my_commands = AsyncMock()
        self.session = SimpleNamespace(close=AsyncMock())
        self.sent = []
        self.started = asyncio.Event()
        self.block_send = block_send
        self.lose_ack = lose_ack

    async def send_message(self, user_id, text, **kwargs):
        self.started.set()
        if self.block_send:
            await asyncio.Event().wait()
        self.sent.append((user_id, text))
        if self.lose_ack:
            raise TimeoutError("Remote acceptance succeeded but the acknowledgement was lost")


@unittest.skipUnless(HAS_RUNTIME, "Install requirements.txt to run scheduler integration tests")
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 10, 11, 12, 0)
        self.path = Path(self.directory.name) / "runtime.db"
        self.settings = SimpleNamespace(
            database_path=self.path, now=lambda: self.now, timezone="Asia/Novosibirsk",
            llm_api_key="", llm_model="", llm_timeout=5,
            llm_api_url="https://api.anthropic.com/v1/messages", bot_token="unused",
            scheduler_interval=0.03,
        )

    def tearDown(self):
        self.directory.cleanup()

    def add_deadline(self):
        with Database(self.path, clock=lambda: self.now) as db:
            return db.create_deadline(1, "Базы данных", "Лабораторная работа №3", self.now + timedelta(days=2))

    def add_publication(self):
        with Database(self.path, clock=lambda: self.now) as db:
            group = db.create_group(1, "ИВТ-31")
            db.join_group(2, group["invite_code"])
            return db.create_deadline(1, "БД", "Лаба №3", self.now + timedelta(days=10), group_id=group["id"])

    async def test_startup_recovers_publication_without_replaying_it(self):
        self.add_publication()
        bot = FakeBot()

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        self.assertEqual(bot.sent[0][0], 2)
        self.assertIn("Новое задание группы", bot.sent[0][1])

        async def idle(*args, **kwargs):
            await asyncio.sleep(0.07)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=idle):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)

    async def test_confirmation_trigger_delivers_without_waiting_for_interval(self):
        from bot import create_router
        bot = FakeBot()
        captured = {}

        def router(*args, **kwargs):
            captured["notify"] = kwargs["on_publication"]
            return create_router(*args, **kwargs)

        async def polling(*args, **kwargs):
            await asyncio.sleep(0.01)
            item = self.add_publication()
            await captured["notify"](item["id"])
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        self.settings.scheduler_interval = 300
        with patch("aiogram.Bot", return_value=bot), patch("bot.create_router", side_effect=router), \
                patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        self.assertEqual(bot.sent[0][0], 2)

    async def test_shutdown_waits_for_update_handlers_and_defers_their_late_publications(self):
        from bot import create_router
        bot = FakeBot()
        database = Database(self.path, clock=self.settings.now)
        captured = {}
        entered = asyncio.Event()

        def router(*args, **kwargs):
            captured["notify"] = kwargs["on_publication"]
            return create_router(*args, **kwargs)

        async def handler():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Update cleanup still needs the open database. A callback that
                # arrives during shutdown must leave its durable work for restart.
                try:
                    group = database.create_group(1, "Сохранение при остановке")
                    database.join_group(2, group["invite_code"])
                    item = database.create_deadline(1, "БД", "Лаба", self.now + timedelta(days=10),
                                                    group_id=group["id"])
                except sqlite3.Error as exc:
                    captured["cleanup_error"] = type(exc).__name__
                else:
                    captured["item_id"] = item["id"]
                    await captured["notify"](item["id"])

        async def polling(dispatcher, *args, **kwargs):
            task = asyncio.create_task(handler())
            captured["handler"] = task
            # aiogram keeps these tasks separate from its polling tasks and does
            # not await them in start_polling's shutdown implementation.
            dispatcher._handle_update_tasks.add(task)
            task.add_done_callback(dispatcher._handle_update_tasks.discard)
            await entered.wait()

        try:
            with patch("aiogram.Bot", return_value=bot), patch("bot.create_router", side_effect=router), \
                    patch("main.Database", return_value=database), \
                    patch("aiogram.Dispatcher.start_polling", new=polling):
                await run(self.settings)
            self.assertTrue(captured["handler"].cancelled(), "Update handler outlived the runtime")
            self.assertNotIn("cleanup_error", captured)
            self.assertEqual(bot.sent, [])
            with Database(self.path, clock=self.settings.now) as reopened:
                self.assertEqual([item["id"] for item in reopened.get_pending_publications(self.now)],
                                 [captured["item_id"]])
        finally:
            task = captured.get("handler")
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_publication_flood_wait_pauses_other_publications_and_reminders(self):
        from aiogram.exceptions import TelegramRetryAfter
        from aiogram.methods import SendMessage
        with Database(self.path, clock=self.settings.now) as db:
            group = db.create_group(1, "Группа")
            for user_id in (2, 3):
                db.join_group(user_id, group["invite_code"])
            db.create_deadline(1, "БД", "Общее", self.now + timedelta(days=10), group_id=group["id"])
            db.create_deadline(4, "Физика", "Личное", self.now + timedelta(days=2))
        bot = FakeBot()
        attempts = []

        async def send(user_id, text, **kwargs):
            attempts.append(user_id)
            bot.started.set()
            if len(attempts) == 1:
                raise TelegramRetryAfter(method=SendMessage(chat_id=user_id, text=text),
                                         message="Flood wait", retry_after=60)
            bot.sent.append((user_id, text))

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0.12)

        bot.send_message = send
        self.settings.scheduler_interval = 300
        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(attempts, [2], "Flood wait must pause all outbound deliveries")
        self.assertEqual(bot.sent, [])
        with Database(self.path, clock=self.settings.now) as db:
            self.assertEqual([row["user_id"] for row in db.get_pending_publications(self.now)], [3])
            self.assertEqual([row["user_id"] for row in db.get_due_reminders(self.now)], [4])

    async def test_reminder_flood_wait_pauses_other_recipients(self):
        from aiogram.exceptions import TelegramRetryAfter
        from aiogram.methods import SendMessage
        with Database(self.path, clock=self.settings.now) as db:
            for user_id in (1, 2):
                db.create_deadline(user_id, "БД", "Личное", self.now + timedelta(days=2))
        bot = FakeBot()
        attempts = []

        async def send(user_id, text, **kwargs):
            attempts.append(user_id)
            bot.started.set()
            if len(attempts) == 1:
                raise TelegramRetryAfter(method=SendMessage(chat_id=user_id, text=text),
                                         message="Flood wait", retry_after=60)
            bot.sent.append((user_id, text))

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0.08)

        bot.send_message = send
        self.settings.scheduler_interval = 300
        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(attempts, [1], "Flood wait must pause requests to other recipients")
        self.assertEqual(bot.sent, [])
        with Database(self.path, clock=self.settings.now) as db:
            self.assertEqual([row["user_id"] for row in db.get_due_reminders(self.now)], [2])

    async def test_flood_wait_is_rechecked_when_another_delivery_is_already_pacing(self):
        from aiogram.exceptions import TelegramRetryAfter
        from aiogram.methods import SendMessage
        from bot import create_router
        self.add_deadline()
        bot = FakeBot()
        captured = {}
        attempts = []
        release_first = asyncio.Event()
        pacing_started = asyncio.Event()
        sleep = asyncio.sleep

        def router(*args, **kwargs):
            captured["notify"] = kwargs["on_publication"]
            return create_router(*args, **kwargs)

        async def observed_sleep(delay):
            if 0 < delay <= 0.1:
                pacing_started.set()
            await sleep(delay)

        async def send(user_id, text, **kwargs):
            attempts.append(user_id)
            bot.started.set()
            if user_id == 1:
                await release_first.wait()
                raise TelegramRetryAfter(method=SendMessage(chat_id=user_id, text=text),
                                         message="Flood wait", retry_after=60)
            bot.sent.append((user_id, text))

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            item = self.add_publication()
            captured["publication_id"] = item["id"]
            await captured["notify"](item["id"])
            await asyncio.wait_for(pacing_started.wait(), 2)
            release_first.set()
            await sleep(0.08)

        bot.send_message = send
        self.settings.scheduler_interval = 300
        with patch("aiogram.Bot", return_value=bot), patch("bot.create_router", side_effect=router), \
                patch("main.asyncio.sleep", new=observed_sleep), \
                patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(attempts, [1], "A newly extended cooldown must stop an already waiting sender")
        with Database(self.path, clock=self.settings.now) as db:
            self.assertEqual([row["id"] for row in db.get_pending_publications(self.now)],
                             [captured["publication_id"]])

    async def test_startup_restores_pending_reminder_and_cleans_previous_semester(self):
        self.add_deadline()
        with Database(self.path, clock=lambda: datetime(2026, 8, 1)) as db:
            old = db.create_deadline(1, "История", "Сданная работа", "2026-08-02 18:00")
            db.complete_deadline(1, old["id"])
        bot = FakeBot()

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        bot.session.close.assert_awaited_once()
        with Database(self.path) as db:
            self.assertEqual(db.get_due_reminders(self.now), [])
            self.assertIsNone(db.get_deadline(1, old["id"]))

    async def test_interval_discovers_task_saved_after_startup(self):
        bot = FakeBot()

        async def polling(*args, **kwargs):
            await asyncio.sleep(0.01)  # Allow the initial empty batch to finish.
            self.add_deadline()
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        with Database(self.path) as db:
            self.assertEqual(db.get_due_reminders(self.now), [])

    async def test_shutdown_spends_window_after_transport_started(self):
        self.add_deadline()
        bot = FakeBot(block_send=True)

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(bot.sent, [])
        bot.session.close.assert_awaited_once()
        with Database(self.path) as db:
            self.assertEqual(db.get_due_reminders(self.now), [])
            self.assertEqual(db.get_due_reminders(self.now + timedelta(minutes=10)), [])

    async def test_shutdown_during_pacing_leaves_unattempted_window_available(self):
        first = self.add_deadline()
        second = self.add_deadline()
        bot = FakeBot()

        async def polling(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=polling):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        bot.session.close.assert_awaited_once()
        with Database(self.path) as db:
            eligible = db.get_due_reminders(self.now)
            self.assertEqual([item["id"] for item in eligible], [second["id"]])
            self.assertNotEqual(first["id"], second["id"])

    async def test_lost_ack_and_restarts_never_exceed_two_actual_reminders(self):
        self.add_deadline()
        bot = FakeBot(lose_ack=True)

        async def until_send(*args, **kwargs):
            await asyncio.wait_for(bot.started.wait(), 2)
            await asyncio.sleep(0)

        async def idle(*args, **kwargs):
            # Let startup restoration and two scheduler ticks observe the journal.
            await asyncio.sleep(0.07)

        with self.assertLogs("notifications", level="WARNING"):
            with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=until_send):
                await run(self.settings)
        self.assertEqual(len(bot.sent), 1)
        bot.lose_ack = False
        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=idle):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 1)

        self.now += timedelta(days=1)
        bot.started.clear()
        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=until_send):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 2)
        with patch("aiogram.Bot", return_value=bot), patch("aiogram.Dispatcher.start_polling", new=idle):
            await run(self.settings)
        self.assertEqual(len(bot.sent), 2)
        self.assertEqual(bot.session.close.await_count, 4)

    async def test_connection_failure_closes_resources(self):
        bot = FakeBot()
        bot.get_me.side_effect = ConnectionError("offline")
        with patch("aiogram.Bot", return_value=bot), self.assertRaises(ConnectionError):
            await run(self.settings)
        bot.session.close.assert_awaited_once()

    async def test_failed_startup_reminder_task_does_not_skip_resource_cleanup(self):
        bot = FakeBot()
        database = Database(self.path, clock=self.settings.now)

        async def polling(*args, **kwargs):
            await asyncio.sleep(0.01)

        with patch("main.Database", return_value=database), \
                patch.object(database, "get_due_reminders", side_effect=sqlite3.OperationalError("private-error-body")), \
                patch.object(database, "close", wraps=database.close) as close_database, \
                patch("aiogram.Bot", return_value=bot), \
                patch("aiogram.Dispatcher.start_polling", new=polling), \
                self.assertLogs("main", level="ERROR") as diagnostics:
            await run(self.settings)
        close_database.assert_called_once_with()
        bot.session.close.assert_awaited_once()
        self.assertIn("OperationalError", "\n".join(diagnostics.output))
        self.assertNotIn("private-error-body", "\n".join(diagnostics.output))

    async def test_failure_of_individual_closes_does_not_skip_remaining_resources(self):
        storage_close = AsyncMock(side_effect=RuntimeError("private-storage-body"))
        isolation_close = AsyncMock()
        bot = FakeBot()
        bot.session.close.side_effect = OSError("private-session-body")
        database = SimpleNamespace(close=Mock())
        dispatcher = SimpleNamespace(
            storage=SimpleNamespace(close=storage_close),
            fsm=SimpleNamespace(events_isolation=SimpleNamespace(close=isolation_close)),
        )
        with self.assertLogs("main", level="ERROR") as diagnostics:
            await _close_runtime_resources(bot, database, dispatcher)
        storage_close.assert_awaited_once()
        isolation_close.assert_awaited_once()
        bot.session.close.assert_awaited_once()
        database.close.assert_called_once_with()
        output = "\n".join(diagnostics.output)
        self.assertIn("RuntimeError", output)
        self.assertIn("OSError", output)
        self.assertNotIn("private-storage-body", output)
        self.assertNotIn("private-session-body", output)

    async def test_parser_setup_failure_closes_previously_created_database_and_bot(self):
        bot = FakeBot()
        database = Database(self.path, clock=self.settings.now)
        with patch("main.Database", return_value=database), \
                patch.object(database, "close", wraps=database.close) as close_database, \
                patch("aiogram.Bot", return_value=bot), \
                patch("main.create_parser", side_effect=ValueError("Invalid parser settings")), \
                self.assertRaises(ValueError):
            await run(self.settings)
        close_database.assert_called_once_with()
        bot.session.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
