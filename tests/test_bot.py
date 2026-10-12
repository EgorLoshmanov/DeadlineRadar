"""Conversation tests through the real dispatcher, without Telegram requests."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from db import Database
from parser import ParserEngine
from services import DeadlineService

try:
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.client.session.base import BaseSession
    from aiogram.exceptions import TelegramBadRequest
    from aiogram.fsm.storage.memory import MemoryStorage, SimpleEventIsolation
    from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, GetMe, SendMessage
    from aiogram.types import (
        CallbackQuery, Chat, InaccessibleMessage, InlineKeyboardMarkup, Message, MessageEntity,
        MessageOriginUser, PhotoSize, Update, User,
    )
    from bot import Flow, answer_callback, create_router
    HAS_AIOGRAM = True
except ModuleNotFoundError:
    BaseSession = object
    HAS_AIOGRAM = False


class RecordingSession(BaseSession):
    """Return Telegram-shaped responses and record calls without an HTTP client."""

    def __init__(self, now: datetime) -> None:
        super().__init__()
        self.now = now
        self.requests = []
        self.messages = []
        self.unhandled = []
        self.callback_error = None

    async def close(self) -> None:
        pass

    async def make_request(self, bot, method, timeout=None):
        self.requests.append(method)
        if isinstance(method, AnswerCallbackQuery) and self.callback_error:
            raise TelegramBadRequest(method=method, message=self.callback_error)
        if isinstance(method, GetMe):
            return User(id=bot.id, is_bot=True, first_name="DeadlineRadar", username="deadline_radar_test_bot")
        if isinstance(method, SendMessage):
            chat_id = int(method.chat_id)
            data = {
                "message_id": 10000 + len(self.messages), "date": self.now,
                "chat": Chat(id=chat_id, type="private" if chat_id > 0 else "supergroup"),
                "from_user": User(id=bot.id, is_bot=True, first_name="DeadlineRadar"),
                "text": method.text,
            }
            if isinstance(method.reply_markup, InlineKeyboardMarkup):
                data["reply_markup"] = method.reply_markup
            message = Message(**data).as_(bot)
            self.messages.append(message)
            return message
        if isinstance(method, (AnswerCallbackQuery, EditMessageReplyMarkup)):
            return True
        self.unhandled.append(type(method).__name__)
        raise AssertionError("The test session does not support this Telegram method")

    async def stream_content(self, *args, **kwargs):
        raise AssertionError("File downloads are not allowed in conversation tests")
        yield b""  # The session interface requires an asynchronous iterator.

    def last_message(self, chat_id: int = 1):
        return next(message for message in reversed(self.messages) if message.chat.id == chat_id)

    def last_callback_answer(self):
        return next(method for method in reversed(self.requests) if isinstance(method, AnswerCallbackQuery))


@unittest.skipUnless(HAS_AIOGRAM, "Install requirements.txt to run Telegram conversation tests")
class BotConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 10, 11, 12, 0)
        self.db = Database(Path(self.directory.name) / "bot.db", clock=lambda: self.now)
        self.settings = SimpleNamespace(
            now=lambda: self.now, timezone="Asia/Novosibirsk", llm_api_key="", llm_model="",
        )
        self.service = DeadlineService(self.db, ParserEngine(), clock=lambda: self.now)
        self.session = RecordingSession(self.now)
        self.bot = Bot(
            token="123456789:" + "x" * 35,
            session=self.session,
            default=DefaultBotProperties(parse_mode="HTML"),
        )
        self.dispatcher = Dispatcher(
            storage=MemoryStorage(), events_isolation=SimpleEventIsolation(),
        )
        self.dispatcher.include_router(create_router(self.db, self.service, self.settings, "deadline_radar_test_bot"))
        self.sequence = 0

    async def asyncTearDown(self) -> None:
        await self.dispatcher.storage.close()
        await self.dispatcher.fsm.events_isolation.close()
        await self.session.close()
        self.db.close()
        self.directory.cleanup()
        self.assertEqual(self.session.unhandled, [])

    def state(self, user_id: int = 1):
        return self.dispatcher.fsm.get_context(bot=self.bot, chat_id=user_id, user_id=user_id)

    async def send(self, text: str | None, user_id: int = 1, *, chat_id: int | None = None, forward: bool = False, photo: bool = False):
        self.sequence += 1
        chat_id = user_id if chat_id is None else chat_id
        data = {
            "message_id": self.sequence, "date": self.now,
            "chat": Chat(id=chat_id, type="private" if chat_id > 0 else "supergroup"),
            "from_user": User(id=user_id, is_bot=False, first_name=f"Student {user_id}", username=f"student{user_id}"),
        }
        if text is not None:
            data["text"] = text
            if text.startswith("/"):
                data["entities"] = [MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
        if forward:
            data["forward_origin"] = MessageOriginUser(
                type="user", date=self.now,
                sender_user=User(id=99, is_bot=False, first_name="Teacher"),
            )
        if photo:
            data["photo"] = [PhotoSize(file_id="offline-photo", file_unique_id="offline-photo", width=10, height=10)]
        await self.dispatcher.feed_update(self.bot, Update(update_id=self.sequence, message=Message(**data)))
        return self.session.last_message(chat_id)

    async def click(self, data: str, user_id: int = 1, message=None):
        self.sequence += 1
        if message is None:
            try:
                message = self.session.last_message(user_id)
            except StopIteration:
                message = Message(
                    message_id=999, date=self.now, chat=Chat(id=user_id, type="private"),
                    from_user=User(id=self.bot.id, is_bot=True, first_name="DeadlineRadar"), text="Menu",
                )
        query = CallbackQuery(
            id=f"callback-{self.sequence}", chat_instance=f"chat-{message.chat.id}",
            from_user=User(id=user_id, is_bot=False, first_name=f"Student {user_id}"),
            message=message, data=data,
        )
        await self.dispatcher.feed_update(self.bot, Update(update_id=self.sequence, callback_query=query))
        return self.session.last_callback_answer()

    def button(self, message, prefix: str) -> str:
        self.assertIsNotNone(message.reply_markup)
        for row in message.reply_markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
        self.fail(f"No button starts with {prefix}")

    async def preview(self, user_id: int = 1):
        return await self.send("Лаба 3 по БД до 15.10.2026 18:00", user_id, forward=True)

    async def shared_group(self):
        await self.send("/newgroup ИВТ-31", 1)
        group = self.db.list_groups(1)[0]
        await self.send(f"/join {group['invite_code']}", 2)
        return group

    async def test_forwarded_message_is_only_a_preview_until_confirmed(self) -> None:
        await self.send("/start")
        message = await self.preview()
        self.assertIn("Проверьте дедлайн", message.text)
        self.assertIn("Базы данных", message.text)
        self.assertEqual(await self.state().get_state(), Flow.preview.state)
        self.assertEqual(self.db.list_deadlines(1), [])
        self.assertTrue(self.button(message, "draft:save:"))

    async def test_concurrent_repeated_save_creates_one_deadline(self) -> None:
        message = await self.preview()
        callback = self.button(message, "draft:save:")
        await asyncio.gather(self.click(callback, message=message), self.click(callback, message=message))
        items = self.db.list_deadlines(1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["subject"], "Базы данных")
        self.assertEqual(items[0]["due_date"], "2026-10-15 18:00")
        self.assertIn("Уже сохранено", self.session.last_callback_answer().text)

    async def test_new_active_list_restarts_after_all_previous_tasks_are_completed(self) -> None:
        for number in range(4):
            item = self.db.create_deadline(1, "Физика", f"Сданная работа {number}",
                                           self.now + timedelta(days=1))
            self.db.complete_deadline(1, item["id"])
        preview = await self.preview()
        save = self.button(preview, "draft:save:")
        await self.click(save, message=preview)
        saved = self.session.last_message()
        self.assertIn("Дедлайн сохранён", saved.text)
        self.assertNotIn("#5", saved.text)
        item = self.db.list_deadlines(1)[0]
        self.assertEqual(item["id"], 5)

        listing = await self.send("/deadlines")
        self.assertEqual(re.findall(r"<b>#(\d+) ·", listing.text), ["1"])
        button = next(button for row in listing.reply_markup.inline_keyboard for button in row
                      if button.callback_data.startswith("done:"))
        self.assertEqual(button.text, "✅ Я сдал #1")
        self.assertEqual(button.callback_data, "done:5")
        await self.click(button.callback_data, message=listing)
        self.assertTrue(self.db.get_deadline(1, 5)["completed"])
        self.assertIn("Пока пусто", self.session.last_message().text)

        history = await self.send("/history")
        self.assertEqual(re.findall(r"<b>#(\d+) ·", history.text), ["1", "2", "3", "4"])
        await self.click("list:1:1", message=history)
        self.assertEqual(re.findall(r"<b>#(\d+) ·", self.session.last_message().text), ["5"])

    async def test_numbering_spans_pages_and_old_buttons_keep_their_original_target(self) -> None:
        for number in range(3):
            self.db.create_deadline(2, "Другой студент", f"Работа {number}",
                                    self.now + timedelta(days=1))
        items = [self.db.create_deadline(1, "Физика", f"Работа {number}",
                                        self.now + timedelta(days=number + 1))
                 for number in range(6)]
        first = await self.send("/deadlines")
        self.assertEqual(re.findall(r"<b>#(\d+) ·", first.text), ["1", "2", "3", "4"])
        await self.click("list:0:1", message=first)
        second = self.session.last_message()
        self.assertEqual(re.findall(r"<b>#(\d+) ·", second.text), ["5", "6"])
        buttons = [button for row in second.reply_markup.inline_keyboard for button in row
                   if button.callback_data.startswith("done:")]
        self.assertEqual([button.text for button in buttons], ["✅ Я сдал #5", "✅ Я сдал #6"])
        self.assertEqual(buttons[0].callback_data, f"done:{items[4]['id']}")

        await self.click(self.button(first, "done:"), message=first)
        self.assertEqual(re.findall(r"<b>#(\d+) ·", self.session.last_message().text),
                         ["1", "2", "3", "4"])
        await self.click(buttons[0].callback_data, message=second)
        self.assertTrue(self.db.get_deadline(1, items[4]["id"])["completed"])
        self.assertFalse(self.db.get_deadline(1, items[5]["id"])["completed"])

    async def test_delete_confirmation_names_task_and_numbers_close_gap(self) -> None:
        old = self.db.create_deadline(1, "Физика", "Сданная работа", self.now + timedelta(days=1))
        self.db.complete_deadline(1, old["id"])
        remove = self.db.create_deadline(1, "<БД>", "Код & отчёт", self.now + timedelta(days=2))
        remaining = self.db.create_deadline(1, "Питон", "Код", self.now + timedelta(days=3))
        listing = await self.send("/deadlines")
        buttons = [button for row in listing.reply_markup.inline_keyboard for button in row
                   if button.callback_data.startswith("remove:ask:")]
        self.assertEqual([button.text for button in buttons], ["🗑 #1", "🗑 #2"])
        self.assertEqual(buttons[0].callback_data, f"remove:ask:{remove['id']}")
        await self.click(buttons[0].callback_data, message=listing)
        confirmation = self.session.last_message()
        self.assertIn("&lt;БД&gt;", confirmation.text)
        self.assertIn("Код &amp; отчёт", confirmation.text)
        await self.click(self.button(confirmation, "remove:yes:"), message=confirmation)
        self.assertIsNone(self.db.get_deadline(1, remove["id"]))
        refreshed = self.session.last_message()
        self.assertEqual(re.findall(r"<b>#(\d+) ·", refreshed.text), ["1"])
        self.assertEqual(self.button(refreshed, "done:"), f"done:{remaining['id']}")

    async def test_publication_notifies_member_once_and_done_is_individual(self) -> None:
        group = await self.shared_group()
        await self.send("/start", 3)
        before = len(self.session.messages)
        await self.click(f"publish:{group['id']}", 1)
        preview = await self.send("Лаба 3 по БД до 25.10.2026", 1)
        save = self.button(preview, "draft:save:")
        self.assertFalse(any("Новое задание группы" in m.text for m in self.session.messages[before:]))
        await self.click(save, 1, message=preview)
        publications = [m for m in self.session.messages[before:] if "Новое задание группы" in m.text]
        self.assertEqual(len(publications), 1)
        self.assertEqual(publications[0].chat.id, 2)
        self.assertIn("ИВТ-31", publications[0].text)
        await self.click(save, 1, message=preview)
        self.assertEqual(sum("Новое задание группы" in m.text for m in self.session.messages[before:]), 1)
        item = self.db.list_deadlines(2)[0]
        await self.click(self.button(publications[0], "done:"), 2, message=publications[0])
        self.assertTrue(self.db.get_deadline(2, item["id"])["completed"])
        self.assertFalse(self.db.get_deadline(1, item["id"])["completed"])
        self.assertEqual(self.db.list_deadlines(3), [])

    async def test_limit_error_preserves_draft_and_plan_upgrade_allows_retry(self) -> None:
        for number in range(15):
            self.db.create_deadline(1, "БД", f"Задание {number}", self.now + timedelta(days=10))
        preview = await self.preview()
        save = self.button(preview, "draft:save:")
        answer = await self.click(save, message=preview)
        self.assertTrue(answer.show_alert)
        self.assertIn("15/15", answer.text)
        self.assertEqual(await self.state().get_state(), Flow.preview.state)
        self.assertEqual(len(self.db.list_deadlines(1)), 15)
        plan = await self.send("/plan student")
        self.assertIn("Студент", plan.text)
        self.assertEqual(await self.state().get_state(), Flow.preview.state)
        await self.click(save, message=preview)
        self.assertEqual(len(self.db.list_deadlines(1)), 16)
        await self.click("plan:free", message=plan)
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "free")
        self.assertEqual(len(self.db.list_deadlines(1)), 16)

    async def test_plan_choices_are_local_and_invalid_plan_changes_nothing(self) -> None:
        message = await self.send("/plan")
        self.assertIn("Бесплатный", message.text)
        self.assertIn("15", message.text)
        self.assertIn("оплаты нет", message.text)
        await self.click("plan:group", message=message)
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "group")
        await self.send("/plan unknown")
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "group")
        self.assertIn("План не найден", self.session.last_message().text)
        self.assertEqual(self.db.list_groups(1), [])

    async def test_expired_callback_toast_does_not_abort_or_duplicate_save(self) -> None:
        preview = await self.preview()
        save = self.button(preview, "draft:save:")
        self.session.callback_error = "Bad Request: query is too old and response timeout expired or query ID is invalid"
        with self.assertNoLogs("bot", level="ERROR"):
            await self.click(save, message=preview)
            await self.click(save, message=preview)
        self.assertEqual(len(self.db.list_deadlines(1)), 1)
        self.assertIsNone(await self.state().get_state())
        self.assertIn("сохранён", self.session.last_message().text)

    async def test_expired_plan_toast_keeps_draft_and_refreshes_plan(self) -> None:
        preview = await self.preview()
        draft_before = deepcopy(await self.state().get_data())
        plan = await self.send("/plan")
        self.session.callback_error = "Bad Request: query ID is invalid"
        with self.assertNoLogs("bot", level="ERROR"):
            await self.click("plan:student", message=plan)
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "student")
        self.assertIn("Студент", self.session.last_message().text)
        self.assertEqual(await self.state().get_data(), draft_before)
        self.assertEqual(await self.state().get_state(), Flow.preview.state)
        await self.click(self.button(preview, "draft:save:"), message=preview)
        self.assertEqual(len(self.db.list_deadlines(1)), 1)

    async def test_expired_account_toast_still_clears_confirmation_and_replies(self) -> None:
        self.db.create_deadline(1, "БД", "Задание", self.now + timedelta(days=3))
        confirmation = await self.send("/delete_me")
        self.session.callback_error = "Bad Request: query is too old and response timeout expired"
        with self.assertNoLogs("bot", level="ERROR"):
            await self.click(self.button(confirmation, "account:delete:"), message=confirmation)
        self.assertIsNone(self.db.get_user(1))
        self.assertEqual(self.db.list_deadlines(1, include_completed=True), [])
        self.assertIsNone(await self.state().get_state())
        self.assertEqual(await self.state().get_data(), {})
        self.assertIn("Ваши данные удалены", self.session.last_message().text)

    async def test_callback_helper_does_not_hide_other_bad_requests(self) -> None:
        self.session.callback_error = "Bad Request: BUTTON_DATA_INVALID"
        query = CallbackQuery(
            id="invalid-request", chat_instance="chat-1",
            from_user=User(id=1, is_bot=False, first_name="Student"), data="dismiss",
        ).as_(self.bot)
        with self.assertRaises(TelegramBadRequest):
            await answer_callback(query, "Ответ")

    async def test_save_from_inaccessible_preview_still_confirms_and_replies(self) -> None:
        preview = await self.preview()
        inaccessible = InaccessibleMessage(chat=preview.chat, message_id=preview.message_id, date=0)
        before = len(self.session.messages)
        answer = await self.click(self.button(preview, "draft:save:"), message=inaccessible)
        self.assertEqual(answer.text, "Сохранено")
        self.assertEqual(len(self.db.list_deadlines(1)), 1)
        self.assertIsNone(await self.state().get_state())
        self.assertEqual(len(self.session.messages), before + 1)
        self.assertIn("сохранён", self.session.last_message().text)

    async def test_done_from_inaccessible_listing_still_completes_and_refreshes(self) -> None:
        item = self.db.create_deadline(1, "Физика", "Лабораторная", self.now + timedelta(days=2))
        listing = await self.send("/deadlines")
        inaccessible = InaccessibleMessage(chat=listing.chat, message_id=listing.message_id, date=0)
        before = len(self.session.messages)
        answer = await self.click(self.button(listing, "done:"), message=inaccessible)
        self.assertIn("Отмечено", answer.text)
        self.assertTrue(self.db.get_deadline(1, item["id"])["completed"])
        self.assertEqual(len(self.session.messages), before + 1)
        self.assertIn("Пока пусто", self.session.last_message().text)

    async def test_subject_and_date_correction_keep_one_confirmation(self) -> None:
        original = await self.preview()
        await self.click(self.button(original, "draft:subject:"), message=original)
        corrected_subject = await self.send("Проектирование баз данных")
        self.assertIn("Проектирование баз данных", corrected_subject.text)
        await self.click(self.button(corrected_subject, "draft:date:"), message=corrected_subject)
        corrected_date = await self.send("20.10.2026 17:30")
        self.assertEqual(self.db.list_deadlines(1), [])
        await self.click(self.button(corrected_date, "draft:save:"), message=corrected_date)
        item = self.db.list_deadlines(1)[0]
        self.assertEqual(item["subject"], "Проектирование баз данных")
        self.assertEqual(item["due_date"], "2026-10-20 17:30")
        stale = await self.click(self.button(original, "draft:save:"), message=original)
        self.assertTrue(stale.show_alert)
        self.assertEqual(len(self.db.list_deadlines(1)), 1)

    async def test_ambiguous_message_requests_explicit_date(self) -> None:
        message = await self.send("Физика: подготовить доклад к следующей паре", forward=True)
        self.assertIn("Не удалось однозначно определить срок", message.text)
        self.assertEqual(await self.state().get_state(), Flow.date.state)
        self.assertEqual(self.db.list_deadlines(1), [])
        rejected = await self.send("к следующей паре")
        self.assertIn("формате", rejected.text)
        self.assertEqual(await self.state().get_state(), Flow.date.state)
        preview = await self.send("16.10.2026")
        await self.click(self.button(preview, "draft:save:"), message=preview)
        item = self.db.list_deadlines(1)[0]
        self.assertEqual(item["subject"], "Физика")
        self.assertEqual(item["due_date"], "2026-10-16 23:59")

    async def test_cancel_invalidates_old_preview_and_foreign_callback(self) -> None:
        message = await self.preview()
        callback = self.button(message, "draft:save:")
        foreign = await self.click(callback, user_id=2)
        self.assertTrue(foreign.show_alert)
        self.assertEqual(self.db.list_deadlines(2), [])
        await self.send("/cancel")
        answer = await self.click(callback, message=message)
        self.assertIn("устарела", answer.text)
        self.assertIsNone(await self.state().get_state())
        self.assertEqual(self.db.list_deadlines(1), [])

    async def test_out_of_range_callback_ids_do_not_raise_or_change_data(self) -> None:
        item = self.db.create_deadline(1, "Физика", "Лабораторная", self.now + timedelta(days=2))
        preview = await self.preview()
        state_before = deepcopy(await self.state().get_data())
        for value in ("0", "-1", str(2**63), "9" * 30):
            for prefix in ("done:", "remove:ask:", "remove:yes:", "publish:", "invite:", "leave:ask:", "leave:yes:"):
                with self.subTest(prefix=prefix, value=value):
                    callback = prefix + value
                    self.assertLessEqual(len(callback.encode("utf-8")), 64)
                    with self.assertNoLogs("bot", level="ERROR"):
                        answer = await self.click(callback, message=preview)
                    self.assertNotIn("Не удалось выполнить действие", answer.text)
                    self.assertEqual(await self.state().get_state(), Flow.preview.state)
                    self.assertEqual(await self.state().get_data(), state_before)
                    self.assertEqual(len(self.db.list_deadlines(1)), 1)
                    self.assertFalse(self.db.get_deadline(1, item["id"])["completed"])

    async def test_largest_sqlite_id_can_still_be_completed(self) -> None:
        item = self.db.create_deadline(1, "Физика", "Лабораторная", self.now + timedelta(days=2))
        maximum_id = 2**63 - 1
        with self.db._transaction() as connection:
            connection.execute("UPDATE deadlines SET id = ? WHERE id = ?", (maximum_id, item["id"]))
        listing = await self.send("/deadlines")
        with self.assertNoLogs("bot", level="ERROR"):
            answer = await self.click(self.button(listing, "done:"), message=listing)
        self.assertIn("Отмечено", answer.text)
        self.assertTrue(self.db.get_deadline(1, maximum_id)["completed"])

    async def test_group_owner_publishes_but_member_cannot_use_owner_callback(self) -> None:
        group = await self.shared_group()
        rejected = await self.click(f"publish:{group['id']}", user_id=2)
        self.assertTrue(rejected.show_alert)
        self.assertIn("старосте", rejected.text)
        self.assertIsNone(await self.state(2).get_state())
        await self.click(f"publish:{group['id']}", user_id=1)
        self.assertEqual(await self.state(1).get_state(), Flow.publish_text.state)
        preview = await self.preview()
        self.assertIn("ИВТ-31", preview.text)
        self.assertEqual(self.db.list_deadlines(2), [])
        await self.click(self.button(preview, "draft:save:"), message=preview)
        owner_items, member_items = self.db.list_deadlines(1), self.db.list_deadlines(2)
        self.assertEqual(len(owner_items), 1)
        self.assertEqual(owner_items[0]["id"], member_items[0]["id"])
        self.assertEqual(member_items[0]["group_id"], group["id"])

    async def test_group_completion_applies_only_to_clicking_student(self) -> None:
        group = await self.shared_group()
        deadline = self.db.create_deadline(1, "Физика", "Лабораторная", self.now + timedelta(days=2), group_id=group["id"])
        listing = await self.send("/deadlines", user_id=2)
        await self.click(self.button(listing, "done:"), user_id=2, message=listing)
        self.assertEqual(self.db.get_deadline(2, deadline["id"])["status"], "completed")
        self.assertTrue(self.db.get_deadline(2, deadline["id"])["completed_at"].startswith("2026-10-11 12:00"))
        self.assertEqual(self.db.get_deadline(1, deadline["id"])["status"], "pending")
        self.assertEqual([item["user_id"] for item in self.db.get_due_reminders(self.now)], [1])
        history = await self.send("/history", user_id=2)
        self.assertIn("Лабораторная", history.text)
        self.assertIn("Сдано", history.text)

    async def test_archived_group_completion_is_hidden_from_history_without_reviving_task(self) -> None:
        self.now = datetime(2026, 8, 20, 12)
        group = await self.shared_group()
        shared = self.db.create_deadline(
            1, "Базы данных", "Архивная общая работа",
            datetime(2026, 8, 25, 18), group_id=group["id"],
        )
        old_listing = await self.send("/deadlines", user_id=1)
        await self.click(f"done:{shared['id']}", user_id=1, message=old_listing)
        self.now = datetime(2026, 10, 11, 12)
        personal = self.db.create_deadline(
            1, "Физика", "Текущая личная работа", self.now + timedelta(days=2),
        )
        current_listing = await self.send("/deadlines", user_id=1)
        await self.click(f"done:{personal['id']}", user_id=1, message=current_listing)
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 0)
        archived = self.db.get_deadline(1, shared["id"])
        self.assertTrue(archived["completed"])
        self.assertIsNone(archived["completed_at"])
        self.assertIsNotNone(self.db.get_deadline(1, personal["id"])["completed_at"])
        history = await self.send("/history", user_id=1)
        self.assertIn("Текущая личная работа", history.text)
        self.assertNotIn("Архивная общая работа", history.text)
        self.assertNotIn("Не удалось выполнить действие", history.text)
        owner_active = await self.send("/deadlines", user_id=1)
        self.assertNotIn("Архивная общая работа", owner_active.text)
        self.assertNotIn("Текущая личная работа", owner_active.text)
        member_active = await self.send("/deadlines", user_id=2)
        self.assertIn("Архивная общая работа", member_active.text)
        self.assertEqual(self.db.get_deadline(2, shared["id"])["status"], "pending")

    async def test_member_cannot_delete_shared_deadline_and_owner_confirms_deletion(self) -> None:
        group = await self.shared_group()
        item = self.db.create_deadline(1, "Физика", "Лабораторная", self.now + timedelta(days=2), group_id=group["id"])
        listing = await self.send("/deadlines", user_id=2)
        callbacks = [button.callback_data for row in listing.reply_markup.inline_keyboard for button in row]
        self.assertFalse(any(data.startswith("remove:") for data in callbacks))
        for action in ("ask", "yes"):
            with self.subTest(action=action):
                rejected = await self.click(f"remove:{action}:{item['id']}", user_id=2, message=listing)
                self.assertTrue(rejected.show_alert)
                self.assertIsNotNone(self.db.get_deadline(1, item["id"]))
        owner_list = await self.send("/deadlines", user_id=1)
        await self.click(self.button(owner_list, "remove:ask:"), message=owner_list)
        confirmation = self.session.last_message(1)
        self.assertIn("у всех участников", confirmation.text)
        self.assertIsNotNone(self.db.get_deadline(2, item["id"]))
        await self.click(self.button(confirmation, "remove:yes:"), message=confirmation)
        self.assertIsNone(self.db.get_deadline(1, item["id"]))
        self.assertIsNone(self.db.get_deadline(2, item["id"]))

    async def test_owner_can_delete_shared_deadline_after_own_completion(self) -> None:
        group = await self.shared_group()
        item = self.db.create_deadline(
            1, "Физика", "Лабораторная", self.now + timedelta(days=2), group_id=group["id"],
        )
        listing = await self.send("/deadlines")
        await self.click(self.button(listing, "done:"), message=listing)
        self.assertEqual(self.db.list_deadlines(1), [])
        self.assertFalse(self.db.get_deadline(2, item["id"])["completed"])

        history = await self.send("/history")
        self.assertIn("Лабораторная", history.text)
        await self.click(self.button(history, "remove:ask:"), message=history)
        confirmation = self.session.last_message()
        self.assertIn("у всех участников", confirmation.text)
        self.assertIsNotNone(self.db.get_deadline(2, item["id"]))
        await self.click(self.button(confirmation, "remove:yes:"), message=confirmation)
        self.assertIsNone(self.db.get_deadline(1, item["id"]))
        self.assertIsNone(self.db.get_deadline(2, item["id"]))

    async def test_completed_group_history_keeps_member_delete_restricted(self) -> None:
        group = await self.shared_group()
        item = self.db.create_deadline(
            1, "Физика", "Лабораторная", self.now + timedelta(days=2), group_id=group["id"],
        )
        listing = await self.send("/deadlines", user_id=2)
        await self.click(self.button(listing, "done:"), user_id=2, message=listing)
        history = await self.send("/history", user_id=2)
        callbacks = [button.callback_data for row in history.reply_markup.inline_keyboard for button in row]
        self.assertFalse(any(data.startswith(("remove:", "done:")) for data in callbacks))
        rejected = await self.click(f"remove:yes:{item['id']}", user_id=2, message=history)
        self.assertTrue(rejected.show_alert)
        self.assertIsNotNone(self.db.get_deadline(1, item["id"]))

    async def test_group_chat_content_is_not_parsed_or_persisted(self) -> None:
        group_reply = await self.send("Лаба 3 по БД до 15.10.2026", chat_id=-100123)
        self.assertIn("личном диалоге", group_reply.text)
        self.assertIsNone(self.db.get_user(1))
        self.assertEqual(self.db.list_deadlines(1), [])
        self.assertIsNone(await self.state().get_state())
        answer = await self.click("done:1", message=group_reply)
        self.assertIn("личный диалог", answer.text)
        self.assertIsNone(self.db.get_user(1))

    async def test_invalid_manual_inputs_do_not_advance_or_save_draft(self) -> None:
        oversized = await self.send("x" * 1001)
        self.assertIn("1000", oversized.text)
        await self.send("/add")
        for value in ("   ", "x" * 121):
            with self.subTest(value=value):
                await self.send(value)
                self.assertEqual(await self.state().get_state(), Flow.subject.state)
        await self.send("Физика")
        await self.send("Подготовить доклад")
        for value in ("31.02.2026", "10.10.2026", "завтра"):
            with self.subTest(value=value):
                await self.send(value)
                self.assertEqual(await self.state().get_state(), Flow.date.state)
                self.assertEqual(self.db.list_deadlines(1), [])
        preview = await self.send("16.10.2026 19:00")
        await self.click(self.button(preview, "draft:save:"), message=preview)
        self.assertEqual(self.db.list_deadlines(1)[0]["task"], "Подготовить доклад")

    async def test_unknown_commands_do_not_become_manual_fields(self) -> None:
        for stage in ("subject", "task", "date"):
            with self.subTest(stage=stage):
                await self.send("/add")
                if stage in {"task", "date"}:
                    await self.send("Физика")
                if stage == "date":
                    await self.send("Подготовить доклад")
                old_state = await self.state().get_state()
                old_data = deepcopy(await self.state().get_data())
                response = await self.send("/unavailable_command")
                self.assertIn("Неизвестная команда", response.text)
                self.assertEqual(await self.state().get_state(), old_state)
                self.assertEqual(await self.state().get_data(), old_data)
                self.assertEqual(self.db.list_deadlines(1), [])

    async def test_known_commands_keep_manual_subject_unchanged(self) -> None:
        await self.send("/add")
        old_data = deepcopy(await self.state().get_data())
        await self.send("/help")
        self.assertEqual(await self.state().get_state(), Flow.subject.state)
        self.assertEqual(await self.state().get_data(), old_data)
        await self.send("Физика")
        self.assertEqual((await self.state().get_data())["draft"]["subject"], "Физика")

    async def test_unknown_command_does_not_create_a_group(self) -> None:
        await self.send("/newgroup")
        response = await self.send("/unavailable_command")
        self.assertIn("Неизвестная команда", response.text)
        self.assertEqual(await self.state().get_state(), Flow.group_name.state)
        self.assertEqual(self.db.list_groups(1), [])

    async def test_photos_are_rejected_without_creating_deadline(self) -> None:
        response = await self.send(None, photo=True)
        self.assertIn("Пришлите текст", response.text)
        self.assertEqual(self.db.list_deadlines(1), [])
        self.assertIsNone(await self.state().get_state())

    async def test_settings_toggle_delivery_and_start_preserves_user_choice(self) -> None:
        settings = await self.send("/settings")
        await self.click(self.button(settings, "settings:delivery:0"), message=settings)
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        await self.send("/deadlines")
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        settings = await self.send("/settings")
        await self.click(self.button(settings, "settings:delivery:1"), message=settings)
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])
        await self.click("settings:delivery:0")
        await self.send("/start")
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        settings = await self.send("/settings")
        await self.click(self.button(settings, "settings:delivery:1"), message=settings)
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])

    async def test_settings_names_configured_provider_and_preserves_legacy_default(self) -> None:
        self.settings.llm_api_key = "offline-test-key"
        self.settings.llm_model = "offline-test-model"
        for provider, api_url, expected, excluded in (
            ("gigachat", "https://api.giga.chat/v1/chat/completions", "GigaChat", "Anthropic"),
            ("anthropic", "https://api.anthropic.com/v1/messages", "Anthropic", "GigaChat"),
            ("openai", "https://deepcode.ci.nsu.ru/api/chat/completions", "API НГУ", "Anthropic"),
            ("openai", "https://other.example/v1/chat/completions", "совместимый API", "API НГУ"),
            (None, "https://api.anthropic.com/v1/messages", "Anthropic", "GigaChat"),
        ):
            with self.subTest(provider=provider, api_url=api_url):
                if provider is None:
                    del self.settings.llm_provider
                else:
                    self.settings.llm_provider = provider
                self.settings.llm_api_url = api_url
                message = await self.send("/settings")
                self.assertIn(f"отправляется провайдеру {expected}", message.text)
                self.assertNotIn(excluded, message.text)
                self.assertNotIn("Распознавание выполняется локально", message.text)

    async def test_settings_describes_local_parsing_when_key_or_model_is_empty(self) -> None:
        for provider in ("gigachat", "openai"):
            self.settings.llm_provider = provider
            for key, model in (("", "offline-test-model"), ("offline-test-key", "")):
                with self.subTest(provider=provider, key_present=bool(key), model_present=bool(model)):
                    self.settings.llm_api_key = key
                    self.settings.llm_model = model
                    message = await self.send("/settings")
                    self.assertIn("Распознавание выполняется локально", message.text)
                    self.assertNotIn("отправляется провайдеру", message.text)

    async def test_start_deep_link_joins_group_without_enabling_paused_delivery(self) -> None:
        group = self.db.create_group(2, "ИВТ-32")
        self.db.set_delivery_enabled(1, False)
        welcome = await self.send(f"/start g_{group['invite_code']}")
        self.assertIn("Вы в группе", welcome.text)
        self.assertEqual(self.db.list_groups(1)[0]["id"], group["id"])
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        settings = await self.send("/settings")
        await self.click(self.button(settings, "settings:delivery:1"), message=settings)
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])

    async def test_join_command_preserves_entire_generated_invite_code(self) -> None:
        for user_id, code in enumerate(("plaincode123", "g_1234567890", "g_g_12345678"), 2):
            with self.subTest(code=code):
                with patch("db.secrets.token_urlsafe", return_value=code):
                    invitation = await self.send("/newgroup ИВТ-31")
                group = next(group for group in self.db.list_groups(1) if group["invite_code"] == code)
                self.assertIn(f"/join {code}", invitation.text)
                response = await self.send(f"/join {code}", user_id)
                self.assertIn("Вы в группе", response.text)
                self.assertEqual([entry["id"] for entry in self.db.list_groups(user_id)], [group["id"]])

    async def test_join_prompt_preserves_raw_codes_with_start_prefix(self) -> None:
        for user_id, code in enumerate(("g_1234567890", "g_g_12345678"), 2):
            with self.subTest(code=code):
                with patch("db.secrets.token_urlsafe", return_value=code):
                    group = self.db.create_group(1, "ИВТ-31")
                await self.send("/start", user_id)
                await self.click("group:join", user_id)
                response = await self.send(code, user_id)
                self.assertIn("Вы в группе", response.text)
                self.assertEqual(self.db.list_groups(user_id)[0]["id"], group["id"])
                self.assertIsNone(await self.state(user_id).get_state())

    async def test_group_links_strip_only_the_added_start_wrapper(self) -> None:
        user_id = 2
        for code in ("plaincode123", "g_1234567890", "g_g_12345678"):
            with patch("db.secrets.token_urlsafe", return_value=code):
                group = self.db.create_group(1, "ИВТ-31")
            link = f"https://t.me/deadline_radar_test_bot?start=g_{code}"
            for entry in ("start", "join_url", "prompt_url"):
                with self.subTest(code=code, entry=entry):
                    if entry == "start":
                        response = await self.send(f"/start g_{code}", user_id)
                    elif entry == "join_url":
                        response = await self.send(f"/join {link}", user_id)
                    else:
                        await self.send("/start", user_id)
                        await self.click("group:join", user_id)
                        response = await self.send(link, user_id)
                    self.assertIn("Вы в группе", response.text)
                    self.assertEqual(self.db.list_groups(user_id)[0]["id"], group["id"])
                    user_id += 1

    async def test_join_command_still_accepts_a_bare_start_payload(self) -> None:
        for user_id, code in enumerate(("plaincode123", "g_1234567890", "g_g_12345678"), 2):
            with self.subTest(code=code):
                with patch("db.secrets.token_urlsafe", return_value=code):
                    group = self.db.create_group(1, "ИВТ-31")
                response = await self.send(f"/join g_{code}", user_id)
                self.assertIn("Вы в группе", response.text)
                self.assertEqual(self.db.list_groups(user_id)[0]["id"], group["id"])


if __name__ == "__main__":
    unittest.main()
