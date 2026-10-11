"""Use-case validation and safe rendering tests, independent of Telegram."""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from db import Database
from models import DeadlineDraft, MAX_SUBJECT_LENGTH, MAX_TASK_LENGTH
from parser import ParserEngine
from presentation import deadline_list, draft_card, reminder_text
from services import DeadlineService


class DeadlineServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "services.db")
        self.now = datetime(2026, 10, 11, 12, 0)
        self.parser = Mock()
        self.parser.parse = AsyncMock(return_value={
            "subject": "Математика", "task": "Решить задачи 1–5",
            "due_date": "2026-10-15 23:59", "parsing_method": "local",
        })
        self.service = DeadlineService(self.db, self.parser, clock=lambda: self.now)

    def tearDown(self) -> None:
        self.db.close()
        self.directory.cleanup()

    def draft(self, **changes: object) -> dict:
        return {
            "subject": "Математика", "task": "Решить задачи",
            "due_date": "2026-10-15 23:59", "group_id": None,
            "creation_key": "confirmation-123", **changes,
        }

    async def test_propose_has_stable_payload_and_unique_confirmation_key(self) -> None:
        proposed = await self.service.propose(1, "Математика, задачи к четвергу")
        self.parser.parse.assert_awaited_once_with("Математика, задачи к четвергу", now=self.now)
        self.assertEqual(proposed["subject"], "Математика")
        self.assertIsNone(proposed["group_id"])
        self.assertEqual(len(proposed["creation_key"]), 32)
        self.assertEqual(len(self.db.list_deadlines(1)), 0)
        next_proposal = await self.service.propose(1, "Другое сообщение")
        self.assertNotEqual(next_proposal["creation_key"], proposed["creation_key"])

    async def test_non_owner_cannot_propose_group_deadline(self) -> None:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        for user_id in (2, 3):
            with self.subTest(user_id=user_id), self.assertRaises(ValueError):
                await self.service.propose(user_id, "Задание", group["id"])
        self.parser.parse.assert_not_awaited()

    async def test_owner_can_propose_and_confirm_group_deadline(self) -> None:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        proposed = await self.service.propose(1, "Задание", group["id"])
        self.assertEqual(proposed["group_id"], group["id"])
        saved = self.service.confirm(1, proposed)
        self.assertEqual(saved["group_id"], group["id"])
        self.assertEqual(self.db.get_deadline(2, saved["id"])["status"], "pending")

    async def test_confirm_checks_owner_even_if_proposal_is_forged(self) -> None:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        with self.assertRaises(ValueError):
            self.service.confirm(2, self.draft(group_id=group["id"]))
        self.assertEqual(self.db.list_deadlines(1), [])

    async def test_repeated_confirmation_creates_exactly_one_deadline(self) -> None:
        proposed = await self.service.propose(1, "Задание")
        first = self.service.confirm(1, proposed)
        second = self.service.confirm(1, proposed)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.db.list_deadlines(1)), 1)
        altered = {**proposed, "task": "Другая работа"}
        with self.assertRaises(ValueError):
            self.service.confirm(1, altered)

    async def test_confirm_requires_future_date_at_time_of_confirmation(self) -> None:
        for value in ("2026-10-10 23:59", "2026-10-11 12:00", "2026-02-30 12:00", "2026-10-11"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.confirm(1, self.draft(due_date=value))
        saved = self.service.confirm(1, self.draft(due_date="2026-10-11 12:01"))
        self.assertEqual(saved["due_date"], "2026-10-11 12:01")

    async def test_draft_can_expire_while_waiting_for_confirmation(self) -> None:
        self.parser.parse.return_value["due_date"] = "2026-10-11 12:01"
        proposed = await self.service.propose(1, "Задание")
        self.now += timedelta(minutes=1)
        with self.assertRaises(ValueError):
            self.service.confirm(1, proposed)
        self.assertEqual(self.db.list_deadlines(1), [])

    async def test_confirmation_normalizes_fields_and_rejects_invalid_payloads(self) -> None:
        saved = self.service.confirm(1, self.draft(subject="  Высшая\nматематика  ", task="  Решить\t задачи   "))
        self.assertEqual(saved["subject"], "Высшая математика")
        self.assertEqual(saved["task"], "Решить задачи")
        invalid = [
            {"subject": "   "}, {"subject": "S" * 121}, {"task": "T" * 501},
            {"task": None}, {"group_id": True}, {"group_id": "1"}, {"group_id": -1},
            {"creation_key": 123}, {"creation_key": "x" * 65},
            {"due_date": "2026-1-15 23:59"},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DeadlineDraft.from_dict(self.draft(**changes), now=self.now)

    async def test_real_parser_preserves_assignment_details_through_confirmation(self) -> None:
        service = DeadlineService(self.db, ParserEngine(use_llm=False), clock=lambda: self.now)
        for text, expected_task, expected_due in (
            ("Лаба 3 по БД: создать таблицы завтра в 18 часов 30 минут",
             "Лабораторная работа №3: создать таблицы", "2026-10-12 18:30"),
            ("ДЗ по БД: решить задачи 10-18 до пятницы",
             "Домашнее задание: решить задачи 10-18", "2026-10-16 23:59"),
            ("Реферат по истории на тему Древняя Греция до пятницы",
             "Реферат: на тему Древняя Греция", "2026-10-16 23:59"),
        ):
            with self.subTest(text=text):
                proposed = await service.propose(1, text)
                self.assertEqual(proposed["status"], "ok")
                saved = service.confirm(1, proposed)
                self.assertEqual(saved["task"], expected_task)
                self.assertEqual(saved["due_date"], expected_due)

    async def test_uncertain_time_or_class_schedule_cannot_be_confirmed(self) -> None:
        service = DeadlineService(self.db, ParserEngine(use_llm=False), clock=lambda: self.now)
        for text in ("Лаба по БД завтра в 18:3", "Лаба по БД к следующей паре завтра"):
            with self.subTest(text=text):
                proposed = await service.propose(1, text)
                self.assertEqual(proposed["status"], "manual_input_required")
                with self.assertRaises(ValueError):
                    service.confirm(1, proposed)
        self.assertEqual(self.db.list_deadlines(1), [])

    async def test_shared_field_limits_accept_boundary_and_reject_one_more(self) -> None:
        saved = self.service.confirm(1, self.draft(subject="п" * MAX_SUBJECT_LENGTH,
                                                  task="з" * MAX_TASK_LENGTH))
        self.assertEqual(len(saved["subject"]), MAX_SUBJECT_LENGTH)
        self.assertEqual(len(saved["task"]), MAX_TASK_LENGTH)
        for changes in ({"subject": "п" * (MAX_SUBJECT_LENGTH + 1)},
                        {"task": "з" * (MAX_TASK_LENGTH + 1)}):
            with self.subTest(field=next(iter(changes))), self.assertRaises(ValueError):
                self.service.confirm(1, self.draft(creation_key="another-draft", **changes))

    async def test_overlong_remote_fields_fall_back_before_user_can_save(self) -> None:
        parser = ParserEngine(api_key="test", model="configured-model")
        service = DeadlineService(self.db, parser, clock=lambda: self.now)
        remote = {"subject": "п" * (MAX_SUBJECT_LENGTH + 1), "task": "Работа",
                  "due_date": "2026-10-16 23:59", "confidence": 0.99}
        with patch.object(parser, "_request_llm", return_value=remote):
            proposed = await service.propose(1, "Лаба 3 по БД до пятницы")
        self.assertEqual(proposed["parsing_method"], "regex_fallback")
        saved = service.confirm(1, proposed)
        self.assertEqual(saved["subject"], "Базы данных")
        self.assertEqual(saved["task"], "Лабораторная работа №3")

    async def test_user_markup_is_escaped_in_cards_lists_and_reminders(self) -> None:
        subject = '<b>Физика & "математика"</b>'
        task = '<script>alert("test")</script> & работа'
        destination = '<a href="https://example.invalid">Группа</a>'
        data = self.draft(subject=subject, task=task)
        saved = self.service.confirm(1, data)
        outputs = [
            draft_card(data, destination),
            deadline_list([{**saved, "group_name": destination}], self.now),
            reminder_text({**saved, "days_before": 3}),
        ]
        for output in outputs:
            with self.subTest(output=output):
                self.assertNotIn(subject, output)
                self.assertNotIn("<script>", output)
                self.assertIn("&lt;b&gt;Физика &amp;", output)
                self.assertIn("&lt;script&gt;", output)
        self.assertIn("&lt;a href=", outputs[0])
        self.assertIn("&lt;a href=", outputs[1])


if __name__ == "__main__":
    unittest.main()
