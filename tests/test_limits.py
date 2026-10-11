"""Local plan limits with real temporary SQLite databases and concurrent writers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from db import Database
from plans import PlanLimitExceeded


class PlanLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "limits.db"
        self.now = datetime(2026, 10, 11, 12)
        self.db = Database(self.path, clock=lambda: self.now)

    def tearDown(self) -> None:
        self.db.close()
        self.directory.cleanup()

    def deadline(self, user_id: int = 1, **kwargs: object) -> dict:
        return self.db.create_deadline(
            user_id, "Базы данных", "Лабораторная работа", self.now + timedelta(days=3), **kwargs
        )

    def fill(self, count: int, user_id: int = 1) -> list[dict]:
        return [self.deadline(user_id) for _ in range(count)]

    def test_default_plan_is_free_and_usage_read_does_not_create_user(self) -> None:
        self.assertEqual(self.db.get_plan_usage(1),
                         {"plan": "free", "active_count": 0, "limit": 15, "remaining": 15})
        self.assertIsNone(self.db.get_user(1))
        self.deadline()
        self.assertEqual(self.db.get_user(1)["plan"], "free")
        self.assertEqual(self.db.get_plan_usage(1)["remaining"], 14)

    def test_free_limit_rejects_sixteenth_active_task_without_saving_it(self) -> None:
        self.fill(15)
        with self.assertRaises(PlanLimitExceeded) as raised:
            self.deadline()
        self.assertEqual((raised.exception.plan, raised.exception.active_count, raised.exception.limit),
                         ("free", 15, 15))
        self.assertEqual(self.db.get_plan_usage(1)["remaining"], 0)
        self.assertEqual(len(self.db.list_deadlines(1, include_completed=True)), 15)

    def test_overdue_pending_tasks_still_consume_slots(self) -> None:
        self.db.create_deadline(1, "Физика", "Просроченная работа", self.now - timedelta(days=1))
        self.fill(14)
        with self.assertRaises(PlanLimitExceeded):
            self.deadline()

    def test_personal_completion_and_deletion_release_slots(self) -> None:
        items = self.fill(15)
        self.assertFalse(self.db.complete_deadline(2, items[0]["id"]))
        self.assertFalse(self.db.delete_deadline(2, items[0]["id"]))
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)
        self.assertTrue(self.db.complete_deadline(1, items[0]["id"]))
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 14)
        self.deadline()
        self.assertTrue(self.db.delete_deadline(1, items[1]["id"]))
        self.deadline()
        with self.assertRaises(PlanLimitExceeded):
            self.deadline()
        self.assertEqual(len(self.db.list_deadlines(1, include_completed=True)), 16)

    def test_group_task_counts_once_until_all_current_members_complete_it(self) -> None:
        group = self.db.create_group(1, "Учебная группа")
        self.db.join_group(2, group["invite_code"])
        shared = self.deadline(group_id=group["id"])
        self.fill(14)
        self.db.complete_deadline(1, shared["id"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)
        with self.assertRaises(PlanLimitExceeded):
            self.deadline()
        self.db.complete_deadline(2, shared["id"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 14)
        self.deadline()

    def test_incoming_group_tasks_do_not_consume_recipient_quota_or_block_join(self) -> None:
        self.fill(15, user_id=2)
        group = self.db.create_group(1, "Входящие задания")
        self.db.join_group(2, group["invite_code"])
        self.deadline(group_id=group["id"])
        self.assertEqual(len(self.db.list_deadlines(2)), 16)
        self.assertEqual(self.db.get_plan_usage(2)["active_count"], 15)
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 1)

    def test_leaving_and_rejoining_recompute_shared_activity_without_deleting_tasks(self) -> None:
        group = self.db.create_group(1, "Состав группы")
        self.db.join_group(2, group["invite_code"])
        shared = self.deadline(group_id=group["id"])
        self.fill(14)
        self.db.complete_deadline(1, shared["id"])
        self.db.leave_group(2, group["id"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 14)
        self.deadline()
        self.db.join_group(2, group["invite_code"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 16)
        self.assertEqual(self.db.get_plan_usage(1)["remaining"], 0)
        self.assertIsNotNone(self.db.get_deadline(2, shared["id"]))
        with self.assertRaises(PlanLimitExceeded):
            self.deadline()
        self.db.delete_user(2)
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)

    def test_archived_completion_flag_still_releases_group_slot(self) -> None:
        group = self.db.create_group(1, "Архив")
        self.db.join_group(2, group["invite_code"])
        shared = self.deadline(group_id=group["id"])
        self.db.complete_deadline(1, shared["id"])
        self.db.cleanup_completed_before(self.now + timedelta(days=1))
        self.assertIsNone(self.db.get_deadline(1, shared["id"])["completed_at"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 1)
        self.db.complete_deadline(2, shared["id"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 0)

    def test_unlimited_plans_persist_and_downgrade_keeps_existing_tasks(self) -> None:
        for plan in ("student", "group"):
            with self.subTest(plan=plan):
                user_id = 1 if plan == "student" else 2
                self.db.set_delivery_enabled(user_id, False)
                self.assertEqual(self.db.set_plan(user_id, plan)["limit"], None)
                self.fill(17, user_id=user_id)
                usage = self.db.set_plan(user_id, "free")
                self.assertEqual(usage, {"plan": "free", "active_count": 17, "limit": 15, "remaining": 0})
                self.assertEqual(len(self.db.list_deadlines(user_id)), 17)
                self.assertFalse(self.db.get_user(user_id)["delivery_enabled"])
                with self.assertRaises(PlanLimitExceeded):
                    self.deadline(user_id)
                self.db.set_plan(user_id, plan)
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "student")
        self.assertEqual(self.db.get_plan_usage(2)["plan"], "group")
        self.db.upsert_user(1, "new_username")
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "student")
        self.assertIsNone(self.db.get_plan_usage(1)["remaining"])

    def test_invalid_plan_never_mutates_or_creates_an_account(self) -> None:
        for plan in ("premium", "", None, 123):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.db.set_plan(1, plan)
        self.assertIsNone(self.db.get_user(1))
        self.db.set_plan(1, "student")
        with self.assertRaises(ValueError):
            self.db.set_plan(1, "invalid")
        self.assertEqual(self.db.get_plan_usage(1)["plan"], "student")

    def test_all_plans_keep_both_reminder_windows(self) -> None:
        for user_id, plan in enumerate(("free", "student", "group"), 1):
            with self.subTest(plan=plan):
                self.db.set_plan(user_id, plan)
                item = self.deadline(user_id)
                for days_before, now in ((3, self.now), (1, self.now + timedelta(days=2))):
                    token = self.db.claim_reminder(item["id"], user_id, days_before, now)
                    self.assertIsNotNone(token)
                    self.assertTrue(self.db.begin_reminder_send(item["id"], user_id, days_before, now,
                                                               claim_token=token))
                    self.assertTrue(self.db.mark_reminder_sent(item["id"], user_id, days_before, now,
                                                              claim_token=token))

    def test_repeat_confirmation_at_limit_reuses_original_and_rejects_changed_payload(self) -> None:
        self.fill(14)
        item = self.deadline(idempotency_key="confirm:boundary")
        repeated = self.deadline(idempotency_key="confirm:boundary")
        self.assertEqual(repeated["id"], item["id"])
        self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)
        with self.assertRaisesRegex(ValueError, "Idempotency key"):
            self.db.create_deadline(1, "Физика", "Другая работа", self.now,
                                    idempotency_key="confirm:boundary")
        with self.assertRaises(PlanLimitExceeded):
            self.deadline(idempotency_key="confirm:new")

    def test_two_connections_cannot_both_take_the_final_slot(self) -> None:
        self.fill(14)
        second = Database(self.path, clock=lambda: self.now)
        barrier = threading.Barrier(2)

        def create(database: Database) -> bool:
            barrier.wait(timeout=5)
            try:
                database.create_deadline(1, "Физика", "Конкурентное создание", self.now)
            except PlanLimitExceeded:
                return False
            return True

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(create, (self.db, second)))
            self.assertEqual(sorted(results), [False, True])
            self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)
        finally:
            second.close()

    def test_two_connections_repeating_same_confirmation_share_final_slot(self) -> None:
        self.fill(14)
        second = Database(self.path, clock=lambda: self.now)
        barrier = threading.Barrier(2)

        def create(database: Database) -> int:
            barrier.wait(timeout=5)
            return database.create_deadline(1, "Физика", "Повтор подтверждения", self.now,
                                            idempotency_key="confirm:shared")["id"]

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                ids = list(pool.map(create, (self.db, second)))
            self.assertEqual(ids[0], ids[1])
            self.assertEqual(self.db.get_plan_usage(1)["active_count"], 15)
        finally:
            second.close()

    def test_legacy_users_migrate_with_tasks_preferences_and_completions_preserved(self) -> None:
        legacy_path = Path(self.directory.name) / "legacy.db"
        with closing(sqlite3.connect(legacy_path)) as legacy, legacy:
            legacy.executescript("""
                CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT,
                    delivery_enabled INTEGER NOT NULL DEFAULT 1 CHECK (delivery_enabled IN (0, 1)),
                    created_at TEXT NOT NULL);
                CREATE TABLE groups (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, invite_code TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
                CREATE TABLE group_members (group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    joined_at TEXT NOT NULL, PRIMARY KEY (group_id, user_id));
                CREATE TABLE deadlines (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creator_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    group_id INTEGER REFERENCES groups(id) ON DELETE CASCADE,
                    subject TEXT NOT NULL, task TEXT NOT NULL, due_date TEXT NOT NULL,
                    created_at TEXT NOT NULL, idempotency_key TEXT, UNIQUE (creator_id, idempotency_key));
                CREATE TABLE deadline_completions (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    completed_at TEXT, PRIMARY KEY (deadline_id, user_id));
                INSERT INTO users VALUES (1, 'owner', 0, '2026-10-01 12:00:00.000000');
                INSERT INTO users VALUES (2, 'member', 1, '2026-10-01 12:00:00.000000');
                INSERT INTO groups VALUES (5, 1, 'Группа', 'legacy-invite', '2026-10-01 12:00:00.000000');
                INSERT INTO group_members VALUES (5, 1, '2026-10-01 12:00:00.000000');
                INSERT INTO group_members VALUES (5, 2, '2026-10-01 12:00:00.000000');
                INSERT INTO deadlines VALUES (10, 1, NULL, 'БД', 'Сдано', '2026-10-14 12:00',
                    '2026-10-01 12:00:00.000000', 'old-personal');
                INSERT INTO deadlines VALUES (11, 1, 5, 'БД', 'Общее', '2026-10-14 12:00',
                    '2026-10-01 12:00:00.000000', 'old-shared');
                INSERT INTO deadline_completions VALUES (10, 1, '2026-10-10 12:00:00.000000');
                INSERT INTO deadline_completions VALUES (11, 1, NULL);
            """)
        with Database(legacy_path, clock=lambda: self.now) as migrated:
            self.assertEqual(migrated.get_plan_usage(1),
                             {"plan": "free", "active_count": 1, "limit": 15, "remaining": 14})
            self.assertEqual(migrated.get_user(1)["username"], "owner")
            self.assertFalse(migrated.get_user(1)["delivery_enabled"])
            self.assertEqual(migrated.get_user(1)["created_at"], "2026-10-01 12:00:00.000000")
            self.assertEqual(migrated.get_deadline(1, 10)["status"], "completed")
            self.assertEqual(migrated.get_deadline(2, 11)["status"], "pending")
            migrated.set_plan(1, "student")
            migrated.initialize()
        with Database(legacy_path, clock=lambda: self.now) as reopened:
            self.assertEqual(reopened.get_plan_usage(1)["plan"], "student")
            self.assertEqual(reopened.list_group_member_ids(5), [1, 2])
            self.assertEqual(len(reopened.list_deadlines(1, include_completed=True)), 2)
        with closing(sqlite3.connect(legacy_path)) as migrated, migrated:
            self.assertEqual(migrated.execute("PRAGMA foreign_key_check").fetchall(), [])
            with self.assertRaises(sqlite3.IntegrityError):
                migrated.execute("UPDATE users SET plan = 'invalid' WHERE id = 1")


if __name__ == "__main__":
    unittest.main()
