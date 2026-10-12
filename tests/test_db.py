"""Behavior tests for access control, individual completion, and delivery state."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from db import Database, REMINDER_LEASE_SECONDS


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "test.db"
        self.db = Database(self.path)
        self.now = datetime(2026, 10, 11, 12, 0)

    def tearDown(self) -> None:
        self.db.close()
        self.directory.cleanup()

    def deadline(self, user_id: int = 1, days: float = 2, **kwargs: object) -> dict:
        return self.db.create_deadline(user_id, "Математика", "Решить задачи", self.now + timedelta(days=days), **kwargs)

    def group(self) -> dict:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        return group

    def test_data_and_delivery_preference_survive_reopen(self) -> None:
        self.db.upsert_user(1, "student")
        deadline = self.deadline()
        self.db.set_delivery_enabled(1, False)
        self.db.close()
        self.db = Database(self.path)
        self.db.initialize()
        self.assertEqual(self.db.get_deadline(1, deadline["id"])["subject"], "Математика")
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        self.db.upsert_user(1, "new_name")
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        self.assertEqual(self.db.get_user(1)["username"], "new_name")

    def test_personal_deadlines_are_private(self) -> None:
        deadline = self.deadline()
        self.db.upsert_user(2)
        self.assertEqual(self.db.list_deadlines(2), [])
        self.assertIsNone(self.db.get_deadline(2, deadline["id"]))
        self.assertFalse(self.db.complete_deadline(2, deadline["id"]))
        self.assertFalse(self.db.delete_deadline(2, deadline["id"]))
        self.assertIsNotNone(self.db.get_deadline(1, deadline["id"]))

    def test_completion_is_idempotent_and_personal(self) -> None:
        group = self.group()
        deadline = self.deadline(group_id=group["id"])
        self.assertTrue(self.db.complete_deadline(2, deadline["id"]))
        completed_at = self.db.get_deadline(2, deadline["id"])["completed_at"]
        self.assertTrue(self.db.complete_deadline(2, deadline["id"]))
        self.assertEqual(self.db.get_deadline(2, deadline["id"])["completed_at"], completed_at)
        self.assertEqual(self.db.get_deadline(2, deadline["id"])["status"], "completed")
        self.assertEqual(self.db.get_deadline(1, deadline["id"])["status"], "pending")
        self.assertEqual(self.db.list_deadlines(2), [])
        self.assertEqual(len(self.db.list_deadlines(2, include_completed=True)), 1)
        self.assertEqual([row["user_id"] for row in self.db.get_due_reminders(self.now)], [1])

    def test_group_invite_join_and_leave(self) -> None:
        group = self.group()
        self.db.join_group(2, group["invite_code"])
        self.assertEqual(self.db.list_group_member_ids(group["id"]), [1, 2])
        self.assertTrue(self.db.list_groups(1)[0]["is_owner"])
        self.assertFalse(self.db.list_groups(2)[0]["is_owner"])
        self.assertIsNone(self.db.get_group_for_owner(2, group["id"]))
        with self.assertRaises(ValueError):
            self.db.join_group(3, "invalid")
        self.assertIsNone(self.db.get_user(3))
        with self.assertRaises(ValueError):
            self.db.leave_group(1, group["id"])
        deadline = self.deadline(group_id=group["id"])
        self.db.leave_group(2, group["id"])
        self.assertIsNone(self.db.get_deadline(2, deadline["id"]))
        self.assertEqual(self.db.list_groups(2), [])

    def test_only_owner_can_create_and_delete_group_deadline(self) -> None:
        group = self.group()
        with self.assertRaises(ValueError):
            self.deadline(2, group_id=group["id"])
        with self.assertRaises(ValueError):
            self.deadline(3, group_id=group["id"])
        deadline = self.deadline(group_id=group["id"])
        self.assertFalse(self.db.delete_deadline(2, deadline["id"]))
        self.assertTrue(self.db.delete_deadline(1, deadline["id"]))
        self.assertIsNone(self.db.get_deadline(2, deadline["id"]))

    def test_duplicate_event_is_reused_and_payload_conflicts_rejected(self) -> None:
        first = self.deadline(idempotency_key="chat:1:message:50")
        repeated = self.deadline(idempotency_key="chat:1:message:50")
        self.assertEqual(first["id"], repeated["id"])
        self.assertEqual(len(self.db.list_deadlines(1)), 1)
        with self.assertRaises(ValueError):
            self.db.create_deadline(1, "Физика", "Лабораторная", self.now, idempotency_key="chat:1:message:50")
        # Keys are scoped to the creating user, not globally.
        other = self.deadline(2, idempotency_key="chat:1:message:50")
        self.assertNotEqual(first["id"], other["id"])

    def test_deadline_order_and_validation(self) -> None:
        later = self.deadline(days=3)
        earlier = self.deadline(days=1)
        self.assertEqual([row["id"] for row in self.db.list_deadlines(1)], [earlier["id"], later["id"]])
        for subject, task, date in [
            ("", "Task", self.now), ("S" * 121, "Task", self.now),
            ("Subject", "T" * 501, self.now), ("Subject", "Task", "2026-02-30 10:00"),
            ("Subject", "Task", self.now.replace(tzinfo=timezone.utc)),
        ]:
            with self.subTest(subject=subject, date=date), self.assertRaises(ValueError):
                self.db.create_deadline(1, subject, task, date)

    def test_reminder_window_boundaries(self) -> None:
        ids = {days: self.deadline(days=days)["id"] for days in (-1, 0, 1, 2, 3, 4)}
        reminders = {row["id"]: row["days_before"] for row in self.db.get_due_reminders(self.now)}
        self.assertEqual(reminders, {ids[1]: 1, ids[2]: 3, ids[3]: 3})
        # A target minute immediately beyond the exact three-day window is excluded.
        edge = self.db.create_deadline(1, "Физика", "Тест", self.now + timedelta(days=3, minutes=1))
        self.assertNotIn(edge["id"], [row["id"] for row in self.db.get_due_reminders(self.now + timedelta(seconds=59))])

    def test_sent_reminders_persist_and_next_window_is_independent(self) -> None:
        deadline = self.deadline(days=3)
        token = self.db.claim_reminder(deadline["id"], 1, 3, self.now)
        self.assertIsNotNone(token)
        self.assertTrue(self.db.begin_reminder_send(deadline["id"], 1, 3, self.now, claim_token=token))
        self.db.mark_reminder_sent(deadline["id"], 1, 3, self.now, claim_token=token)
        self.db.close()
        self.db = Database(self.path)
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.assertFalse(self.db.claim_reminder(deadline["id"], 1, 3, self.now))
        next_window = self.db.get_due_reminders(self.now + timedelta(days=2))
        self.assertEqual([(row["id"], row["days_before"]) for row in next_window], [(deadline["id"], 1)])

    def test_restart_catches_current_window_without_obsolete_reminders(self) -> None:
        deadline = self.deadline(days=0.5)
        self.db.close()
        self.db = Database(self.path)
        self.assertEqual(self.db.get_due_reminders(self.now)[0]["days_before"], 1)
        self.assertFalse(self.db.claim_reminder(deadline["id"], 1, 3, self.now))
        self.assertEqual(self.db.get_due_reminders(self.now + timedelta(days=1)), [])

    def test_claim_has_durable_lease_and_expired_lease_can_be_recovered(self) -> None:
        deadline = self.deadline()
        self.assertTrue(self.db.claim_reminder(deadline["id"], 1, 3, self.now))
        self.db.close()
        self.db = Database(self.path)
        self.assertFalse(self.db.claim_reminder(deadline["id"], 1, 3, self.now))
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        retry = self.now + timedelta(seconds=REMINDER_LEASE_SECONDS)
        self.assertTrue(self.db.claim_reminder(deadline["id"], 1, 3, retry))

    def test_two_workers_cannot_claim_the_same_delivery(self) -> None:
        deadline = self.deadline()
        second = Database(self.path)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(
                    lambda db: db.claim_reminder(deadline["id"], 1, 3, self.now),
                    [self.db, second],
                ))
            self.assertEqual(sorted(bool(result) for result in results), [False, True])
        finally:
            second.close()

    def test_release_delays_retry_and_never_reopens_sent_delivery(self) -> None:
        deadline = self.deadline()
        token = self.db.claim_reminder(deadline["id"], 1, 3, self.now)
        self.assertIsNotNone(token)
        retry_at = self.now + timedelta(minutes=2)
        self.db.release_reminder(deadline["id"], 1, 3, retry_at=retry_at, claim_token=token)
        self.assertFalse(self.db.claim_reminder(deadline["id"], 1, 3, self.now))
        token = self.db.claim_reminder(deadline["id"], 1, 3, retry_at)
        self.assertIsNotNone(token)
        self.assertTrue(self.db.begin_reminder_send(deadline["id"], 1, 3, retry_at, claim_token=token))
        self.db.mark_reminder_sent(deadline["id"], 1, 3, retry_at, claim_token=token)
        self.db.release_reminder(deadline["id"], 1, 3, claim_token=token)
        self.assertEqual(self.db.get_due_reminders(retry_at), [])

    def test_delivery_disabled_completed_and_departed_members_are_skipped(self) -> None:
        group = self.group()
        deadline = self.deadline(group_id=group["id"])
        self.db.set_delivery_enabled(1, False)
        self.assertEqual([row["user_id"] for row in self.db.get_due_reminders(self.now)], [2])
        self.db.leave_group(2, group["id"])
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.assertFalse(self.db.claim_reminder(deadline["id"], 2, 3, self.now))
        self.db.set_delivery_enabled(1, True)
        self.db.complete_deadline(1, deadline["id"])
        self.assertFalse(self.db.claim_reminder(deadline["id"], 1, 3, self.now))

    def test_cleanup_preserves_group_deadlines_needed_by_other_members(self) -> None:
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        personal = self.deadline()
        self.db.complete_deadline(1, shared["id"])
        self.db.complete_deadline(1, personal["id"])
        future_cutoff = datetime.now() + timedelta(days=1)
        self.assertEqual(self.db.cleanup_completed_before(future_cutoff), 1)
        self.assertIsNotNone(self.db.get_deadline(2, shared["id"]))
        self.db.complete_deadline(2, shared["id"])
        self.assertEqual(self.db.cleanup_completed_before(datetime(2000, 1, 1)), 0)
        self.assertEqual(self.db.cleanup_completed_before(future_cutoff), 1)

    def test_delete_user_cascades_private_data_and_owned_groups(self) -> None:
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        self.deadline()
        self.deadline(2)
        self.db.delete_user(1)
        self.assertIsNone(self.db.get_user(1))
        self.assertEqual(self.db.list_groups(2), [])
        self.assertIsNone(self.db.get_deadline(2, shared["id"]))
        self.assertEqual(len(self.db.list_deadlines(2)), 1)
        self.assertEqual(self.db.list_group_member_ids(group["id"]), [])

    def test_leave_and_rejoin_does_not_repeat_a_sent_group_reminder(self) -> None:
        group = self.group()
        deadline = self.deadline(group_id=group["id"])
        token = self.db.claim_reminder(deadline["id"], 2, 3, self.now)
        self.assertIsNotNone(token)
        self.assertTrue(self.db.begin_reminder_send(deadline["id"], 2, 3, self.now, claim_token=token))
        self.db.mark_reminder_sent(deadline["id"], 2, 3, self.now, claim_token=token)
        self.db.leave_group(2, group["id"])
        self.db.join_group(2, group["invite_code"])
        self.assertFalse(self.db.claim_reminder(deadline["id"], 2, 3, self.now))
        self.assertEqual([item["user_id"] for item in self.db.get_due_reminders(self.now)], [1])
        # The separate one-day reminder remains eligible after rejoining.
        next_window = self.db.get_due_reminders(self.now + timedelta(days=1))
        self.assertIn((2, 1), [(item["user_id"], item["days_before"]) for item in next_window])

    def test_completion_timestamps_use_the_supplied_application_clock(self) -> None:
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        deadline = self.deadline()
        self.db.complete_deadline(1, deadline["id"])
        completed = self.db.get_deadline(1, deadline["id"])
        self.assertEqual(datetime.fromisoformat(completed["completed_at"]), self.now)
        self.assertEqual(datetime.fromisoformat(self.db.get_user(1)["created_at"]), self.now)
        self.assertEqual(datetime.fromisoformat(completed["created_at"]), self.now)

    def test_cleanup_archives_group_completion_without_forgetting_done_status(self) -> None:
        self.db.close()
        self.now = datetime(2026, 8, 20, 12)
        self.db = Database(self.path, clock=lambda: self.now)
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        self.db.complete_deadline(1, shared["id"])
        self.now = datetime(2026, 10, 11, 12)
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 0)
        archived = self.db.get_deadline(1, shared["id"])
        self.assertEqual(archived["status"], "completed")
        self.assertTrue(archived["completed"])
        self.assertIsNone(archived["completed_at"])
        self.assertEqual(self.db.list_deadlines(1), [])
        self.assertEqual(self.db.get_deadline(2, shared["id"])["status"], "pending")
        self.assertTrue(self.db.complete_deadline(1, shared["id"]))
        self.assertIsNone(self.db.get_deadline(1, shared["id"])["completed_at"])
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(self.db.get_deadline(1, shared["id"])["status"], "completed")
        self.assertIsNone(self.db.get_deadline(1, shared["id"])["completed_at"])
        self.assertEqual(self.db.list_deadlines(1), [])

    def test_cleanup_preserves_current_history_and_archives_old_group_dates(self) -> None:
        self.db.close()
        self.now = datetime(2026, 8, 20, 12)
        self.db = Database(self.path, clock=lambda: self.now)
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        personal = self.deadline()
        self.db.complete_deadline(1, shared["id"])
        self.db.complete_deadline(1, personal["id"])
        self.now = datetime(2026, 10, 11, 12)
        self.db.complete_deadline(2, shared["id"])
        current_timestamp = self.db.get_deadline(2, shared["id"])["completed_at"]
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 1)
        self.assertIsNone(self.db.get_deadline(1, personal["id"]))
        self.assertIsNone(self.db.get_deadline(1, shared["id"])["completed_at"])
        self.assertEqual(self.db.get_deadline(2, shared["id"])["completed_at"], current_timestamp)
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 0)
        self.assertEqual(self.db.cleanup_completed_before(datetime(2027, 2, 1)), 1)
        self.assertIsNone(self.db.get_deadline(2, shared["id"]))

    def test_cleanup_removes_group_when_remaining_members_have_archived_done_flags(self) -> None:
        self.db.close()
        self.now = datetime(2026, 8, 20, 12)
        self.db = Database(self.path, clock=lambda: self.now)
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        self.db.complete_deadline(1, shared["id"])
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 0)
        self.assertIsNone(self.db.get_deadline(1, shared["id"])["completed_at"])
        self.db.leave_group(2, group["id"])
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 1)
        self.assertIsNone(self.db.get_deadline(1, shared["id"]))

    def test_cleanup_removes_personal_deadline_with_archived_done_flag(self) -> None:
        personal = self.deadline()
        self.db.complete_deadline(1, personal["id"])
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE deadline_completions SET completed_at = NULL WHERE deadline_id = ?", (personal["id"],))
        self.assertEqual(self.db.get_deadline(1, personal["id"])["status"], "completed")
        self.assertEqual(self.db.cleanup_completed_before(datetime(2026, 9, 1)), 1)
        self.assertIsNone(self.db.get_deadline(1, personal["id"]))

    def test_old_not_null_completion_schema_migrates_without_losing_data(self) -> None:
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        personal = self.deadline()
        self.db.complete_deadline(1, personal["id"])
        self.db.complete_deadline(2, shared["id"])
        before = self.db.get_deadline(2, shared["id"])["completed_at"]
        self.db.close()
        with sqlite3.connect(self.path) as legacy:
            rows = legacy.execute("SELECT deadline_id, user_id, completed_at FROM deadline_completions").fetchall()
            legacy.execute("DROP TABLE deadline_completions")
            legacy.execute("""CREATE TABLE deadline_completions (
                deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (deadline_id, user_id))""")
            legacy.executemany("INSERT INTO deadline_completions VALUES (?, ?, ?)", rows)
            legacy.execute("CREATE INDEX idx_completions_date ON deadline_completions(completed_at)")
            legacy.execute("CREATE INDEX idx_completions_user ON deadline_completions(user_id)")
        self.db = Database(self.path)
        self.assertEqual(self.db.get_deadline(2, shared["id"])["completed_at"], before)
        self.assertEqual(self.db.get_deadline(2, shared["id"])["status"], "completed")
        self.assertEqual(self.db.get_deadline(1, personal["id"])["status"], "completed")
        self.assertEqual(len(self.db.list_deadlines(1, include_completed=True)), 2)
        self.assertEqual(self.db.list_groups(2)[0]["id"], group["id"])
        with sqlite3.connect(self.path) as migrated:
            columns = {row[1]: row for row in migrated.execute("PRAGMA table_info(deadline_completions)")}
            self.assertEqual(columns["completed_at"][3], 0)
            indexes = {row[1] for row in migrated.execute("PRAGMA index_list(deadline_completions)")}
            self.assertTrue({"idx_completions_date", "idx_completions_user"}.issubset(indexes))
            self.assertEqual(migrated.execute("PRAGMA foreign_key_check").fetchall(), [])
        # Repeated initialization and process restart do not repeat or lose migration data.
        self.db.initialize()
        self.db.close()
        self.db = Database(self.path)
        self.assertEqual(self.db.get_deadline(2, shared["id"])["completed_at"], before)
        self.assertTrue(self.db.delete_deadline(1, shared["id"]))
        with sqlite3.connect(self.path) as migrated:
            self.assertEqual(migrated.execute("SELECT COUNT(*) FROM deadline_completions WHERE deadline_id = ?", (shared["id"],)).fetchone()[0], 0)

    def test_failed_legacy_completion_migration_rolls_back_its_schema_and_rows(self) -> None:
        personal = self.deadline()
        self.db.close()
        with sqlite3.connect(self.path) as legacy:
            legacy.execute("DROP TABLE deadline_completions")
            legacy.execute("""CREATE TABLE deadline_completions (
                deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (deadline_id, user_id))""")
            # A damaged legacy row cannot be copied with foreign-key enforcement.
            legacy.execute("INSERT INTO deadline_completions VALUES (99999, 1, '2026-08-20 12:00:00.000000')")
        failed = Database.__new__(Database)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                failed.__init__(self.path)
        finally:
            failed.close()
        with sqlite3.connect(self.path) as legacy:
            columns = {row[1]: row for row in legacy.execute("PRAGMA table_info(deadline_completions)")}
            self.assertEqual(columns["completed_at"][3], 1)
            self.assertEqual(legacy.execute("SELECT deadline_id FROM deadline_completions").fetchall(), [(99999,)])
            self.assertEqual(legacy.execute("SELECT id FROM deadlines").fetchall(), [(personal["id"],)])
            self.assertIsNone(legacy.execute("SELECT name FROM sqlite_master WHERE name = 'deadline_completions_legacy'").fetchone())
            legacy.execute("DELETE FROM deadline_completions WHERE deadline_id = 99999")
        self.db = Database(self.path)

    def test_background_delivery_refusal_requires_current_claim_owner(self) -> None:
        deadline = self.deadline()
        token = self.db.claim_reminder(deadline["id"], 1, 3, self.now)
        self.assertIsNotNone(token)
        self.assertFalse(self.db.disable_delivery_for_reminder(deadline["id"], 1, 3, claim_token="wrong-owner"))
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])
        self.assertTrue(self.db.disable_delivery_for_reminder(deadline["id"], 1, 3, claim_token=token))
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        self.db.set_delivery_enabled(1, True)
        self.assertTrue(self.db.release_reminder(deadline["id"], 1, 3, claim_token=token))
        self.assertFalse(self.db.disable_delivery_for_reminder(deadline["id"], 1, 3, claim_token=token))
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])

    def test_delivery_revision_migrates_without_replaying_completed_deliveries(self) -> None:
        personal = self.deadline()
        group = self.group()
        shared = self.deadline(group_id=group["id"])
        self.db.set_plan(1, "student")
        token = self.db.claim_reminder(personal["id"], 1, 3, self.now)
        self.assertTrue(self.db.begin_reminder_send(personal["id"], 1, 3, self.now, claim_token=token))
        self.db.mark_reminder_sent(personal["id"], 1, 3, self.now, claim_token=token)
        publication_token = self.db.begin_publication_send(shared["id"], 2, self.now)
        self.assertTrue(self.db.finish_publication(shared["id"], 2, token=publication_token, result="disabled"))
        self.db.close()
        # Rebuild only the three affected tables with their previous columns;
        # this fixture works without SQLite's newer DROP COLUMN syntax.
        with sqlite3.connect(self.path) as legacy:
            legacy.execute("PRAGMA foreign_keys = OFF")
            for table in ("users", "reminder_deliveries", "group_publications"):
                sql = legacy.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
                previous_sql = "\n".join(line for line in sql.splitlines() if "delivery_revision" not in line)
                columns = ", ".join(row[1] for row in legacy.execute(f"PRAGMA table_info({table})")
                                    if row[1] != "delivery_revision")
                legacy.execute(f"CREATE TEMP TABLE previous_rows AS SELECT {columns} FROM {table}")
                legacy.execute(f"DROP TABLE {table}")
                legacy.execute(previous_sql)
                legacy.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM previous_rows")
                legacy.execute("DROP TABLE previous_rows")
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(self.db.get_user(1)["plan"], "student")
        self.assertFalse(self.db.get_user(2)["delivery_enabled"])
        self.assertEqual(self.db.list_group_member_ids(group["id"]), [1, 2])
        self.assertNotIn(personal["id"], [item["id"] for item in self.db.get_due_reminders(self.now)])
        self.assertEqual(self.db.get_pending_publications(self.now), [])
        self.db.set_delivery_enabled(2, True)
        self.assertEqual([item["id"] for item in self.db.get_pending_publications(self.now)], [shared["id"]])
        revision = self.db.get_user(2)["delivery_revision"]
        self.db.initialize()
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(self.db.get_user(2)["delivery_revision"], revision)
        self.assertEqual(self.db._connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_old_refusal_cannot_disable_recreated_account_or_replacement_claim(self) -> None:
        group = self.db.create_group(2, "ИВТ-32")
        self.db.join_group(1, group["invite_code"])
        deadline = self.deadline(user_id=2, group_id=group["id"])
        old_token = self.db.claim_reminder(deadline["id"], 1, 3, self.now)
        self.assertTrue(self.db.begin_reminder_send(deadline["id"], 1, 3, self.now, claim_token=old_token))
        # Another connection represents account deletion/restart while transport is pending.
        with Database(self.path) as other:
            other.delete_user(1)
            other.upsert_user(1, "restarted")
            other.join_group(1, group["invite_code"])
            new_token = other.claim_reminder(deadline["id"], 1, 3, self.now)
        self.assertNotEqual(old_token, new_token)
        self.assertFalse(self.db.disable_delivery_for_reminder(deadline["id"], 1, 3, claim_token=old_token))
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])
        self.assertEqual(self.db.get_user(1)["username"], "restarted")
        self.assertTrue(self.db.disable_delivery_for_reminder(deadline["id"], 1, 3, claim_token=new_token))


if __name__ == "__main__":
    unittest.main()
