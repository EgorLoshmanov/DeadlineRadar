"""Durable send gates, owner fencing and conservative legacy-journal migration."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest

from db import Database, REMINDER_LEASE_SECONDS


class ReminderDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "delivery.db"
        self.now = datetime(2026, 10, 11, 12, 0)
        self.db = Database(self.path, clock=lambda: self.now)

    def tearDown(self) -> None:
        self.db.close()
        self.directory.cleanup()

    def deadline(self, *, user_id: int = 1, days: int = 2, group_id: int | None = None) -> dict:
        return self.db.create_deadline(user_id, "Базы данных", "Лабораторная работа №3",
                                       self.now + timedelta(days=days), group_id=group_id)

    def claim(self, args: tuple[int, int, int]) -> str:
        token = self.db.claim_reminder(*args, self.now)
        self.assertIsInstance(token, str)
        self.assertTrue(token)
        return token

    def reopen(self) -> None:
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)

    def test_sent_ack_requires_send_gate_and_released_claim_gets_fresh_token(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        first = self.claim(args)
        self.assertFalse(self.db.mark_reminder_sent(*args, self.now, claim_token=first))
        self.assertTrue(self.db.release_reminder(*args, claim_token=first))
        second = self.claim(args)
        self.assertNotEqual(first, second)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=second))
        self.assertTrue(self.db.mark_reminder_sent(*args, self.now, claim_token=second))
        self.assertEqual(self.db.get_due_reminders(self.now), [])

    def test_wrong_owner_cannot_change_claim_or_gate(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        token = self.claim(args)
        wrong = "unrelated-owner-token"
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=wrong))
        self.assertFalse(self.db.release_reminder(*args, claim_token=wrong))
        self.assertFalse(self.db.mark_reminder_sent(*args, self.now, claim_token=wrong))
        self.assertFalse(self.db.mark_reminder_uncertain(*args, claim_token=wrong))
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        self.assertFalse(self.db.release_reminder(*args, claim_token=wrong, known_not_sent=True))
        self.assertFalse(self.db.mark_reminder_uncertain(*args, claim_token=wrong))
        self.assertTrue(self.db.mark_reminder_sent(*args, self.now, claim_token=token))

    def test_expired_owner_cannot_begin_even_before_replacement_worker_arrives(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        token = self.claim(args)
        self.now += timedelta(seconds=REMINDER_LEASE_SECONDS)
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        replacement = self.claim(args)
        self.assertNotEqual(token, replacement)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=replacement))

    def test_stale_worker_cannot_release_or_finish_replacement_workers_delivery(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        stale = self.claim(args)
        self.now += timedelta(seconds=REMINDER_LEASE_SECONDS)
        other = Database(self.path, clock=lambda: self.now)
        try:
            current = other.claim_reminder(*args, self.now)
            self.assertIsInstance(current, str)
            self.assertNotEqual(stale, current)
            self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=stale))
            self.assertFalse(self.db.release_reminder(*args, claim_token=stale))
            self.assertFalse(self.db.release_reminder(*args, claim_token=stale, known_not_sent=True))
            self.assertFalse(self.db.mark_reminder_sent(*args, self.now, claim_token=stale))
            self.assertFalse(self.db.mark_reminder_uncertain(*args, claim_token=stale))
            self.assertTrue(other.begin_reminder_send(*args, self.now, claim_token=current))
            self.assertFalse(self.db.release_reminder(*args, claim_token=stale, known_not_sent=True))
            self.assertFalse(self.db.mark_reminder_uncertain(*args, claim_token=stale))
            self.assertTrue(other.mark_reminder_sent(*args, self.now, claim_token=current))
            self.assertEqual(self.db.get_due_reminders(self.now), [])
        finally:
            other.close()

    def test_restart_before_send_gate_recovers_only_after_claim_lease_expires(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        first = self.claim(args)
        self.reopen()
        self.assertIsNone(self.db.claim_reminder(*args, self.now))
        self.now += timedelta(seconds=REMINDER_LEASE_SECONDS - 1)
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.now += timedelta(seconds=1)
        second = self.claim(args)
        self.assertNotEqual(first, second)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=second))

    def test_crash_after_gate_never_reopens_window_even_when_lease_expires(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        token = self.claim(args)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        self.reopen()
        self.now += timedelta(hours=1)
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.assertIsNone(self.db.claim_reminder(*args, self.now))
        self.assertFalse(self.db.release_reminder(*args, claim_token=token))
        self.assertEqual(self.db.get_due_reminders(self.now), [])

    def test_uncertain_result_is_terminal_even_with_known_not_sent_override(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        token = self.claim(args)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        self.assertTrue(self.db.mark_reminder_uncertain(*args, claim_token=token))
        self.assertFalse(self.db.release_reminder(*args, claim_token=token, known_not_sent=True))
        self.reopen()
        self.now += timedelta(hours=1)
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.assertIsNone(self.db.claim_reminder(*args, self.now))

    def test_known_refusal_releases_gate_and_retry_uses_new_owner(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        rejected = self.claim(args)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=rejected))
        self.assertTrue(self.db.release_reminder(*args, retry_at=self.now + timedelta(seconds=60),
                                                 claim_token=rejected, known_not_sent=True))
        self.assertIsNone(self.db.claim_reminder(*args, self.now))
        self.now += timedelta(seconds=59)
        self.assertIsNone(self.db.claim_reminder(*args, self.now))
        self.now += timedelta(seconds=1)
        accepted = self.claim(args)
        self.assertNotEqual(rejected, accepted)
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=rejected))
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=accepted))
        self.assertTrue(self.db.mark_reminder_sent(*args, self.now, claim_token=accepted))
        self.assertFalse(self.db.release_reminder(*args, claim_token=accepted, known_not_sent=True))

    def test_begin_gate_rechecks_completion_delivery_preference_and_window(self) -> None:
        completed = self.deadline()
        args = (completed["id"], 1, 3)
        token = self.claim(args)
        self.db.complete_deadline(1, completed["id"])
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=token))

        paused = self.deadline(user_id=2)
        args = (paused["id"], 2, 3)
        token = self.claim(args)
        self.db.set_delivery_enabled(2, False)
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=token))

        boundary = self.db.create_deadline(3, "Физика", "Отчёт", self.now + timedelta(days=1, minutes=1))
        args = (boundary["id"], 3, 3)
        token = self.claim(args)
        self.now += timedelta(minutes=2)
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        current_args = (boundary["id"], 3, 1)
        current = self.claim(current_args)
        self.assertTrue(self.db.begin_reminder_send(*current_args, self.now, claim_token=current))

    def test_begin_gate_rechecks_group_membership(self) -> None:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        shared = self.deadline(group_id=group["id"])
        args = (shared["id"], 2, 3)
        token = self.claim(args)
        self.db.leave_group(2, group["id"])
        self.assertFalse(self.db.begin_reminder_send(*args, self.now, claim_token=token))

    def test_ambiguous_group_send_does_not_repeat_after_leave_rejoin_or_settings_toggle(self) -> None:
        group = self.db.create_group(1, "ИВТ-31")
        self.db.join_group(2, group["invite_code"])
        shared = self.deadline(group_id=group["id"])
        self.db.complete_deadline(1, shared["id"])
        args = (shared["id"], 2, 3)
        token = self.claim(args)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        self.assertTrue(self.db.mark_reminder_uncertain(*args, claim_token=token))
        self.db.leave_group(2, group["id"])
        self.db.join_group(2, group["invite_code"])
        self.db.set_delivery_enabled(2, False)
        self.db.set_delivery_enabled(2, True)
        self.reopen()
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.assertIsNone(self.db.claim_reminder(*args, self.now))

    def test_terminal_three_day_window_does_not_block_independent_one_day_window(self) -> None:
        deadline = self.deadline(days=3)
        args = (deadline["id"], 1, 3)
        token = self.claim(args)
        self.assertTrue(self.db.begin_reminder_send(*args, self.now, claim_token=token))
        self.assertTrue(self.db.mark_reminder_uncertain(*args, claim_token=token))
        self.now += timedelta(days=2)
        self.reopen()
        current_args = (deadline["id"], 1, 1)
        current = self.claim(current_args)
        self.assertTrue(self.db.begin_reminder_send(*current_args, self.now, claim_token=current))
        self.assertTrue(self.db.mark_reminder_sent(*current_args, self.now, claim_token=current))
        self.assertEqual(self.db.get_due_reminders(self.now), [])

    def test_two_connections_cannot_begin_the_same_window_twice(self) -> None:
        args = (self.deadline()["id"], 1, 3)
        token = self.claim(args)
        other = Database(self.path, clock=lambda: self.now)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(
                    lambda db: db.begin_reminder_send(*args, self.now, claim_token=token),
                    [self.db, other],
                ))
            self.assertEqual(sorted(results), [False, True])
        finally:
            other.close()

    def test_legacy_sent_claimed_and_retry_rows_migrate_conservatively_and_idempotently(self) -> None:
        deadlines = {status: self.deadline()["id"] for status in ("sent", "claimed", "retry")}
        self.db.close()
        with sqlite3.connect(self.path) as legacy:
            legacy.executescript(
                """
                DROP TABLE IF EXISTS reminder_attempts;
                DROP TABLE reminder_deliveries;
                CREATE TABLE reminder_deliveries (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    days_before INTEGER NOT NULL CHECK (days_before IN (3, 1)),
                    status TEXT NOT NULL CHECK (status IN ('claimed', 'retry', 'sent')),
                    lease_until TEXT,
                    next_attempt_at TEXT,
                    sent_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    PRIMARY KEY (deadline_id, user_id, days_before)
                );
                """
            )
            stamp = (self.now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S.%f")
            for status, deadline_id in deadlines.items():
                legacy.execute(
                    """INSERT INTO reminder_deliveries
                       (deadline_id, user_id, days_before, status, lease_until, next_attempt_at, sent_at, attempts)
                       VALUES (?, 1, 3, ?, ?, ?, ?, 1)""",
                    (deadline_id, status, stamp, stamp, stamp if status == "sent" else None),
                )
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        for deadline_id in deadlines.values():
            self.assertIsNone(self.db.claim_reminder(deadline_id, 1, 3, self.now))
        with sqlite3.connect(self.path) as migrated:
            columns = {row[1] for row in migrated.execute("PRAGMA table_info(reminder_deliveries)")}
            self.assertIn("claim_token", columns)
            results = dict(migrated.execute("SELECT deadline_id, result FROM reminder_attempts"))
        self.assertEqual(results, {deadlines["sent"]: "sent", deadlines["claimed"]: "uncertain",
                                   deadlines["retry"]: "uncertain"})
        self.reopen()
        self.assertEqual(self.db.get_due_reminders(self.now), [])
        self.now += timedelta(days=1)
        remaining = self.db.get_due_reminders(self.now)
        self.assertCountEqual([item["id"] for item in remaining], list(deadlines.values()))
        self.assertEqual({item["days_before"] for item in remaining}, {1})


if __name__ == "__main__":
    unittest.main()
