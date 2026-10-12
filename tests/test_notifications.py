"""Asynchronous reminder delivery tests with a real temporary database."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from db import Database, REMINDER_LEASE_SECONDS
from notifications import DeliveryDisabled, DeliverySkipped, NotificationService, RetryDelivery


class NotificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "notifications.db"
        self.db = Database(self.path)
        self.now = datetime(2026, 10, 11, 12, 0)
        self.service = NotificationService(self.db, clock=lambda: self.now)

    def tearDown(self) -> None:
        self.db.close()
        self.directory.cleanup()

    def deadline(self, user_id: int = 1, days: int = 2) -> dict:
        return self.db.create_deadline(user_id, "Физика", "Лабораторная работа", self.now + timedelta(days=days))

    async def test_success_is_deduplicated_and_persisted_after_restart(self) -> None:
        first, second = self.deadline(), self.deadline(2, days=1)
        sent = []

        async def sender(item: dict) -> None:
            sent.append((item["id"], item["user_id"], item["days_before"]))

        self.assertEqual(await self.service.tick(sender), 2)
        self.assertCountEqual(sent, [(first["id"], 1, 3), (second["id"], 2, 1)])
        self.assertEqual(await self.service.tick(sender), 0)
        self.db.close()
        self.db = Database(self.path)
        restarted = NotificationService(self.db, clock=lambda: self.now)
        self.assertEqual(await restarted.tick(sender), 0)
        self.assertEqual(len(sent), 2)

    async def test_concurrent_ticks_of_one_service_skip_overlapping_run(self) -> None:
        self.deadline()
        entered, proceed = asyncio.Event(), asyncio.Event()
        sent = []

        async def sender(item: dict) -> None:
            sent.append(item["id"])
            entered.set()
            await proceed.wait()

        first = asyncio.create_task(self.service.tick(sender))
        await asyncio.wait_for(entered.wait(), timeout=1)
        try:
            self.assertEqual(await self.service.tick(sender), 0)
        finally:
            proceed.set()
        self.assertEqual(await first, 1)
        self.assertEqual(len(sent), 1)

    async def test_separate_services_share_durable_claims(self) -> None:
        self.deadline()
        other_db = Database(self.path)
        other = NotificationService(other_db, clock=lambda: self.now)
        sent = []

        async def sender(item: dict) -> None:
            sent.append(item["id"])
            await asyncio.sleep(0)

        try:
            results = await asyncio.gather(self.service.tick(sender), other.tick(sender))
            self.assertEqual(sorted(results), [0, 1])
            self.assertEqual(len(sent), 1)
        finally:
            other_db.close()

    async def test_delivery_disabled_pauses_user_and_continues_other_deliveries(self) -> None:
        self.deadline(1)
        second = self.deadline(2)
        attempts, sent = [], []

        async def sender(item: dict) -> None:
            attempts.append(item["user_id"])
            if item["user_id"] == 1:
                raise DeliveryDisabled()
            sent.append(item["id"])

        self.assertEqual(await self.service.tick(sender), 1)
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])
        self.assertTrue(self.db.get_user(2)["delivery_enabled"])
        self.assertEqual(sent, [second["id"]])
        self.assertEqual(await self.service.tick(sender), 0)
        self.assertEqual(attempts, [1, 2])

    async def test_late_forbidden_does_not_override_newer_explicit_enable(self) -> None:
        for user_id, toggle_off in ((1, False), (2, True)):
            with self.subTest(toggle_off=toggle_off):
                deadline = self.deadline(user_id)

                async def late_refusal(item):
                    with Database(self.path) as settings_connection:
                        if toggle_off:
                            settings_connection.set_delivery_enabled(item["user_id"], False)
                        settings_connection.set_delivery_enabled(item["user_id"], True)
                    raise DeliveryDisabled

                self.assertEqual(await self.service.tick(late_refusal), 0)
                self.assertTrue(self.db.get_user(user_id)["delivery_enabled"])
                sent = []

                async def sender(item):
                    sent.append(item["id"])

                self.assertEqual(await self.service.tick(sender), 1)
                self.assertEqual(await self.service.tick(sender), 0)
                self.assertEqual(sent, [deadline["id"]])

    async def test_reminder_send_gate_uses_setting_after_preparation(self) -> None:
        self.deadline()

        async def reenable(_):
            self.db.set_delivery_enabled(1, True)

        async def refused(_):
            raise DeliveryDisabled

        self.assertEqual(await self.service.tick(refused, prepare=reenable), 0)
        # This refusal belongs to the new setting: it must still pause delivery.
        self.assertFalse(self.db.get_user(1)["delivery_enabled"])

    async def test_explicit_retry_delay_is_respected(self) -> None:
        self.deadline()
        attempts = []

        async def sender(item: dict) -> None:
            attempts.append(item["id"])
            if len(attempts) == 1:
                raise RetryDelivery(retry_after=60)

        self.assertEqual(await self.service.tick(sender), 0)
        self.now += timedelta(seconds=59)
        self.assertEqual(await self.service.tick(sender), 0)
        self.now += timedelta(seconds=1)
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(await self.service.tick(sender), 0)

    async def test_explicit_retry_delay_starts_after_slow_sender_fails(self) -> None:
        self.deadline()
        attempts = []

        async def sender(item: dict) -> None:
            attempts.append(item["id"])
            if len(attempts) == 1:
                self.now += timedelta(seconds=60)
                raise RetryDelivery(retry_after=60)

        self.assertEqual(await self.service.tick(sender), 0)
        self.assertEqual(await self.service.tick(sender), 0)
        self.now += timedelta(seconds=59)
        self.assertEqual(await self.service.tick(sender), 0)
        self.assertEqual(len(attempts), 1)
        self.now += timedelta(seconds=1)
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(len(attempts), 2)

    async def test_preparation_error_uses_five_minute_retry_without_spending_window(self) -> None:
        self.deadline()
        preparations, sent = [], []

        async def prepare(item: dict) -> None:
            preparations.append(item["id"])
            if len(preparations) == 1:
                raise OSError("Temporary connection failure")

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        with self.assertLogs("notifications", level="WARNING") as captured:
            self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertNotIn("Лабораторная работа", " ".join(captured.output))
        self.assertEqual(sent, [])
        self.now += timedelta(minutes=4, seconds=59)
        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.now += timedelta(seconds=1)
        self.assertEqual(await self.service.tick(sender, prepare=prepare), 1)
        self.assertEqual(len(preparations), 2)
        self.assertEqual(len(sent), 1)

    async def test_preparation_retry_delay_starts_after_slow_preparation_fails(self) -> None:
        self.deadline()
        preparations, sent = [], []

        async def prepare(item: dict) -> None:
            preparations.append(item["id"])
            if len(preparations) == 1:
                self.now += timedelta(seconds=60)
                raise OSError("Temporary connection failure")

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        with self.assertLogs("notifications", level="WARNING"):
            self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.now += timedelta(minutes=4, seconds=59)
        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(len(preparations), 1)
        self.now += timedelta(seconds=1)
        self.assertEqual(await self.service.tick(sender, prepare=prepare), 1)
        self.assertEqual(len(preparations), 2)
        self.assertEqual(len(sent), 1)

    async def test_completion_after_batch_selection_prevents_delivery(self) -> None:
        deadline = self.deadline()
        original_claim = self.db.claim_reminder
        sent = []

        def complete_then_claim(*args: object) -> str | None:
            self.db.complete_deadline(1, deadline["id"])
            return original_claim(*args)

        async def sender(item: dict) -> None:
            sent.append(item)

        with patch.object(self.db, "claim_reminder", side_effect=complete_then_claim):
            self.assertEqual(await self.service.tick(sender), 0)
        self.assertEqual(sent, [])

    async def test_completion_during_preparation_prevents_starting_transport(self) -> None:
        deadline = self.deadline()
        sent = []

        async def prepare(item: dict) -> None:
            self.db.complete_deadline(item["user_id"], item["id"])

        async def sender(item: dict) -> None:
            sent.append(item)

        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(sent, [])
        self.assertEqual(self.db.get_deadline(1, deadline["id"])["status"], "completed")

    async def test_cancellation_after_send_gate_spends_window_even_without_ack(self) -> None:
        self.deadline()
        delivered = []

        async def cancelled_sender(item: dict) -> None:
            # The remote side accepted the request; cancellation loses its acknowledgement.
            delivered.append(item["id"])
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await self.service.tick(cancelled_sender)

        async def sender(item: dict) -> None:
            delivered.append(item["id"])

        self.assertEqual(await self.service.tick(sender), 0)
        self.now += timedelta(minutes=10)
        self.db.close()
        self.db = Database(self.path)
        restarted = NotificationService(self.db, clock=lambda: self.now)
        self.assertEqual(await restarted.tick(sender), 0)
        self.assertEqual(len(delivered), 1)

    async def test_cancellation_before_send_gate_leaves_window_available(self) -> None:
        self.deadline()
        sent = []

        async def prepare(item: dict) -> None:
            raise asyncio.CancelledError()

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        with self.assertRaises(asyncio.CancelledError):
            await self.service.tick(sender, prepare=prepare)
        self.assertEqual(sent, [])
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(len(sent), 1)

    async def test_lost_ack_is_not_retried_and_next_window_caps_actual_messages_at_two(self) -> None:
        deadline = self.deadline(days=3)
        delivered = []

        async def sender(item: dict) -> None:
            delivered.append((item["id"], item["user_id"], item["days_before"]))
            if len(delivered) == 1:
                raise TimeoutError("Acknowledgement was lost after remote acceptance")

        with self.assertLogs("notifications", level="WARNING"):
            self.assertEqual(await self.service.tick(sender), 0)
        self.assertEqual(delivered, [(deadline["id"], 1, 3)])
        self.now += timedelta(minutes=10)
        self.db.close()
        self.db = Database(self.path)
        restarted = NotificationService(self.db, clock=lambda: self.now)
        for _ in range(3):
            self.assertEqual(await restarted.tick(sender), 0)
        self.assertEqual(len(delivered), 1)

        self.now = datetime(2026, 10, 13, 12, 0)
        self.assertEqual(await restarted.tick(sender), 1)
        self.assertEqual(delivered, [(deadline["id"], 1, 3), (deadline["id"], 1, 1)])
        self.db.close()
        self.db = Database(self.path)
        restarted = NotificationService(self.db, clock=lambda: self.now)
        for _ in range(3):
            self.assertEqual(await restarted.tick(sender), 0)
        self.now += timedelta(days=1)
        self.assertEqual(await restarted.tick(sender), 0)
        self.assertEqual(len(delivered), 2)

    async def test_slow_preparation_crossing_window_sends_only_current_window(self) -> None:
        deadline = self.db.create_deadline(
            1, "Физика", "Отчёт", self.now + timedelta(days=1, minutes=1))
        sent = []

        async def prepare(item: dict) -> None:
            self.now += timedelta(minutes=2)

        async def sender(item: dict) -> None:
            sent.append((item["id"], item["days_before"]))

        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(sent, [])
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(sent, [(deadline["id"], 1)])
        self.assertEqual(await self.service.tick(sender), 0)

    async def test_disabling_delivery_during_preparation_does_not_spend_window(self) -> None:
        deadline = self.deadline()
        sent = []

        async def prepare(item: dict) -> None:
            self.db.set_delivery_enabled(item["user_id"], False)

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(sent, [])
        self.db.set_delivery_enabled(1, True)
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(sent, [deadline["id"]])

    async def test_preparation_outliving_claim_lease_cannot_start_transport(self) -> None:
        deadline = self.deadline()
        sent = []

        async def prepare(item: dict) -> None:
            self.now += timedelta(seconds=REMINDER_LEASE_SECONDS)

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        self.assertEqual(await self.service.tick(sender, prepare=prepare), 0)
        self.assertEqual(sent, [])
        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(sent, [deadline["id"]])

    async def test_skipped_delivery_releases_claim_without_marking_sent(self) -> None:
        deadline = self.deadline()

        async def skipped_sender(item: dict) -> None:
            raise DeliverySkipped()

        self.assertEqual(await self.service.tick(skipped_sender), 0)
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])
        self.assertEqual([item["id"] for item in self.db.get_due_reminders(self.now)], [deadline["id"]])
        sent = []

        async def sender(item: dict) -> None:
            sent.append(item["id"])

        self.assertEqual(await self.service.tick(sender), 1)
        self.assertEqual(sent, [deadline["id"]])

    async def test_forbidden_response_after_account_deletion_does_not_recreate_user(self) -> None:
        self.deadline(1)
        second = self.deadline(2)
        attempted, delivered = [], []

        async def sender(item: dict) -> None:
            attempted.append(item["user_id"])
            if item["user_id"] == 1:
                self.db.delete_user(1)
                raise DeliveryDisabled
            delivered.append(item["id"])

        self.assertEqual(await self.service.tick(sender), 1)
        self.assertIsNone(self.db.get_user(1))
        self.assertEqual(attempted, [1, 2])
        self.assertEqual(delivered, [second["id"]])
        self.assertTrue(self.db.get_user(2)["delivery_enabled"])

    async def test_old_forbidden_response_does_not_disable_account_recreated_by_start(self) -> None:
        self.deadline(1)
        second = self.deadline(2)
        attempted, delivered = [], []

        async def sender(item: dict) -> None:
            attempted.append(item["user_id"])
            if item["user_id"] == 1:
                self.db.delete_user(1)
                # PrivateUserMiddleware performs this upsert on a subsequent /start.
                self.db.upsert_user(1, "restarted")
                raise DeliveryDisabled
            delivered.append(item["id"])

        self.assertEqual(await self.service.tick(sender), 1)
        self.assertTrue(self.db.get_user(1)["delivery_enabled"])
        self.assertEqual(self.db.get_user(1)["username"], "restarted")
        self.assertEqual(attempted, [1, 2])
        self.assertEqual(delivered, [second["id"]])
        self.assertTrue(self.db.get_user(2)["delivery_enabled"])


if __name__ == "__main__":
    unittest.main()
