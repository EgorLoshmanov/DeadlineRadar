"""Publication delivery, recovery and access control without external services."""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from db import Database
from notifications import DeliveryDisabled, RetryDelivery
from publications import PublicationService


class PublicationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "publications.db"
        self.now = datetime(2026, 10, 11, 12)
        self.db = Database(self.path, clock=lambda: self.now)
        self.group = self.db.create_group(1, "ИВТ <31>")
        self.db.join_group(2, self.group["invite_code"])
        self.service = PublicationService(self.db, lambda: self.now)
        self.sent = []

    def tearDown(self):
        self.db.close()
        self.directory.cleanup()

    def publish(self, **kwargs):
        return self.db.create_deadline(1, "Базы данных", "Лаба №3", self.now + timedelta(days=10),
                                       group_id=self.group["id"], **kwargs)

    async def sender(self, item):
        self.sent.append((item["id"], item["user_id"]))

    async def test_only_current_members_are_queued_once_after_confirmation(self):
        self.db.upsert_user(3)
        item = self.publish(idempotency_key="once")
        self.db.join_group(4, self.group["invite_code"])
        self.publish(idempotency_key="once")
        self.db.create_deadline(1, "Физика", "Личное", self.now + timedelta(days=10))
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(self.sent, [(item["id"], 2)])
        self.assertEqual(await self.service.tick(self.sender), 0)

    async def test_pending_and_sent_delivery_survive_reopening_database(self):
        item = self.publish()
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.service = PublicationService(self.db, lambda: self.now)
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.service = PublicationService(self.db, lambda: self.now)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.assertEqual(self.sent, [(item["id"], 2)])

    async def test_upgrade_does_not_broadcast_historical_group_tasks(self):
        old = self.publish()
        # A previous installation has deadlines but no publication queue.
        self.db._connection.execute("DROP TABLE group_publications")
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.service = PublicationService(self.db, lambda: self.now)
        self.assertIsNotNone(self.db.get_deadline(2, old["id"]))
        self.assertEqual(await self.service.tick(self.sender), 0)
        new = self.publish()
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(self.sent, [(new["id"], 2)])

    async def test_eligibility_is_rechecked_after_pacing(self):
        item = self.publish()

        async def leave(_):
            self.db.leave_group(2, self.group["id"])

        self.assertEqual(await self.service.tick(self.sender, prepare=leave), 0)
        self.assertEqual(self.sent, [])
        self.assertIsNone(self.db.begin_publication_send(item["id"], 2, self.now))

    async def test_disabled_completed_expired_and_deleted_targets_are_skipped(self):
        first = self.publish()
        self.db.set_delivery_enabled(2, False)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.db.set_delivery_enabled(2, True)
        self.db.complete_deadline(2, first["id"])
        second = self.publish()
        self.db.delete_deadline(1, second["id"])
        self.now += timedelta(days=11)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.assertEqual(self.sent, [])

    async def test_two_connections_cannot_send_the_same_publication(self):
        self.publish()
        other = Database(self.path, clock=lambda: self.now)
        try:
            second_service = PublicationService(other, lambda: self.now)

            async def pause(_):
                await asyncio.sleep(0)

            results = await asyncio.gather(
                self.service.tick(self.sender, prepare=pause),
                second_service.tick(self.sender, prepare=pause),
            )
            self.assertEqual(sum(results), 1)
            self.assertEqual(len(self.sent), 1)
        finally:
            other.close()

    async def test_lost_ack_consumes_attempt_across_restart(self):
        self.publish()

        async def lose_ack(item):
            await self.sender(item)
            raise TimeoutError("Acknowledgement lost")

        with self.assertLogs("publications", level="WARNING"):
            self.assertEqual(await self.service.tick(lose_ack), 0)
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(await PublicationService(self.db, lambda: self.now).tick(self.sender), 0)
        self.assertEqual(len(self.sent), 1)

    async def test_confirmed_retry_after_waits_then_sends_once(self):
        self.publish()

        async def retry(_):
            raise RetryDelivery(60)

        self.assertEqual(await self.service.tick(retry), 0)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.now += timedelta(seconds=60)
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(await self.service.tick(self.sender), 0)

    async def test_slow_retry_after_uses_failure_time_and_survives_restart(self):
        item = self.publish()

        async def slow_refusal(_):
            self.now += timedelta(seconds=60)
            raise RetryDelivery(60)

        self.assertEqual(await self.service.tick(slow_refusal), 0)
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.service = PublicationService(self.db, lambda: self.now)
        self.now += timedelta(seconds=59)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.now += timedelta(seconds=1)
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(await self.service.tick(self.sender), 0)
        self.assertEqual(self.sent, [(item["id"], 2)])

    async def test_cancel_before_send_keeps_pending_but_after_send_spends_attempt(self):
        self.publish()

        async def cancel(_):
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.service.tick(self.sender, prepare=cancel)
        self.assertEqual(len(self.db.get_pending_publications(self.now)), 1)
        with self.assertRaises(asyncio.CancelledError):
            await self.service.tick(cancel)
        self.assertEqual(self.db.get_pending_publications(self.now), [])

    async def test_blocked_recipient_does_not_interrupt_other_members(self):
        self.db.join_group(3, self.group["invite_code"])
        self.publish()

        async def block_one(item):
            if item["user_id"] == 2:
                raise DeliveryDisabled
            await self.sender(item)

        self.assertEqual(await self.service.tick(block_one), 1)
        self.assertFalse(self.db.get_user(2)["delivery_enabled"])
        self.assertEqual([user for _, user in self.sent], [3])

    async def test_late_forbidden_preserves_new_enable_and_publication_is_retried_once(self):
        for toggle_off in (False, True):
            with self.subTest(toggle_off=toggle_off):
                item = self.publish()

                async def late_refusal(_):
                    with Database(self.path) as settings_connection:
                        if toggle_off:
                            settings_connection.set_delivery_enabled(2, False)
                        settings_connection.set_delivery_enabled(2, True)
                    raise DeliveryDisabled

                self.assertEqual(await self.service.tick(late_refusal), 0)
                self.assertTrue(self.db.get_user(2)["delivery_enabled"])
                self.assertEqual(await self.service.tick(self.sender), 1)
                self.assertEqual(await self.service.tick(self.sender), 0)
                self.assertEqual(self.sent[-1], (item["id"], 2))
        self.assertEqual(len(self.sent), 2)

    async def test_publication_send_gate_uses_setting_after_preparation(self):
        self.publish()

        async def reenable(_):
            self.db.set_delivery_enabled(2, True)

        async def refused(_):
            raise DeliveryDisabled

        self.assertEqual(await self.service.tick(refused, prepare=reenable), 0)
        self.assertFalse(self.db.get_user(2)["delivery_enabled"])
        self.db.set_delivery_enabled(2, True)
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(await self.service.tick(self.sender), 0)

    async def test_explicit_enable_recovers_only_confirmed_disabled_publications(self):
        items = {status: self.publish() for status in ("sent", "uncertain", "sending", "disabled")}
        for status, item in items.items():
            token = self.db.begin_publication_send(item["id"], 2, self.now)
            self.assertIsNotNone(token)
            if status != "sending":
                self.db.finish_publication(item["id"], 2, token=token, result=status)
        self.assertFalse(self.db.get_user(2)["delivery_enabled"])
        self.db.set_delivery_enabled(2, True)
        self.assertEqual(await self.service.tick(self.sender), 1)
        self.assertEqual(self.sent, [(items["disabled"]["id"], 2)])
        self.assertEqual(await self.service.tick(self.sender), 0)

    async def test_stale_refusal_does_not_disable_recreated_account(self):
        item = self.publish()
        token = self.db.begin_publication_send(item["id"], 2, self.now)
        self.db.delete_user(2)
        self.db.join_group(2, self.group["invite_code"])
        self.assertFalse(self.db.finish_publication(item["id"], 2, token=token, result="disabled"))
        self.assertTrue(self.db.get_user(2)["delivery_enabled"])

    async def test_unfinished_network_attempt_is_not_replayed_after_crash(self):
        item = self.publish()
        self.assertIsNotNone(self.db.begin_publication_send(item["id"], 2, self.now))
        self.db.close()
        self.db = Database(self.path, clock=lambda: self.now)
        self.assertEqual(await PublicationService(self.db, lambda: self.now).tick(self.sender), 0)
