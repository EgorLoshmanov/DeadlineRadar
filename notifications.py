"""Durable notification delivery; all schedule state lives in SQLite."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class DeliveryDisabled(Exception):
    """Telegram definitively rejected delivery: blocked bot or deleted account."""


class DeliverySkipped(Exception):
    """The request was not sent, e.g. eligibility changed during pacing."""


class RetryDelivery(Exception):
    """A confirmed rejection without delivery (Telegram RetryAfter), safe to retry."""

    def __init__(self, retry_after: int = 60):
        self.retry_after = max(1, retry_after)
        super().__init__("Temporary delivery error")


class NotificationService:
    def __init__(self, database, clock: Callable[[], datetime] = datetime.now):
        self.db, self.clock = database, clock
        self._lock = asyncio.Lock()

    async def tick(
        self, sender: Callable[[dict], Awaitable[None]], *,
        prepare: Callable[[dict], Awaitable[None]] | None = None,
    ) -> int:
        if self._lock.locked():
            return 0
        sent = 0
        async with self._lock:
            now = self.clock()
            for item in self.db.get_due_reminders(now):
                now = self.clock()
                args = (item["id"], item["user_id"], item["days_before"])
                claim_token = self.db.claim_reminder(*args, now)
                if claim_token is None:
                    continue
                sending = False
                try:
                    current = self.db.get_deadline(item["user_id"], item["id"])
                    if not current or current["completed"]:
                        raise DeliverySkipped
                    if prepare is not None:
                        await prepare(item)
                    # Commit after pacing, immediately before transport. A crash after
                    # this gate consumes the window even if no ACK reaches the process.
                    if not self.db.begin_reminder_send(*args, self.clock(), claim_token=claim_token):
                        self.db.release_reminder(*args, claim_token=claim_token)
                        continue
                    sending = True
                    await sender(item)
                except DeliverySkipped:
                    self.db.release_reminder(*args, claim_token=claim_token, known_not_sent=True)
                except DeliveryDisabled:
                    disabled = self.db.disable_delivery_for_reminder(*args, claim_token=claim_token)
                    self.db.release_reminder(*args, claim_token=claim_token, known_not_sent=True)
                    if disabled:
                        logger.info("Delivery disabled for user %s", item["user_id"])
                except RetryDelivery as exc:
                    self.db.release_reminder(
                        *args, retry_at=self.clock() + timedelta(seconds=exc.retry_after),
                        claim_token=claim_token, known_not_sent=True,
                    )
                except asyncio.CancelledError:
                    if sending:
                        self.db.mark_reminder_uncertain(*args, claim_token=claim_token)
                    else:
                        self.db.release_reminder(*args, claim_token=claim_token)
                    raise
                except Exception:
                    if sending:
                        self.db.mark_reminder_uncertain(*args, claim_token=claim_token)
                        logger.warning("Reminder delivery uncertain; window consumed: deadline=%s user=%s", *args[:2])
                    else:
                        self.db.release_reminder(
                            *args, retry_at=self.clock() + timedelta(minutes=5), claim_token=claim_token,
                        )
                        logger.warning("Reminder preparation failed: deadline=%s user=%s", *args[:2])
                else:
                    self.db.mark_reminder_sent(*args, self.clock(), claim_token=claim_token)
                    sent += 1
        return sent
