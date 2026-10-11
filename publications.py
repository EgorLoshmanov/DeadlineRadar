"""Deliver newly published group deadlines from a durable SQLite queue."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from notifications import DeliveryDisabled, DeliverySkipped, RetryDelivery

logger = logging.getLogger(__name__)


class PublicationService:
    def __init__(self, database, clock: Callable[[], datetime] = datetime.now):
        self.db, self.clock = database, clock
        self._lock = asyncio.Lock()

    async def tick(
        self, sender: Callable[[dict], Awaitable[None]], *,
        prepare: Callable[[dict], Awaitable[None]] | None = None,
        deadline_id: int | None = None,
    ) -> int:
        sent = 0
        # A publication queued during another batch waits for that batch instead
        # of being postponed until the next scheduled tick.
        async with self._lock:
            for item in self.db.get_pending_publications(self.clock(), deadline_id=deadline_id):
                args = (item["id"], item["user_id"])
                token = None
                try:
                    if prepare is not None:
                        await prepare(item)
                    token = self.db.begin_publication_send(*args, self.clock())
                    if token is None:
                        continue
                    await sender(item)
                except DeliverySkipped:
                    continue
                except DeliveryDisabled:
                    if token is not None:
                        self.db.finish_publication(*args, token=token, result="disabled")
                except RetryDelivery as exc:
                    if token is not None:
                        self.db.retry_publication(
                            *args, token=token,
                            retry_at=self.clock() + timedelta(seconds=exc.retry_after),
                        )
                except asyncio.CancelledError:
                    if token is not None:
                        self.db.finish_publication(*args, token=token, result="uncertain")
                    raise
                except Exception:
                    if token is not None:
                        self.db.finish_publication(*args, token=token, result="uncertain")
                        logger.warning("Publication delivery uncertain: deadline=%s user=%s", *args)
                    else:
                        logger.warning("Publication preparation failed: deadline=%s user=%s", *args)
                else:
                    self.db.finish_publication(*args, token=token, result="sent")
                    sent += 1
        return sent
