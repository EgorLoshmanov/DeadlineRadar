"""Use cases shared by Telegram handlers and the offline demonstration."""
from __future__ import annotations

from datetime import datetime
from typing import Callable
from uuid import uuid4

from models import DeadlineDraft


class DeadlineService:
    def __init__(self, database, parser, clock: Callable[[], datetime] = datetime.now):
        self.db, self.parser, self.clock = database, parser, clock

    async def propose(self, user_id: int, text: str, group_id: int | None = None) -> dict:
        if group_id is not None and not self.db.get_group_for_owner(user_id, group_id):
            raise ValueError("Добавлять дедлайны группы может только её староста")
        result = await self.parser.parse(text, now=self.clock())
        return {**result, "group_id": group_id, "creation_key": uuid4().hex}

    def confirm(self, user_id: int, data: dict) -> dict:
        draft = DeadlineDraft.from_dict(data, now=self.clock())
        return self.db.create_deadline(
            user_id, draft.subject, draft.task, draft.due_date,
            group_id=draft.group_id, idempotency_key=draft.creation_key,
        )
