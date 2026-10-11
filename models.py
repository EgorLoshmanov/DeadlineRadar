"""Validation of confirmed deadline drafts, independent from Telegram."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

DATE_FORMAT = "%Y-%m-%d %H:%M"
MAX_SUBJECT_LENGTH = 120
MAX_TASK_LENGTH = 500


def parse_due_date(value: str) -> datetime:
    try:
        due = datetime.strptime(value, DATE_FORMAT)
    except (TypeError, ValueError) as exc:
        raise ValueError("Введите дату и время в формате ДД.ММ.ГГГГ ЧЧ:ММ") from exc
    if due.strftime(DATE_FORMAT) != value:
        raise ValueError("Дата должна содержать год, месяц, день, часы и минуты")
    return due


def clean_field(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: требуется текст")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name}: некорректные символы Unicode") from None
    value = " ".join(value.split())
    if not value or len(value) > maximum:
        raise ValueError(f"{name}: введите от 1 до {maximum} символов")
    return value


@dataclass(frozen=True)
class DeadlineDraft:
    subject: str
    task: str
    due_date: str
    group_id: int | None = None
    creation_key: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, now: datetime) -> "DeadlineDraft":
        subject = clean_field(data.get("subject"), "Предмет", MAX_SUBJECT_LENGTH)
        task = clean_field(data.get("task"), "Задание", MAX_TASK_LENGTH)
        due = parse_due_date(data.get("due_date"))
        if due <= now.replace(tzinfo=None):
            raise ValueError("Срок уже прошёл. Укажите будущую дату и время")
        group_id = data.get("group_id")
        if group_id is not None and (type(group_id) is not int or group_id <= 0):
            raise ValueError("Некорректная учебная группа")
        key = data.get("creation_key") or uuid4().hex
        if not isinstance(key, str) or len(key) > 64:
            raise ValueError("Некорректный ключ подтверждения")
        return cls(subject, task, due.strftime(DATE_FORMAT), group_id, key)
