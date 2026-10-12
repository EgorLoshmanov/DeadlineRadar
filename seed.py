"""Add repeatable example deadlines without deleting or replacing existing data."""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sqlite3
import sys

from config import ROOT, Settings
from db import Database
from presentation import format_date


def positive_user_id(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Telegram ID должен быть целым числом") from exc
    if not 1 <= result <= 2**63 - 1:
        raise argparse.ArgumentTypeError("Telegram ID должен быть от 1 до 9223372036854775807")
    return result


def add_examples(database: Database, user_id: int, settings: Settings) -> tuple[int, int]:
    """Reuse seed keys and preserve the original dates and completion statuses."""
    now = settings.now().replace(second=0, microsecond=0)
    database.upsert_user(user_id)
    group_name = f"Демонстрационная группа · {user_id}"
    group = next((item for item in database.list_groups(user_id)
                  if item["is_owner"] and item["name"] == group_name), None)
    if group is None:
        group = database.create_group(user_id, group_name)

    examples = (
        ("personal-databases", "Базы данных", "Лабораторная работа №3",
         (now + timedelta(days=3)).replace(hour=18, minute=0), None, False),
        ("personal-math", "Математический анализ", "Домашнее задание №2",
         (now + timedelta(days=1)).replace(hour=23, minute=59), None, False),
        ("personal-oop", "ООП", "Проект: обработка исключений",
         (now + timedelta(days=6)).replace(hour=18, minute=0), None, False),
        ("personal-physics", "Физика", "Отчёт по лабораторной работе №1",
         now + timedelta(hours=2), None, False),
        ("personal-history-completed", "История", "Реферат №1 — пример сданной работы",
         (now + timedelta(days=2)).replace(hour=23, minute=59), None, True),
        ("group-networks", "Компьютерные сети", "Лабораторная работа №4",
         (now + timedelta(days=4)).replace(hour=18, minute=0), group["id"], False),
        ("group-operating-systems", "Операционные системы", "Практическая работа №2",
         (now + timedelta(days=7)).replace(hour=23, minute=59), group["id"], False),
    )
    existing = {item["idempotency_key"] for item in database.list_deadlines(user_id, include_completed=True)
                if item["creator_id"] == user_id}
    created = reused = 0
    for label, subject, task, due, group_id, completed in examples:
        key = f"demo:v1:{label}"
        if key in existing:
            reused += 1
            continue
        item = database.create_deadline(user_id, subject, task, due,
                                        group_id=group_id, idempotency_key=key)
        if completed:
            database.complete_deadline(user_id, item["id"])
        created += 1
        existing.add(key)
    return created, reused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Добавить демонстрационные дедлайны. Повторный запуск сохраняет исходные даты и статусы.",
        epilog="Для бота укажите его рабочую БД и свой Telegram ID: --database deadlines.db --user-id 123456789")
    parser.add_argument("--database", type=Path, default=ROOT / "demo.db",
                        help="путь SQLite (по умолчанию отдельный файл demo.db рядом со скриптом)")
    parser.add_argument("--user-id", type=positive_user_id, default=1001,
                        help="Telegram ID владельца примеров (по умолчанию демонстрационный 1001)")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        path = args.database.expanduser().resolve()
        with Database(path, clock=settings.now) as database:
            created, reused = add_examples(database, args.user_id, settings)
            print(f"База: {path}")
            print(f"Пользователь: {args.user_id}; часовой пояс: {settings.timezone}")
            print(f"Добавлено дедлайнов: {created}. Уже существовало: {reused}.")
            print("\nДемонстрационные задания:")
            for item in database.list_deadlines(args.user_id, include_completed=True):
                if not (item.get("idempotency_key") or "").startswith("demo:v1:"):
                    continue
                if item["creator_id"] != args.user_id:
                    continue
                status = "сдано" if item["completed"] else "активно"
                scope = "группа" if item["group_id"] else "личное"
                print(f"  #{item['id']} · {item['subject']}: {item['task']} · "
                      f"{format_date(item['due_date'])} · {scope}, {status}")
            group_name = f"Демонстрационная группа · {args.user_id}"
            group = next(item for item in database.list_groups(args.user_id)
                         if item["is_owner"] and item["name"] == group_name)
            print(f"\nКод демонстрационной группы: {group['invite_code']}")
            print("Повторный запуск не меняет существующие задания, сроки и отметки выполнения.")
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Не удалось подготовить примеры: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
