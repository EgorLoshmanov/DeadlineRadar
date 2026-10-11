"""Run the complete offline acceptance scenario using a temporary SQLite file."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import traceback

from db import Database, REMINDER_LEASE_SECONDS
from notifications import NotificationService, RetryDelivery
from parser import ParserEngine, parse_manual_date
from presentation import format_date, urgency
from services import DeadlineService


FIXED_NOW = datetime(2026, 10, 11, 12, 0)
OWNER, MEMBER, OUTSIDER = 1001, 1002, 1003


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def passed(message: str) -> None:
    print(f"  [ОК] {message}")


async def check_deadline_flow(path: Path) -> None:
    parser = ParserEngine(use_llm=False)
    with Database(path) as database:
        service = DeadlineService(database, parser, clock=lambda: FIXED_NOW)
        print("\n1. Распознавание и подтверждение")
        examples = (
            ("Лаба 3 по БД до пятницы в 18:00", "2026-10-16 18:00"),
            ("ДЗ по матанализу до 14.10.2026", "2026-10-14 23:59"),
            ("Отчёт по физике завтра в 19:00", "2026-10-12 19:00"),
            ("Проект по ООП до 20 октября в 18:30", "2026-10-20 18:30"),
        )
        drafts = []
        for message, expected in examples:
            draft = await service.propose(OWNER, message)
            require(draft["status"] == "ok", f"Не распознано сообщение: {message}")
            require(draft["due_date"] == expected, f"Неверная дата: {message}")
            require(draft["parsing_method"] == "regex_fallback", "Офлайн-парсер вызвал внешнюю модель")
            require(0 <= draft["confidence"] <= 1, "Некорректная уверенность парсера")
            drafts.append(draft)
            print(f"  «{message}» → {draft['subject']}; {format_date(expected)}")

        require(database.list_deadlines(OWNER) == [], "Черновик сохранился без подтверждения")
        first = service.confirm(OWNER, drafts[0])
        repeated = service.confirm(OWNER, drafts[0])
        require(first["id"] == repeated["id"], "Повторное подтверждение создало дубль")
        passed("Запись появляется после подтверждения; повторное сохранение не создаёт дубль")

        count = len(database.list_deadlines(OWNER))
        cancelled = await service.propose(OWNER, "Лаба 9 по ООП до 25.10.2026")
        require(cancelled["due_date"] is not None, "Не создан черновик для проверки отмены")
        # Cancel means discarding the draft without calling confirm, just as in the bot.
        del cancelled
        require(len(database.list_deadlines(OWNER)) == count, "Отмена сохранила дедлайн")
        passed("Отмена карточки оставляет базу без изменений")

        print("\n2. Ручное уточнение и исправление")
        manual = await service.propose(OWNER, "Реферат по истории к следующей паре")
        require(manual["status"] == "manual_input_required", "Выдумана дата следующей пары")
        require(manual["due_date"] is None, "Для неопределённой даты получен готовый срок")
        try:
            service.confirm(OWNER, manual)
        except ValueError:
            pass
        else:
            raise AssertionError("Черновик без даты можно сохранить")
        require(len(database.list_deadlines(OWNER)) == count, "Неполный черновик попал в БД")
        manual.update(subject="История", task="Реферат №2",
                      due_date=parse_manual_date("14.10.2026 19:00", FIXED_NOW))
        corrected = service.confirm(OWNER, manual)
        require(corrected["task"] == "Реферат №2", "Исправление задания не сохранилось")
        passed("Для «к следующей паре» запрошена дата; исправленные поля сохранены")
        for invalid in ("Лаба по БД до 31.02.2027", "Лаба по БД до 01.10.2026"):
            result = await service.propose(OWNER, invalid)
            require(result["status"] == "manual_input_required", f"Принята недопустимая дата: {invalid}")
        passed("Невозможные и прошедшие даты требуют уточнения")

        # Save out of order to check that the list sorts by due date, not insertion.
        for draft in reversed(drafts[1:]):
            service.confirm(OWNER, draft)
        items = database.list_deadlines(OWNER)
        dates = [item["due_date"] for item in items]
        require(dates == sorted(dates), "Дедлайны не отсортированы по сроку сдачи")
        print("\n3. Личный список по срочности")
        for item in items:
            print(f"  {urgency(item['due_date'], FIXED_NOW)} · {item['subject']}: "
                  f"{item['task']} · {format_date(item['due_date'])}")
        passed("Дедлайны упорядочены по дате")

        print("\n4. Учебная группа и индивидуальное выполнение")
        group = database.create_group(OWNER, "ИВТ-31 · проверка")
        database.join_group(MEMBER, group["invite_code"])
        database.join_group(MEMBER, group["invite_code"])
        require(database.list_group_member_ids(group["id"]) == [OWNER, MEMBER], "Повторный вход создал дубль участника")
        shared_draft = await service.propose(
            OWNER, "Лаба 4 по сетям до 14.10.2026 в 12:00", group_id=group["id"])
        shared = service.confirm(OWNER, shared_draft)
        require(database.get_deadline(MEMBER, shared["id"]) is not None, "Участник не видит общий дедлайн")
        require(database.get_deadline(MEMBER, first["id"]) is None, "Участнику виден чужой личный дедлайн")
        require(database.get_deadline(OUTSIDER, shared["id"]) is None, "Постороннему виден дедлайн группы")
        require(not database.complete_deadline(OUTSIDER, shared["id"]), "Посторонний изменил общий дедлайн")
        try:
            await service.propose(MEMBER, "Лаба по БД завтра", group_id=group["id"])
        except ValueError:
            pass
        else:
            raise AssertionError("Участник может публиковать дедлайны за старосту")
        require(database.complete_deadline(MEMBER, shared["id"]), "Не удалось отметить выполнение")
        require(database.get_deadline(MEMBER, shared["id"])["status"] == "completed", "Выполнение участника не сохранено")
        require(database.get_deadline(OWNER, shared["id"])["status"] == "pending", "Выполнение участника закрыло задачу всей группе")
        require(database.list_deadlines(MEMBER) == [], "Сданный дедлайн остался в активном списке участника")
        passed("Приглашение работает; личные задачи защищены; «Сдал» действует для одного участника")

    with Database(path) as reopened:
        require(reopened.get_deadline(MEMBER, shared["id"])["status"] == "completed", "Перезапуск потерял индивидуальное выполнение")
        require(reopened.get_deadline(OWNER, first["id"]) is not None, "Перезапуск потерял дедлайны")
        passed("Списки, группа и индивидуальное выполнение пережили открытие БД заново")


async def check_notifications(path: Path) -> None:
    print("\n5. Напоминания без Telegram и сети")
    current = [FIXED_NOW]
    delivered: list[tuple[int, int, int]] = []

    async def record(item: dict) -> None:
        delivered.append((item["id"], item["user_id"], item["days_before"]))

    with Database(path) as database:
        due = FIXED_NOW + timedelta(days=3)
        deadline = database.create_deadline(OWNER, "Базы данных", "Лабораторная №3", due)
        completed = database.create_deadline(OWNER, "Физика", "Уже сданный отчёт", due)
        database.complete_deadline(OWNER, completed["id"])
        scheduler = NotificationService(database, clock=lambda: current[0])
        current[0] = FIXED_NOW - timedelta(minutes=1)
        require(await scheduler.tick(record) == 0, "Напоминание отправлено раньше трёхдневного окна")
        current[0] = FIXED_NOW
        require(await scheduler.tick(record) == 1, "Не отправлено напоминание за 3 дня")
        require(delivered == [(deadline["id"], OWNER, 3)], "Напоминание пришло по сданной задаче или с неверным интервалом")
        require(await scheduler.tick(record) == 0, "Повторная проверка дублирует отправку")
        passed("За 3 дня отправлено одно напоминание; сданные задания пропущены")

    with Database(path) as database:
        scheduler = NotificationService(database, clock=lambda: current[0])
        require(await scheduler.tick(record) == 0, "Перезапуск повторил доставленное напоминание")
        current[0] = FIXED_NOW + timedelta(days=2)
        require(await scheduler.tick(record) == 1, "Не восстановлено напоминание за 1 день")
        require(delivered[-1] == (deadline["id"], OWNER, 1), "Восстановлен неверный интервал напоминания")
        require(await scheduler.tick(record) == 0, "Напоминание за 1 день продублировано")
        passed("После перезапуска восстановлено напоминание за 1 день; дублей нет")

    with Database(path) as database:
        scheduler = NotificationService(database, clock=lambda: current[0])
        require(await scheduler.tick(record) == 0, "Повторное открытие БД сбросило историю доставки")
        current[0] = FIXED_NOW + timedelta(days=3)
        require(await scheduler.tick(record) == 0, "Напоминание отправлено после срока сдачи")
        require(len(delivered) == 2, "Число напоминаний на задачу превышает два")
        passed("История доставки сохраняется; после срока напоминания не отправляются")

        current[0] = FIXED_NOW
        retry = database.create_deadline(OWNER, "ООП", "Проверка повторной доставки", FIXED_NOW + timedelta(days=2))
        attempts = 0

        async def temporary_error(item: dict) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RetryDelivery(retry_after=60)
            await record(item)

        require(await scheduler.tick(temporary_error) == 0, "Ошибка доставки ошибочно отмечена успешной")
        require(await scheduler.tick(temporary_error) == 0 and attempts == 1, "Нарушен интервал повторной попытки")
        current[0] += timedelta(seconds=60)
        require(await scheduler.tick(temporary_error) == 1, "Доставка не повторилась после временной ошибки")
        require(delivered[-1] == (retry["id"], OWNER, 3), "Повторная доставка перепутала задачу")
        database.complete_deadline(OWNER, retry["id"])
        current[0] = FIXED_NOW + timedelta(days=1)
        require(await scheduler.tick(record) == 0, "Выполнение не отключило дальнейшие напоминания")
        passed("Временная ошибка повторяется через минуту; «Сдал» отключает следующий интервал")

        current[0] = FIXED_NOW
        interrupted = database.create_deadline(OWNER, "Сети", "Проверка восстановления после сбоя", FIXED_NOW + timedelta(days=2))
        require(database.claim_reminder(interrupted["id"], OWNER, 3, current[0]), "Не зарезервирована доставка")

    with Database(path) as database:
        scheduler = NotificationService(database, clock=lambda: current[0])
        current[0] = FIXED_NOW + timedelta(seconds=REMINDER_LEASE_SECONDS - 1)
        require(await scheduler.tick(record) == 0, "Резерв доставки сброшен при перезапуске")
        current[0] += timedelta(seconds=1)
        require(await scheduler.tick(record) == 1, "Зависшая после сбоя доставка не восстановилась")
        require(delivered[-1] == (interrupted["id"], OWNER, 3), "Восстановлена неверная доставка")
        require(await scheduler.tick(record) == 0, "Восстановленная доставка продублирована")
        passed("Незавершённая доставка восстановилась после истечения резерва")


async def run() -> None:
    print("DeadlineRadar — автономная проверка MVP")
    print(f"Фиксированное локальное время: {FIXED_NOW:%d.%m.%Y %H:%M}")
    print("Внешние API отключены. Данные создаются во временных БД.")
    with tempfile.TemporaryDirectory(prefix="deadline-radar-check-") as directory:
        root = Path(directory)
        await check_deadline_flow(root / "scenario.db")
        await check_notifications(root / "notifications.db")
    print("\nВсе сценарии пройдены. Временные данные удалены.")


def main() -> int:
    try:
        asyncio.run(run())
    except Exception as exc:
        print(f"\n[ОШИБКА] Проверка не пройдена: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
