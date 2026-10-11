"""Russian text views. User content is escaped before rendering as HTML."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from models import parse_due_date

WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def format_date(value: str) -> str:
    due = parse_due_date(value)
    return f"{due:%d.%m.%Y} ({WEEKDAYS[due.weekday()]}) {due:%H:%M}"


def draft_card(draft: dict, destination: str = "Личный список") -> str:
    method = {"llm": "модель", "manual": "ручной ввод"}.get(draft.get("parsing_method"), "локальные правила")
    return ("📌 <b>Проверьте дедлайн</b>\n\n"
            f"📚 Предмет: <b>{escape(draft['subject'])}</b>\n"
            f"📝 Задание: {escape(draft['task'])}\n"
            f"⏰ Срок: <b>{format_date(draft['due_date'])}</b>\n"
            f"👥 Куда: {escape(destination)}\n\n"
            f"Распознавание: {method}. Дата без времени означает 23:59.\n"
            "Сохраните карточку или исправьте поля.")


def urgency(value: str, now: datetime) -> str:
    due = parse_due_date(value)
    if due <= now:
        return "🔴 Просрочено"
    if due.date() == now.date():
        return "🟠 Сегодня"
    if due <= now + timedelta(days=3):
        return "🟡 В ближайшие 3 дня"
    if due <= now + timedelta(days=7):
        return "🟢 В течение недели"
    return "🔵 Позже"


def deadline_list(items: list[dict], now: datetime, *, completed: bool = False) -> str:
    title = "✅ Сданные задания" if completed else "📋 Активные дедлайны"
    if not items:
        return f"<b>{title}</b>\n\n" + ("Сданных заданий пока нет." if completed else
            "Пока пусто. Перешлите сообщение с заданием или нажмите «Добавить вручную».")
    lines = [f"<b>{title}</b>", ""]
    previous = None
    for item in items:
        heading = "✅ Сдано" if completed else urgency(item["due_date"], now)
        if heading != previous:
            lines.extend([f"<b>{heading}</b>", ""])
            previous = heading
        scope = " · " + escape(item["group_name"]) if item.get("group_name") else " · личное"
        lines.extend([f"<b>#{item['id']} · {escape(item['subject'])}</b>{scope}",
                      escape(item["task"]), f"⏰ {format_date(item['due_date'])}", ""])
    return "\n".join(lines).rstrip()


def reminder_text(item: dict) -> str:
    days = "3 дня" if item["days_before"] == 3 else "1 день"
    return (f"🔔 <b>До дедлайна не больше {days}</b>\n\n"
            f"📚 {escape(item['subject'])}\n📝 {escape(item['task'])}\n"
            f"⏰ {format_date(item['due_date'])}\n\n"
            "Если задание готово, отметьте «Я сдал». После отметки напоминания прекратятся.")


def publication_text(item: dict) -> str:
    return ("📣 <b>Новое задание группы</b>\n\n"
            f"👥 {escape(item['group_name'])}\n"
            f"📚 {escape(item['subject'])}\n📝 {escape(item['task'])}\n"
            f"⏰ {format_date(item['due_date'])}\n\n"
            "Задание добавлено в /deadlines. Отметьте «Я сдал», когда закончите.")
