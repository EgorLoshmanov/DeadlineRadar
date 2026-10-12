"""Private Telegram conversations, draft confirmation and group management."""
from __future__ import annotations

import logging
from html import escape
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from aiogram import BaseMiddleware, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery, ErrorEvent, InaccessibleMessage, InlineKeyboardButton, InlineKeyboardMarkup,
    KeyboardButton, Message, ReplyKeyboardMarkup,
)

from config import semester_start
from models import clean_field
from notifications import DeliveryDisabled, RetryDelivery
from parser import parse_manual_date, sanitize_text
from presentation import deadline_list, draft_card, publication_text
from publications import PublicationService

logger = logging.getLogger(__name__)
PAGE_SIZE = 4
MAX_ENTITY_ID = 2**63 - 1

HELP = (
    "<b>DeadlineRadar — учебные дедлайны</b>\n\n"
    "Перешлите сюда текст преподавателя или напишите, например: "
    "«Лаба 3 по БД до пятницы 18:00». Проверьте карточку и нажмите «Сохранить». "
    "Пока карточка не подтверждена, в списке её нет.\n\n"
    "/deadlines — активные задания и отметка «Я сдал»\n"
    "/add — добавить вручную\n"
    "/history — сданные в этом семестре\n"
    "/group — мои учебные группы\n"
    "/newgroup Название — создать группу старосты\n"
    "/join КОД — вступить по приглашению\n"
    "/publish — добавить задание для своей группы\n"
    "/settings — часовой пояс и обработка данных\n"
    "/plan — учебные планы и лимит активных задач\n"
    "/cancel — отменить ввод\n"
    "/delete_me — удалить свои данные\n\n"
    "Напоминания приходят за 3 дня и за 1 день, пока задание не сдано. "
    "После публикации старостой участники также получают новое задание сообщением. "
    "У группового задания каждый студент отмечает выполнение за себя. "
    "Бот работает в личном диалоге и не читает учебные чаты."
)


class Flow(StatesGroup):
    preview = State()
    subject = State()
    task = State()
    date = State()
    group_name = State()
    join_code = State()
    publish_text = State()
    delete_user = State()


def inline(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=data) for label, data in row]
        for row in rows
    ])


def parse_callback_id(value: str) -> int:
    entity_id = int(value)
    if not 1 <= entity_id <= MAX_ENTITY_ID:
        raise ValueError("Некорректный номер")
    return entity_id


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📋 Мои дедлайны"), KeyboardButton(text="➕ Добавить вручную")],
        [KeyboardButton(text="👥 Группа"), KeyboardButton(text="❓ Помощь")],
    ], resize_keyboard=True, input_field_placeholder="Перешлите текст с заданием")


async def answer_callback(query: CallbackQuery, *args, **kwargs) -> None:
    """An expired toast must not interrupt the action or its chat response."""
    try:
        await query.answer(*args, **kwargs)
    except TelegramBadRequest as exc:
        error = exc.message.lower()
        if "query is too old" not in error and "query id is invalid" not in error:
            raise


async def send_group_publication(bot, item: dict) -> None:
    try:
        await bot.send_message(item["user_id"], publication_text(item), reply_markup=inline([
            [("✅ Я сдал", f"done:{item['id']}")],
        ]))
    except TelegramForbiddenError as exc:
        raise DeliveryDisabled from exc
    except TelegramRetryAfter as exc:
        raise RetryDelivery(exc.retry_after) from exc


class PrivateUserMiddleware(BaseMiddleware):
    def __init__(self, database):
        self.db = database

    async def __call__(self, handler, event, data):
        message = event.message if isinstance(event, CallbackQuery) else event
        if not message or message.chat.type != "private":
            if isinstance(event, CallbackQuery):
                await answer_callback(event, "Откройте личный диалог с ботом", show_alert=True)
            elif isinstance(event, Message):
                await event.answer("DeadlineRadar работает в личном диалоге. Откройте профиль бота и нажмите Start.")
            return None
        if event.from_user is None:
            return None
        self.db.upsert_user(event.from_user.id, event.from_user.username)
        return await handler(event, data)


def create_router(database, service, settings, bot_username: str, *, on_publication=None) -> Router:
    router = Router(name="deadline_radar")
    router.message.outer_middleware(PrivateUserMiddleware(database))
    router.callback_query.outer_middleware(PrivateUserMiddleware(database))
    publications = PublicationService(database, settings.now)

    async def clear_buttons(message: Message | InaccessibleMessage) -> None:
        if isinstance(message, InaccessibleMessage):
            return
        try:
            await message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            # A second click or a deleted/old message does not undo a saved task.
            pass

    async def preview(message: Message, state: FSMContext, draft: dict) -> None:
        token = uuid4().hex[:12]
        destination = "Личный список"
        if draft.get("group_id"):
            group = database.get_group_for_owner(message.chat.id, draft["group_id"])
            if not group:
                await state.clear()
                await message.answer("Группа больше недоступна. Начните заново: /group")
                return
            destination = group["group_name"]
        await state.set_data({"draft": draft, "preview_token": token})
        await state.set_state(Flow.preview)
        await message.answer(draft_card(draft, destination), reply_markup=inline([
            [("✅ Сохранить", f"draft:save:{token}")],
            [("📚 Предмет", f"draft:subject:{token}"), ("📝 Задание", f"draft:task:{token}")],
            [("⏰ Дата", f"draft:date:{token}"), ("✖ Отмена", f"draft:cancel:{token}")],
        ]))

    async def show_deadlines(message: Message, user_id: int, page: int = 0, history: bool = False) -> None:
        items = database.list_deadlines(user_id, include_completed=history)
        if history:
            cutoff = semester_start(settings.now()).strftime("%Y-%m-%d")
            items = [item for item in items if item["completed"] and item["completed_at"] is not None
                     and item["completed_at"] >= cutoff]
        pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(max(page, 0), pages - 1)
        visible = items[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        start_number = page * PAGE_SIZE + 1
        rows = []
        for number, item in enumerate(visible, start=start_number):
            buttons = [] if history else [(f"✅ Я сдал #{number}", f"done:{item['id']}")]
            if item["can_delete"]:
                buttons.append((f"🗑 #{number}", f"remove:ask:{item['id']}"))
            if buttons:
                rows.append(buttons)
        navigation = []
        if page > 0:
            navigation.append(("← Назад", f"list:{int(history)}:{page - 1}"))
        if page + 1 < pages:
            navigation.append(("Далее →", f"list:{int(history)}:{page + 1}"))
        if navigation:
            rows.append(navigation)
        rows.append([("🔄 Обновить", f"list:{int(history)}:{page}")])
        text = deadline_list(visible, settings.now(), completed=history, start_number=start_number)
        if len(items) > PAGE_SIZE:
            text += f"\n\nСтраница {page + 1}/{pages} · всего {len(items)}"
        await message.answer(text, reply_markup=inline(rows))

    async def show_groups(message: Message, user_id: int, page: int = 0, owned: bool = False) -> None:
        groups = database.list_groups(user_id)
        if owned:
            groups = [group for group in groups if group["is_owner"]]
        pages = max(1, (len(groups) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(max(page, 0), pages - 1)
        rows, lines = [], ["<b>👥 Мои учебные группы</b>", ""]
        for group in groups[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
            role = "староста" if group["is_owner"] else "студент"
            lines.append(f"#{group['id']} · <b>{escape(group['name'])}</b> · {role}")
            if group["is_owner"]:
                rows.append([(f"Добавить в #{group['id']}", f"publish:{group['id']}"),
                             ("Приглашение", f"invite:{group['id']}")])
            else:
                rows.append([(f"Выйти из #{group['id']}", f"leave:ask:{group['id']}")])
        if not groups:
            lines.append("Групп пока нет. Создайте свою группу или вступите по приглашению старосты.")
        navigation = []
        if page > 0:
            navigation.append(("← Назад", f"groups:{int(owned)}:{page - 1}"))
        if page + 1 < pages:
            navigation.append(("Далее →", f"groups:{int(owned)}:{page + 1}"))
        if navigation:
            rows.append(navigation)
        rows.append([("Создать группу", "group:new"), ("Вступить", "group:join")])
        await message.answer("\n".join(lines), reply_markup=inline(rows))

    async def invitation(message: Message, group: dict) -> None:
        link = f"https://t.me/{bot_username}?start=g_{group['invite_code']}"
        await message.answer(
            f"<b>{escape(group['group_name'])}</b>\n\n"
            f"Отправьте одногруппникам ссылку:\n{escape(link)}\n\n"
            f"Или попросите ввести <code>/join {group['invite_code']}</code>. "
            "Каждый участник должен открыть бота и нажать Start."
        )

    async def join(message: Message, user_id: int, text: str, *, start_payload: bool = False) -> bool:
        code = text.strip()
        if code.startswith("https://t.me/"):
            code = parse_qs(urlparse(code).query).get("start", [""])[0]
            start_payload = True
        if start_payload:
            code = code.removeprefix("g_")
        codes = [code]
        if not start_payload and code.startswith("g_"):
            # A generated raw code may itself start with g_. Prefer it in full;
            # retain support for a pasted bare start payload if it is not found.
            codes.append(code[2:])
        for candidate in codes:
            try:
                group = database.join_group(user_id, candidate)
            except ValueError:
                continue
            break
        else:
            await message.answer("Приглашение не найдено. Проверьте код или запросите ссылку у старосты.")
            return False
        await message.answer(f"Вы в группе <b>{escape(group['name'])}</b>. Задания доступны в /deadlines.")
        return True

    async def show_settings(message: Message, user_id: int) -> None:
        enabled = database.get_user(user_id)["delivery_enabled"]
        provider = getattr(settings, "llm_provider", "anthropic")
        provider_name = {"gigachat": "GigaChat", "anthropic": "Anthropic"}.get(provider, "совместимый API")
        if (provider == "openai" and getattr(settings, "llm_api_url", "").rstrip("/")
                == "https://deepcode.ci.nsu.ru/api/chat/completions"):
            provider_name = "API НГУ"
        await message.answer(
            f"<b>Настройки и данные</b>\n\nЧасовой пояс: {escape(settings.timezone)}. "
            "Все участники используют этот часовой пояс. Если время не указано, срок — 23:59.\n"
            f"Уведомления: {'включены' if enabled else 'выключены'}. "
            "Настройка действует на напоминания и новые задания группы.\n\n"
            "Хранятся ваш Telegram ID, username, группы, подтверждённые задания и отметки выполнения. "
            "Исходные сообщения и история чата не сохраняются в БД. "
            + (f"Для распознавания введённый текст отправляется провайдеру {provider_name}. "
               if settings.llm_api_key and settings.llm_model else "Распознавание выполняется локально. ")
            + "Черновики находятся в памяти до перезапуска. "
            "История выполненных заданий доступна за текущий семестр. /delete_me — удалить данные.",
            reply_markup=inline([[("Выключить уведомления" if enabled else "Включить уведомления",
                                   f"settings:delivery:{int(not enabled)}")]]),
        )

    async def show_plan(message: Message, user_id: int) -> None:
        usage = database.get_plan_usage(user_id)
        names = {"free": "Бесплатный", "student": "Студент", "group": "Группа"}
        limit = str(usage["limit"]) if usage["limit"] is not None else "без лимита"
        await message.answer(
            f"<b>Учебный план: {names[usage['plan']]}</b>\n\n"
            f"Ваших активных задач: {usage['active_count']}. Лимит: {limit}.\n"
            "Учитываются созданные вами личные и групповые задания. Общее задание "
            "занимает слот, пока нужно хотя бы одному участнику. Входящие задания "
            "других старост квоту не расходуют.\n\n"
            "Бесплатный — 15 активных задач. Студент и Группа — без лимита. "
            "Это учебный режим: переключение бесплатно, оплаты нет. "
            "На всех планах два напоминания: за 3 и за 1 день. "
            "Права старосты определяются владением группой.",
            reply_markup=inline([
                [("Бесплатный", "plan:free"), ("Студент", "plan:student"), ("Группа", "plan:group")],
            ]),
        )

    @router.message(CommandStart())
    async def start(message: Message, state: FSMContext, command: CommandObject):
        await state.clear()
        await message.answer(
            "Привет! Я помогу собрать учебные дедлайны в одном списке. "
            f"Часовой пояс: <b>{escape(settings.timezone)}</b>.\n\n"
            "Перешлите текст с заданием — я предложу карточку для проверки. /help — все команды.",
            reply_markup=main_keyboard(),
        )
        if not database.get_user(message.from_user.id)["delivery_enabled"]:
            await message.answer("Уведомления выключены. Их можно включить в /settings.")
        if command.args and command.args.startswith("g_"):
            await join(message, message.from_user.id, command.args, start_payload=True)

    @router.message(Command("help"))
    @router.message(F.text == "❓ Помощь")
    async def help_message(message: Message):
        await message.answer(HELP, reply_markup=main_keyboard())

    @router.message(Command("cancel"))
    async def cancel(message: Message, state: FSMContext):
        await state.clear()
        await message.answer("Ввод отменён. Черновик не сохранён.", reply_markup=main_keyboard())

    @router.message(Command("deadlines"))
    @router.message(F.text == "📋 Мои дедлайны")
    async def deadlines(message: Message):
        await show_deadlines(message, message.from_user.id)

    @router.message(Command("history"))
    async def history(message: Message):
        await show_deadlines(message, message.from_user.id, history=True)

    @router.message(Command("settings"))
    async def settings_message(message: Message):
        await show_settings(message, message.from_user.id)

    @router.message(Command("plan"))
    async def plan_message(message: Message, command: CommandObject):
        if command.args:
            try:
                database.set_plan(message.from_user.id, command.args.strip().lower())
            except ValueError:
                await message.answer("План не найден. Доступны: /plan free, /plan student, /plan group.")
                return
        await show_plan(message, message.from_user.id)

    @router.callback_query(F.data.startswith("plan:"))
    async def plan_choice(query: CallbackQuery):
        try:
            database.set_plan(query.from_user.id, query.data.split(":", 1)[1])
        except ValueError:
            await answer_callback(query, "План не найден", show_alert=True)
            return
        await answer_callback(query, "Учебный план сохранён")
        await clear_buttons(query.message)
        await show_plan(query.message, query.from_user.id)

    @router.callback_query(F.data.in_({"settings:delivery:0", "settings:delivery:1"}))
    async def delivery_setting(query: CallbackQuery):
        database.set_delivery_enabled(query.from_user.id, query.data.endswith(":1"))
        await answer_callback(query, "Настройка сохранена")
        await clear_buttons(query.message)
        await show_settings(query.message, query.from_user.id)

    @router.message(Command("add"))
    @router.message(F.text == "➕ Добавить вручную")
    async def add(message: Message, state: FSMContext):
        await state.set_data({"draft": {"creation_key": uuid4().hex, "group_id": None,
                                        "parsing_method": "manual", "status": "ok"}, "manual_new": True})
        await state.set_state(Flow.subject)
        await message.answer("Введите предмет (до 120 символов). /cancel — отмена.")

    @router.message(Command("group"))
    @router.message(F.text == "👥 Группа")
    async def groups(message: Message):
        await show_groups(message, message.from_user.id)

    @router.message(Command("newgroup"))
    async def new_group(message: Message, state: FSMContext, command: CommandObject):
        await state.clear()
        if not command.args:
            await state.set_state(Flow.group_name)
            await message.answer("Введите название учебной группы (до 120 символов). /cancel — отмена.")
            return
        try:
            name = clean_field(sanitize_text(command.args), "Название группы", 120)
            group = database.create_group(message.from_user.id, name)
        except ValueError as exc:
            await message.answer(escape(str(exc)))
            return
        await invitation(message, group)

    @router.message(Flow.group_name, F.text, ~F.text.startswith("/"))
    async def group_name(message: Message, state: FSMContext):
        try:
            name = clean_field(sanitize_text(message.text), "Название группы", 120)
            group = database.create_group(message.from_user.id, name)
        except ValueError as exc:
            await message.answer(escape(str(exc)))
            return
        await state.clear()
        await invitation(message, group)

    @router.message(Command("join"))
    async def join_command(message: Message, state: FSMContext, command: CommandObject):
        await state.clear()
        if command.args:
            await join(message, message.from_user.id, command.args)
        else:
            await state.set_state(Flow.join_code)
            await message.answer("Введите код или ссылку приглашения старосты. /cancel — отмена.")

    @router.message(Flow.join_code, F.text, ~F.text.startswith("/"))
    async def join_code(message: Message, state: FSMContext):
        if await join(message, message.from_user.id, message.text):
            await state.clear()

    @router.message(Command("publish"))
    async def publish_command(message: Message):
        await show_groups(message, message.from_user.id, owned=True)

    @router.message(Command("delete_me"))
    async def delete_me(message: Message, state: FSMContext):
        token = uuid4().hex[:12]
        await state.set_data({"delete_token": token})
        await state.set_state(Flow.delete_user)
        await message.answer(
            "Удалить все ваши данные? Личные задания, отметки выполнения и созданные вами группы "
            "с их заданиями будут удалены, в том числе у участников. Это действие необратимо.",
            reply_markup=inline([[("Удалить мои данные", f"account:delete:{token}"),
                                  ("Отмена", f"account:cancel:{token}")]]),
        )

    @router.message(Flow.subject, F.text, ~F.text.startswith("/"))
    @router.message(Flow.task, F.text, ~F.text.startswith("/"))
    @router.message(Flow.date, F.text, ~F.text.startswith("/"))
    async def draft_field(message: Message, state: FSMContext):
        data = await state.get_data()
        draft = data.get("draft")
        if not draft:
            await state.clear()
            await message.answer("Черновик утрачен. Начните заново: /add")
            return
        current = await state.get_state()
        try:
            if current == Flow.date.state:
                draft["due_date"] = parse_manual_date(message.text, now=settings.now())
                draft["status"] = "ok"
            else:
                field = "subject" if current == Flow.subject.state else "task"
                maximum = 120 if field == "subject" else 500
                draft[field] = clean_field(sanitize_text(message.text), "Предмет" if field == "subject" else "Задание", maximum)
        except ValueError as exc:
            await message.answer(escape(str(exc)))
            return
        await state.update_data(draft=draft)
        if data.get("manual_new") and current == Flow.subject.state:
            await state.set_state(Flow.task)
            await message.answer("Введите задание (до 500 символов).")
        elif data.get("manual_new") and current == Flow.task.state:
            await state.set_state(Flow.date)
            await message.answer("Введите срок: ДД.ММ.ГГГГ ЧЧ:ММ. Без времени будет 23:59.")
        else:
            await preview(message, state, draft)

    @router.callback_query(F.data.startswith("draft:"))
    async def draft_action(query: CallbackQuery, state: FSMContext):
        _, action, token = query.data.split(":", 2)
        data = await state.get_data()
        if data.get("saved_token") == token and action == "save":
            await answer_callback(query, "Уже сохранено. Задание доступно в /deadlines.")
            return
        if token != data.get("preview_token") or await state.get_state() != Flow.preview.state:
            await answer_callback(query, "Эта карточка устарела. Откройте текущую или начните заново.", show_alert=True)
            return
        draft = data["draft"]
        if action == "save":
            try:
                item = service.confirm(query.from_user.id, draft)
            except ValueError as exc:
                await answer_callback(query, str(exc)[:190], show_alert=True)
                return
            await state.set_state(None)
            await state.set_data({"saved_token": token, "saved_id": item["id"]})
            if item["group_id"] is not None and on_publication is not None:
                await on_publication(item["id"])
            await answer_callback(query, "Сохранено")
            await clear_buttons(query.message)
            await query.message.answer("✅ Дедлайн сохранён. /deadlines — список заданий.")
            if item["group_id"] is not None and on_publication is None:
                await publications.tick(lambda row: send_group_publication(query.bot, row), deadline_id=item["id"])
        elif action == "cancel":
            await state.clear()
            await answer_callback(query, "Черновик отменён")
            await clear_buttons(query.message)
            await query.message.answer("Черновик не сохранён.")
        elif action in {"subject", "task", "date"}:
            await state.set_state(getattr(Flow, action))
            await answer_callback(query)
            await clear_buttons(query.message)
            prompts = {"subject": "Введите предмет (до 120 символов).", "task": "Введите задание (до 500 символов).",
                       "date": "Введите срок: ДД.ММ.ГГГГ ЧЧ:ММ. Без времени — 23:59."}
            await query.message.answer(prompts[action] + " /cancel — отмена.")

    @router.callback_query(F.data.startswith("list:"))
    async def list_page(query: CallbackQuery):
        try:
            _, history_value, page = query.data.split(":")
            await answer_callback(query)
            await show_deadlines(query.message, query.from_user.id, int(page), history_value == "1")
        except ValueError:
            await answer_callback(query, "Некорректная страница")

    @router.callback_query(F.data.startswith("done:"))
    async def done(query: CallbackQuery):
        try:
            deadline_id = parse_callback_id(query.data.split(":")[1])
        except ValueError:
            await answer_callback(query, "Некорректный номер")
            return
        if not database.complete_deadline(query.from_user.id, deadline_id):
            await answer_callback(query, "Задание больше недоступно", show_alert=True)
            return
        await answer_callback(query, "Отмечено: вы сдали. Напоминания отключены.")
        await clear_buttons(query.message)
        await show_deadlines(query.message, query.from_user.id)

    @router.callback_query(F.data.startswith("remove:"))
    async def remove(query: CallbackQuery):
        try:
            _, action, value = query.data.split(":")
            deadline_id = parse_callback_id(value)
        except ValueError:
            await answer_callback(query, "Некорректный номер")
            return
        item = database.get_deadline(query.from_user.id, deadline_id)
        if not item or not item["can_delete"]:
            await answer_callback(query, "Удалять может только создатель задания", show_alert=True)
            return
        if action == "ask":
            suffix = " Удалится у всех участников группы." if item["group_id"] else ""
            await answer_callback(query)
            await query.message.answer(
                f"Удалить задание?{suffix}\n\n📚 {escape(item['subject'])}\n📝 {escape(item['task'])}",
                reply_markup=inline([
                    [("Удалить", f"remove:yes:{deadline_id}"), ("Отмена", "dismiss")],
                ]))
        elif action == "yes":
            database.delete_deadline(query.from_user.id, deadline_id)
            await answer_callback(query, "Удалено")
            await clear_buttons(query.message)
            await show_deadlines(query.message, query.from_user.id)

    @router.callback_query(F.data.startswith("groups:"))
    async def groups_page(query: CallbackQuery):
        try:
            _, owned, page = query.data.split(":")
            page = int(page)
        except ValueError:
            await answer_callback(query, "Некорректная страница")
            return
        await answer_callback(query)
        await show_groups(query.message, query.from_user.id, page, owned == "1")

    @router.callback_query(F.data.in_({"group:new", "group:join"}))
    async def group_input(query: CallbackQuery, state: FSMContext):
        await state.clear()
        await answer_callback(query)
        if query.data == "group:new":
            await state.set_state(Flow.group_name)
            await query.message.answer("Введите название учебной группы (до 120 символов). /cancel — отмена.")
        else:
            await state.set_state(Flow.join_code)
            await query.message.answer("Введите код или ссылку приглашения. /cancel — отмена.")

    @router.callback_query(F.data.startswith("invite:"))
    @router.callback_query(F.data.startswith("publish:"))
    async def group_owner_action(query: CallbackQuery, state: FSMContext):
        try:
            action, value = query.data.split(":")
            group = database.get_group_for_owner(query.from_user.id, parse_callback_id(value))
        except ValueError:
            group = None
        if not group:
            await answer_callback(query, "Действие доступно только старосте этой группы", show_alert=True)
            return
        await answer_callback(query)
        if action == "invite":
            await invitation(query.message, group)
        else:
            await state.set_data({"publish_group_id": group["id"]})
            await state.set_state(Flow.publish_text)
            await query.message.answer(f"Перешлите или введите задание для <b>{escape(group['name'])}</b>. "
                                       "Его увидят все участники после вашего подтверждения. /cancel — отмена.")

    @router.callback_query(F.data.startswith("leave:"))
    async def leave(query: CallbackQuery):
        try:
            _, action, value = query.data.split(":")
            group_id = parse_callback_id(value)
            group = next((g for g in database.list_groups(query.from_user.id) if g["id"] == group_id), None)
            if not group or group["is_owner"]:
                raise ValueError
        except ValueError:
            await answer_callback(query, "Эта группа недоступна или вы её староста", show_alert=True)
            return
        await answer_callback(query)
        if action == "ask":
            await query.message.answer(f"Выйти из <b>{escape(group['name'])}</b>? Её задания исчезнут из вашего списка.",
                                       reply_markup=inline([[("Выйти", f"leave:yes:{group_id}"), ("Отмена", "dismiss")]]))
        elif action == "yes":
            database.leave_group(query.from_user.id, group_id)
            await clear_buttons(query.message)
            await show_groups(query.message, query.from_user.id)

    @router.callback_query(F.data.startswith("account:"))
    async def account(query: CallbackQuery, state: FSMContext):
        _, action, token = query.data.split(":", 2)
        data = await state.get_data()
        if token != data.get("delete_token") or await state.get_state() != Flow.delete_user.state:
            await answer_callback(query, "Подтверждение устарело. Введите /delete_me заново.", show_alert=True)
            return
        if action == "delete":
            database.delete_user(query.from_user.id)
            await answer_callback(query, "Данные удалены")
            await query.message.answer("Ваши данные удалены. /start — начать заново.")
        else:
            await answer_callback(query, "Удаление отменено")
        await state.clear()
        await clear_buttons(query.message)

    @router.callback_query(F.data == "dismiss")
    async def dismiss(query: CallbackQuery):
        await answer_callback(query, "Отменено")
        await clear_buttons(query.message)

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message):
        await message.answer("Неизвестная команда. /help — список команд, /cancel — отменить ввод.")

    @router.message(F.text)
    async def forwarded_text(message: Message, state: FSMContext):
        data = await state.get_data()
        group_id = data.get("publish_group_id") if await state.get_state() == Flow.publish_text.state else None
        try:
            draft = await service.propose(message.from_user.id, message.text, group_id=group_id)
        except ValueError as exc:
            await message.answer(escape(str(exc)))
            return
        if draft["status"] == "manual_input_required":
            await state.set_data({"draft": draft})
            await state.set_state(Flow.date)
            await message.answer(
                f"Не удалось однозначно определить срок.\n📚 {escape(draft['subject'])}\n📝 {escape(draft['task'])}\n\n"
                "Введите дату: ДД.ММ.ГГГГ ЧЧ:ММ (без времени — 23:59). "
                "Расписание пар мне неизвестно. /cancel — отмена."
            )
        else:
            await preview(message, state, draft)

    @router.message()
    async def unsupported(message: Message):
        await message.answer("Пришлите текст сообщения. Фото, голосовые сообщения и файлы пока не распознаются. "
                             "Можно добавить задание вручную: /add")

    @router.callback_query()
    async def stale_callback(query: CallbackQuery):
        await answer_callback(query, "Кнопка устарела. Откройте /deadlines или /group заново.")

    @router.error()
    async def unexpected_error(event: ErrorEvent):
        logger.error("Update failed: %s", type(event.exception).__name__)
        update = event.update
        if update.callback_query:
            await answer_callback(update.callback_query, "Не удалось выполнить действие. Попробуйте снова.", show_alert=True)
        elif update.message:
            await update.message.answer("Не удалось выполнить действие. Данные в БД сохранены; попробуйте снова. /cancel — сброс ввода.")
        return True

    return router
