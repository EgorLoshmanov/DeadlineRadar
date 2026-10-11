"""SQLite persistence, authorization, and durable reminder delivery state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Callable, Iterator

from models import MAX_SUBJECT_LENGTH, MAX_TASK_LENGTH
from plans import DEFAULT_PLAN, PLAN_LIMITS, PlanLimitExceeded, validate_plan


REMINDER_LEASE_SECONDS = 300


def _naive(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError("Expected a local datetime without a timezone")
    return value


def _stamp(value: datetime) -> str:
    return _naive(value).strftime("%Y-%m-%d %H:%M:%S.%f")


def _due(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return _naive(value).strftime("%Y-%m-%d %H:%M")
    if not isinstance(value, str):
        raise ValueError("Expected due_date in YYYY-MM-DD HH:MM format")
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError("Expected due_date in YYYY-MM-DD HH:MM format") from exc


def _text(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{name} must contain between 1 and {maximum} characters")
    return value.strip()


class Database:
    """One connection per instance; writes use short, serialized transactions.

    Deadline dates use the application's local wall clock. Group completion is
    stored for each member separately. Only reservations before the durable
    send gate can be retried after a crash; ambiguous sends consume their window.
    """

    def __init__(self, path: str | Path = "deadlines.db", clock: Callable[[], datetime] = datetime.now) -> None:
        self.path = str(path)
        self.clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=5, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self.initialize()

    def initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    delivery_enabled INTEGER NOT NULL DEFAULT 1
                        CHECK (delivery_enabled IN (0, 1)),
                    plan TEXT NOT NULL DEFAULT 'free'
                        CHECK (plan IN ('free', 'student', 'group')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
                    invite_code TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_members (
                    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    joined_at TEXT NOT NULL,
                    PRIMARY KEY (group_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS deadlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creator_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    group_id INTEGER REFERENCES groups(id) ON DELETE CASCADE,
                    subject TEXT NOT NULL CHECK (length(subject) BETWEEN 1 AND 120),
                    task TEXT NOT NULL CHECK (length(task) BETWEEN 1 AND 500),
                    due_date TEXT NOT NULL CHECK (length(due_date) = 16),
                    created_at TEXT NOT NULL,
                    idempotency_key TEXT,
                    UNIQUE (creator_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS deadline_completions (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    completed_at TEXT,
                    PRIMARY KEY (deadline_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS reminder_deliveries (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    days_before INTEGER NOT NULL CHECK (days_before IN (3, 1)),
                    status TEXT NOT NULL CHECK (status IN ('claimed', 'retry', 'sent')),
                    lease_until TEXT,
                    next_attempt_at TEXT,
                    sent_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    claim_token TEXT,
                    PRIMARY KEY (deadline_id, user_id, days_before)
                );
                CREATE TABLE IF NOT EXISTS reminder_attempts (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    days_before INTEGER NOT NULL CHECK (days_before IN (3, 1)),
                    claim_token TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    result TEXT NOT NULL CHECK (result IN ('sending', 'sent', 'uncertain')),
                    PRIMARY KEY (deadline_id, user_id, days_before)
                );
                CREATE TABLE IF NOT EXISTS group_publications (
                    deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'sending', 'sent', 'uncertain', 'disabled')),
                    retry_at TEXT,
                    claim_token TEXT,
                    attempted_at TEXT,
                    finished_at TEXT,
                    PRIMARY KEY (deadline_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_publications_pending ON group_publications(status, retry_at);
                CREATE INDEX IF NOT EXISTS idx_members_user ON group_members(user_id);
                CREATE INDEX IF NOT EXISTS idx_deadlines_due ON deadlines(due_date);
                CREATE INDEX IF NOT EXISTS idx_deadlines_creator ON deadlines(creator_id);
                CREATE INDEX IF NOT EXISTS idx_deadlines_group ON deadlines(group_id);
                CREATE INDEX IF NOT EXISTS idx_completions_date ON deadline_completions(completed_at);
                """
            )
            completion_columns = self._connection.execute("PRAGMA table_info(deadline_completions)").fetchall()
            if any(column["name"] == "completed_at" and column["notnull"] for column in completion_columns):
                # Archived group completion needs only the done flag, not an old history date.
                # Rebuild the previous NOT NULL table atomically, preserving rows and indexes.
                with self._transaction() as connection:
                    indexes = connection.execute(
                        "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'deadline_completions' AND sql IS NOT NULL"
                    ).fetchall()
                    connection.execute("ALTER TABLE deadline_completions RENAME TO deadline_completions_legacy")
                    connection.execute(
                        """CREATE TABLE deadline_completions (
                            deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
                            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            completed_at TEXT,
                            PRIMARY KEY (deadline_id, user_id)
                        )"""
                    )
                    connection.execute(
                        """INSERT INTO deadline_completions(deadline_id, user_id, completed_at)
                           SELECT deadline_id, user_id, completed_at FROM deadline_completions_legacy"""
                    )
                    connection.execute("DROP TABLE deadline_completions_legacy")
                    for index in indexes:
                        connection.execute(index["sql"])
            with self._transaction() as connection:
                user_columns = connection.execute("PRAGMA table_info(users)").fetchall()
                if not any(column["name"] == "plan" for column in user_columns):
                    connection.execute(
                        """ALTER TABLE users ADD COLUMN plan TEXT NOT NULL DEFAULT 'free'
                           CHECK (plan IN ('free', 'student', 'group'))"""
                    )
                delivery_columns = connection.execute("PRAGMA table_info(reminder_deliveries)").fetchall()
                if not any(column["name"] == "claim_token" for column in delivery_columns):
                    connection.execute("ALTER TABLE reminder_deliveries ADD COLUMN claim_token TEXT")
                    connection.execute("UPDATE reminder_deliveries SET claim_token = lower(hex(randomblob(16)))")
                    # Old retry/claimed rows may already have reached Telegram without an ACK.
                    # Conservatively consume these windows instead of replaying a possible send.
                    connection.execute(
                        """INSERT INTO reminder_attempts
                           (deadline_id, user_id, days_before, claim_token, attempted_at, result)
                           SELECT deadline_id, user_id, days_before, claim_token,
                                  COALESCE(sent_at, ?),
                                  CASE WHEN status = 'sent' THEN 'sent' ELSE 'uncertain' END
                           FROM reminder_deliveries""",
                        (_stamp(self.clock()),),
                    )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_user(self, connection: sqlite3.Connection, user_id: int) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO users(id, created_at) VALUES (?, ?)",
            (user_id, _stamp(self.clock())),
        )

    def upsert_user(self, user_id: int, username: str | None = None) -> dict:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO users(id, username, created_at) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   username = COALESCE(excluded.username, users.username)""",
                (user_id, username, _stamp(self.clock())),
            )
        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: int) -> dict | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        user = dict(row)
        user["user_id"] = user["id"]
        user["delivery_enabled"] = bool(user["delivery_enabled"])
        return user

    def set_delivery_enabled(self, user_id: int, enabled: bool) -> None:
        with self._transaction() as connection:
            self._ensure_user(connection, user_id)
            connection.execute("UPDATE users SET delivery_enabled = ? WHERE id = ?", (bool(enabled), user_id))

    @staticmethod
    def _plan_usage(connection: sqlite3.Connection, user_id: int) -> dict:
        # Only tasks created by this user consume their allowance. A group task
        # stays active until every current member has a completion flag.
        row = connection.execute(
            """SELECT COALESCE((SELECT plan FROM users WHERE id = :user_id), :default_plan) AS plan,
                      (SELECT COUNT(*) FROM deadlines d WHERE d.creator_id = :user_id AND (
                          (d.group_id IS NULL AND NOT EXISTS (
                              SELECT 1 FROM deadline_completions c
                              WHERE c.deadline_id = d.id AND c.user_id = d.creator_id))
                          OR (d.group_id IS NOT NULL AND EXISTS (
                              SELECT 1 FROM group_members gm WHERE gm.group_id = d.group_id
                                AND NOT EXISTS (SELECT 1 FROM deadline_completions c
                                    WHERE c.deadline_id = d.id AND c.user_id = gm.user_id))))) AS active_count""",
            {"user_id": user_id, "default_plan": DEFAULT_PLAN},
        ).fetchone()
        plan, active_count = row["plan"], row["active_count"]
        limit = PLAN_LIMITS[plan]
        return {"plan": plan, "active_count": active_count, "limit": limit,
                "remaining": None if limit is None else max(0, limit - active_count)}

    def get_plan_usage(self, user_id: int) -> dict:
        """Read the local plan and a consistent usage snapshot without creating a user."""
        with self._lock:
            return self._plan_usage(self._connection, user_id)

    def set_plan(self, user_id: int, plan: str) -> dict:
        """Switch a simulated plan without deleting tasks or changing delivery preferences."""
        plan = validate_plan(plan)
        with self._transaction() as connection:
            self._ensure_user(connection, user_id)
            connection.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
            return self._plan_usage(connection, user_id)

    def disable_delivery_for_reminder(
        self, deadline_id: int, user_id: int, days_before: int, *, claim_token: str,
    ) -> bool:
        """Apply a transport refusal only while its original reservation still exists."""
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE users SET delivery_enabled = 0 WHERE id = ?
                   AND EXISTS (SELECT 1 FROM reminder_deliveries
                       WHERE deadline_id = ? AND user_id = ? AND days_before = ?
                         AND status = 'claimed' AND claim_token = ?)""",
                (user_id, deadline_id, user_id, days_before, claim_token),
            ).rowcount > 0

    @staticmethod
    def _group(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["group_id"] = result["id"]
        result["group_name"] = result["name"]
        if "is_owner" in result:
            result["is_owner"] = bool(result["is_owner"])
        return result

    def create_group(self, owner_id: int, name: str) -> dict:
        name = _text(name, "Group name", 120)
        with self._transaction() as connection:
            self._ensure_user(connection, owner_id)
            # Database uniqueness also protects against the unlikely token collision.
            for _ in range(5):
                invite_code = secrets.token_urlsafe(9)
                try:
                    cursor = connection.execute(
                        "INSERT INTO groups(owner_id, name, invite_code, created_at) VALUES (?, ?, ?, ?)",
                        (owner_id, name, invite_code, _stamp(self.clock())),
                    )
                    break
                except sqlite3.IntegrityError:
                    if not connection.execute("SELECT 1 FROM groups WHERE invite_code = ?", (invite_code,)).fetchone():
                        raise
            else:
                raise RuntimeError("Could not generate a unique invite code")
            group_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO group_members(group_id, user_id, joined_at) VALUES (?, ?, ?)",
                (group_id, owner_id, _stamp(self.clock())),
            )
            row = connection.execute("SELECT *, 1 AS is_owner FROM groups WHERE id = ?", (group_id,)).fetchone()
            return self._group(row)

    def join_group(self, user_id: int, invite_code: str) -> dict:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM groups WHERE invite_code = ?", (invite_code.strip(),)).fetchone()
            if row is None:
                raise ValueError("Invalid invite code")
            self._ensure_user(connection, user_id)
            connection.execute(
                "INSERT OR IGNORE INTO group_members(group_id, user_id, joined_at) VALUES (?, ?, ?)",
                (row["id"], user_id, _stamp(self.clock())),
            )
            result = self._group(row)
            result["is_owner"] = row["owner_id"] == user_id
            return result

    def list_groups(self, user_id: int) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT g.*, g.owner_id = ? AS is_owner
                   FROM groups g JOIN group_members gm ON gm.group_id = g.id
                   WHERE gm.user_id = ? ORDER BY g.name, g.id""",
                (user_id, user_id),
            ).fetchall()
        return [self._group(row) for row in rows]

    def leave_group(self, user_id: int, group_id: int) -> None:
        with self._transaction() as connection:
            group = connection.execute("SELECT owner_id FROM groups WHERE id = ?", (group_id,)).fetchone()
            if group and group["owner_id"] == user_id:
                raise ValueError("The owner cannot leave their group")
            connection.execute("DELETE FROM group_members WHERE group_id = ? AND user_id = ?", (group_id, user_id))
            # Keep sent deliveries: leaving and rejoining must not repeat a reminder.

    def get_group_for_owner(self, owner_id: int, group_id: int) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT *, 1 AS is_owner FROM groups WHERE id = ? AND owner_id = ?", (group_id, owner_id)
            ).fetchone()
        return self._group(row) if row else None

    def create_deadline(
        self, user_id: int, subject: str, task: str, due_date: str | datetime,
        group_id: int | None = None, idempotency_key: str | None = None,
    ) -> dict:
        subject = _text(subject, "Subject", MAX_SUBJECT_LENGTH)
        task = _text(task, "Task", MAX_TASK_LENGTH)
        due_date = _due(due_date)
        if idempotency_key is not None:
            idempotency_key = _text(idempotency_key, "Idempotency key", 256)
        with self._transaction() as connection:
            if group_id is not None:
                owner = connection.execute("SELECT 1 FROM groups WHERE id = ? AND owner_id = ?", (group_id, user_id)).fetchone()
                if owner is None:
                    raise ValueError("Only a group owner may add group deadlines")
            self._ensure_user(connection, user_id)
            existing = None
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM deadlines WHERE creator_id = ? AND idempotency_key = ?", (user_id, idempotency_key)
                ).fetchone()
            if existing:
                if (existing["subject"], existing["task"], existing["due_date"], existing["group_id"]) != (subject, task, due_date, group_id):
                    raise ValueError("Idempotency key was already used for a different deadline")
                deadline_id = existing["id"]
            else:
                usage = self._plan_usage(connection, user_id)
                if usage["limit"] is not None and usage["active_count"] >= usage["limit"]:
                    raise PlanLimitExceeded(usage["plan"], usage["active_count"], usage["limit"])
                cursor = connection.execute(
                    """INSERT INTO deadlines(creator_id, group_id, subject, task, due_date, created_at, idempotency_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, group_id, subject, task, due_date, _stamp(self.clock()), idempotency_key),
                )
                deadline_id = cursor.lastrowid
                if group_id is not None:
                    # Capture the audience atomically with creation. A repeated
                    # confirmation or a later member must not create new sends.
                    connection.execute(
                        """INSERT INTO group_publications(deadline_id, user_id)
                           SELECT ?, user_id FROM group_members
                           WHERE group_id = ? AND user_id != ?""",
                        (deadline_id, group_id, user_id),
                    )
            row = self._deadline_rows(connection, user_id, deadline_id=deadline_id)[0]
            return self._deadline(row)

    @staticmethod
    def _deadline_rows(
        connection: sqlite3.Connection, user_id: int,
        *, include_completed: bool = True, deadline_id: int | None = None,
    ) -> list[sqlite3.Row]:
        sql = """SELECT d.*, g.name AS group_name, c.completed_at,
                        CASE WHEN c.deadline_id IS NULL THEN 'pending' ELSE 'completed' END AS status,
                        d.creator_id = :user_id OR g.owner_id = :user_id AS can_delete
                 FROM deadlines d
                 LEFT JOIN groups g ON g.id = d.group_id
                 LEFT JOIN deadline_completions c ON c.deadline_id = d.id AND c.user_id = :user_id
                 WHERE ((d.group_id IS NULL AND d.creator_id = :user_id)
                   OR EXISTS (SELECT 1 FROM group_members gm WHERE gm.group_id = d.group_id AND gm.user_id = :user_id))"""
        if not include_completed:
            sql += " AND c.deadline_id IS NULL"
        if deadline_id is not None:
            sql += " AND d.id = :deadline_id"
        sql += " ORDER BY d.due_date, d.id"
        return connection.execute(sql, {"user_id": user_id, "deadline_id": deadline_id}).fetchall()

    @staticmethod
    def _deadline(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["user_id"] = result["creator_id"]
        result["completed"] = result["is_completed"] = result["status"] == "completed"
        result["can_delete"] = bool(result["can_delete"])
        return result

    def list_deadlines(self, user_id: int, include_completed: bool = False) -> list[dict]:
        with self._lock:
            rows = self._deadline_rows(self._connection, user_id, include_completed=include_completed)
        return [self._deadline(row) for row in rows]

    def get_deadline(self, user_id: int, deadline_id: int) -> dict | None:
        with self._lock:
            rows = self._deadline_rows(self._connection, user_id, deadline_id=deadline_id)
        return self._deadline(rows[0]) if rows else None

    def complete_deadline(self, user_id: int, deadline_id: int) -> bool:
        with self._transaction() as connection:
            if not self._deadline_rows(connection, user_id, deadline_id=deadline_id):
                return False
            connection.execute(
                "INSERT OR IGNORE INTO deadline_completions(deadline_id, user_id, completed_at) VALUES (?, ?, ?)",
                (deadline_id, user_id, _stamp(self.clock())),
            )
            return True

    def delete_deadline(self, user_id: int, deadline_id: int) -> bool:
        with self._transaction() as connection:
            rows = self._deadline_rows(connection, user_id, deadline_id=deadline_id)
            if not rows or not rows[0]["can_delete"]:
                return False
            return connection.execute("DELETE FROM deadlines WHERE id = ?", (deadline_id,)).rowcount > 0

    def list_group_member_ids(self, group_id: int) -> list[int]:
        with self._lock:
            rows = self._connection.execute("SELECT user_id FROM group_members WHERE group_id = ? ORDER BY user_id", (group_id,)).fetchall()
        return [row["user_id"] for row in rows]

    @staticmethod
    def _publication_rows(
        connection: sqlite3.Connection, now: datetime, *,
        deadline_id: int | None = None, user_id: int | None = None,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """SELECT d.*, g.name AS group_name, p.user_id
               FROM group_publications p
               JOIN deadlines d ON d.id = p.deadline_id
               JOIN groups g ON g.id = d.group_id
               JOIN group_members gm ON gm.group_id = d.group_id AND gm.user_id = p.user_id
               JOIN users u ON u.id = p.user_id
               WHERE p.status = 'pending' AND (p.retry_at IS NULL OR p.retry_at <= :now)
                 AND u.delivery_enabled = 1 AND d.due_date > :due_now
                 AND NOT EXISTS (SELECT 1 FROM deadline_completions c
                                 WHERE c.deadline_id = d.id AND c.user_id = p.user_id)
                 AND (:deadline_id IS NULL OR d.id = :deadline_id)
                 AND (:user_id IS NULL OR p.user_id = :user_id)
               ORDER BY d.id, p.user_id""",
            {"now": _stamp(now), "due_now": _due(now), "deadline_id": deadline_id, "user_id": user_id},
        ).fetchall()

    def get_pending_publications(self, now: datetime, *, deadline_id: int | None = None) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._publication_rows(self._connection, now, deadline_id=deadline_id)]

    def begin_publication_send(self, deadline_id: int, user_id: int, now: datetime) -> str | None:
        """Spend the one publication attempt before any request reaches Telegram."""
        with self._transaction() as connection:
            if not self._publication_rows(connection, now, deadline_id=deadline_id, user_id=user_id):
                return None
            token = secrets.token_hex(16)
            connection.execute(
                """UPDATE group_publications SET status = 'sending', claim_token = ?, attempted_at = ?
                   WHERE deadline_id = ? AND user_id = ? AND status = 'pending'""",
                (token, _stamp(now), deadline_id, user_id),
            )
            return token

    def finish_publication(self, deadline_id: int, user_id: int, *, token: str, result: str) -> bool:
        if result not in {"sent", "uncertain", "disabled"}:
            raise ValueError("Invalid publication result")
        with self._transaction() as connection:
            updated = connection.execute(
                """UPDATE group_publications SET status = ?, finished_at = ?
                   WHERE deadline_id = ? AND user_id = ? AND status = 'sending' AND claim_token = ?""",
                (result, _stamp(self.clock()), deadline_id, user_id, token),
            ).rowcount > 0
            if updated and result == "disabled":
                connection.execute("UPDATE users SET delivery_enabled = 0 WHERE id = ?", (user_id,))
            return updated

    def retry_publication(self, deadline_id: int, user_id: int, *, token: str, retry_at: datetime) -> bool:
        """Retry only a definitive refusal without a delivered message."""
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE group_publications SET status = 'pending', retry_at = ?, claim_token = NULL
                   WHERE deadline_id = ? AND user_id = ? AND status = 'sending' AND claim_token = ?""",
                (_stamp(retry_at), deadline_id, user_id, token),
            ).rowcount > 0

    def delete_user(self, user_id: int) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def cleanup_completed_before(self, cutoff: datetime) -> int:
        """Remove old history while preserving minimal completion flags on active group tasks."""
        cutoff_string = _stamp(cutoff)
        with self._transaction() as connection:
            deleted = connection.execute(
                """DELETE FROM deadlines WHERE
                   (group_id IS NULL AND EXISTS (
                       SELECT 1 FROM deadline_completions c WHERE c.deadline_id = deadlines.id
                       AND c.user_id = deadlines.creator_id
                       AND (c.completed_at IS NULL OR c.completed_at < :cutoff)))
                   OR (group_id IS NOT NULL AND EXISTS (
                       SELECT 1 FROM group_members gm WHERE gm.group_id = deadlines.group_id)
                       AND NOT EXISTS (
                       SELECT 1 FROM group_members gm WHERE gm.group_id = deadlines.group_id
                       AND NOT EXISTS (SELECT 1 FROM deadline_completions c
                           WHERE c.deadline_id = deadlines.id AND c.user_id = gm.user_id
                           AND (c.completed_at IS NULL OR c.completed_at < :cutoff))))""",
                {"cutoff": cutoff_string},
            ).rowcount
            connection.execute(
                """UPDATE deadline_completions SET completed_at = NULL
                   WHERE completed_at < :cutoff AND deadline_id IN (
                       SELECT id FROM deadlines WHERE group_id IS NOT NULL)""",
                {"cutoff": cutoff_string},
            )
            return deleted

    @staticmethod
    def _reminder_rows(
        connection: sqlite3.Connection, now: datetime, *,
        deadline_id: int | None = None, user_id: int | None = None,
        days_before: int | None = None, ignore_claim: bool = False,
    ) -> list[sqlite3.Row]:
        sql = """WITH recipients AS (
                 SELECT id AS deadline_id, creator_id AS user_id FROM deadlines WHERE group_id IS NULL
                 UNION ALL
                 SELECT d.id, gm.user_id FROM deadlines d JOIN group_members gm ON gm.group_id = d.group_id
               )
               SELECT d.id, r.user_id, d.subject, d.task, d.due_date,
                      CASE WHEN d.due_date > :one_day THEN 3 ELSE 1 END AS days_before
               FROM recipients r JOIN deadlines d ON d.id = r.deadline_id
               JOIN users u ON u.id = r.user_id
               LEFT JOIN deadline_completions c ON c.deadline_id = d.id AND c.user_id = r.user_id
               LEFT JOIN reminder_deliveries rd ON rd.deadline_id = d.id AND rd.user_id = r.user_id
                    AND rd.days_before = CASE WHEN d.due_date > :one_day THEN 3 ELSE 1 END
               WHERE u.delivery_enabled = 1 AND c.deadline_id IS NULL
                 AND d.due_date > :now_minute AND d.due_date <= :three_days
                 AND NOT EXISTS (SELECT 1 FROM reminder_attempts ra
                     WHERE ra.deadline_id = d.id AND ra.user_id = r.user_id
                       AND ra.days_before = CASE WHEN d.due_date > :one_day THEN 3 ELSE 1 END)"""
        if not ignore_claim:
            sql += """ AND (rd.status IS NULL OR (rd.status <> 'sent'
                       AND (rd.lease_until IS NULL OR rd.lease_until <= :now)
                       AND (rd.next_attempt_at IS NULL OR rd.next_attempt_at <= :now)))"""
        parameters = {
            "now": _stamp(now), "now_minute": _due(now),
            "one_day": _due(now + timedelta(days=1)),
            "three_days": _due(now + timedelta(days=3)),
            "deadline_id": deadline_id, "user_id": user_id, "days_before": days_before,
        }
        if deadline_id is not None:
            sql += " AND d.id = :deadline_id"
        if user_id is not None:
            sql += " AND r.user_id = :user_id"
        if days_before is not None:
            sql += " AND CASE WHEN d.due_date > :one_day THEN 3 ELSE 1 END = :days_before"
        sql += " ORDER BY d.due_date, d.id, r.user_id"
        return connection.execute(sql, parameters).fetchall()

    def get_due_reminders(self, now: datetime) -> list[dict]:
        _naive(now)
        with self._lock:
            return [dict(row) for row in self._reminder_rows(self._connection, now)]

    def claim_reminder(self, deadline_id: int, user_id: int, days_before: int, now: datetime) -> str | None:
        _naive(now)
        if days_before not in (1, 3):
            raise ValueError("days_before must be 1 or 3")
        with self._transaction() as connection:
            eligible = self._reminder_rows(
                connection, now, deadline_id=deadline_id, user_id=user_id,
                days_before=days_before,
            )
            if not eligible:
                return None
            claim_token = secrets.token_hex(16)
            connection.execute(
                """INSERT INTO reminder_deliveries
                   (deadline_id, user_id, days_before, status, lease_until, attempts, claim_token)
                   VALUES (?, ?, ?, 'claimed', ?, 1, ?)
                   ON CONFLICT(deadline_id, user_id, days_before) DO UPDATE SET
                       status = 'claimed', lease_until = excluded.lease_until,
                       next_attempt_at = NULL, attempts = reminder_deliveries.attempts + 1,
                       claim_token = excluded.claim_token""",
                (deadline_id, user_id, days_before,
                 _stamp(now + timedelta(seconds=REMINDER_LEASE_SECONDS)), claim_token),
            )
            return claim_token

    def begin_reminder_send(
        self, deadline_id: int, user_id: int, days_before: int, now: datetime,
        *, claim_token: str,
    ) -> bool:
        """Commit the one permitted send attempt before any transport request."""
        with self._transaction() as connection:
            owned = connection.execute(
                """SELECT 1 FROM reminder_deliveries WHERE deadline_id = ? AND user_id = ?
                   AND days_before = ? AND status = 'claimed' AND claim_token = ? AND lease_until > ?""",
                (deadline_id, user_id, days_before, claim_token, _stamp(now)),
            ).fetchone()
            if not owned or not self._reminder_rows(
                connection, now, deadline_id=deadline_id, user_id=user_id,
                days_before=days_before, ignore_claim=True,
            ):
                return False
            connection.execute(
                """INSERT INTO reminder_attempts
                   (deadline_id, user_id, days_before, claim_token, attempted_at, result)
                   VALUES (?, ?, ?, ?, ?, 'sending')""",
                (deadline_id, user_id, days_before, claim_token, _stamp(now)),
            )
            return True

    def _finish_reminder(
        self, deadline_id: int, user_id: int, days_before: int,
        claim_token: str, result: str, now: datetime | None = None,
    ) -> bool:
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE reminder_attempts SET result = ?
                   WHERE deadline_id = ? AND user_id = ? AND days_before = ?
                     AND claim_token = ? AND result = 'sending'
                     AND EXISTS (SELECT 1 FROM reminder_deliveries rd
                         WHERE rd.deadline_id = reminder_attempts.deadline_id
                           AND rd.user_id = reminder_attempts.user_id
                           AND rd.days_before = reminder_attempts.days_before
                           AND rd.claim_token = reminder_attempts.claim_token AND rd.status = 'claimed')""",
                (result, deadline_id, user_id, days_before, claim_token),
            ).rowcount
            if not changed:
                return False
            connection.execute(
                """UPDATE reminder_deliveries SET status = ?, sent_at = ?,
                          lease_until = NULL, next_attempt_at = NULL
                   WHERE deadline_id = ? AND user_id = ? AND days_before = ? AND claim_token = ?""",
                ("sent" if result == "sent" else "claimed",
                 _stamp(now) if now is not None else None,
                 deadline_id, user_id, days_before, claim_token),
            )
            return True

    def mark_reminder_sent(
        self, deadline_id: int, user_id: int, days_before: int, now: datetime,
        *, claim_token: str,
    ) -> bool:
        return self._finish_reminder(deadline_id, user_id, days_before, claim_token, "sent", now)

    def mark_reminder_uncertain(
        self, deadline_id: int, user_id: int, days_before: int, *, claim_token: str,
    ) -> bool:
        return self._finish_reminder(deadline_id, user_id, days_before, claim_token, "uncertain")

    def release_reminder(
        self, deadline_id: int, user_id: int, days_before: int,
        retry_at: datetime | None = None, *, claim_token: str,
        known_not_sent: bool = False,
    ) -> bool:
        retry_string = _stamp(retry_at) if retry_at is not None else None
        with self._transaction() as connection:
            owned = connection.execute(
                """SELECT 1 FROM reminder_deliveries WHERE deadline_id = ? AND user_id = ?
                   AND days_before = ? AND status = 'claimed' AND claim_token = ?""",
                (deadline_id, user_id, days_before, claim_token),
            ).fetchone()
            if not owned:
                return False
            attempt = connection.execute(
                """SELECT claim_token, result FROM reminder_attempts
                   WHERE deadline_id = ? AND user_id = ? AND days_before = ?""",
                (deadline_id, user_id, days_before),
            ).fetchone()
            if attempt:
                if not known_not_sent or attempt["claim_token"] != claim_token or attempt["result"] != "sending":
                    return False
                connection.execute(
                    "DELETE FROM reminder_attempts WHERE deadline_id = ? AND user_id = ? AND days_before = ?",
                    (deadline_id, user_id, days_before),
                )
            return connection.execute(
                """UPDATE reminder_deliveries SET status = 'retry', lease_until = NULL, next_attempt_at = ?
                   WHERE deadline_id = ? AND user_id = ? AND days_before = ?
                     AND status = 'claimed' AND claim_token = ?""",
                (retry_string, deadline_id, user_id, days_before, claim_token),
            ).rowcount > 0
