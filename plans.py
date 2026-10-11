"""Local study plans; changing a plan never charges money or changes reminders."""

DEFAULT_PLAN = "free"
PLAN_LIMITS: dict[str, int | None] = {"free": 15, "student": None, "group": None}


def validate_plan(plan: str) -> str:
    if not isinstance(plan, str) or plan not in PLAN_LIMITS:
        raise ValueError("Неизвестный план. Доступны free, student и group.")
    return plan


class PlanLimitExceeded(ValueError):
    """A new task would exceed the creator's active-task allowance."""

    def __init__(self, plan: str, active_count: int, limit: int) -> None:
        self.plan = plan
        self.active_count = active_count
        self.limit = limit
        super().__init__(
            f"Достигнут лимит плана {plan}: {active_count}/{limit} активных заданий. "
            "Освободите место или переключите локальный план через /plan."
        )
