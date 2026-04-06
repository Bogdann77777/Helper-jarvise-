# board/data_gate.py — Валидация входных данных перед запуском board-сессии

from board.models import DataGateInput


class DataGateError(Exception):
    """Ошибка валидации данных."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Data Gate validation failed: {'; '.join(errors)}")


def validate_gate_input(data: dict) -> DataGateInput:
    """
    Валидирует входные данные и возвращает DataGateInput.

    Finance requirement is RELAXED when loaded_from_memory=True:
    - If company profile is in memory, financials are already known
    - User only needs to provide company_name + problem_statement
    """
    errors = []

    # Required fields
    if not data.get("company_name", "").strip():
        errors.append("company_name is required")

    problem = data.get("problem_statement", "").strip()
    if not problem:
        errors.append("problem_statement is required")
    elif len(problem) < 10:
        errors.append("problem_statement must be at least 10 characters")

    # Financial requirement: skip if company profile was loaded from memory
    loaded_from_memory = data.get("loaded_from_memory", False)
    if not loaded_from_memory:
        financial_fields = ["revenue", "expenses", "runway_months", "funding"]
        has_financial = any(
            data.get(f) not in (None, "", 0)
            for f in financial_fields
        )
        if not has_financial:
            errors.append(
                "At least one financial metric is required "
                "(revenue, expenses, runway_months, or funding) — "
                "or select a known company to auto-load profile"
            )

    if errors:
        raise DataGateError(errors)

    # Clean data
    clean = {}
    for key in DataGateInput.model_fields:
        val = data.get(key)
        if isinstance(val, str):
            val = val.strip() or None
        clean[key] = val

    return DataGateInput(**clean)
