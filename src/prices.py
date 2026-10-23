from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

OPERATION_READ = "read"
OPERATION_CHUNK = "chunk"
OPERATION_EMBED = "embed"
OPERATION_RERANK = "rerank"
OPERATION_SEARCH = "search"

BILLABLE_OPERATIONS: tuple[str, ...] = (
    OPERATION_READ,
    OPERATION_CHUNK,
    OPERATION_EMBED,
    OPERATION_RERANK,
    OPERATION_SEARCH,
)

UNIT_PAGE = "page"
UNIT_TOKEN = "token"
UNIT_REQUEST = "request"

BILLABLE_UNITS: tuple[str, ...] = (UNIT_PAGE, UNIT_TOKEN, UNIT_REQUEST)

UNITS_BY_OPERATION: dict[str, str] = {
    OPERATION_READ: UNIT_PAGE,
    OPERATION_CHUNK: UNIT_PAGE,
    OPERATION_EMBED: UNIT_TOKEN,
    OPERATION_RERANK: UNIT_REQUEST,
    OPERATION_SEARCH: UNIT_REQUEST,
}

PROVIDER_RANK = 2
MODEL_RANK = 1

DEFAULT_CURRENCY = "USD"

KEY_SHAPE = "operation/provider/model/unit"


@dataclass(frozen=True, slots=True)
class Price:
    operation: str
    provider: str | None
    model: str | None
    unit: str
    amount: Decimal
    currency: str = DEFAULT_CURRENCY

    def cost_of(self, units: Decimal | int | float | str) -> Decimal:
        return self.amount * Decimal(str(units))


class UnknownPrice(Exception):
    pass


def price_key(operation: str, provider: str, model: str, unit: str) -> str:
    return f"{operation}/{provider}/{model}/{unit}"


def parse_price_key(key: str) -> tuple[str, str, str, str]:
    parts = key.split("/")
    if len(parts) < 4:
        raise ValueError(
            f"price key {key!r} is not '{KEY_SHAPE}'; "
            f"a model of '{ANY_MODEL}' matches any model"
        )
    operation, provider, *model_parts, unit = parts
    return operation, provider, "/".join(model_parts), unit


class PriceBook:
    __slots__ = ("_entries",)

    def __init__(self, entries=None) -> None:
        parsed: list[Price] = []
        seen: set[tuple[str, str | None, str | None, str]] = set()
        for position, entry in enumerate(entries or ()):
            price = _price_of(entry, position)
            shape = (price.operation, price.provider, price.model, price.unit)
            if shape in seen:
                raise ValueError(
                    f"two price entries name the same operation, provider, model and unit "
                    f"({price.operation!r}, {price.provider!r}, {price.model!r}, "
                    f"{price.unit!r}); a file saying two things about one price is a file "
                    f"whose answer depends on which line was read first"
                )
            seen.add(shape)
            parsed.append(price)
        self._entries = tuple(parsed)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[Price, ...]:
        return self._entries

    def price_for(self, *, operation: str, provider: str, model: str, unit: str) -> Price:
        price = self.price_or_none(operation=operation, provider=provider, model=model, unit=unit)
        if price is None:
            raise UnknownPrice(
                f"no price configured for operation {operation!r} at provider {provider!r} "
                f"with model {model!r} per {unit!r}; add an entry to usage.prices naming "
                "at least that operation and unit — there is no built-in price, because a "
                "stale one is worse than none"
            )
        return price

    def price_or_none(
        self, *, operation: str, provider: str, model: str, unit: str
    ) -> Price | None:
        best: Price | None = None
        best_rank = -1
        for entry in self._entries:
            if entry.operation != operation or entry.unit != unit:
                continue
            if entry.provider is not None and entry.provider != provider:
                continue
            if entry.model is not None and entry.model != model:
                continue
            rank = (PROVIDER_RANK if entry.provider is not None else 0) + (
                MODEL_RANK if entry.model is not None else 0
            )
            if rank > best_rank:
                best, best_rank = entry, rank
        return best


def _price_of(entry, position: int) -> Price:
    def named(field: str):
        if isinstance(entry, Mapping):
            return entry.get(field)
        return getattr(entry, field, None)

    operation, unit = named("operation"), named("unit")
    where = f"price entry {position}"
    if operation not in BILLABLE_OPERATIONS:
        raise ValueError(
            f"{where} names operation {operation!r}, which is not billable; the billable "
            f"operations are {', '.join(BILLABLE_OPERATIONS)}"
        )
    if unit not in BILLABLE_UNITS:
        raise ValueError(
            f"{where} names unit {unit!r}, which is not billable; the units are "
            f"{', '.join(BILLABLE_UNITS)}"
        )
    expected = UNITS_BY_OPERATION[operation]
    if unit != expected:
        raise ValueError(
            f"{where} prices {operation!r} per {unit!r}, and {operation!r} is billed per "
            f"{expected!r}. Both words are real, which is what makes this worth refusing: "
            f"the unit is decided where the operation is recorded, so a price naming the "
            f"other one matches nothing and says nothing"
        )
    value = Decimal(str(named("amount")))
    if value < 0:
        raise ValueError(f"{where} carries a negative amount, {value}")
    return Price(
        operation=operation,
        provider=named("provider") or None,
        model=named("model") or None,
        unit=unit,
        amount=value,
    )
