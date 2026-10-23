import pathlib
from decimal import Decimal

import pytest
import yaml

from src.prices import (
    BILLABLE_OPERATIONS,
    BILLABLE_UNITS,
    OPERATION_CHUNK,
    OPERATION_EMBED,
    OPERATION_READ,
    OPERATION_SEARCH,
    UNIT_PAGE,
    UNIT_REQUEST,
    UNIT_TOKEN,
    PriceBook,
    UnknownPrice,
)

READER = "pretrained-layout-parser-v1.0-2024-06-03"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMITTED = REPOSITORY_ROOT / "apps" / "base" / "config.yaml"


def _entry(operation, unit, amount, *, provider=None, model=None) -> dict:
    entry = {"operation": operation, "unit": unit, "amount": amount}
    if provider is not None:
        entry["provider"] = provider
    if model is not None:
        entry["model"] = model
    return entry


def _configured() -> list[dict]:
    document = yaml.safe_load(COMMITTED.read_text()) or {}
    prices = (document.get("usage") or {}).get("prices")
    if not prices:
        raise AssertionError(f"{COMMITTED} declares no usage.prices")
    return prices


def test_a_deployment_that_configured_nothing_prices_nothing():
    book = PriceBook()

    assert len(book) == 0
    assert book.price_or_none(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    ) is None


def test_it_prices_exactly_what_it_was_configured_with():
    book = PriceBook([_entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google")])

    price = book.price_for(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    )

    assert price.amount == Decimal("0.01")
    assert price.operation == OPERATION_READ
    assert price.unit == UNIT_PAGE


def test_the_operation_is_part_of_the_entry_because_the_vendor_prices_it():
    book = PriceBook([
        _entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google"),
        _entry(OPERATION_CHUNK, UNIT_PAGE, "0.00002", provider="google"),
    ])

    read = book.price_for(operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE)
    chunk = book.price_for(operation=OPERATION_CHUNK, provider="google", model=READER, unit=UNIT_PAGE)

    assert read.amount / chunk.amount == Decimal("500")


def test_an_omitted_field_matches_whatever_the_caller_named():
    book = PriceBook([_entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google")])

    for version in (READER, "pretrained-layout-parser-v1.5", "anything-at-all"):
        price = book.price_for(
            operation=OPERATION_READ, provider="google", model=version, unit=UNIT_PAGE
        )
        assert price.amount == Decimal("0.01"), version


def test_the_most_specific_entry_wins():
    book = PriceBook([
        _entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000015", provider="openrouter"),
        _entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000001", provider="openrouter", model=EMBED_MODEL),
    ])

    special = book.price_for(
        operation=OPERATION_EMBED, provider="openrouter", model=EMBED_MODEL, unit=UNIT_TOKEN
    )
    ordinary = book.price_for(
        operation=OPERATION_EMBED, provider="openrouter", model="outro/modelo", unit=UNIT_TOKEN
    )

    assert special.amount == Decimal("0.00000001")
    assert ordinary.amount == Decimal("0.00000015")


def test_the_provider_outranks_the_model_so_two_entries_never_tie():
    book = PriceBook([
        _entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000009", provider="openrouter"),
        _entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000002", model=EMBED_MODEL),
    ])

    price = book.price_for(
        operation=OPERATION_EMBED, provider="openrouter", model=EMBED_MODEL, unit=UNIT_TOKEN
    )

    assert price.amount == Decimal("0.00000009")


def test_two_entries_saying_the_same_thing_are_refused():
    with pytest.raises(ValueError) as excinfo:
        PriceBook([
            _entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google"),
            _entry(OPERATION_READ, UNIT_PAGE, "0.02", provider="google"),
        ])

    assert "same operation, provider, model and unit" in str(excinfo.value)


def test_the_lenient_lookup_returns_nothing_and_the_strict_one_raises():
    book = PriceBook()

    assert book.price_or_none(
        operation=OPERATION_EMBED, provider="openrouter", model=EMBED_MODEL, unit=UNIT_TOKEN
    ) is None
    with pytest.raises(UnknownPrice) as excinfo:
        book.price_for(
            operation=OPERATION_EMBED, provider="openrouter", model=EMBED_MODEL, unit=UNIT_TOKEN
        )

    message = str(excinfo.value)
    assert OPERATION_EMBED in message and EMBED_MODEL in message
    assert "stale one is worse than none" in message


def test_the_model_may_contain_slashes_because_a_gateway_names_it_that_way():
    book = PriceBook([
        _entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000001", provider="openrouter", model=EMBED_MODEL)
    ])

    price = book.price_for(
        operation=OPERATION_EMBED, provider="openrouter", model=EMBED_MODEL, unit=UNIT_TOKEN
    )

    assert price.model == EMBED_MODEL
    assert price.amount == Decimal("0.00000001")


def test_an_operation_that_is_not_billable_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        PriceBook([_entry("reed", UNIT_PAGE, "0.01", provider="google")])

    message = str(excinfo.value)
    assert "reed" in message
    for operation in BILLABLE_OPERATIONS:
        assert operation in message


def test_a_unit_that_is_not_billable_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        PriceBook([_entry(OPERATION_READ, "paginas", "0.01", provider="google")])

    message = str(excinfo.value)
    assert "paginas" in message
    for unit in BILLABLE_UNITS:
        assert unit in message


def test_a_negative_price_is_refused_at_the_boundary():
    with pytest.raises(ValueError) as excinfo:
        PriceBook([_entry(OPERATION_READ, UNIT_PAGE, "-0.01", provider="google")])

    assert "negative" in str(excinfo.value)


def test_a_price_is_read_as_an_exact_decimal_whatever_it_arrives_as():
    book = PriceBook([_entry(OPERATION_EMBED, UNIT_TOKEN, "0.00000013", provider="openai")])

    price = book.price_for(
        operation=OPERATION_EMBED, provider="openai", model="x", unit=UNIT_TOKEN
    )

    assert price.amount == Decimal("0.00000013")
    assert price.cost_of(1_000_000) == Decimal("0.13")


def test_cost_of_multiplies_by_the_units_consumed():
    book = PriceBook([_entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google")])

    price = book.price_for(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    )

    assert price.cost_of(540) == Decimal("5.40")


def test_two_books_do_not_share_anything():
    first = PriceBook([_entry(OPERATION_READ, UNIT_PAGE, "0.01", provider="google")])
    second = PriceBook()

    assert len(first) == 1
    assert len(second) == 0


def test_the_committed_prices_build_a_valid_price_book():
    book = PriceBook(_configured())

    assert len(book) == 5
    assert book.price_for(
        operation="read", provider="internal", model="", unit="page"
    ).amount == 0


def test_the_committed_amounts_are_quoted_not_binary_doubles():
    for entry in _configured():
        amount = entry["amount"]
        assert isinstance(amount, str), entry
        assert "e" not in amount.lower(), entry


def test_the_committed_amounts_are_the_vendor_quotes_converted_to_one_unit():
    book = PriceBook(_configured())

    def amount(operation, provider, model, unit):
        return book.price_for(
            operation=operation, provider=provider, model=model, unit=unit
        ).amount

    assert amount(OPERATION_READ, "google", READER, UNIT_PAGE) * 1000 == Decimal("10")
    assert amount(OPERATION_EMBED, "openrouter", EMBED_MODEL, UNIT_TOKEN) * 1_000_000 == Decimal("0.01")


def test_cutting_again_costs_nothing_and_reading_again_does_not():
    book = PriceBook(_configured())

    read = book.price_for(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    )
    chunk = book.price_for(
        operation=OPERATION_CHUNK, provider="internal", model="local", unit=UNIT_PAGE
    )

    assert read.amount > 0
    assert chunk.amount == 0
    assert read.cost_of(540) == Decimal("5.40")
    assert chunk.cost_of(540) == Decimal("0")


def test_the_cut_that_actually_runs_is_priced_at_zero_rather_than_left_out():
    book = PriceBook(_configured())

    price = book.price_for(
        operation=OPERATION_CHUNK, provider="internal", model="local", unit=UNIT_PAGE
    )

    assert price.amount == Decimal("0")


def test_a_search_costs_no_vendor_money_and_says_so_with_a_number():
    book = PriceBook(_configured())

    price = book.price_for(
        operation=OPERATION_SEARCH, provider="internal", model="x", unit=UNIT_REQUEST
    )

    assert price.amount == Decimal("0")


def test_the_configured_embedding_model_has_a_price():
    document = yaml.safe_load(COMMITTED.read_text())
    model = document["embed"]["model"]
    book = PriceBook(_configured())

    price = book.price_or_none(
        operation=OPERATION_EMBED, provider="openrouter", model=model, unit=UNIT_TOKEN
    )

    assert price is not None, f"{model} has no configured price"


def test_every_committed_entry_dates_the_figure_it_names():
    text = COMMITTED.read_text()
    prices = text[text.index("  prices:"):text.index("\nschedule:")]

    dated = [line for line in prices.splitlines() if "(20" in line]

    assert len(dated) >= len(_configured()), prices


def test_the_unit_is_decided_by_the_operation_and_the_two_declarations_agree():
    from src.config import PRICED_BY_PAGE, PRICED_BY_REQUEST, PRICED_BY_TOKEN
    from src.prices import UNITS_BY_OPERATION

    declared = {
        **{operation: UNIT_PAGE for operation in PRICED_BY_PAGE},
        **{operation: UNIT_TOKEN for operation in PRICED_BY_TOKEN},
        **{operation: UNIT_REQUEST for operation in PRICED_BY_REQUEST},
    }

    assert declared == UNITS_BY_OPERATION
    assert set(declared) == set(BILLABLE_OPERATIONS)


def test_an_operation_priced_in_the_wrong_unit_is_refused_although_both_words_are_real():
    with pytest.raises(ValueError) as excinfo:
        PriceBook([_entry(OPERATION_READ, UNIT_TOKEN, "0.01", provider="google")])

    message = str(excinfo.value)
    assert "'read'" in message and "'token'" in message and "'page'" in message
