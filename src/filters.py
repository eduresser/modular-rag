from __future__ import annotations

import re as _re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from src.config import REGISTERED_PROCESS_MODULES
from src.processing.preflight import SUPPORTED_MEDIA_TYPES


TYPE_TEXT = "text"
TYPE_TEXT_SEARCHABLE = "text_searchable"
TYPE_ENUM = "enum"
TYPE_ENUM_OPEN = "enum_open"
TYPE_NUMBER = "number"
TYPE_DATE = "date"
TYPE_TEXT_LIST = "text_list"
TYPE_RANGE = "range"

FIELD_TYPES: tuple[str, ...] = (
    TYPE_TEXT,
    TYPE_TEXT_SEARCHABLE,
    TYPE_ENUM,
    TYPE_ENUM_OPEN,
    TYPE_NUMBER,
    TYPE_DATE,
    TYPE_TEXT_LIST,
    TYPE_RANGE,
)


OP_IS = "is"
OP_IS_NOT = "is_not"
OP_IN = "in"
OP_NOT_IN = "not_in"
OP_EXISTS = "exists"
OP_CONTAINS = "contains"
OP_STARTS_WITH = "starts_with"
OP_GLOB = "glob"
OP_REGEX = "regex"
OP_GT = "gt"
OP_GTE = "gte"
OP_LT = "lt"
OP_LTE = "lte"
OP_BETWEEN = "between"
OP_HAS_ANY = "has_any"
OP_HAS_ALL = "has_all"
OP_HAS_NONE = "has_none"
OP_PATH_STARTS_WITH = "path_starts_with"

OPERATORS: dict[str, tuple[str, ...]] = {
    TYPE_TEXT: (OP_IS, OP_IS_NOT, OP_IN, OP_NOT_IN, OP_EXISTS),
    TYPE_TEXT_SEARCHABLE: (
        OP_IS, OP_IS_NOT, OP_IN, OP_NOT_IN, OP_EXISTS,
        OP_CONTAINS, OP_STARTS_WITH, OP_GLOB, OP_REGEX,
    ),
    TYPE_ENUM: (OP_IS, OP_IS_NOT, OP_IN, OP_NOT_IN),
    TYPE_ENUM_OPEN: (OP_IS, OP_IS_NOT, OP_IN, OP_NOT_IN),
    TYPE_NUMBER: (OP_IS, OP_GT, OP_GTE, OP_LT, OP_LTE, OP_BETWEEN, OP_EXISTS),
    TYPE_DATE: (OP_IS, OP_GT, OP_GTE, OP_LT, OP_LTE, OP_BETWEEN, OP_EXISTS),
    TYPE_TEXT_LIST: (
        OP_HAS_ANY, OP_HAS_ALL, OP_HAS_NONE, OP_PATH_STARTS_WITH, OP_EXISTS,
        OP_CONTAINS, OP_STARTS_WITH, OP_GLOB, OP_REGEX,
    ),
    TYPE_RANGE: (OP_BETWEEN, OP_GTE, OP_LTE),
}


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    type: str
    values: tuple[str, ...] | None = None
    published: bool = False
    open_values: bool = False


FIELDS: dict[str, FieldSpec] = {
    "document_id": FieldSpec("document_id", TYPE_TEXT),
    "title": FieldSpec("title", TYPE_TEXT_SEARCHABLE),
    "folder": FieldSpec("folder", TYPE_TEXT_LIST, open_values=True),
    "media_type": FieldSpec(
        "media_type", TYPE_ENUM, values=SUPPORTED_MEDIA_TYPES, published=True,
    ),
    "modified_at": FieldSpec("modified_at", TYPE_DATE),
    "size_bytes": FieldSpec("size_bytes", TYPE_NUMBER),
    "page": FieldSpec("page", TYPE_RANGE),
    "heading_path": FieldSpec("heading_path", TYPE_TEXT_LIST, open_values=True),
    "ordinal": FieldSpec("ordinal", TYPE_NUMBER),
    "content": FieldSpec("content", TYPE_TEXT_SEARCHABLE),
    "block_kind": FieldSpec("block_kind", TYPE_ENUM_OPEN, published=True),
    "indexed_at": FieldSpec("indexed_at", TYPE_DATE),
    "process_module": FieldSpec(
        "process_module", TYPE_ENUM, values=REGISTERED_PROCESS_MODULES, published=True,
    ),
    "reader_version": FieldSpec("reader_version", TYPE_TEXT),
}


DEFAULT_MAX_FILTER_CLAUSES: Final[int] = 16
DEFAULT_MAX_FILTER_DEPTH: Final[int] = 3
DEFAULT_MAX_PATTERN_LENGTH: Final[int] = 200
DEFAULT_FILTER_TIMEOUT_MS: Final[int] = 2000

COMBINATORS: tuple[str, ...] = ("all", "any", "none")

_CLAUSE_KEYS = frozenset({"field", "op", "value"})

# Only these two operators reach a backtracking engine; the length limit
# exists for them and for no other operator (see the module docstring).
_PATTERN_OPERATORS = frozenset({OP_GLOB, OP_REGEX})

_KIND_BOOL = "bool"
_KIND_SCALAR_TEXT = "scalar_text"
_KIND_LIST_TEXT = "list_text"
_KIND_SCALAR_NUMBER = "scalar_number"
_KIND_PAIR_NUMBER = "pair_number"
_KIND_SCALAR_DATE = "scalar_date"
_KIND_PAIR_DATE = "pair_date"


class FilterError(Exception):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True, slots=True)
class Clause:
    field: str
    op: str
    value: Any
    path: str


@dataclass(frozen=True, slots=True)
class Combinator:
    kind: str
    children: tuple["FilterNode", ...]
    path: str


FilterNode = Clause | Combinator


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_date_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_text_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(item, str) for item in value)
    )


def _expected_kind(field_type: str, op: str) -> str:
    if op == OP_EXISTS:
        return _KIND_BOOL
    if op in (OP_IN, OP_NOT_IN, OP_HAS_ANY, OP_HAS_ALL, OP_HAS_NONE, OP_PATH_STARTS_WITH):
        return _KIND_LIST_TEXT
    if op == OP_BETWEEN:
        return _KIND_PAIR_DATE if field_type == TYPE_DATE else _KIND_PAIR_NUMBER
    if op in (OP_GT, OP_GTE, OP_LT, OP_LTE):
        return _KIND_SCALAR_DATE if field_type == TYPE_DATE else _KIND_SCALAR_NUMBER
    if op in (OP_IS, OP_IS_NOT):
        if field_type in (TYPE_NUMBER, TYPE_RANGE):
            return _KIND_SCALAR_NUMBER
        if field_type == TYPE_DATE:
            return _KIND_SCALAR_DATE
        return _KIND_SCALAR_TEXT
    # contains, starts_with, glob, regex — text_searchable fields only.
    return _KIND_SCALAR_TEXT


def _validate_value(
    path: str, spec: FieldSpec, op: str, value: Any, *, max_pattern_length: int,
) -> None:
    kind = _expected_kind(spec.type, op)

    if kind == _KIND_BOOL:
        if not isinstance(value, bool):
            raise FilterError(
                path, f"{spec.name!r} takes true or false for {op!r}; got {value!r}",
            )
        return

    if kind == _KIND_SCALAR_TEXT:
        if not isinstance(value, str):
            raise FilterError(
                path,
                f"{spec.name!r} is a {spec.type} field and takes text for {op!r}; "
                f"got {type(value).__name__}",
            )
        if op in _PATTERN_OPERATORS and len(value) > max_pattern_length:
            raise FilterError(
                path,
                f"the pattern for {spec.name!r} is {len(value)} characters, over this "
                f"deployment's max_pattern_length ({max_pattern_length})",
            )
        if spec.type == TYPE_ENUM and spec.values is not None and value not in spec.values:
            raise FilterError(
                path,
                f"{value!r} is not a value {spec.name!r} holds; it takes "
                f"{', '.join(spec.values)}",
            )
        return

    if kind == _KIND_LIST_TEXT:
        if not _is_text_list(value):
            raise FilterError(
                path, f"{spec.name!r} takes a non-empty list of text for {op!r}; got {value!r}",
            )
        if spec.type == TYPE_ENUM and spec.values is not None:
            unknown = [item for item in value if item not in spec.values]
            if unknown:
                raise FilterError(
                    path,
                    f"{unknown!r} are not values {spec.name!r} holds; it takes "
                    f"{', '.join(spec.values)}",
                )
        return

    if kind == _KIND_SCALAR_NUMBER:
        if not _is_number(value):
            raise FilterError(
                path,
                f"{spec.name!r} is a {spec.type} field and takes a number for {op!r}; "
                f"got {type(value).__name__}",
            )
        return

    if kind == _KIND_PAIR_NUMBER:
        if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value)):
            raise FilterError(
                path, f"{spec.name!r} takes a [min, max] pair of numbers for {op!r}; got {value!r}",
            )
        return

    if kind == _KIND_SCALAR_DATE:
        if not _is_date_string(value):
            raise FilterError(
                path, f"{spec.name!r} is a date field and takes an ISO date for {op!r}; got {value!r}",
            )
        return

    if kind == _KIND_PAIR_DATE:  # pragma: no branch — every kind above is exhaustive
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(_is_date_string(v) for v in value)
        ):
            raise FilterError(
                path,
                f"{spec.name!r} takes a [from, to] pair of ISO dates for {op!r}; got {value!r}",
            )
        return


def _parse(
    node: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_clauses: int,
    max_pattern_length: int,
    clause_count: list[int],
) -> FilterNode:
    if not isinstance(node, Mapping):
        raise FilterError(path, f"a clause must be an object, not {type(node).__name__}")

    keys = set(node.keys())

    if len(keys) == 1:
        only_key = next(iter(keys))
        if only_key not in _CLAUSE_KEYS and only_key not in COMBINATORS:
            candidate = node[only_key]
            if isinstance(candidate, (list, tuple)):
                raise FilterError(
                    path,
                    f"{only_key!r} is not a combinator this deployment recognises; "
                    f"it takes {', '.join(COMBINATORS)}",
                )

    combinator_keys = keys & set(COMBINATORS)
    if combinator_keys:
        if len(combinator_keys) != 1 or keys != combinator_keys:
            raise FilterError(
                path,
                "a combinator node carries exactly one of "
                f"{', '.join(COMBINATORS)} and nothing else",
            )
        kind = next(iter(combinator_keys))
        if depth > max_depth:
            raise FilterError(
                f"{path}.{kind}",
                f"nesting exceeds this deployment's max_filter_depth ({max_depth})",
            )
        children_raw = node[kind]
        if not isinstance(children_raw, Sequence) or isinstance(children_raw, (str, bytes)):
            raise FilterError(f"{path}.{kind}", f"{kind!r} takes a list of clauses")
        if not children_raw:
            raise FilterError(f"{path}.{kind}", f"{kind!r} must carry at least one clause")
        children = tuple(
            _parse(
                child,
                path=f"{path}.{kind}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_clauses=max_clauses,
                max_pattern_length=max_pattern_length,
                clause_count=clause_count,
            )
            for index, child in enumerate(children_raw)
        )
        return Combinator(kind=kind, children=children, path=path)

    missing = _CLAUSE_KEYS - keys
    extra = keys - _CLAUSE_KEYS
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            problems.append(f"unknown key(s) {', '.join(sorted(extra))}")
        raise FilterError(
            path, f"a clause must carry exactly field, op and value ({'; '.join(problems)})",
        )

    field_name = node["field"]
    op = node["op"]
    value = node["value"]

    clause_count[0] += 1
    if clause_count[0] > max_clauses:
        raise FilterError(
            path, f"the filter carries more clauses than this deployment's max_filter_clauses ({max_clauses})",
        )

    if field_name not in FIELDS:
        raise FilterError(
            path,
            f"{field_name!r} is not a field this archive publishes; the vocabulary is "
            f"{', '.join(sorted(FIELDS))}",
        )
    spec = FIELDS[field_name]

    allowed_ops = OPERATORS[spec.type]
    if op not in allowed_ops:
        raise FilterError(
            path,
            f"{field_name!r} is a {spec.type} field and does not take {op!r}; "
            f"it takes {', '.join(allowed_ops)}",
        )

    _validate_value(path, spec, op, value, max_pattern_length=max_pattern_length)

    return Clause(field=field_name, op=op, value=value, path=path)


def validate_filter(
    filter_: Mapping[str, Any] | None,
    *,
    max_clauses: int = DEFAULT_MAX_FILTER_CLAUSES,
    max_depth: int = DEFAULT_MAX_FILTER_DEPTH,
    max_pattern_length: int = DEFAULT_MAX_PATTERN_LENGTH,
) -> FilterNode | None:
    if filter_ is None:
        return None
    return _parse(
        filter_,
        path="<filter>",
        depth=1,
        max_depth=max_depth,
        max_clauses=max_clauses,
        max_pattern_length=max_pattern_length,
        clause_count=[0],
    )


_DOCUMENT_COLUMNS: dict[str, str] = {
    "document_id": "d.external_id",
    "title": "d.title",
    "folder": "d.location",
    "media_type": "d.media_type",
    "modified_at": "d.modified_at",
    "size_bytes": "d.size_bytes",
}

_CHUNK_COLUMNS: dict[str, str] = {
    "heading_path": "c.heading_path",
    "ordinal": "c.ordinal",
    "content": "c.content",
}

_JOINED_COLUMNS: dict[str, tuple[str, str, str]] = {
    "indexed_at": ("document_ingest", "di", "indexed_at"),
    "process_module": ("document_ingest", "di", "process_module"),
    "reader_version": ("document_reads", "dr", "reader_version"),
}


def _glob_to_regex(pattern: str) -> str:
    out = ["^"]
    for char in pattern:
        if char == "*":
            out.append(".*")
        elif char == "?":
            out.append(".")
        else:
            out.append(_re.escape(char))
    out.append("$")
    return "".join(out)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _compile_operator(column: str, field_type: str, op: str, value: Any) -> tuple[str, list[Any]]:
    if op == OP_IS:
        return f"{column} = %s", [value]
    if op == OP_IS_NOT:
        return f"{column} <> %s", [value]
    if op == OP_IN:
        return f"{column} = ANY(%s)", [list(value)]
    if op == OP_NOT_IN:
        return f"{column} <> ALL(%s)", [list(value)]
    if op == OP_EXISTS:
        return (f"{column} IS NOT NULL" if value else f"{column} IS NULL"), []
    if op == OP_CONTAINS:
        return f"{column} ILIKE %s", [f"%{_escape_like(value)}%"]
    if op == OP_STARTS_WITH:
        return f"{column} ILIKE %s", [f"{_escape_like(value)}%"]
    if op == OP_GLOB:
        return f"{column} ~ %s", [_glob_to_regex(value)]
    if op == OP_REGEX:
        return f"{column} ~ %s", [value]
    if op == OP_GT:
        return f"{column} > %s", [value]
    if op == OP_GTE:
        return f"{column} >= %s", [value]
    if op == OP_LT:
        return f"{column} < %s", [value]
    if op == OP_LTE:
        return f"{column} <= %s", [value]
    if op == OP_BETWEEN:
        low, high = value
        return f"{column} BETWEEN %s AND %s", [low, high]
    raise AssertionError(f"unreachable: {op!r} on {column!r} (validated before compilation)")


def _compile_text_list(column: str, op: str, value: Any) -> tuple[str, list[Any]]:
    if op == OP_HAS_ANY:
        return f"{column} && %s", [list(value)]
    if op == OP_HAS_ALL:
        return f"{column} @> %s", [list(value)]
    if op == OP_HAS_NONE:
        return f"NOT ({column} && %s)", [list(value)]
    if op == OP_EXISTS:
        return (
            f"array_length({column}, 1) IS NOT NULL" if value else f"array_length({column}, 1) IS NULL"
        ), []
    if op == OP_PATH_STARTS_WITH:
        # A prefix of the *path* — the first len(value) elements, compared as
        # an array — never a prefix of a flattened string (AC-281).
        segments = list(value)
        return f"{column}[1:%s] = %s", [len(segments), segments]
    if op == OP_CONTAINS:
        return f"EXISTS (SELECT 1 FROM unnest({column}) AS segment WHERE segment ILIKE %s)", [
            f"%{_escape_like(value)}%"
        ]
    if op == OP_STARTS_WITH:
        return f"EXISTS (SELECT 1 FROM unnest({column}) AS segment WHERE segment ILIKE %s)", [
            f"{_escape_like(value)}%"
        ]
    if op == OP_GLOB:
        return f"EXISTS (SELECT 1 FROM unnest({column}) AS segment WHERE segment ~ %s)", [
            _glob_to_regex(value)
        ]
    if op == OP_REGEX:
        return f"EXISTS (SELECT 1 FROM unnest({column}) AS segment WHERE segment ~ %s)", [value]
    raise AssertionError(f"unreachable: {op!r} on {column!r} (validated before compilation)")


def _compile_page(op: str, value: Any) -> tuple[str, list[Any]]:
    if op == OP_BETWEEN:
        low, high = value
        return (
            "(c.page_start IS NOT NULL AND c.page_end IS NOT NULL "
            "AND c.page_end >= %s AND c.page_start <= %s)"
        ), [low, high]
    if op == OP_GTE:
        return "(c.page_end IS NOT NULL AND c.page_end >= %s)", [value]
    if op == OP_LTE:
        return "(c.page_start IS NOT NULL AND c.page_start <= %s)", [value]
    raise AssertionError(f"unreachable: {op!r} on page (validated before compilation)")


def _compile_block_kind(op: str, value: Any) -> tuple[str, list[Any]]:
    column = "c.metadata -> 'block_kinds'"
    if op == OP_IS:
        return f"({column} ? %s)", [value]
    if op == OP_IS_NOT:
        return f"NOT ({column} ? %s)", [value]
    if op == OP_IN:
        return f"({column} ?| %s)", [list(value)]
    if op == OP_NOT_IN:
        return f"NOT ({column} ?| %s)", [list(value)]
    raise AssertionError(f"unreachable: {op!r} on block_kind (validated before compilation)")


def _compile_joined(table: str, alias: str, column: str, field_type: str, op: str, value: Any) -> tuple[str, list[Any]]:
    predicate, params = _compile_operator(f"{alias}.{column}", field_type, op, value)
    return (
        f"EXISTS (SELECT 1 FROM {table} {alias} WHERE {alias}.document_id = c.document_id AND {predicate})",
        params,
    )


def _compile_clause(clause: Clause) -> tuple[str, list[Any]]:
    field_name, op, value = clause.field, clause.op, clause.value
    spec = FIELDS[field_name]

    if field_name == "page":
        return _compile_page(op, value)
    if field_name == "block_kind":
        return _compile_block_kind(op, value)
    if field_name in _JOINED_COLUMNS:
        table, alias, column = _JOINED_COLUMNS[field_name]
        return _compile_joined(table, alias, column, spec.type, op, value)

    column = _DOCUMENT_COLUMNS.get(field_name) or _CHUNK_COLUMNS[field_name]
    if spec.type == TYPE_TEXT_LIST:
        return _compile_text_list(column, op, value)
    return _compile_operator(column, spec.type, op, value)


def _compile_node(node: FilterNode) -> tuple[str, list[Any]]:
    if isinstance(node, Clause):
        return _compile_clause(node)

    parts = [_compile_node(child) for child in node.children]
    fragments = [fragment for fragment, _ in parts]
    params = [value for _, child_params in parts for value in child_params]

    if node.kind == "all":
        return " AND ".join(f"({fragment})" for fragment in fragments), params
    if node.kind == "any":
        return " OR ".join(f"({fragment})" for fragment in fragments), params
    # "none": true when not one of the children holds.
    return "NOT (" + " OR ".join(f"({fragment})" for fragment in fragments) + ")", params


def compile_filter(node: FilterNode | None) -> tuple[str, list[Any]] | None:
    if node is None:
        return None
    return _compile_node(node)
