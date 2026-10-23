from __future__ import annotations

import re
from typing import Any, Callable, Mapping

SECRET_SUFFIX = "_secret"

Resolver = Callable[[str], str]

SCHEME_SEPARATOR = ":"

SCHEME_ESCAPE = SCHEME_SEPARATOR * 2

_TARGET = "[A-Za-z_][A-Za-z0-9_]*"

SCHEME_ENV = "env"

REGISTERED_SCHEMES: tuple[str, ...] = (SCHEME_ENV,)


_REFERENCE = re.compile(
    r"(?P<scheme>" + "|".join(re.escape(name) for name in REGISTERED_SCHEMES) + r")"
    r"(?P<separator>" + re.escape(SCHEME_ESCAPE) + r"|" + re.escape(SCHEME_SEPARATOR) + r")"
    r"(?P<target>" + _TARGET + r")"
)


def scheme_of(value: object) -> str:
    if not isinstance(value, str) or SCHEME_SEPARATOR not in value:
        return ""
    return value.split(SCHEME_SEPARATOR, 1)[0]


def is_reference(value: object) -> bool:
    if not isinstance(value, str):
        return False
    found = _REFERENCE.fullmatch(value)
    return found is not None and found.group("separator") == SCHEME_SEPARATOR


def points_outside(value: object) -> bool:
    return isinstance(value, str) and any(
        found.group("separator") == SCHEME_SEPARATOR for found in _REFERENCE.finditer(value)
    )


def target_of(reference: str) -> str:
    return reference.split(SCHEME_SEPARATOR, 1)[1]


def resolve_text(name: str, value: str, resolver: Resolver) -> str:
    def substitute(found: "re.Match[str]") -> str:
        scheme, separator, target = found.group("scheme", "separator", "target")
        if separator == SCHEME_ESCAPE:
            return f"{scheme}{SCHEME_SEPARATOR}{target}"
        reference = f"{scheme}{SCHEME_SEPARATOR}{target}"
        try:
            return resolver(reference)
        except Exception as exc:
            raise SecretResolutionError(
                f"could not resolve secret reference for '{name}' ({reference})"
            ) from exc

    return _REFERENCE.sub(substitute, value)


def environment_backend(target: str) -> str:
    import os

    value = os.environ.get(target, "")
    if not value:
        raise KeyError(f"${target} is unset or empty")
    return value


def resolver_over(
    backends: "Mapping[str, Resolver] | None" = None, **overrides: Resolver
) -> Resolver:
    table: dict[str, Resolver] = {SCHEME_ENV: environment_backend}
    table.update(backends or {})
    table.update(overrides)

    def resolve(reference: str) -> str:
        scheme = scheme_of(reference)
        backend = table.get(scheme)
        if backend is None:
            known = ", ".join(sorted(table)) or "none"
            raise KeyError(
                f"no backend for '{scheme}{SCHEME_SEPARATOR}' in this process; "
                f"this build resolves {known}"
            )
        return backend(target_of(reference))

    return resolve


class SecretResolutionError(Exception):
    pass


def resolve_secrets(settings: Mapping[str, str], resolver: Resolver) -> dict[str, str]:
    return {name: resolve_text(name, value, resolver) for name, value in settings.items()}
