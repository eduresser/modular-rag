from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass

from fastapi import Request

logger = logging.getLogger(__name__)

BEARER_PREFIX = "bearer "

END_USER_HEADERS: tuple[str, ...] = (
    "x-user-id",
    "x-user-email",
    "x-on-behalf-of",
    "x-end-user",
    "x-forwarded-user",
)


class Unauthenticated(Exception):
    pass


class IdentityNotAccepted(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Credential:
    token: str

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError(
                "the resolved API credential is empty; an empty token would accept "
                "every request that sends an empty one"
            )

    def matches(self, presented: str) -> bool:
        return hmac.compare_digest(self.token, presented)


def presented_token(request: Request) -> str:
    # Header only: proxies log query strings, which would leak a token in the URL
    header = request.headers.get("authorization", "")
    if not header.lower().startswith(BEARER_PREFIX):
        raise Unauthenticated(
            "this route requires a bearer credential in the Authorization header"
        )
    token = header[len(BEARER_PREFIX):].strip()
    if not token:
        raise Unauthenticated("the Authorization header carries no token")
    return token


def refuse_end_user_identity(request: Request) -> None:
    named = sorted(name for name in END_USER_HEADERS if name in request.headers)
    if named:
        raise IdentityNotAccepted(
            f"this service accepts no end-user identity and the request carries {named}; "
            "there is no per-user access control in this version (principle P-014), so a "
            "plaintext user identifier is refused rather than trusted with a filter that "
            "would never be applied"
        )


def authorise(request: Request, credential: Credential) -> None:
    refuse_end_user_identity(request)
    if not credential.matches(presented_token(request)):
        raise Unauthenticated("the credential presented is not this deployment's")


def host_of(request: Request) -> str:
    header = request.headers.get("host", "")
    name = header.split(",")[0].strip()
    if name.startswith("["):  # an IPv6 literal keeps its brackets, loses its port
        closing = name.find("]")
        return name[: closing + 1] if closing != -1 else name
    return name.rsplit(":", 1)[0] if ":" in name else name


def host_allowed(host: str, allowed: "tuple[str, ...] | list[str]") -> bool:
    if "*" in allowed:
        return True
    return host in {name.strip() for name in allowed}


def host_refusal(request: Request, allowed_hosts: "tuple[str, ...] | list[str]") -> str | None:
    host = host_of(request)
    if host_allowed(host, allowed_hosts):
        return None
    logger.warning(
        "refused a request for host %r on %s; configured hosts are %s",
        host,
        request.url.path,
        ", ".join(allowed_hosts),
    )
    return (
        f"this deployment does not answer to host {host!r}; it is configured for "
        f"{', '.join(allowed_hosts)}"
    )
