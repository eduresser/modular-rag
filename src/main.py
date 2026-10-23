from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings

from uuid import uuid4

from src.auth import Credential, IdentityNotAccepted, Unauthenticated, authorise, host_refusal
from src.boot import BootCompatibilityError, check_boot_compatibility
from src.config import AppConfig, load_config, resolve_database_config, resolve_references
from src.db import DatabaseUnavailable, connect, stored_dimension
from src.embedding.pinning import pinned_client
from src.mcp import build_server
from src.rerank import Scorer
from src.responses import REQUEST_ID, failure_payload
from src.routes import health as health_routes
from src.routes import v1 as v1_routes
from src.routes.health import StartupChecks
from src.secrets import resolver_over

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"

OPEN_PATHS: frozenset[str] = frozenset({"/healthz"})


@dataclass(frozen=True, slots=True)
class Service:
    config: AppConfig
    embedder: Any
    credential: Credential
    mcp_credential: Credential
    scorer: Scorer | None = None
    checks: StartupChecks = StartupChecks(passed=True)


def startup_checks(config: AppConfig) -> StartupChecks:
    try:
        with connect(config.db.url) as conn:
            width = stored_dimension(conn)
    except DatabaseUnavailable as exc:
        logger.warning("the startup checks could not reach the archive: %s", exc)
        return StartupChecks(passed=False, detail=str(exc))
    check_boot_compatibility(config, stored_dimension=width)
    return StartupChecks(passed=True)


SERVING_SECTIONS: tuple[str, ...] = ("api", "mcp", "embed", "db")


def build_service(
    config: AppConfig,
    *,
    resolver: Callable[[str], str],
    http: httpx.Client,
    scorer: Scorer | None = None,
    checks: StartupChecks | None = None,
) -> Service:
    config = resolve_references(config, resolver, sections=SERVING_SECTIONS)
    embed_token = config.embed.credentials_secret
    return Service(
        config=config,
        embedder=pinned_client(config, http=http, token=lambda: embed_token),
        credential=Credential(token=config.api.credentials_secret),
        mcp_credential=Credential(token=config.mcp.credentials_secret),
        scorer=scorer,
        checks=checks if checks is not None else StartupChecks(passed=True),
    )


def create_app(service: Service) -> FastAPI:
    tools = build_server(service)
    tool_app = tools.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        transport_security=transport_security(service.config.api.allowed_hosts),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with tools.session_manager.run():
            yield

    app = FastAPI(title="modular-rag", lifespan=lifespan)
    app.state.service = service
    app.state.startup_checks = service.checks
    app.state.mcp = tools

    app.middleware("http")(_gate(service))

    app.include_router(health_routes.router)
    app.include_router(v1_routes.router)
    app.mount(MCP_PATH, tool_app)
    return app


def transport_security(allowed_hosts: "list[str] | tuple[str, ...]") -> TransportSecuritySettings:
    if "*" in allowed_hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    names: list[str] = []
    for host in allowed_hosts:
        names.extend((host, f"{host}:*"))
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=names)


def _gate(service: Service):
    allowed_hosts = service.config.api.allowed_hosts
    api_credential = service.credential
    mcp_credential = service.mcp_credential

    async def gate(request: Request, call_next):
        identifier = uuid4().hex
        token = REQUEST_ID.set(identifier)
        credential = (
            mcp_credential
            if request.url.path == MCP_PATH or request.url.path.startswith(f"{MCP_PATH}/")
            else api_credential
        )
        try:
            response = await _gated(request, call_next, allowed_hosts, credential)
        finally:
            REQUEST_ID.reset(token)
        if response.status_code >= 400:
            logger.warning(
                "request failed: request_id=%s method=%s path=%s status=%s",
                identifier,
                request.method,
                request.url.path,
                response.status_code,
            )
        response.headers["X-Request-Id"] = identifier
        return response

    return gate


async def _gated(request: Request, call_next, allowed_hosts, credential):
    refusal = host_refusal(request, allowed_hosts)
    if refusal is not None:
        return JSONResponse(status_code=400, content=failure_payload(refusal))
    if request.url.path in OPEN_PATHS:
        return await call_next(request)
    try:
        authorise(request, credential)
    except IdentityNotAccepted as exc:
        return JSONResponse(status_code=400, content=failure_payload(str(exc)))
    except Unauthenticated as exc:
        return JSONResponse(
            status_code=401,
            content=failure_payload(str(exc)),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


def main_app(
    *,
    resolver: Callable[[str], str] | None = None,
    http: httpx.Client | None = None,
    scorer: Scorer | None = None,
    app: str | None = None,
    path: str | None = None,
) -> FastAPI:
    config = resolve_database_config(load_config(path, app=app), resolver)
    checks = startup_checks(config)
    service = build_service(
        config,
        resolver=resolver if resolver is not None else resolver_over(),
        http=http if http is not None else httpx.Client(timeout=30.0),
        scorer=scorer,
        checks=checks,
    )
    return create_app(service)


__all__ = [
    "MCP_PATH",
    "transport_security",
    "BootCompatibilityError",
    "Service",
    "StartupChecks",
    "build_service",
    "create_app",
    "main_app",
    "startup_checks",
]
