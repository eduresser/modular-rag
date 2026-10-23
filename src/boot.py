from __future__ import annotations

from urllib.parse import urlparse

from src.config import AppConfig

RESERVED_BODY_KEYS: dict[str, str] = {
    "model": "'embed.model' names it, and the answer is checked against that name",
    "dimensions": "'embed.dimension' names it, and it is also the store's column type (ASM-031)",
    "allow_fallbacks": "a substituted provider is invisible from every row (AC-052)",
    "input": "the texts are what the caller asked to embed",
}

RESERVED_HEADERS: dict[str, str] = {
    "authorization": "built from 'embed.credentials_secret'; name a different header instead",
    "content-type": "the body is JSON, and the client says so",
}


class BootCompatibilityError(Exception):
    pass


def check_boot_compatibility(config: AppConfig, *, stored_dimension: int | None = None) -> None:
    messages: list[str] = []
    messages += _check_chunk_size_against_window(config)
    messages += _check_dimension_against_stored_column(config, stored_dimension)
    messages += _check_endpoint_allowlisted(config)
    messages += _check_request_collisions(config)
    messages += _check_surface_window_ceilings(config)
    if messages:
        raise BootCompatibilityError("\n".join(messages))


def _check_chunk_size_against_window(config: AppConfig) -> list[str]:
    size = getattr(config.chunk, "size", None)
    if size is None:
        return []
    window = config.embed.context_window
    if size <= window:
        return []
    return [
        f"[chunk] chunk size {size} exceeds the context window "
        f"({window} tokens) of embedding model '{config.embed.model}'; "
        "reduce the chunk size or choose a model with a larger window"
    ]


def _check_surface_window_ceilings(config: AppConfig) -> list[str]:
    messages: list[str] = []
    surfaces = [(name, value) for name, value in config if hasattr(value, "defaults") and hasattr(value, "limits")]

    for surface_name, surface in surfaces:
        operations = sorted(set(type(surface.defaults).model_fields) & set(type(surface.limits).model_fields))
        for operation in operations:
            default_limit = getattr(surface.defaults, operation).limit
            ceiling = getattr(surface.limits, operation).limit
            if default_limit > ceiling:
                messages.append(
                    f"[{surface_name}] {surface_name}.defaults.{operation}.limit ({default_limit}) is "
                    f"above {surface_name}.limits.{operation}.limit ({ceiling}); a default may not "
                    f"exceed its own surface's ceiling — lower the default or raise the limit"
                )

    if "api" in dict(surfaces) and "mcp" in dict(surfaces):
        api, mcp = config.api, config.mcp
        operations = sorted(set(type(mcp.defaults).model_fields) & set(type(api.limits).model_fields))
        for operation in operations:
            mcp_default = getattr(mcp.defaults, operation).limit
            api_ceiling = getattr(api.limits, operation).limit
            if mcp_default > api_ceiling:
                messages.append(
                    f"[mcp] mcp.defaults.{operation}.limit ({mcp_default}) is above "
                    f"api.limits.{operation}.limit ({api_ceiling}); the tool surface's window may not "
                    f"exceed the conventional surface's ceiling — lower the default or raise the limit"
                )

    return messages


def _check_dimension_against_stored_column(config: AppConfig, stored_dimension: int | None) -> list[str]:
    if stored_dimension is None or stored_dimension == config.embed.dimension:
        return []
    return [
        f"[embed] configured embedding dimension {config.embed.dimension} disagrees with "
        f"the stored column width {stored_dimension}; this requires a schema migration, "
        "not a configuration change"
    ]


def _check_endpoint_allowlisted(config: AppConfig) -> list[str]:
    host = urlparse(config.embed.base_url).hostname
    if host in config.embed.allowed_hosts:
        return []
    permitted = ", ".join(config.embed.allowed_hosts) or "(none)"
    return [
        f"[embed] embedding base URL host '{host}' is not on this deployment's "
        f"allowlist ('embed.allowed_hosts': {permitted}); a host nobody listed is "
        "refused rather than tried, because the corpus is what travels to it"
    ]


def _check_request_collisions(config: AppConfig) -> list[str]:
    messages = []
    for key, because in RESERVED_BODY_KEYS.items():
        if key in config.embed.request.body:
            messages.append(
                f"[embed] 'request.body.{key}' is set and this service decides it: {because}"
            )
    for header in config.embed.request.headers:
        because = RESERVED_HEADERS.get(header.lower())
        if because is not None:
            messages.append(
                f"[embed] 'request.headers.{header}' is set and this service decides it: {because}"
            )
    return messages


#: The surface that answers requests over HTTP and the mounted tool transport.
ROLE_SERVE = "serve"
#: The worker that fires this deployment's schedules and indexes the corpus.
ROLE_WORKER = "worker"

ROLES: tuple[str, ...] = (ROLE_SERVE, ROLE_WORKER)


class UnknownRole(Exception):
    pass


def check_role(role: str) -> str:
    if role not in ROLES:
        known = ", ".join(ROLES)
        raise UnknownRole(f"no such role: {role!r}; this build defines: {known}")
    return role
