from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, Union, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.prices import (
    OPERATION_CHUNK,
    OPERATION_EMBED,
    OPERATION_READ,
    OPERATION_RERANK,
    OPERATION_SEARCH,
    UNIT_PAGE,
    UNIT_REQUEST,
    UNIT_TOKEN,
)

PRICED_BY_PAGE = (OPERATION_READ, OPERATION_CHUNK)
PRICED_BY_TOKEN = (OPERATION_EMBED,)
PRICED_BY_REQUEST = (OPERATION_RERANK, OPERATION_SEARCH)
from src.schedule import InvalidScheduleError, next_fire
from src.secrets import (
    SECRET_SUFFIX,
    REGISTERED_SCHEMES, SCHEME_SEPARATOR, SecretResolutionError,
    is_reference, resolve_secrets, resolve_text, resolver_over,
)

SOURCE_GDRIVE = "gdrive"
READER_DOCAI = "docai"
CHUNK_DOCAI = "docai"
PROCESS_DOCAI = "docai"
PROCESS_PDFTEXT = "pdftext"

DISCRIMINATOR = "type"

#: The paths under which a mapping is a SECTION and therefore merges. Anywhere
#: else a mapping is a setting and is replaced whole (AC-170, AC-175).
SUB_SECTIONS: frozenset[tuple[str, ...]] = frozenset(
    {("process", "read"), ("process", "chunk"), ("embed", "request"), ("embed", "prompts")}
)

APP_VARIABLE = "APP"

#: Superseded by APP_VARIABLE, and refused rather than ignored — a path left in
#: a shell would otherwise quietly beat the app that was asked for.
RETIRED_PATH_VARIABLE = "RAG_CONFIG"

#: Where the deployments live, and what each one's configuration is called.
APPS_DIRECTORY = Path(__file__).resolve().parent.parent / "apps"
CONFIG_FILENAME = "config.yaml"

BASE_APP = "base"

SECRET_FIELD_SUFFIX = SECRET_SUFFIX


class ModuleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSettings(ModuleSettings):
    pass


class GDriveSourceSettings(SourceSettings):
    type: Literal[SOURCE_GDRIVE]
    folder_ids: Annotated[list[str], Field(min_length=1)]
    credentials_secret: str
    units_per_minute: int = 325_000

    @field_validator("folder_ids")
    @classmethod
    def _folders_are_named(cls, value: list[str]) -> list[str]:
        folders = [folder.strip() for folder in value]
        if any(not folder for folder in folders):
            raise ValueError(
                "every entry of source.folder_ids has to name a folder; one of them is "
                "empty, which is usually a list item left without a value"
            )
        return folders


class ReadSettings(ModuleSettings):
    pass


class DocaiReadSettings(ReadSettings):
    processor_id: str
    processor_version: str
    artifact_bucket: str
    page_limit: int = 200
    max_document_bytes: int = 1_073_741_824
    max_documents_per_batch: int = 1_000
    max_concurrent_batches: int = 5
    enable_table_annotation: bool = True
    enable_image_annotation: bool = True


def _registered(modules: "tuple[type[ModuleSettings], ...]") -> tuple[str, ...]:
    return tuple(get_args(module.model_fields["type"].annotation)[0] for module in modules)


SOURCE_MODULES: tuple[type[SourceSettings], ...] = (GDriveSourceSettings,)

REGISTERED_SOURCE_MODULES: tuple[str, ...] = _registered(SOURCE_MODULES)

SourceConfig = GDriveSourceSettings
ReadConfig = DocaiReadSettings


class ChunkSettings(ModuleSettings):
    pass


class DocaiChunkSettings(ChunkSettings):
    size: int
    include_headings: bool = True


CHUNK_MODULES: tuple[type[ChunkSettings], ...] = (DocaiChunkSettings,)
REGISTERED_CHUNK_MODULES: tuple[str, ...] = (CHUNK_DOCAI,)
READ_MODULES: tuple[type[ReadSettings], ...] = (DocaiReadSettings,)
REGISTERED_READ_MODULES: tuple[str, ...] = (READER_DOCAI,)

ChunkConfig = DocaiChunkSettings


class ProcessSettings(ModuleSettings):
    read: ReadSettings
    chunk: ChunkSettings


class DocaiProcessSettings(ProcessSettings):
    type: Literal[PROCESS_DOCAI]
    credentials_secret: str
    region: Literal["us", "eu"] = "us"
    read: DocaiReadSettings
    chunk: DocaiChunkSettings


class PdfTextReadSettings(ReadSettings):
    pass


class PdfTextChunkSettings(ChunkSettings):
    pass


class PdfTextProcessSettings(ProcessSettings):
    type: Literal[PROCESS_PDFTEXT]
    read: PdfTextReadSettings = Field(default_factory=PdfTextReadSettings)
    chunk: PdfTextChunkSettings = Field(default_factory=PdfTextChunkSettings)


PROCESS_MODULES: tuple[type[ProcessSettings], ...] = (
    DocaiProcessSettings,
    PdfTextProcessSettings,
)
REGISTERED_PROCESS_MODULES: tuple[str, ...] = _registered(PROCESS_MODULES)

ProcessConfig = Annotated[
    DocaiProcessSettings | PdfTextProcessSettings, Field(discriminator=DISCRIMINATOR)
]


class PromptSettings(ModuleSettings):
    query: str = ""
    document: str = ""


class RequestSettings(ModuleSettings):
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class EmbedSettings(ModuleSettings):
    model: str
    dimension: int
    context_window: int
    base_url: str
    credentials_secret: str
    allowed_hosts: list[str] = Field(default_factory=list)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    understood_models: list[str] = Field(default_factory=list)
    max_inputs_per_request: int = Field(default=2048, ge=1)
    provider: str = ""
    batch_size: int = 64
    request: RequestSettings = Field(default_factory=RequestSettings)


class DbSettings(ModuleSettings):
    url: str = Field(default="", repr=False)
    url_secret: str | None = None

    @model_validator(mode="after")
    def _one_connection(self) -> "DbSettings":
        if self.url and self.url_secret is not None:
            raise ValueError("choose exactly one of 'url' and 'url_secret'")
        if not self.url and self.url_secret is None:
            raise ValueError("missing required setting 'url' or 'url_secret'")
        return self


class SearchSettings(ModuleSettings):
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    index_search_effort: int = 40
    # How deep each retrieval leg goes before fusion, which is a separate knob
    # from how many results come back (AC-079). Read-time, like the rest.
    candidate_depth: int = 50
    max_candidate_depth: int = 500
    max_index_search_effort: int = 1_000

    max_filter_clauses: int = 16
    max_filter_depth: int = 3
    max_pattern_length: int = 200
    filter_timeout_ms: int = 2000


class RerankSettings(ModuleSettings):
    enabled: bool = False
    model: str = ""


class PriceEntryBase(ModuleSettings):
    amount: Decimal
    provider: str | None = None
    model: str | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _quoted(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError(
                f"price amount {value!r} is a number, which YAML reads as a binary double "
                f"and which rounds where money must not — quote it: \"{value}\""
            )
        return value


class PagePrice(PriceEntryBase):
    operation: Literal[PRICED_BY_PAGE]  # type: ignore[valid-type]
    unit: Literal[UNIT_PAGE]  # type: ignore[valid-type]


class TokenPrice(PriceEntryBase):
    operation: Literal[PRICED_BY_TOKEN]  # type: ignore[valid-type]
    unit: Literal[UNIT_TOKEN]  # type: ignore[valid-type]


class RequestPrice(PriceEntryBase):
    operation: Literal[PRICED_BY_REQUEST]  # type: ignore[valid-type]
    unit: Literal[UNIT_REQUEST]  # type: ignore[valid-type]


PriceEntry = Annotated[
    Union[PagePrice, TokenPrice, RequestPrice], Field(discriminator="operation")
]


class UsageSettings(ModuleSettings):
    raw_retention_days: int = 30
    record_question_text: bool = False
    # Decimal, not float: a price with eight decimal places read through a binary float is
    # the classic silent-rounding bug, and the column it lands in is numeric.
    prices: list[PriceEntry] = Field(default_factory=list)


class SearchWindow(ModuleSettings):
    limit: int = Field(default=10, ge=1)
    brief_length: int = Field(default=500, gt=4)


class FetchWindow(ModuleSettings):
    limit: int = Field(default=50, ge=1)
    passage_window: int = Field(default=3, ge=1)


class OperationLimit(ModuleSettings):
    limit: int = Field(ge=1)


class SurfaceDefaults(ModuleSettings):
    search: SearchWindow = SearchWindow()
    fetch: FetchWindow = FetchWindow()


class SurfaceLimits(ModuleSettings):
    search: OperationLimit = OperationLimit(limit=50)
    fetch: OperationLimit = OperationLimit(limit=50)


class ToolSurfaceDefaults(SurfaceDefaults):
    search: SearchWindow = SearchWindow(limit=10)
    fetch: FetchWindow = FetchWindow(limit=10)


class ToolSurfaceLimits(SurfaceLimits):
    search: OperationLimit = OperationLimit(limit=10)
    fetch: OperationLimit = OperationLimit(limit=10)


class McpSettings(ModuleSettings):
    credentials_secret: str
    defaults: ToolSurfaceDefaults = ToolSurfaceDefaults()
    limits: ToolSurfaceLimits = ToolSurfaceLimits()


class ApiSettings(ModuleSettings):
    credentials_secret: str
    allowed_hosts: list[str]
    defaults: SurfaceDefaults = SurfaceDefaults()
    limits: SurfaceLimits = SurfaceLimits()

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: object) -> object:
        if isinstance(value, str):
            return [host.strip() for host in value.split(",") if host.strip()]
        return value


class TelemetrySettings(ModuleSettings):
    client: str
    export_endpoint: str = ""


class ScheduleSettings(ModuleSettings):
    full_sweep: str = ""
    incremental: str = ""

    @field_validator("full_sweep", "incremental")
    @classmethod
    def _must_be_a_readable_schedule(cls, value: str) -> str:
        if not value:
            return value
        try:
            next_fire(value, now=datetime.now(timezone.utc))
        except InvalidScheduleError as exc:
            raise ValueError(str(exc)) from exc
        return value


_MODULES: tuple[tuple[str, type[ModuleSettings]], ...] = (
    ("source", SourceConfig),
    ("process", ProcessConfig),
    ("embed", EmbedSettings),
    ("db", DbSettings),
    ("search", SearchSettings),
    ("rerank", RerankSettings),
    ("api", ApiSettings),
    ("mcp", McpSettings),
    ("usage", UsageSettings),
    ("telemetry", TelemetrySettings),
    ("schedule", ScheduleSettings),
)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceConfig
    process: ProcessConfig
    embed: EmbedSettings
    db: DbSettings
    search: SearchSettings
    rerank: RerankSettings
    api: ApiSettings
    mcp: McpSettings
    usage: UsageSettings
    telemetry: TelemetrySettings
    schedule: ScheduleSettings

    @property
    def read(self) -> ReadSettings:
        return self.process.read

    @property
    def chunk(self) -> ChunkSettings:
        return self.process.chunk


class ConfigError(Exception):
    pass


def _resolved_value(name: str, value: Any, resolver: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return resolve_text(name, value, resolver)
    if isinstance(value, list):
        return [_resolved_value(f"{name}[{index}]", item, resolver) for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _resolved_value(f"{name}.{key}", item, resolver) for key, item in value.items()}
    if isinstance(value, ModuleSettings):
        return _resolved_section(name, value, resolver)
    return value


def _resolved_section(
    name: str, section: ModuleSettings, resolver: Callable[[str], str]
) -> ModuleSettings:
    updates = {
        field: _resolved_value(f"{name}.{field}", value, resolver)
        for field, value in section
    }
    return section.model_copy(update=updates)


def resolve_references(
    config: AppConfig,
    resolver: Callable[[str], str],
    *,
    sections: "Sequence[str] | None" = None,
) -> AppConfig:
    wanted = set(sections) if sections is not None else None
    updates = {
        name: _resolved_section(name, section, resolver)
        for name, section in config
        if isinstance(section, ModuleSettings) and (wanted is None or name in wanted)
    }
    return config.model_copy(update=updates)


def resolve_database_config(
    config: AppConfig, resolver: Callable[[str], str] | None = None,
) -> AppConfig:
    if config.db.url_secret is None:
        return config
    value = resolve_secrets(
        {"DB_URL_SECRET": config.db.url_secret},
        resolver if resolver is not None else resolver_over(),
    )["DB_URL_SECRET"]
    try:
        from psycopg.conninfo import conninfo_to_dict

        if not value or not value.startswith(("postgresql://", "postgres://")):
            raise ValueError("expected a PostgreSQL connection URL")
        conninfo_to_dict(value)
    except Exception:
        raise SecretResolutionError(
            "DB_URL_SECRET must resolve to a valid PostgreSQL connection URL (value omitted)"
        ) from None
    return config.model_copy(update={"db": DbSettings(url=value)})


def config_path_for(app: str) -> Path:
    if not app.strip() or "/" in app or app.startswith("."):
        raise ConfigError(
            f"{app!r} is not a deployment name; a name selects "
            f"{APPS_DIRECTORY.name}/<app>/{CONFIG_FILENAME} and carries no path of its own"
        )
    return APPS_DIRECTORY / app / CONFIG_FILENAME


def _config_layers(path: str | Path | None, app: str | None) -> tuple[Path, ...]:
    if os.environ.get(RETIRED_PATH_VARIABLE):
        raise ConfigError(
            f"${RETIRED_PATH_VARIABLE} was replaced by ${APP_VARIABLE}, which names the "
            f"deployment rather than a path — unset it and use "
            f"${APP_VARIABLE}=<app>, whose configuration is read from "
            f"{APPS_DIRECTORY.name}/<app>/{CONFIG_FILENAME}"
        )
    if path is not None:
        return (Path(path).expanduser(),)
    named = app if app is not None else os.environ.get(APP_VARIABLE, "")
    if not str(named).strip():
        available = ", ".join(sorted(
            entry.name for entry in APPS_DIRECTORY.glob("*")
            if entry.is_dir() and not entry.name.startswith("_")
        )) if APPS_DIRECTORY.is_dir() else ""
        known = f"; the deployments in this build are {available}" if available else ""
        raise ConfigError(
            f"no deployment was named: set ${APP_VARIABLE}=<app> or pass app= to "
            f"load_config(){known}"
        )
    own = config_path_for(str(named))
    base = config_path_for(BASE_APP)
    if str(named) == BASE_APP or not base.is_file():
        return (own,)
    return (base, own)


def _read_document(location: Path) -> dict[str, Any]:
    try:
        text = location.read_text()
    except OSError as exc:
        raise ConfigError(f"the configuration file {location} could not be read: {exc}") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigError(f"the configuration file {location} is not valid YAML{where}") from exc
    if document is None:
        raise ConfigError(f"the configuration file {location} is empty")
    if not isinstance(document, dict):
        raise ConfigError(
            f"the configuration file {location} has to be a mapping of section to "
            f"settings, and carries {type(document).__name__} instead"
        )
    return document


def merge_documents(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = {section: dict(values) for section, values in base.items()}
    for section, values in overlay.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = _merge_section(merged[section], values, (section,))
        else:
            merged[section] = values
    return merged


def _merge_section(
    base: dict[str, Any], overlay: dict[str, Any], path: tuple[str, ...]
) -> dict[str, Any]:
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in overlay.items():
        here = (*path, key)
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and here in SUB_SECTIONS:
            merged[key] = _merge_section(merged[key], value, here)
        else:
            merged[key] = value
    return merged


def _unknown_sections(document: dict[str, Any]) -> list[str]:
    claimed = {name for name, _ in _MODULES}
    return sorted(key for key in document if key not in claimed)


def _reference_problems(
    module: str, instance: ModuleSettings, location: Path, *, prefix: str = ""
) -> list[str]:
    problems: list[str] = []
    registered = ", ".join(f"{scheme}{SCHEME_SEPARATOR}" for scheme in REGISTERED_SCHEMES)
    for field, value in instance:
        if value is None:
            continue
        if isinstance(value, ModuleSettings):
            problems += _reference_problems(
                module, value, location, prefix=f"{prefix}{field}."
            )
            continue
        name = f"{prefix}{field}"
        if field.endswith(SECRET_FIELD_SUFFIX) and not is_reference(value):
            problems.append(
                f"[{module}] {location}: setting '{name}' has to name where its value "
                f"is kept, not carry it — a reference beginning {registered} (the value "
                f"is not shown, and this file is committed: if one was pasted here, "
                f"rotate it)"
            )
    return problems


_ADAPTERS: dict[str, TypeAdapter] = {}


def adapter_for(module: str) -> TypeAdapter:
    if module not in _ADAPTERS:
        _ADAPTERS[module] = TypeAdapter(dict(_MODULES)[module])
    return _ADAPTERS[module]


def _build_namespace(
    module: str, cls: Any, section: Any, location: Path
) -> tuple[ModuleSettings | None, list[str]]:
    if section is None:
        section = {}
    if not isinstance(section, dict):
        return None, [
            f"[{module}] {location}: the section has to be a mapping of setting to "
            f"value, and carries {type(section).__name__} instead"
        ]
    try:
        instance = adapter_for(module).validate_python(section)
    except ValidationError as exc:
        return None, [_format_errors(module, exc, location)]
    problems = _reference_problems(module, instance, location)
    return (None, problems) if problems else (instance, [])


def load_config(path: str | Path | None = None, *, app: str | None = None) -> AppConfig:
    layers = _config_layers(path, app)
    document: dict[str, Any] = {}
    for layer in layers:
        document = merge_documents(document, _read_document(layer))
    location = layers[-1]
    values: dict[str, ModuleSettings] = {}
    messages: list[str] = [
        f"[file] {location}: unrecognised section '{key}' — no module claims it; "
        f"the sections are {', '.join(name for name, _ in _MODULES)}"
        for key in _unknown_sections(document)
    ]
    for name, cls in _MODULES:
        instance, section_messages = _build_namespace(name, cls, document.get(name), location)
        messages.extend(section_messages)
        if instance is not None:
            values[name] = instance
    if messages:
        raise ConfigError("\n".join(messages))
    return AppConfig(**values)


def _registered_for(module: str) -> tuple[str, ...]:
    declared = dict(_MODULES)[module]
    variants = get_args(get_args(declared)[0]) if get_args(declared) else (declared,)
    return _registered(
        tuple(
            variant
            for variant in variants
            if hasattr(variant, "model_fields") and DISCRIMINATOR in variant.model_fields
        )
    )


def _format_errors(module: str, exc: ValidationError, location: Path) -> str:
    lines: list[str] = []
    slugs = set(_registered_for(module))
    for error in exc.errors():
        parts = [str(part) for part in error["loc"]]
        if parts and parts[0] in slugs:
            parts = parts[1:]
        setting = ".".join(parts)
        error_type = error["type"]
        if error_type == "missing":
            registered = _registered_for(module) if setting == DISCRIMINATOR else ()
            if registered:
                lines.append(
                    f"[{module}] {location}: missing required setting '{setting}', which "
                    f"chooses the module that runs this stage; registered modules are "
                    f"{', '.join(repr(slug) for slug in registered)}"
                )
            else:
                lines.append(f"[{module}] {location}: missing required setting '{setting}'")
        elif error_type == "union_tag_not_found":
            discriminator = error.get("ctx", {}).get("discriminator", "'type'").strip("'")
            lines.append(
                f"[{module}] {location}: missing required setting '{discriminator}', which "
                f"chooses the module that runs this stage; registered modules are "
                f"{', '.join(repr(slug) for slug in _registered_for(module))}"
            )
        elif error_type == "union_tag_invalid":
            lines.append(
                f"[{module}] {location}: setting 'type' rejected value {error.get('input')!r}; "
                f"registered modules are "
                f"{', '.join(repr(slug) for slug in _registered_for(module))}"
            )
        elif error_type == "extra_forbidden":
            lines.append(f"[{module}] {location}: unrecognised setting '{setting}'")
        elif error_type == "literal_error":
            expected = error.get("ctx", {}).get("expected", "")
            rejected = error.get("input")
            vocabulary = "registered modules are" if setting == "type" else "accepted values are"
            lines.append(
                f"[{module}] {location}: setting '{setting}' rejected value {rejected!r}; "
                f"{vocabulary} {expected}"
            )
        else:
            lines.append(f"[{module}] {location}: setting '{setting}': {error['msg']}")
    return "\n".join(lines)
