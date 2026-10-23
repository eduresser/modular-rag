"""The process: `python -m src <role>`, one command per role (AC-176).

Two roles, a closed set (:data:`src.boot.ROLES`): ``serve`` is the surface that
answers requests over HTTP and the mounted tool transport; ``worker`` is the
one that fires this deployment's schedules and indexes the corpus. They are
separate processes on purpose (ASM-093, confirmed by Q-091) — a sweep that
saturates a container must not make a user's search time out, and the two
have different memory and restart profiles.

**A role nobody defined refuses immediately**, naming the roles that do exist,
rather than starting something arbitrary — :func:`src.boot.check_role` is
called before anything is loaded or constructed.

**Each role reports which deployment and which role it is** before it serves
or indexes, so a log from a container tells an operator what it is looking at
without them having to already know the platform's own start command.

Building this deployment's real collaborators — the Drive source, the reader,
the object store, the pinned embedding client — is this module's own job
rather than something imported from a factory elsewhere: no other module in
``src`` wires all of them together, and this is the one process that needs to.
Every class it reaches for is already proven by its own feature; what is new
here is the order they are built in and the single command that starts them.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from src.boot import (
    ROLE_SERVE,
    ROLE_WORKER,
    ROLES,
    UnknownRole,
    check_boot_compatibility,
    check_role,
)
from src.config import AppConfig, ConfigError, load_config, resolve_database_config, resolve_references
from src.db import connect, create_schema, stored_dimension
from src.main import main_app
from src.secrets import SecretResolutionError, resolver_over
from src.worker import run_schedule

if TYPE_CHECKING:
    from fastapi import FastAPI

    from src.ingest import Pipeline

logger = logging.getLogger(__name__)

# What the READ stage's own key is exchanged for, one scope per collaborator,
# so the object store and the reader each ask for no more than they use. A
# deployment may point `read.credentials_secret` at the same variable `source`
# names — this one does — but the stage is what states it, and these two scopes
# are the read stage's, never the source's.
SCOPE_STORAGE = "https://www.googleapis.com/auth/devstorage.read_write"
SCOPE_CLOUD = "https://www.googleapis.com/auth/cloud-platform"


def _announce(role: str, config: AppConfig) -> None:
    """What this process is, before it serves or indexes (AC-176)."""
    logger.info("starting role %r for deployment %r", role, config.telemetry.client)


# ── The surface role ───────────────────────────────────────────────────────


def main_serve(
    *,
    resolver: Callable[[str], str] | None = None,
    http: httpx.Client | None = None,
    app: str | None = None,
    path: str | None = None,
    serve: "Callable[[FastAPI, AppConfig], None] | None" = None,
) -> None:
    """Boot the surface role and answer requests until the platform stops it.

    Everything about building the application is already written
    (:func:`src.main.main_app`); what this adds is the announcement (AC-176)
    and handing the assembled application to something that actually listens.
    ``serve`` is injectable so a test can capture what it would have served
    without binding a socket — the same shape every other constructor in this
    module takes its collaborators as arguments for.
    """
    fastapi_app = main_app(resolver=resolver, http=http, app=app, path=path)
    config: AppConfig = fastapi_app.state.service.config
    _announce(ROLE_SERVE, config)
    (serve if serve is not None else _listen)(fastapi_app, config)


def _listen(app: "FastAPI", config: AppConfig) -> None:
    """Actually answer requests. ``$PORT`` is the platform's own convention."""
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


# ── The worker role ────────────────────────────────────────────────────────


def _paid_reader(
    config: AppConfig,
    *,
    http: httpx.Client,
    read_credentials: str,
    store: "ObjectStore",
) -> "DocumentAIReader":
    """The paid module, wired from its own half of the deployment's file.

    A function of its own so that :func:`build_pipeline` reads as the one
    decision it makes — which module runs — rather than as one module's wiring
    with a second module appended to it.
    """
    from src.ingest import DocumentAIReader
    from src.processing.docai import ChunkingOptions, LayoutParser
    from src.sources.gdrive import service_account_token_provider

    return DocumentAIReader(
        reader=LayoutParser(
            processor_id=config.read.processor_id,
            processor_version=config.read.processor_version,
            http=http,
            token=service_account_token_provider(read_credentials, http=http, scopes=(SCOPE_CLOUD,)),
            chunking=ChunkingOptions(
                size=config.chunk.size,
                include_ancestor_headings=config.chunk.include_headings,
                cutter=config.process.type,
            ),
            region=config.process.region,
            enable_table_annotation=config.read.enable_table_annotation,
            enable_image_annotation=config.read.enable_image_annotation,
            max_concurrent_batches=config.read.max_concurrent_batches,
            # The dataclass default is a no-op sleep, which would spin the poll
            # loop instead of waiting for a batch that takes minutes.
            sleeper=time.sleep,
        ),
        store=store,
        max_documents_per_batch=config.read.max_documents_per_batch,
        page_limit=config.read.page_limit,
        max_document_bytes=config.read.max_document_bytes,
    )


def build_pipeline(
    config: AppConfig,
    *,
    http: httpx.Client,
    source_credentials: str,
    read_credentials: str,
    embed_token: str,
    checkpoint: Callable[[], None] | None = None,
) -> "Pipeline":
    """This deployment's real collaborators, wired from its own configuration.

    Takes the credentials already resolved rather than a resolver of its own,
    so the boot order stays "resolve, then construct" (AC-177's shape) instead
    of resolving again, quietly, from inside a constructor.

    **One credential per stage, and the argument list is where that is
    enforced.** ``source_credentials`` reaches Drive; ``read_credentials``
    reaches the Layout Parser and the bucket its output is kept in. This
    deployment happens to resolve both from the same variable, and they are
    still two arguments: a reader arriving with a key of its own has a
    parameter to be handed it through, where before it would have been given
    the source's in silence.

    Every class here is a real, already-proven one — never imported by a name
    configuration supplied (principle P-011): ``config.source.type`` and
    ``config.process.type`` are read as plain values, not resolved into symbols.
    A third module registered tomorrow changes what this function dispatches on,
    not what a deployment's file is allowed to say.

    **The two process modules take different arguments and that is the point.**
    The paid one needs a credential that reaches a provider, a pinned processor
    and a region; the free one needs the bucket and nothing else, because
    nothing leaves this process. ``read_credentials`` reaches both, and for the
    free module it opens only the bucket the artifact is kept in (P-009).
    """
    from src.config import PROCESS_DOCAI
    from src.embedding.pinning import pinned_client
    from src.ingest import LocalTextReader, Pipeline
    from src.prices import PriceBook
    from src.processing.artifact import ArchiveStore, ArtifactStore, ObjectStore
    from src.sources.gdrive import SCOPE_READONLY, GoogleDriveSource, service_account_token_provider
    from src.sources.gdrive_activity import SCOPE_ACTIVITY_READONLY, ActivityHistory
    from src.sources.gdrive_resume import DriveResume, service_account_identity
    from src.sources.gdrive_throttle import QuotaBudget

    # Where this deployment keeps what it read, which is the chosen module's
    # own answer (P-009 says THAT it is kept, never where). The paid route
    # provisions a bucket because the provider's batch path writes into one;
    # the free route reaches nothing and keeps its output in the archive it
    # already has — which is why it names no credential (AC-272).
    if config.process.type == PROCESS_DOCAI:
        store: ArtifactStore = ObjectStore(
            bucket=config.read.artifact_bucket,
            http=http,
            token=service_account_token_provider(
                read_credentials, http=http, scopes=(SCOPE_STORAGE,)
            ),
        )
        reader = _paid_reader(
            config, http=http, read_credentials=read_credentials, store=store
        )
    else:
        store = ArchiveStore()
        reader = LocalTextReader(store=store)
    source = GoogleDriveSource(
        folder_ids=config.source.folder_ids,
        http=http,
        token=service_account_token_provider(source_credentials, http=http, scopes=(SCOPE_READONLY,)),
        budget=QuotaBudget(units_per_minute=config.source.units_per_minute),
    )
    return Pipeline(
        config=config,
        source=source,
        # What the source says actually happened, which is what turns a
        # disappearance into a deletion or into a veto (AC-012, AC-013). Its
        # own narrow grant: one key serves several services here and each gets
        # the scope it needs and no more.
        # The folders asked about are the source's OWN — the listed ones with
        # those already covered by another removed — so a folder listed inside
        # another costs no extra query and yields no duplicated event. Reading
        # it here rather than from the configuration is also what keeps one
        # answer to what the watched set is; the lambda only runs when a
        # reconciliation needs the history, so resolving it costs nothing until
        # then.
        activity=lambda since: ActivityHistory(
            folder_ids=source.folders,
            http=http,
            token=service_account_token_provider(
                source_credentials, http=http, scopes=(SCOPE_ACTIVITY_READONLY,)
            ),
        ).events_since(since)
        if since is not None
        else (),
        # Drive's own rules for its own cursor (AC-026). Built here because
        # this is where the credential exists in the clear, and the identity a
        # position belongs to is read from it rather than configured twice.
        resume=DriveResume(
            source=source,
            identity=service_account_identity(source_credentials),
            # The CONFIGURED folders, not the resolved ones: whether the stored
            # position still describes this corpus has to be answerable before
            # the run decides anything, and asking the source would spend a
            # call to find out it is going to sweep anyway.
            folders=source.configured_folders,
        ),
        reader=reader,
        store=store,
        embedder=pinned_client(config, http=http, token=lambda: embed_token),
        prices=PriceBook(config.usage.prices),
        # How a real run makes progress durable. Nothing under ``src/`` calls
        # ``conn.commit()`` — the suite isolates a test as one transaction it
        # rolls back, and every store function nests inside it — so this is
        # where the decision is taken instead, by the role that owns the
        # connection. Left out, the pipeline is the one the tests build: it
        # writes exactly the same rows and commits none of them.
        **({"checkpoint": checkpoint} if checkpoint is not None else {}),
    )


#: The stages the indexing role calls out to. It answers no request, so it is
#: never handed the credential that opens the API — the mirror of
#: :data:`src.main.SERVING_SECTIONS`.
INDEXING_SECTIONS: tuple[str, ...] = ("source", "process", "embed", "db")


def main_worker(
    *,
    resolver: Callable[[str], str] | None = None,
    http: httpx.Client | None = None,
    app: str | None = None,
    path: str | None = None,
    should_stop: Callable[[], bool] | None = None,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> None:
    """Boot the worker role and run its schedule until asked to stop.

    The order is the point (T-183's own note): read the file, resolve the
    credentials, check compatibility against the live schema, apply it, and
    only then construct the pipeline that spends money. Each step already
    exists — :func:`~src.config.load_config`, :func:`~src.secrets.resolve_secrets`,
    :func:`~src.boot.check_boot_compatibility`, :func:`~src.db.create_schema` —
    and this function is what puts them in one order instead of leaving it to
    whoever calls them next to reinvent.
    """
    config = resolve_database_config(load_config(path, app=app), resolver)
    _announce(ROLE_WORKER, config)

    # One pass over the whole document (AC-174), which also settles what the
    # hand-written list had to say out loud: a stage that names no credential is
    # not asked for one. The free process module reaches no provider and keeps
    # what it read in the archive itself, so it carries no reference and there
    # is nothing under it to resolve (AC-272) — a walk finds that by finding
    # nothing, where a list had to remember to leave it out.
    config = resolve_references(
        config, resolver if resolver is not None else resolver_over(), sections=INDEXING_SECTIONS
    )
    http = http if http is not None else httpx.Client(timeout=300.0)

    with connect(config.db.url) as conn:
        check_boot_compatibility(config, stored_dimension=stored_dimension(conn))
        create_schema(conn, dimension=config.embed.dimension)
        conn.commit()

        pipeline = build_pipeline(
            config,
            http=http,
            source_credentials=config.source.credentials_secret,
            read_credentials=getattr(config.process, "credentials_secret", ""),
            embed_token=config.embed.credentials_secret,
            checkpoint=conn.commit,
        )
        schedule_kwargs: dict[str, object] = {
            "should_stop": should_stop if should_stop is not None else _signal_stop_flag(),
        }
        if clock is not None:
            schedule_kwargs["clock"] = clock
        if sleeper is not None:
            schedule_kwargs["sleeper"] = sleeper
        run_schedule(conn, pipeline, **schedule_kwargs)
        conn.commit()


def _signal_stop_flag() -> Callable[[], bool]:
    """A flag a termination signal sets, and nothing else touches (ASM-094).

    The handler only sets the flag — finishing cleanly stays ``run_schedule``'s
    job. A killed process loses nothing ``resume()`` cannot take up, so this is
    an optimisation over that guarantee, never a second implementation of it.
    """
    import signal
    import threading

    stopping = threading.Event()

    def _handle(signum: int, frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    return stopping.is_set


# ── The command ────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src <role>``. A role nobody defined refuses immediately."""
    args = list(sys.argv[1:] if argv is None else argv)
    role = args[0] if args else ""
    try:
        check_role(role)
    except UnknownRole as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        (main_serve if role == ROLE_SERVE else main_worker)()
    except (ConfigError, SecretResolutionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        logger.error("role startup failed: role=%s error_type=%s", role, type(exc).__name__)
        print("The service could not start; check the deployment configuration and logs.", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "ROLES",
    "ROLE_SERVE",
    "ROLE_WORKER",
    "build_pipeline",
    "main",
    "main_serve",
    "main_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
