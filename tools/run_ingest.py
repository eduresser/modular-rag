"""Drive one deployment's ingest from a terminal.

Modes: ``enumerate`` (list the source tree), ``fetch`` (also download and screen
each document), ``ingest`` (full pipeline; paid, asks before starting).

    make enumerate APP=example
    make fetch APP=example ARGS="--limit 3"
    make ingest APP=example ARGS="--yes --limit 3"
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import AppConfig, ConfigError, load_config, resolve_database_config  # noqa: E402
from src.processing.preflight import Candidate, screen  # noqa: E402
from src.secrets import (  # noqa: E402
    REGISTERED_SCHEMES,
    SecretResolutionError,
    resolve_secrets,
    resolver_over,
)
from src.sources.base import DocumentUnavailable, SourceDocument, SourceError  # noqa: E402
from src.sources.gdrive import (  # noqa: E402
    ALL_DRIVES,
    API_ROOT,
    SCOPE_READONLY,
    GoogleDriveSource,
    service_account_token_provider,
)
from src.sources.gdrive_export import (  # noqa: E402
    ExportTooLarge,
    UnsupportedKind,
    export_plan,
)
from src.sources.gdrive_throttle import QuotaBudget  # noqa: E402


def secret(name: str, reference: str) -> str:
    return resolve_secrets({name: reference}, resolver_over())[name]


def drive_token(config: AppConfig, http: httpx.Client, *, scopes: tuple[str, ...]):
    key = secret("RAG_SOURCE_CREDENTIALS_SECRET", config.source.credentials_secret)
    return service_account_token_provider(key, http=http, scopes=scopes)


def build_source(config: AppConfig, http: httpx.Client, token) -> GoogleDriveSource:
    return GoogleDriveSource(
        folder_ids=config.source.folder_ids,
        http=http,
        token=token,
        budget=QuotaBudget(units_per_minute=config.source.units_per_minute),
    )


class FolderPaths:

    def __init__(self, *, http: httpx.Client, token, roots: Sequence[str]) -> None:
        self._http = http
        self._token = token
        self._roots = set(roots)
        self._names: dict[str, str] = {}
        self._parents: dict[str, str | None] = {}

    def path_of(self, document: SourceDocument) -> str:
        if not document.parents:
            return "(fora da árvore varrida)"
        parts: list[str] = []
        folder: str | None = document.parents[0]
        seen: set[str] = set()
        while folder and folder not in seen:
            seen.add(folder)
            if folder in self._roots:
                parts.append(self._name_of(folder) + " ←raiz")
                break
            parts.append(self._name_of(folder))
            folder = self._parent_of(folder)
        return "/".join(reversed(parts))

    def _name_of(self, folder: str) -> str:
        if folder not in self._names:
            self._load(folder)
        return self._names.get(folder, folder)

    def _parent_of(self, folder: str) -> str | None:
        if folder not in self._parents:
            self._load(folder)
        return self._parents.get(folder)

    def _load(self, folder: str) -> None:
        response = self._http.get(
            f"{API_ROOT}/files/{folder}",
            params={"fields": "id,name,parents", **ALL_DRIVES},
            headers={"Authorization": f"Bearer {self._token()}"},
        )
        if response.status_code >= 400:
            self._names[folder] = f"<{folder}: {response.status_code}>"
            self._parents[folder] = None
            return
        entry = response.json()
        self._names[folder] = str(entry.get("name", folder))
        parents = entry.get("parents") or []
        self._parents[folder] = str(parents[0]) if parents else None


@dataclass(frozen=True, slots=True)
class Mapped:

    document: SourceDocument
    path: str
    retrieval: str
    media_type: str
    byte_size: int | None
    verdict: str
    detail: str = ""

    @property
    def size_is_known(self) -> bool:
        return self.byte_size is not None


def map_document(document: SourceDocument, path: str) -> Mapped:
    try:
        plan = export_plan(document.kind)
    except UnsupportedKind as exc:
        return Mapped(
            document=document,
            path=path,
            retrieval="—",
            media_type=document.kind,
            byte_size=document.size_bytes,
            verdict="SEM MAPEAMENTO",
            detail=str(exc),
        )
    candidate = Candidate(
        document_id=0,
        external_id=document.external_id,
        media_type=plan.media_type,
        byte_size=document.size_bytes or 0,
    )
    screening = screen([candidate])
    if screening.rejected:
        return Mapped(
            document=document,
            path=path,
            retrieval=plan.retrieval.value,
            media_type=plan.media_type,
            byte_size=document.size_bytes,
            verdict="RECUSADO",
            detail=screening.rejected[0].reason,
        )
    return Mapped(
        document=document,
        path=path,
        retrieval=plan.retrieval.value,
        media_type=plan.media_type,
        byte_size=document.size_bytes,
        verdict="aceito",
    )


SHORT_MEDIA: dict[str, str] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
    "application/pdf": "PDF",
    "text/html": "HTML",
    "text/plain": "TXT",
    "text/markdown": "MD",
    "text/csv": "CSV",
}


def short(media_type: str) -> str:
    known = SHORT_MEDIA.get(media_type)
    if known is not None:
        return known
    return media_type.rsplit(".", 1)[-1].rsplit("/", 1)[-1][:6]


def size_of(byte_size: int | None) -> str:
    if byte_size is None:
        return "—"
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if byte_size >= scale:
            return f"{byte_size / scale:.1f} {unit}"
    return f"{byte_size} B"


def report_tree(mapped: Sequence[Mapped]) -> None:
    by_path: dict[str, list[Mapped]] = {}
    for item in mapped:
        by_path.setdefault(item.path, []).append(item)
    for path in sorted(by_path):
        items = sorted(by_path[path], key=lambda item: item.document.name.lower())
        print(f"\n  {path}/   ({len(items)})")
        for item in items:
            marker = " " if item.verdict == "aceito" else "!"
            print(
                f"   {marker} {item.document.name[:52]:<52} "
                f"{short(item.media_type):<6} {item.retrieval:<8} "
                f"{size_of(item.byte_size):>9}  {item.verdict}"
            )
            if item.detail:
                print(f"       └─ {item.detail}")


def report_totals(mapped: Sequence[Mapped]) -> None:
    folders = {item.path for item in mapped}
    accepted = [item for item in mapped if item.verdict == "aceito"]
    refused = [item for item in mapped if item.verdict != "aceito"]
    unknown_size = [item for item in accepted if not item.size_is_known]
    print(f"\n  {len(mapped)} documento(s) em {len(folders)} pasta(s)")
    print(f"  {len(accepted)} o leitor aceita · {len(refused)} não")
    if unknown_size:
        print(
            f"  {len(unknown_size)} sem tamanho conhecido (nativos do Drive): o teto de "
            "10 MB do export só é verificável no modo fetch"
        )
    for item in refused:
        print(f"  ! {item.document.name}: {item.verdict}")


def sweep_mapped(
    source: GoogleDriveSource, paths: FolderPaths, *, limit: int | None
) -> list[Mapped]:
    mapped: list[Mapped] = []
    for position, document in enumerate(source.sweep(), start=1):
        mapped.append(map_document(document, paths.path_of(document)))
        if limit is not None and position >= limit:
            break
    return mapped


def mode_enumerate(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=60.0)
    token = drive_token(config, http, scopes=(SCOPE_READONLY,))
    source = build_source(config, http, token)
    paths = FolderPaths(http=http, token=token, roots=source.folders)
    print(f"varrendo {', '.join(source.folders)} …")
    mapped = sweep_mapped(source, paths, limit=args.limit)
    if not mapped:
        print(
            "\n  nada. As pastas listadas estão vazias — se alguma não estivesse "
            "compartilhada com a service account, a varredura teria parado "
            "nomeando qual (AC-238), em vez de chegar aqui como listagem vazia."
        )
        return 1
    report_tree(mapped)
    report_totals(mapped)
    if args.limit is not None and len(mapped) >= args.limit:
        print(f"  parado em --limit {args.limit}: pode haver mais")
    print(f"\n  posição do feed agora: {source.start_position()}")
    return 0


def mode_fetch(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=120.0)
    token = drive_token(config, http, scopes=(SCOPE_READONLY,))
    source = build_source(config, http, token)
    paths = FolderPaths(http=http, token=token, roots=source.folders)
    print(f"varrendo e baixando de {', '.join(source.folders)} …")
    mapped = sweep_mapped(source, paths, limit=args.limit)
    names = {item.document.external_id: item.document.name for item in mapped}
    candidates: list[Candidate] = []
    unmapped: list[Mapped] = []
    failures: list[str] = []
    for position, item in enumerate(mapped, start=1):
        if item.verdict == "SEM MAPEAMENTO":
            unmapped.append(item)
            continue
        try:
            fetched = source.fetch(item.document)
        except (ExportTooLarge, DocumentUnavailable, SourceError) as exc:
            failures.append(f"{item.document.name}: {exc}")
            continue
        print(
            f"   {item.document.name[:52]:<52} {short(fetched.media_type):<6} "
            f"{size_of(len(fetched.content)):>9}  {fetched.content_digest[:12]}…"
        )
        candidates.append(
            Candidate(
                document_id=position,
                external_id=item.document.external_id,
                media_type=fetched.media_type,
                byte_size=len(fetched.content),
            )
        )
    screening = screen(
        candidates,
        page_limit=config.read.page_limit,
        byte_limit=config.read.max_document_bytes,
    )
    print(
        f"\n  {len(screening.accepted)} iria(m) ao leitor · "
        f"{len(screening.rejected)} recusado(s) · {len(unmapped)} sem mapeamento de export"
    )
    for verdict in screening.rejected:
        print(f"  ! {names.get(verdict.candidate.external_id, verdict.candidate.external_id)}: "
              f"{verdict.reason}")
    for item in unmapped:
        print(f"  ! {item.document.name}: {item.detail}")
    for failure in failures:
        print(f"  ✗ {failure}")
    return 1 if failures else 0


@dataclass
class WatchedSource:

    source: GoogleDriveSource
    limit: int | None = None
    names: dict[str, str] = field(default_factory=dict)

    def name_of(self, external_id: str) -> str:
        return self.names.get(external_id, external_id)

    def sweep(self) -> Iterator[SourceDocument]:
        for position, document in enumerate(self.source.sweep(), start=1):
            self.names[document.external_id] = document.name
            yield document
            if self.limit is not None and position >= self.limit:
                return

    def changes_since(self, position: str | None):
        page = self.source.changes_since(position)
        self.names.update({document.external_id: document.name for document in page.documents})
        return page

    def start_position(self) -> str:
        return self.source.start_position()

    def fetch(self, document: SourceDocument):
        return self.source.fetch(document)

    def probe(self, external_id: str) -> SourceDocument:
        document = self.source.probe(external_id)
        self.names[document.external_id] = document.name
        return document


def mode_ingest(config: AppConfig, args: argparse.Namespace) -> int:
    from dataclasses import replace

    from src.__main__ import build_pipeline
    from src.boot import check_boot_compatibility
    from src.db import connect, create_schema, stored_dimension
    from src.worker import sweep

    if not args.yes:
        print(
            "este modo paga: o leitor cobra por página e o provedor de embedding por "
            "token. Reexecute com --yes (e considere --limit)."
        )
        return 2

    http = httpx.Client(timeout=300.0)
    source_credentials = secret("RAG_SOURCE_CREDENTIALS_SECRET", config.source.credentials_secret)
    read_credentials = secret("RAG_READ_CREDENTIALS_SECRET", config.process.credentials_secret)
    embed_token = secret("RAG_EMBED_CREDENTIALS_SECRET", config.embed.credentials_secret)

    config = resolve_database_config(config)
    with connect(config.db.url) as conn:
        check_boot_compatibility(config, stored_dimension=stored_dimension(conn))
        create_schema(conn, dimension=config.embed.dimension)
        conn.commit()

        pipeline = build_pipeline(
            config,
            http=http,
            source_credentials=source_credentials,
            read_credentials=read_credentials,
            embed_token=embed_token,
            checkpoint=conn.commit,
        )
        watched = WatchedSource(pipeline.source, limit=args.limit)
        outcome = sweep(conn, replace(pipeline, source=watched))
        conn.commit()

    report_run(outcome, watched, run_detail(config, outcome.run_id))
    return 0 if outcome.completed and not outcome.errored else 1


def run_detail(config: AppConfig, run_id: int) -> str:
    from src.db import DatabaseUnavailable, connect

    try:
        config = resolve_database_config(config)
        with connect(config.db.url) as conn:
            row = conn.execute("select detail from sync_runs where id = %s", (run_id,)).fetchone()
    except (DatabaseUnavailable, Exception):  # noqa: BLE001 — a report, never the run
        return ""
    return str(row[0]) if row and row[0] else ""


def report_run(outcome, watched: "WatchedSource", detail: str = "") -> None:
    print(f"\nrun {outcome.run_id} — {'completo' if outcome.completed else 'INTERROMPIDO'}")
    if detail:
        print(f"  motivo: {detail}")
    counters = outcome.counters
    print(
        f"  vistos {counters.documents_seen} · novos {counters.documents_added} · "
        f"atualizados {counters.documents_updated} · pulados {counters.documents_skipped} · "
        f"com erro {counters.documents_errored}"
    )
    for run in outcome.runs:
        for result in run.results:
            name = watched.name_of(result.external_id)[:44]
            if result.failure is None:
                mark = "·" if result.duplicate else " "
                print(f" {mark} {name:<44} {result.action:<8} {result.pieces:>4} peça(s)")
                continue
            failure = result.failure
            print(f" ✗ {name:<44} {failure.stage:<8} {failure.kind}")
            print(f"     └─ {failure.reason}")
    records = [record for run in outcome.runs for record in run.failures.failures]
    if records:
        transient = [record for record in records if record.retryable]
        print(
            f"\n  {len(records)} falha(s): {len(transient)} transiente(s), que a próxima "
            f"execução reabre por conta própria, e {len(records) - len(transient)} "
            "permanente(s), que ficam liquidadas e não são retentadas."
        )
        stages = sorted({record.stage for record in records})
        print(f"  etapa(s) onde pararam: {', '.join(stages)}")
        print(
            "  a recusa de uma leitura também fica em document_reads (estado, versão do "
            "leitor e motivo), que é onde olhar depois:\n"
            "    psql \"$RAG_DB_URL\" -c \"select external_id, state, reader_version, "
            'reason from document_reads r join documents d on d.id = r.document_id"'
        )

MODES = {"enumerate": mode_enumerate, "fetch": mode_fetch, "ingest": mode_ingest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_ingest.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=sorted(MODES), nargs="?", default="enumerate")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="pare depois de N documentos (recomendado antes de qualquer modo que paga)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="confirma que o modo ingest pode gastar",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit tem de ser 1 ou mais")

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"a configuração não carregou:\n{exc}", file=sys.stderr)
        return 2
    try:
        return MODES[args.mode](config, args)
    except SecretResolutionError as exc:
        print(f"{exc}\n(uma referência nomeia seu backend: "
              f"{', '.join(f'{s}:' for s in REGISTERED_SCHEMES)})", file=sys.stderr)
        return 2
    except SourceError as exc:
        print(f"a fonte parou: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
