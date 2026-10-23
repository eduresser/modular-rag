"""Probe a deployment's embedding stage from a terminal.

Modes: ``request``, ``call``, ``providers``, ``roles``, ``batches``, ``neighbours``.

    make embed APP=example MODE=neighbours ARGS='--sample 0 --question "rescisão antecipada"'
    make embed APP=example MODE=providers ARGS="--repeat 8"
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import unquote

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import AppConfig, ConfigError, load_config  # noqa: E402
from src.embedding.batch import Group, plan_batches  # noqa: E402
from src.embedding.client import (  # noqa: E402
    DIMENSION_SETTING,
    INPUT_ROLE_STYLES,
    NORM_TOLERANCE,
    PERMITTED_EXTRA_PARAM_KEYS,
    QUERY_INSTRUCTION,
    ROLE_DOCUMENT,
    ROLE_QUERY,
    STYLE_INSTRUCTION,
    EmbeddingClient,
    EmbeddingError,
)
from src.embedding.cost import provider_of  # noqa: E402
from src.embedding.pinning import (  # noqa: E402
    FIELD_ROUTING,
    ROUTER_HOSTS,
    is_routed,
    pinned_client,
)
from src.fingerprint import compute_fingerprint  # noqa: E402
from src.models import Chunk, VersionStamps  # noqa: E402
from src.prices import PriceBook  # noqa: E402
from src.processing.artifact import ObjectStore  # noqa: E402
from src.processing.docai import chunks_from_document  # noqa: E402
from src.processing.to_chunks import MappingError  # noqa: E402
from src.sources.gdrive import service_account_token_provider  # noqa: E402

from tools.run_ingest import secret  # noqa: E402

SCOPE_STORAGE = "https://www.googleapis.com/auth/devstorage.read_write"

ARTIFACT_PREFIX = "parsed/"

DEFAULT_QUESTION = "Em que casos o contrato pode ser rescindido antecipadamente?"


def build_client(config: AppConfig, http: httpx.Client) -> EmbeddingClient:
    token = secret("RAG_EMBED_CREDENTIALS_SECRET", config.embed.credentials_secret)
    return pinned_client(config, http=http, token=lambda: token)


def build_store(config: AppConfig, http: httpx.Client) -> ObjectStore:
    bucket = getattr(config.read, "artifact_bucket", "")
    reference = getattr(config.process, "credentials_secret", "")
    if not bucket or not reference:
        raise SystemExit(
            f"this probe walks the reader's bucket, and process module "
            f"{config.process.type!r} keeps what it read in the archive itself — there is "
            "no bucket to list and no credential to open one with"
        )
    key = secret("RAG_READ_CREDENTIALS_SECRET", reference)
    return ObjectStore(
        bucket=bucket,
        http=http,
        token=service_account_token_provider(key, http=http, scopes=(SCOPE_STORAGE,)),
    )


@dataclass(frozen=True, slots=True)
class Kept:

    external_id: str
    reader_version: str
    chunks: tuple[Chunk, ...]
    refusal: str = ""


def kept_documents(config: AppConfig, store: ObjectStore) -> list[Kept]:
    fingerprint = compute_fingerprint(config)
    kept: list[Kept] = []
    for name in store.list_names(ARTIFACT_PREFIX):
        parts = name.split("/")
        if len(parts) != 3 or not parts[2].endswith(".json"):
            continue
        reader_version = unquote(parts[1])
        external_id = unquote(parts[2].removesuffix(".json"))
        payload = json.loads(store.get(None, store.uri_for(name)))
        try:
            chunks = chunks_from_document(
                payload,
                external_id=external_id,
                stamps=VersionStamps(reader_version=reader_version, fingerprint=fingerprint),
            ).chunks
        except MappingError as exc:
            kept.append(Kept(external_id, reader_version, (), refusal=str(exc)))
            continue
        kept.append(Kept(external_id, reader_version, chunks))
    return kept


def citation(chunk: Chunk) -> str:
    where = []
    if chunk.page_start is not None:
        pages = (
            f"p. {chunk.page_start}"
            if chunk.page_end in (None, chunk.page_start)
            else f"p. {chunk.page_start}–{chunk.page_end}"
        )
        where.append(pages)
    if chunk.heading_path:
        where.append(" › ".join(chunk.heading_path))
    return " | ".join(where) or "(sem âncora)"


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def tokens_cost(config: AppConfig, tokens: int) -> str:
    price = PriceBook(config.usage.prices).price_or_none(
        operation="embed",
        provider=provider_of(config.embed.base_url),
        model=config.embed.model,
        unit="token",
    )
    if price is None:
        return f"{tokens} token(s) · preço não configurado (RAG_USAGE_PRICES)"
    return f"{tokens} token(s) · {price.amount * tokens} {'USD'}"


def mode_request(config: AppConfig, args: argparse.Namespace) -> int:
    client = build_client(config, httpx.Client())
    print(f"endpoint      : {client.url}")
    print(f"é um router?  : {is_routed(config.embed.base_url)}  (routers: {', '.join(ROUTER_HOSTS)})")
    print(f"provider fixo : {config.embed.provider or '(NENHUM — ver o modo providers)'}")
    print(f"largura       : {config.embed.dimension}  ({DIMENSION_SETTING})")
    print(f"estilo de papel: {client.role_style}  (modelos registrados: {len(INPUT_ROLE_STYLES)})")
    print(f"extras permitidos: {', '.join(sorted(PERMITTED_EXTRA_PARAM_KEYS))}")
    print(f"extras configurados: {dict(config.embed.extra_params) or '{}'}")
    print(f"lote          : {config.embed.batch_size} peça(s) por requisição")
    for role in (ROLE_DOCUMENT, ROLE_QUERY):
        body = client.body(["Cláusula quinta: da rescisão antecipada."], role)
        print(f"\n--- corpo para um(a) {role} ---")
        print(json.dumps(body, indent=2, ensure_ascii=False)[:900])
    if client.role_style == STYLE_INSTRUCTION:
        print(f"\na instrução que vai na frente de uma pergunta:\n  {QUERY_INSTRUCTION!r}")
    return 0


def mode_call(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=60.0)
    client = build_client(config, http)
    text = args.question or "Cláusula quinta: da rescisão antecipada do contrato."
    try:
        result = client.embed_documents([text])
    except EmbeddingError as exc:
        print(f"recusado pelo cliente:\n  {exc}")
        return 1
    vector = result.vectors[0]
    print(f"modelo pedido : {config.embed.model}")
    print(f"modelo servido: {result.model}")
    print(f"largura       : {len(vector)}  (configurada: {config.embed.dimension})")
    print(f"norma         : {norm(vector):.9f}  (tolerância {NORM_TOLERANCE})")
    print(f"uso           : {tokens_cost(config, result.tokens)}")
    print(f"primeiras 6 componentes: {[round(c, 6) for c in vector[:6]]}")
    return 0


def mode_providers(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=60.0)
    token = secret("RAG_EMBED_CREDENTIALS_SECRET", config.embed.credentials_secret)
    client = build_client(config, http)
    body = client.body(["Cláusula quinta: da rescisão antecipada."], ROLE_DOCUMENT)
    seen: Counter[str] = Counter()
    widths: Counter[int] = Counter()
    for attempt in range(1, args.repeat + 1):
        response = http.post(
            client.url, json=body, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code >= 400:
            print(f" {attempt:>2}. {response.status_code}: {response.text[:120]}")
            continue
        payload = response.json()
        upstream = str(payload.get(FIELD_ROUTING) or "(não informado)")
        width = len(payload["data"][0]["embedding"])
        seen[upstream] += 1
        widths[width] += 1
        print(f" {attempt:>2}. {payload.get('model')} · {upstream} · {width} dims")
    print(f"\nprovedores que atenderam: {dict(seen)}")
    print(f"larguras devolvidas     : {dict(widths)}")
    if len(seen) > 1 and not config.embed.provider:
        print(
            "\n  RAG_EMBED_PROVIDER está vazio e mais de um provedor atendeu. Dois\n"
            "  provedores do mesmo modelo não produzem vetores idênticos, e a\n"
            "  truncagem para a largura configurada é implementação de cada um —\n"
            "  um acervo embedado por dois tem duas nuvens levemente desalinhadas\n"
            "  no mesmo espaço, e isso não dá sintoma: as buscas respondem, só um\n"
            "  pouco pior. Fixe um antes de indexar o acervo."
        )
    return 0


def mode_roles(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=60.0)
    client = build_client(config, http)
    text = args.question or DEFAULT_QUESTION
    try:
        as_passage = client.embed_documents([text]).vectors[0]
        as_question = client.embed_query(text)
    except EmbeddingError as exc:
        print(f"recusado pelo cliente:\n  {exc}")
        return 1
    print(f"texto: {text!r}")
    print(f"estilo deste modelo: {client.role_style}")
    print(f"\ncomo passagem, primeiras 4: {[round(c, 6) for c in as_passage[:4]]}")
    print(f"como pergunta, primeiras 4: {[round(c, 6) for c in as_question[:4]]}")
    print(f"\ncosseno entre os dois papéis: {cosine(as_passage, as_question):.6f}")
    if client.role_style == STYLE_INSTRUCTION:
        print(
            "\n  1.0 aqui significaria que a instrução não mudou nada, e a distinção\n"
            "  de papel seria decoração. Qualquer coisa abaixo disso é a diferença\n"
            "  que a ASM-033 diz que se perde ao fixar o papel num único valor."
        )
    else:
        print("\n  este modelo não oferece distinção por este endpoint; 1.0 é o esperado.")
    return 0


def mode_batches(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=120.0)
    kept = kept_documents(config, build_store(config, http))
    groups = tuple(
        Group(key=item.external_id, texts=tuple(chunk.content for chunk in item.chunks))
        for item in kept
        if item.chunks
    )
    for item in kept:
        if item.refusal:
            print(f" ! {item.external_id}: recusado no corte — {item.refusal[:110]}")
    if not groups:
        print("nenhum artefato com peças no bucket; rode o ingest primeiro")
        return 1
    batches = plan_batches(groups, batch_size=config.embed.batch_size)
    print(f"\n{len(groups)} documento(s) · {sum(len(g) for g in groups)} peça(s) · "
          f"lote configurado {config.embed.batch_size}")
    for position, batch in enumerate(batches, start=1):
        marks = ", ".join(
            f"{group.key[:16]}…({len(group)}{'  PARCIAL' if group.partial else ''})"
            for group in batch.groups
        )
        print(f" requisição {position:>3}: {len(batch):>4} peça(s) · {marks}")
    print(
        f"\n  {len(batches)} requisição(ões). Um lote que falha deixa cada peça dele\n"
        "  sem embedding, e um documento partido falha inteiro — é por isso que\n"
        "  RAG_EMBED_BATCH_SIZE é uma aposta sobre quanto uma recusa joga fora,\n"
        "  não um botão de vazão."
    )
    return 0


def mode_neighbours(config: AppConfig, args: argparse.Namespace) -> int:
    http = httpx.Client(timeout=120.0)
    client = build_client(config, http)
    kept = kept_documents(config, build_store(config, http))
    pieces: list[Chunk] = []
    for item in kept:
        if item.refusal:
            print(f" ! {item.external_id}: recusado no corte — {item.refusal[:110]}")
        pieces.extend(item.chunks)
    if not pieces:
        print("nenhuma peça derivável dos artefatos guardados")
        return 1
    sample = pieces
    if args.sample and len(pieces) > args.sample:
        random.seed(args.seed)
        sample = random.sample(pieces, args.sample)
    question = args.question or DEFAULT_QUESTION
    print(f"\n{len(pieces)} peça(s) no total · {len(sample)} embedada(s) nesta sonda")
    print(f"pergunta: {question!r}\n")
    try:
        passages = client.embed_documents([piece.content for piece in sample])
        asked = client.embed_query(question)
    except EmbeddingError as exc:
        print(f"recusado pelo cliente:\n  {exc}")
        return 1
    ranked = sorted(
        zip(sample, passages.vectors),
        key=lambda pair: cosine(asked, pair[1]),
        reverse=True,
    )
    for position, (piece, vector) in enumerate(ranked[: args.top], start=1):
        score = cosine(asked, vector)
        text = " ".join(piece.content.split())
        print(f" {position:>2}. {score:+.4f}  {piece.document_external_id[:14]}…  {citation(piece)}")
        print(f"     {text[:150]}")
    print(f"\nuso: {tokens_cost(config, passages.tokens + 0)} nas passagens")
    print(
        "  O cosseno vale como ordem, não como nota: o que se lê aqui é se a peça\n"
        "  que responde a pergunta está no topo, e se a citação ao lado dela\n"
        "  aponta para onde um advogado iria conferir."
    )
    return 0


MODES = {
    "request": mode_request,
    "call": mode_call,
    "providers": mode_providers,
    "roles": mode_roles,
    "batches": mode_batches,
    "neighbours": mode_neighbours,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="embed_probe.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=list(MODES), nargs="?", default="request")
    parser.add_argument("--question", default="", metavar="TEXTO",
                        help="a pergunta (neighbours/roles) ou o texto (call)")
    parser.add_argument("--sample", type=int, default=12, metavar="N",
                        help="quantas peças embedar em neighbours; 0 = todas")
    parser.add_argument("--top", type=int, default=8, metavar="N",
                        help="quantas vizinhas mostrar")
    parser.add_argument("--repeat", type=int, default=6, metavar="N",
                        help="quantas requisições em providers")
    parser.add_argument("--seed", type=int, default=0, help="semente da amostragem")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"a configuração não carregou:\n{exc}", file=sys.stderr)
        return 2
    try:
        return MODES[args.mode](config, args)
    except Exception as exc:  # noqa: BLE001 — a probe reports, never a traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("(uma referência nomeia seu backend: env: ou gcp-sm:)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
