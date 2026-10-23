"""Show which vector generation the archive serves, and move the pointer to another.

    make cutover APP=example
    make cutover APP=example ARGS="--to qwen/qwen3-embedding-4b"
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import AppConfig, ConfigError, load_config, resolve_database_config  # noqa: E402
from src.secrets import SecretResolutionError  # noqa: E402
from src.db import DatabaseUnavailable, connect  # noqa: E402
from src.store import ModelStamp  # noqa: E402
from src.store_versions import (  # noqa: E402
    activate_version,
    current_version,
    discard_other_generations,
    pieces_under,
)


def generations(conn) -> list[tuple[str, int, int]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT embed_model, embed_dimension, count(*)
              FROM chunks
             GROUP BY embed_model, embed_dimension
             ORDER BY count(*) DESC, embed_model
            """
        )
        return [(row[0], int(row[1]), int(row[2])) for row in cursor.fetchall()]


def report(conn, config: AppConfig) -> int:
    active = current_version(conn)
    configured = ModelStamp(model=config.embed.model, dimension=config.embed.dimension)
    held = generations(conn)
    if not held:
        print("o arquivo não tem peça nenhuma — nada para ativar.")
        return 1
    print("gerações no arquivo:")
    for model, dimension, count in held:
        marks = []
        if active is not None and (model, dimension) == (active.model, active.dimension):
            marks.append("SERVINDO")
        if (model, dimension) == (configured.model, configured.dimension):
            marks.append("configurada")
        print(f"  {model:32} {dimension:>5}d  {count:>7} peça(s)  {' · '.join(marks)}")
    if active is None:
        print("\nnenhuma geração está sendo servida: toda busca devolve vazio.")
        return 1
    if (active.model, active.dimension) == (configured.model, configured.dimension):
        print("\na geração servida é a que o arquivo do deployment nomeia — nada a fazer.")
        return 0
    print(
        f"\nO arquivo nomeia {configured.model!r} a {configured.dimension}d e o arquivo de "
        f"vetores serve {active.model!r} a {active.dimension}d. Enquanto discordarem a busca "
        "recusa, porque a pergunta seria embedada com um modelo e comparada com vetores de "
        f"outro (AC-052).\n\n  make cutover APP=<app> ARGS=\"--to {configured.model}\""
    )
    return 1


def switch(conn, config: AppConfig, model: str, *, keep: bool) -> int:
    stamp = ModelStamp(model=model, dimension=config.embed.dimension)
    count = pieces_under(conn, stamp)
    if count == 0:
        print(
            f"recusado: nenhuma peça foi escrita sob {model!r} a {config.embed.dimension}d, "
            "então ativá-la esvaziaria o arquivo sem esvaziar tabela nenhuma.\n"
            "as gerações que existem:"
        )
        for held_model, dimension, held in generations(conn):
            print(f"  {held_model:32} {dimension:>5}d  {held:>7} peça(s)")
        return 1
    previous = activate_version(conn, stamp)
    gone = {} if keep else discard_other_generations(conn, stamp)
    conn.commit()
    was = f"{previous.model!r} a {previous.dimension}d" if previous else "nada"
    print(f"servindo {model!r} a {stamp.dimension}d ({count} peças). antes: {was}.")
    if gone:
        for dropped, pieces in sorted(gone.items(), key=lambda item: item[0].model):
            print(f"  descartada: {dropped.model} a {dropped.dimension}d ({pieces} peças)")
        print(
            "  voltar atrás agora custa embedar o corpus de novo — nunca uma releitura, "
            "porque a saída do leitor continua guardada (P-009)."
        )
        try:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute("VACUUM (ANALYZE) chunks")
        except Exception as error:  # noqa: BLE001 — tidying, never the switch
            print(f"  (o vacuum não rodou: {error}; o espaço fica ocupado até o próximo)")
        else:
            print(
                "  o espaço das linhas foi liberado; o índice de vetores só encolhe com um\n"
                "  REINDEX INDEX CONCURRENTLY chunks_embedding_idx  (pesado, escolha a janela)"
            )
        finally:
            conn.autocommit = False
    elif previous is not None:
        print(f"gerações anteriores mantidas (--keep). para desfazer:")
        print(f"  make cutover APP=<app> ARGS=\"--to {previous.model}\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--to",
        metavar="MODELO",
        help="ativar a geração escrita por este modelo; sem isto, apenas relata",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help=(
            "manter as gerações anteriores em vez de descartá-las, comprando uma volta "
            "atrás instantânea ao preço de uma cópia do corpus inteiro"
        ),
    )
    args = parser.parse_args()
    try:
        config = load_config()
    except ConfigError as error:
        print(f"a configuração não carregou:\n{error}", file=sys.stderr)
        return 2
    try:
        config = resolve_database_config(config)
        with connect(config.db.url) as conn:
            if args.to:
                return switch(conn, config, args.to, keep=args.keep)
            return report(conn, config)
    except (DatabaseUnavailable, SecretResolutionError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
