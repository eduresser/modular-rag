"""What the cut produces today, frozen, so a rewrite of it is reviewable.

This proves no acceptance criterion. It is scaffolding for one change — taking
the provider's JSON through a third party's document model and back out again,
and replacing that round trip with one pass over the payload — and it exists
because `cut_version()` is expected to move during that change and therefore
stops being able to tell a deliberate move from an accidental one.

**It freezes the text, not the version.** Every field a piece carries except
`block_kinds`, which is the one this change means to move: the kinds become the
provider's own vocabulary, deliberately (AC-232).

**It fired once, on one piece, and both differences were defects it exposed.**
The rewrite landed with every piece of every watched payload identical except
the described table of `docai/contrato.json`, where the old cut said:

- **the caption reached no piece at all.** The table's caption and the reader's
  description were folded into one field whose fallback preferred the
  description, so `Tabela 1 — Parcelas e reajuste` existed in the document
  model and in nothing that was ever embedded. AC-204 asserted it through a
  chunker nothing runs, which is how it passed while being false of production.
- **the description was embedded three times.** The chunk names three cell
  identifiers, each resolves to the same table block, and the note was
  appended once per identifier. On a corpus where 33% of every embedded
  character is already a reader description (Q-111), that is the same sentence
  paid for three times and ranked as three matches.

Both are recorded here rather than in a commit message because this fixture is
the only place that could have caught either.

**The payloads are built here rather than added to `src/processing/cut_samples/`,**
because a sample added there moves `cut_version()` and re-cuts the corpus. This
file needs breadth, not a version bump: the three committed samples come to 8
pieces and 874 characters between them, which catches a change to what they
exercise and is silent about everything else.

Retire it once the rewrite has landed and the version has settled.
"""

import json
import pathlib

import pytest

from src.models import VersionStamps
from src.processing.docai import ChunkingOptions
from src.processing.docai import chunks_from_document as docai_cut
from src.processing.recut import CUT_SAMPLES, SAMPLE_READERS, cut_payload
from tests.conftest import list_block, reader_chunk, reader_payload, table_block, text_block

BASELINE = pathlib.Path(__file__).parent / "fixtures" / "cut_text_baseline.json"
STAMPS = VersionStamps(reader_version="baseline", fingerprint="baseline")
CUTTING = ChunkingOptions(size=400, cutter="docai", include_ancestor_headings=True)

#: The fields that must survive the rewrite unchanged. `block_kinds` is absent
#: on purpose — it is the one this change moves.
FROZEN = ("ordinal", "content", "page_start", "page_end", "heading_path", "source_block_ids")


def _every_branch() -> dict:
    """A payload reaching every branch the block mapping has.

    Each entry below is a branch that has lost text before, in this service or
    in the pipeline that preceded it: a list flattened away, a table cell that
    overwrote its neighbour, a running header answering a question, a block of
    a kind nobody had met.
    """
    blocks = [
        text_block("t0", "Preâmbulo sem título acima dele.", pages=(1, 1)),
        text_block("t1", "CONTRATO", pages=(1, 1), kind="title"),
        text_block("t2", "Cláusula Primeira", pages=(1, 1), kind="heading-1"),
        text_block("t3", "Do objeto e sua destinação.", pages=(1, 1), kind="heading-2"),
        text_block("t4", "O imóvel destina-se a uso comercial.", pages=(1, 2)),
        # Page furniture: the reader types it, and it must not answer a question.
        text_block("h1", "Contrato nº 123 — página 2", pages=(2, 2), kind="header"),
        text_block("f1", "Rubrica ______", pages=(2, 2), kind="footer"),
        # A nested list, whose entries carry blocks of their own.
        list_block("l1", ["pagar o aluguel", "conservar o imóvel"], pages=(2, 2)),
        # A table with a caption and a describing annotation beside it.
        table_block(
            "tb1",
            ["Parcela", "Valor"],
            [["1", "R$ 12.400,00"], ["2", "R$ 12.400,00"]],
            pages=(2, 3),
            annotation="Tabela de parcelas reajustadas pelo IGP-M",
        ),
        # A block whose kind this version does not know. Today its text is
        # dropped (AC-231 is the criterion that stops that).
        {"blockId": "x1", "pageSpan": {"pageStart": 3, "pageEnd": 3},
         "audioBlock": {"transcript": "gravação da audiência"}},
    ]
    chunks = [
        reader_chunk("c0", "Preâmbulo sem título acima dele.", ["t0"]),
        reader_chunk("c1", "CONTRATO\nCláusula Primeira\nDo objeto e sua destinação.\n"
                           "O imóvel destina-se a uso comercial.", ["t1", "t2", "t3", "t4"]),
        reader_chunk("c2", "1. pagar o aluguel\n2. conservar o imóvel", ["l1"]),
        # A table chunk the reader cut but did not fill, which is rebuilt from
        # the blocks it names rather than dropped.
        reader_chunk("c3", "", ["tb1"]),
    ]
    return reader_payload(blocks, chunks)


def _cut_everything() -> dict:
    """Every payload this file watches, cut, as the records the archive stores."""
    produced: dict = {}
    for reader, module_cut in SAMPLE_READERS:
        for path in sorted((CUT_SAMPLES / reader).glob("*.json")):
            pieces = cut_payload(
                json.loads(path.read_text()),
                cut=module_cut, external_id="sample", stamps=STAMPS,
            )
            produced[f"{reader}/{path.name}"] = _frozen(pieces)
    produced["synthetic/every-branch"] = _frozen(
        cut_payload(_every_branch(), cut=docai_cut, external_id="sample", stamps=STAMPS)
    )
    return produced


def _frozen(pieces) -> list[dict]:
    return [{field: _plain(getattr(piece, field)) for field in FROZEN} for piece in pieces]


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


def test_the_cut_produces_the_text_it_produced_before_the_rewrite():
    """The alarm for a silent fidelity regression, and nothing else.

    A failure here is never "move the fixture so it passes". It is one of two
    facts: the rewrite changed what a piece says, or the change was intended and
    the fixture is the record of the old answer that somebody has to look at
    before replacing.
    """
    if not BASELINE.exists():  # pragma: no cover — regenerated by the script below
        pytest.skip("no baseline frozen yet")
    expected = json.loads(BASELINE.read_text())
    produced = _cut_everything()

    assert set(produced) == set(expected), "a watched payload appeared or disappeared"
    for name in sorted(expected):
        assert produced[name] == expected[name], (
            f"{name}: the cut says something different than it did before the rewrite. "
            f"Compare piece by piece — this file exists because `cut_version()` is moving "
            f"deliberately and cannot tell an intended change from an accident."
        )


if __name__ == "__main__":  # pragma: no cover — how the fixture is regenerated
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(
        json.dumps(_cut_everything(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(f"frozen to {BASELINE}")
