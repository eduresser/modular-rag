"""The alarm the measured cut version needs, and the one thing it cannot see.

No acceptance criterion is proven here. What this file covers is the known
weakness of measuring a version from a sample: **it sees only what the samples
exercise**. A change to the cutting code that no sample reaches leaves every
sample piece identical, leaves ``cut_version()`` identical, and leaves a corpus
looking current while the next document read is cut differently. It is the
weakness of every golden test, and it is why one is not left alone.

So the code of the cut is hashed too — never as the version, only as an alarm.
:data:`CUT_SURFACE_LOCK` holds the two together, and moving it is the one
manual step left in the whole mechanism. What that step is *not* is the step
this replaced: nobody decides a version, nobody edits a deployment's file, and
nobody can forget, because this test fails until somebody looks. The three ways
it fails are three different facts about the change, and each names its own
next move.

**The code hash is normalised before it is taken.** Comments and docstrings are
more than half the bytes of these three modules, and prose is the most common
edit they get; hashing raw bytes would raise the alarm every time somebody
fixed a typo. Tokens rather than the AST or the bytecode because only the
tokens are stable across Python versions — ``ast.dump``, the bytecode and the
marshalled code object all differ between 3.12 and 3.13, and this repository
develops on one and ships the other.
"""

import hashlib
import io
import tokenize
from pathlib import Path

import pytest

from src.processing import docai, recut, to_chunks
from src.processing import pdftext
from src.processing.recut import CUT_SAMPLES, SAMPLE_READERS, cut_version, sample_pieces

#: The modules that decide what a piece says. ``docai`` maps the reader's kept
#: payload into the model, ``to_chunks`` places the reader's own cut back into
#: it, and ``recut`` is the cut that happens here — between them they produce
#: every character that reaches the archive.
CUT_SURFACE: tuple[str, ...] = (
    docai.__file__,
    pdftext.__file__,
    to_chunks.__file__,
    recut.__file__,
)

#: The cut's code and what that code produces, locked together: the normalised
#: digest of :data:`CUT_SURFACE`, and **the version each module's samples cut
#: to under it**. Both are printed by the failure that asks for them — this is
#: a paste, never a decision.
#:
#: One version per module, because the cut's version is per module (AC-271):
#: a single digest over every module's samples would re-cut a corpus indexed by
#: one the day the other arrived.
CUT_SURFACE_LOCK: tuple[str, dict[str, str]] = (
    "ede87829c7b4",
    {"docai": "05f4fe8e64be493d", "pdftext": "80bc1cd407127230"},
)

_IGNORED = frozenset(
    {tokenize.COMMENT, tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)


def minified(source: bytes) -> str:
    """``source`` as its code alone: no comments, no docstrings, no layout."""
    kept: list[str] = []
    at_statement_start = True
    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type in (tokenize.NEWLINE, tokenize.NL):
            at_statement_start = True
            continue
        if token.type in _IGNORED:
            continue
        if token.type == tokenize.STRING and at_statement_start:
            at_statement_start = False  # a string opening a statement is a docstring
            continue
        at_statement_start = False
        kept.append(token.string)
    return " ".join(kept)


def surface_digest() -> str:
    """The digest of the cut's code, normalised, in the declared order."""
    digest = hashlib.sha256()
    for path in CUT_SURFACE:
        digest.update(minified(Path(path).read_bytes()).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def _counts() -> str:
    """What each sample cuts to, for a failure that has to say which one moved."""
    return "\n".join(f"      {key}: {len(pieces)} pieces" for key, pieces in sample_pieces().items())


def _versions() -> dict[str, str]:
    """What every module's samples cut to under this build."""
    return {module: cut_version(module) for module, _cut in SAMPLE_READERS}


def test_a_changed_cut_surface_was_seen_by_a_sample():
    locked_surface, locked_version = CUT_SURFACE_LOCK
    surface, version = surface_digest(), _versions()
    if (surface, version) == CUT_SURFACE_LOCK:
        return

    now = f"the lock is now ({surface!r}, {version!r})"
    if surface != locked_surface and version == locked_version:
        # The failure this file exists for. The samples cut to exactly what
        # they did before, so either nothing about the text changed, or
        # something did and no sample goes near it.
        raise AssertionError(
            f"the cut's code changed ({locked_surface} → {surface}) and the version it "
            f"produces did not ({version}): no sample noticed.\n"
            "  If it was a refactor that changes no text, that is the mechanism working — "
            f"move CUT_SURFACE_LOCK to ({surface!r}, {version!r}).\n"
            f"  If it changes a cut, a sample under {CUT_SAMPLES} has to reach it, or every "
            "document already indexed stays stale with nothing saying so.\n"
            f"  What the samples cut to:\n{_counts()}"
        )
    if surface == locked_surface and version != locked_version:
        # Nothing in the cut's own modules moved, so either something under
        # them did — `pypdf` is the candidate now that the document model is
        # gone, and a library that walks a content stream differently really
        # does change what a piece says — or the samples themselves were
        # edited. Both move the version honestly; the second is the price of
        # adding coverage, and it is one re-cut sweep with no read and no
        # embedding behind it, because every piece comes out identical and the
        # early cutoff stops there.
        raise AssertionError(
            f"the cut's code is unchanged ({surface}) and the version it produces moved "
            f"({locked_version} → {version}): either a dependency changed what a piece says "
            f"(`pypdf` is the one to check) or the samples changed. {now}.\n"
            f"  What the samples cut to:\n{_counts()}"
        )
    raise AssertionError(
        f"the cut changed and so did what it produces ({locked_surface} → {surface}, "
        f"{locked_version} → {version}), which is the ordinary case: {now}.\n"
        f"  Every document already indexed will be cut again — no read, no embedding "
        f"where the text is unchanged.\n  What the samples cut to:\n{_counts()}"
    )


def test_every_sample_produces_a_citable_piece_by_both_cutters():
    # A sample that cuts to nothing contributes nothing to the version: it
    # would keep its digest through any change to the code producing the
    # emptiness. Both cutters, because the version covers both.
    for key, pieces in sample_pieces().items():
        assert pieces, f"{key} cuts to no pieces, so it measures nothing"
        for piece in pieces:
            assert piece["content"].strip(), key
            assert piece["page_start"] is not None or piece["heading_path"], key


def test_the_samples_exercise_every_block_kind_the_reader_emits():
    # The coverage claim, checked rather than asserted in a comment: a block
    # kind absent from every sample is a mapping nothing measures.
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for reader, _ in SAMPLE_READERS
        for path in (CUT_SAMPLES / reader).glob("*.json")
    )
    for kind in ("textBlock", "listBlock", "tableBlock", "imageBlock"):
        assert kind in text, f"no sample carries a {kind}, so its mapping is unmeasured"
    # And the two cases whose whole point is that they are easy to lose: a
    # block the reader named nothing, and a chunk naming a block that is not
    # in the document the same reader returned (ASM-023).
    assert '"blockId": "b1"' in text
    assert '"sourceBlockIds": [\n          "b9"\n        ]' in text.replace("\r\n", "\n")


def test_the_version_is_stable_across_calls():
    # It is cached, and a cache that returned two answers in one process would
    # stamp two halves of one run with two different versions.
    for module, _cut in SAMPLE_READERS:
        assert cut_version(module) == cut_version(module)
        assert len(cut_version(module)) == 16
    # And no two modules share one, because they cut different pieces out of
    # different payloads — a shared value would mean a module's corpus was
    # invalidated by another module's code.
    assert len({cut_version(module) for module, _cut in SAMPLE_READERS}) == len(SAMPLE_READERS)


def test_a_read_module_with_no_samples_is_refused_rather_than_ignored(monkeypatch, tmp_path):
    # An empty directory would make the version a constant, and a constant
    # version is worse than a typed one: nothing moves it and nothing says so.
    monkeypatch.setattr(recut, "CUT_SAMPLES", tmp_path)

    with pytest.raises(recut.RecutError) as excinfo:
        sample_pieces()

    assert "no cut samples" in str(excinfo.value)
