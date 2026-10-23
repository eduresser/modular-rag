from __future__ import annotations

import unicodedata
import urllib.request
from pathlib import Path

SOURCE = "https://www.unicode.org/Public/security/latest/confusables.txt"
TARGET = Path(__file__).resolve().parents[1] / "src" / "processing" / "confusables.py"

SYMBOLS = set("αβγδεζηθικλμνξοπρστυφχψω") | set("ΔΩΣΘΠΛΓΦΨΞ")


def derive(text: str) -> tuple[dict[str, str], str, str]:
    version = date = "unknown"
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("# Version:"):
            version = line.split(":", 1)[1].strip()
            continue
        if line.startswith("# Date:"):
            date = line.split(":", 1)[1].strip()
            continue
        if line.startswith("#") or ";" not in line:
            continue
        fields = [field.strip() for field in line.split("#")[0].split(";")]
        if len(fields) < 2 or " " in fields[0] or " " in fields[1]:
            continue
        try:
            source, target = chr(int(fields[0], 16)), chr(int(fields[1], 16))
        except ValueError:
            continue
        if not (source.isalpha() and target.isalpha()):
            continue
        if "LATIN" not in unicodedata.name(target, ""):
            continue
        name = unicodedata.name(source, "")
        if "GREEK" not in name and "CYRILLIC" not in name:
            continue
        if source in SYMBOLS:
            continue
        mapping[source] = target
    return mapping, version, date


def render(mapping: dict[str, str], version: str, date: str) -> str:
    rows = "\n".join(
        f'    {source!r}: {target!r},  # {unicodedata.name(source, "?")}'
        for source, target in sorted(mapping.items())
    )
    return f'''"""Letters this reader substitutes for Latin ones, and what they stand for.

GENERATED — do not edit by hand. Regenerate with
``uv run python scripts/derive_confusables.py``.

Derived from the Unicode Consortium's ``confusables.txt``, the data behind
UTS #39, **version {version}** ({date}), keeping every Greek or Cyrillic letter
whose prototype is a single Latin letter and dropping the ones this corpus uses
as symbols (see ``SYMBOLS`` in the generating script).

The version is part of this: the table changes between Unicode releases, so
regenerating it changes the text a document is cut into, which is why the cut's
own measured version covers it.
"""

from __future__ import annotations

#: Unicode release this table was derived from.
UNICODE_VERSION = {version!r}

#: A letter the reader may emit, mapped to the Latin letter it stands for.
CONFUSABLES: dict[str, str] = {{
{rows}
}}
'''


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=120) as response:  # noqa: S310 — a named standards host
        text = response.read().decode("utf-8-sig")
    mapping, version, date = derive(text)
    TARGET.write_text(render(mapping, version, date), encoding="utf-8")
    print(f"{TARGET}: {len(mapping)} entries from Unicode {version} ({date})")


if __name__ == "__main__":
    main()
