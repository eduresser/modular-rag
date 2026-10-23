"""Letters this reader substitutes for Latin ones, and what they stand for.

GENERATED — do not edit by hand. Regenerate with
``uv run python scripts/derive_confusables.py``.

Derived from the Unicode Consortium's ``confusables.txt``, the data behind
UTS #39, **version 17.0.0** (2025-07-22, 05:49:37 GMT), keeping every Greek or Cyrillic letter
whose prototype is a single Latin letter and dropping the ones this corpus uses
as symbols (see ``SYMBOLS`` in the generating script).

The version is part of this: the table changes between Unicode releases, so
regenerating it changes the text a document is cut into, which is why the cut's
own measured version covers it.
"""

from __future__ import annotations

#: Unicode release this table was derived from.
UNICODE_VERSION = '17.0.0'

#: A letter the reader may emit, mapped to the Latin letter it stands for.
CONFUSABLES: dict[str, str] = {
    'Ͱ': 'Ⱶ',  # GREEK CAPITAL LETTER HETA
    'ͷ': 'ᴎ',  # GREEK SMALL LETTER PAMPHYLIAN DIGAMMA
    'ͺ': 'i',  # GREEK YPOGEGRAMMENI
    'ͻ': 'ɔ',  # GREEK SMALL REVERSED LUNATE SIGMA SYMBOL
    'ͽ': 'ꜿ',  # GREEK SMALL REVERSED DOTTED LUNATE SIGMA SYMBOL
    'Ϳ': 'J',  # GREEK CAPITAL LETTER YOT
    'Α': 'A',  # GREEK CAPITAL LETTER ALPHA
    'Β': 'B',  # GREEK CAPITAL LETTER BETA
    'Ε': 'E',  # GREEK CAPITAL LETTER EPSILON
    'Ζ': 'Z',  # GREEK CAPITAL LETTER ZETA
    'Η': 'H',  # GREEK CAPITAL LETTER ETA
    'Ι': 'l',  # GREEK CAPITAL LETTER IOTA
    'Κ': 'K',  # GREEK CAPITAL LETTER KAPPA
    'Μ': 'M',  # GREEK CAPITAL LETTER MU
    'Ν': 'N',  # GREEK CAPITAL LETTER NU
    'Ο': 'O',  # GREEK CAPITAL LETTER OMICRON
    'Ρ': 'P',  # GREEK CAPITAL LETTER RHO
    'Τ': 'T',  # GREEK CAPITAL LETTER TAU
    'Υ': 'Y',  # GREEK CAPITAL LETTER UPSILON
    'Χ': 'X',  # GREEK CAPITAL LETTER CHI
    'ϐ': 'ß',  # GREEK BETA SYMBOL
    'ϒ': 'Y',  # GREEK UPSILON WITH HOOK SYMBOL
    'ϕ': 'ɸ',  # GREEK PHI SYMBOL
    'Ϝ': 'F',  # GREEK LETTER DIGAMMA
    'ϰ': 'ĸ',  # GREEK KAPPA SYMBOL
    'ϱ': 'p',  # GREEK RHO SYMBOL
    'ϲ': 'c',  # GREEK LUNATE SIGMA SYMBOL
    'ϳ': 'j',  # GREEK LETTER YOT
    'ϵ': 'ꞓ',  # GREEK LUNATE EPSILON SYMBOL
    'Ϸ': 'Þ',  # GREEK CAPITAL LETTER SHO
    'ϸ': 'p',  # GREEK SMALL LETTER SHO
    'Ϲ': 'C',  # GREEK CAPITAL LUNATE SIGMA SYMBOL
    'Ϻ': 'M',  # GREEK CAPITAL LETTER SAN
    'Ͻ': 'Ɔ',  # GREEK CAPITAL REVERSED LUNATE SIGMA SYMBOL
    'Ͽ': 'Ꜿ',  # GREEK CAPITAL REVERSED DOTTED LUNATE SIGMA SYMBOL
    'Є': 'Ꞓ',  # CYRILLIC CAPITAL LETTER UKRAINIAN IE
    'Ѕ': 'S',  # CYRILLIC CAPITAL LETTER DZE
    'І': 'l',  # CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    'Ј': 'J',  # CYRILLIC CAPITAL LETTER JE
    'А': 'A',  # CYRILLIC CAPITAL LETTER A
    'В': 'B',  # CYRILLIC CAPITAL LETTER VE
    'Е': 'E',  # CYRILLIC CAPITAL LETTER IE
    'К': 'K',  # CYRILLIC CAPITAL LETTER KA
    'Л': 'Ʌ',  # CYRILLIC CAPITAL LETTER EL
    'М': 'M',  # CYRILLIC CAPITAL LETTER EM
    'Н': 'H',  # CYRILLIC CAPITAL LETTER EN
    'О': 'O',  # CYRILLIC CAPITAL LETTER O
    'Р': 'P',  # CYRILLIC CAPITAL LETTER ER
    'С': 'C',  # CYRILLIC CAPITAL LETTER ES
    'Т': 'T',  # CYRILLIC CAPITAL LETTER TE
    'У': 'Y',  # CYRILLIC CAPITAL LETTER U
    'Х': 'X',  # CYRILLIC CAPITAL LETTER HA
    'Ь': 'b',  # CYRILLIC CAPITAL LETTER SOFT SIGN
    'а': 'a',  # CYRILLIC SMALL LETTER A
    'в': 'ʙ',  # CYRILLIC SMALL LETTER VE
    'г': 'r',  # CYRILLIC SMALL LETTER GHE
    'е': 'e',  # CYRILLIC SMALL LETTER IE
    'з': 'ɜ',  # CYRILLIC SMALL LETTER ZE
    'и': 'ᴎ',  # CYRILLIC SMALL LETTER I
    'к': 'ĸ',  # CYRILLIC SMALL LETTER KA
    'м': 'ʍ',  # CYRILLIC SMALL LETTER EM
    'н': 'ʜ',  # CYRILLIC SMALL LETTER EN
    'о': 'o',  # CYRILLIC SMALL LETTER O
    'р': 'p',  # CYRILLIC SMALL LETTER ER
    'с': 'c',  # CYRILLIC SMALL LETTER ES
    'т': 'ᴛ',  # CYRILLIC SMALL LETTER TE
    'у': 'y',  # CYRILLIC SMALL LETTER U
    'ф': 'ɸ',  # CYRILLIC SMALL LETTER EF
    'х': 'x',  # CYRILLIC SMALL LETTER HA
    'ш': 'w',  # CYRILLIC SMALL LETTER SHA
    'ь': 'ƅ',  # CYRILLIC SMALL LETTER SOFT SIGN
    'я': 'ᴙ',  # CYRILLIC SMALL LETTER YA
    'є': 'ꞓ',  # CYRILLIC SMALL LETTER UKRAINIAN IE
    'ѕ': 's',  # CYRILLIC SMALL LETTER DZE
    'і': 'i',  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    'ј': 'j',  # CYRILLIC SMALL LETTER JE
    'ѡ': 'w',  # CYRILLIC SMALL LETTER OMEGA
    'Ѵ': 'V',  # CYRILLIC CAPITAL LETTER IZHITSA
    'ѵ': 'v',  # CYRILLIC SMALL LETTER IZHITSA
    'Ү': 'Y',  # CYRILLIC CAPITAL LETTER STRAIGHT U
    'ү': 'y',  # CYRILLIC SMALL LETTER STRAIGHT U
    'һ': 'h',  # CYRILLIC SMALL LETTER SHHA
    'ҽ': 'e',  # CYRILLIC SMALL LETTER ABKHASIAN CHE
    'Ӏ': 'l',  # CYRILLIC LETTER PALOCHKA
    'ӏ': 'l',  # CYRILLIC SMALL LETTER PALOCHKA
    'Ә': 'Ə',  # CYRILLIC CAPITAL LETTER SCHWA
    'ә': 'ǝ',  # CYRILLIC SMALL LETTER SCHWA
    'ӡ': 'ȝ',  # CYRILLIC SMALL LETTER ABKHASIAN DZE
    'ԁ': 'd',  # CYRILLIC SMALL LETTER KOMI DE
    'Ԋ': 'Ƕ',  # CYRILLIC CAPITAL LETTER KOMI NJE
    'Ԍ': 'G',  # CYRILLIC CAPITAL LETTER KOMI SJE
    'ԍ': 'ɢ',  # CYRILLIC SMALL LETTER KOMI SJE
    'Ԑ': 'Ɛ',  # CYRILLIC CAPITAL LETTER REVERSED ZE
    'ԑ': 'ꞓ',  # CYRILLIC SMALL LETTER REVERSED ZE
    'ԛ': 'q',  # CYRILLIC SMALL LETTER QA
    'Ԝ': 'W',  # CYRILLIC CAPITAL LETTER WE
    'ԝ': 'w',  # CYRILLIC SMALL LETTER WE
    'ᴦ': 'r',  # GREEK LETTER SMALL CAPITAL GAMMA
    'ᴧ': 'ʌ',  # GREEK LETTER SMALL CAPITAL LAMDA
    'ᴩ': 'ᴘ',  # GREEK LETTER SMALL CAPITAL RHO
    'ι': 'i',  # GREEK PROSGEGRAMMENI
    'ꙅ': 'ƨ',  # CYRILLIC SMALL LETTER REVERSED DZE
    'ꙇ': 'i',  # CYRILLIC SMALL LETTER IOTA
    'Ꙩ': 'ʘ',  # CYRILLIC CAPITAL LETTER MONOCULAR O
}
