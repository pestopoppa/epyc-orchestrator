"""Language-gated text normalization for the TTS input (Italian only today).

The frozen qwentts.cpp kernel (production-speech-v1) reads raw text: it BPE-tokenizes
whatever it is given and has no number, unit, abbreviation, phoneme or stress front end.
Digits, unit symbols and letter acronyms therefore reach the talker model verbatim, and the
CS-8.L Italian listen check (operator, 2026-10-22) heard them misread: "1963", "150",
"2.4Ghz", "pH" read with a French/Spanish "hache". This module rewrites those tokens into
Italian words BEFORE the text reaches the TTS kernel.

Contract:
  * ``normalize_for_tts(text, language)`` is a pure, deterministic, stdlib-only function.
  * Only Italian is normalized. Any other language, or ``None``, returns the input object
    unchanged (byte-identical by construction).
  * Unknown tokens pass through unchanged. Rules fire only where the reading is
    unambiguous; anything else is left for the model to read.
  * Output is idempotent: normalizing an already-normalized string changes nothing.

Coverage (Italian):
  * cardinals up to 999 999 999 999, with the compound accent rule (ventitré, centotré,
    millenovecentosessantatré), vowel elision (ventuno, trentotto, centottanta) and
    apocope before ``mila``/``milioni`` (ventunmila, ventun milioni);
  * grouped thousands with dots (40.000), decimal comma (2,5), English-style dot decimals
    before a unit (2.4 GHz -> "due virgola quattro gigahertz"), dotted identifiers without a
    unit (802.11n -> "ottocentodue punto undici enne");
  * numbers with a unit (Hz, kHz, MHz, GHz, km, m, cm, mm, kg, g, mg, l, ml, %, °C, °F, km/h,
    W, kW, kWh, V, dB, MB, GB, TB, h, min, ms, euro/dollar/pound signs), with singular/plural
    and gender agreement for 1 (un chilometro, un'ora) and "di" after round millions;
  * ordinals with º/ª, "1 <month>" -> "primo", d/m/yyyy dates, hh:mm times, decades ('20);
  * letter-name acronyms from an extendable table (pH -> "pi acca") and "lettera X".
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

__all__ = [
    "ACRONYMS_IT",
    "LETTER_NAMES_IT",
    "UNITS_IT",
    "cardinal_it",
    "canonical_language",
    "normalize_for_tts",
    "normalize_italian",
    "RESPELL_IT",
    "ordinal_it",
    "respell_italian",
]


# --------------------------------------------------------------------------- language gate

_LANGUAGE_ALIASES: Mapping[str, str] = {
    "it": "it", "ita": "it", "it-it": "it", "it_it": "it", "italian": "it", "italiano": "it",
}


def canonical_language(language: str | None) -> str | None:
    """Map a language tag or name to the normalizer's canonical code, or None."""
    if not isinstance(language, str):
        return None
    return _LANGUAGE_ALIASES.get(language.strip().lower())


# Respelling table (CS-8.L4): the frozen qwentts kernel has no stress lexicon, so words it
# stresses or pronounces wrongly are respelled with a written accent / hard-g digraph. Only
# forms the operator rated better in the CS-8.L re-listen (5/5) are listed.
RESPELL_IT: Mapping[str, str] = {
    "atomi": "àtomi",
    "idiozie": "idiozìe",
    "analoga": "anàloga",
    "giza": "Ghiza",
    "marianne": "Mariànne",
}
_RESPELL_RE = re.compile(r"(?<![\w])(" + "|".join(RESPELL_IT) + r")(?![\w])", re.IGNORECASE)


def _sub_respell(m: re.Match[str]) -> str:
    word = m.group(1)
    target = RESPELL_IT[word.lower()]
    if word.isupper() and len(word) > 1:
        return target.upper()
    if word[0].isupper():
        return target[0].upper() + target[1:]
    return target


def respell_italian(text: str) -> str:
    """Respell known mis-stressed / mis-pronounced Italian words (idempotent)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _RESPELL_RE.sub(_sub_respell, text)


def normalize_for_tts(
    text: str, language: str | None, *, normalize: bool = True, respell: bool = True
) -> str:
    """Return ``text`` prepared for TTS in ``language``; non-Italian input is returned as is.

    ``normalize`` / ``respell`` switch the two Italian steps independently (live kill-switches).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if canonical_language(language) != "it":
        return text
    if normalize:
        text = normalize_italian(text)
    if respell:
        text = respell_italian(text)
    return text


# --------------------------------------------------------------------------- Italian numbers

_UNITS = (
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
    "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
    "diciassette", "diciotto", "diciannove",
)
_TENS = {2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta", 6: "sessanta",
         7: "settanta", 8: "ottanta", 9: "novanta"}
_MAX_CARDINAL = 10**12 - 1


def _below_100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    word = _TENS[tens]
    if unit == 0:
        return word
    if unit in (1, 8):  # ventuno, trentotto: the tens word drops its final vowel
        word = word[:-1]
    return word + _UNITS[unit]


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest)
    head = "cento" if hundreds == 1 else _UNITS[hundreds] + "cento"
    if rest == 0:
        return head
    if 80 <= rest <= 89:  # centottanta, duecentottantadue (centootto keeps both vowels)
        head = head[:-1]
    return head + _below_100(rest)


def _apocope(word: str) -> str:
    """ventuno -> ventun before mila/milioni/miliardi."""
    return word[:-1] if word.endswith("uno") and word != "uno" else word


def _below_million(n: int) -> str:
    thousands, rest = divmod(n, 1000)
    if thousands == 0:
        return _below_1000(rest)
    head = "mille" if thousands == 1 else _apocope(_below_1000(thousands)) + "mila"
    return head + (_below_1000(rest) if rest else "")


def _scale(n: int, size: int, one: str, many: str, lower: Callable[[int], str]) -> str:
    count, rest = divmod(n, size)
    head = f"un {one}" if count == 1 else f"{_apocope(_below_1000(count))} {many}"
    return head + (f" {lower(rest)}" if rest else "")


def _accent_final_tre(words: str) -> str:
    """Compounds ending in -tre take the accent: ventitré, centotré, milletré; "tre" alone does not."""
    parts = words.split(" ")
    last = parts[-1]
    if last.endswith("tre") and last != "tre":
        parts[-1] = last[:-3] + "tré"
    return " ".join(parts)


def year_it(n: int) -> str:
    """Year reading for 1000 <= n <= 2099: the below-100 tail is a separate word.

    CS-8.L4a: spelled "duemilasei" synthesises a voiced "zei" (+7 dB on the year word); the
    spaced "duemila sei" does not. Other numbers keep the fused form.
    """
    if not 1000 <= n <= 2099 or n % 100 == 0:
        return cardinal_it(n)
    head, tail = n - n % 100, n % 100
    if tail == 0:
        return cardinal_it(n)
    return _accent_final_tre(f"{cardinal_it(head)} {_below_100(tail)}")


def cardinal_it(n: int) -> str:
    """Italian cardinal for 0 <= |n| <= 999 999 999 999."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an int")
    if n < 0:
        return "meno " + cardinal_it(-n)
    if n > _MAX_CARDINAL:
        raise ValueError("number out of range")
    if n < 10**6:
        words = _below_million(n)
    elif n < 10**9:
        words = _scale(n, 10**6, "milione", "milioni", _below_million)
    else:
        words = _scale(n, 10**9, "miliardo", "miliardi",
                       lambda r: _below_million(r) if r < 10**6 else
                       _scale(r, 10**6, "milione", "milioni", _below_million))
    return _accent_final_tre(words)


_ORDINALS = {1: "primo", 2: "secondo", 3: "terzo", 4: "quarto", 5: "quinto", 6: "sesto",
             7: "settimo", 8: "ottavo", 9: "nono", 10: "decimo"}


def ordinal_it(n: int, *, feminine: bool = False) -> str:
    """Italian ordinal for 1 <= n < 10**6 (primo, undicesimo, ventitreesimo, duemillesimo)."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n < 10**6:
        raise ValueError("ordinal out of range")
    if n in _ORDINALS:
        word = _ORDINALS[n]
    else:
        base = cardinal_it(n)
        if base.endswith("tré"):
            word = base[:-1] + "eesimo"
        elif base.endswith("sei"):
            word = base + "esimo"
        elif base.endswith("mila"):
            word = base[:-4] + "millesimo"
        else:
            word = base[:-1] + "esimo"
    return word[:-1] + "a" if feminine else word


def _digits(s: str) -> str:
    return " ".join(_UNITS[int(c)] for c in s)


def _int_words(s: str) -> str | None:
    """Words for a plain digit string, or None when it should pass through."""
    if len(s) > 1 and s[0] == "0":
        return None  # codes, phone numbers, zero-padded ids: not a quantity
    n = int(s)
    return cardinal_it(n) if n <= _MAX_CARDINAL else None


def _fraction_words(frac: str) -> str:
    """Digits after the decimal comma: 2,5 -> cinque, 3,14 -> quattordici, 0,05 -> zero cinque."""
    if frac[0] == "0" or len(frac) > 3:
        return _digits(frac)
    return cardinal_it(int(frac))


# --------------------------------------------------------------------------- tables

LETTER_NAMES_IT: Mapping[str, str] = {
    "a": "a", "b": "bi", "c": "ci", "d": "di", "e": "e", "f": "effe", "g": "gi", "h": "acca",
    "i": "i", "j": "i lunga", "k": "cappa", "l": "elle", "m": "emme", "n": "enne", "o": "o",
    "p": "pi", "q": "cu", "r": "erre", "s": "esse", "t": "ti", "u": "u", "v": "vu",
    "w": "doppia vu", "x": "ics", "y": "ipsilon", "z": "zeta",
}

# Case-sensitive, matched as whole tokens. Only items that Italian reads letter by letter
# (or whose spoken form is fixed) belong here; word-like acronyms (NATO, IVA) must NOT be
# added, the model reads those correctly as words. Extend by adding rows.
ACRONYMS_IT: dict[str, str] = {
    "pH": "pi acca",
    "DNA": "di enne a",
    "RNA": "erre enne a",
    "TV": "ti vu",
    "PC": "pi ci",
    "CD": "ci di",
    "DVD": "di vu di",
    "GPS": "gi pi esse",
    "SMS": "esse emme esse",
    "USB": "u esse bi",
    "Hz": "hertz",
    "kHz": "chilohertz",
    "MHz": "megahertz",
    "GHz": "gigahertz",
}

# symbol -> (singular, plural, feminine). Singular == plural for invariant units.
# Lookup is case-sensitive except for the Hz family, whose miscapitalised spellings
# (Ghz, ghz, Mhz, KHz) are common in source text and unambiguous.
UNITS_IT: dict[str, tuple[str, str, bool]] = {
    "Hz": ("hertz", "hertz", False),
    "kHz": ("chilohertz", "chilohertz", False),
    "MHz": ("megahertz", "megahertz", False),
    "GHz": ("gigahertz", "gigahertz", False),
    "km/h": ("chilometro orario", "chilometri orari", False),
    "m/s": ("metro al secondo", "metri al secondo", False),
    "km²": ("chilometro quadrato", "chilometri quadrati", False),
    "m²": ("metro quadrato", "metri quadrati", False),
    "m³": ("metro cubo", "metri cubi", False),
    "km": ("chilometro", "chilometri", False),
    "m": ("metro", "metri", False),
    "cm": ("centimetro", "centimetri", False),
    "mm": ("millimetro", "millimetri", False),
    "kg": ("chilogrammo", "chilogrammi", False),
    "g": ("grammo", "grammi", False),
    "mg": ("milligrammo", "milligrammi", False),
    "l": ("litro", "litri", False),
    "L": ("litro", "litri", False),
    "ml": ("millilitro", "millilitri", False),
    "mL": ("millilitro", "millilitri", False),
    "°C": ("grado Celsius", "gradi Celsius", False),
    "°F": ("grado Fahrenheit", "gradi Fahrenheit", False),
    "%": ("per cento", "per cento", False),
    "‰": ("per mille", "per mille", False),
    "W": ("watt", "watt", False),
    "kW": ("chilowatt", "chilowatt", False),
    "MW": ("megawatt", "megawatt", False),
    "kWh": ("chilowattora", "chilowattora", False),
    "V": ("volt", "volt", False),
    "dB": ("decibel", "decibel", False),
    "kB": ("kilobyte", "kilobyte", False),
    "MB": ("megabyte", "megabyte", False),
    "GB": ("gigabyte", "gigabyte", False),
    "TB": ("terabyte", "terabyte", False),
    "h": ("ora", "ore", True),
    "min": ("minuto", "minuti", False),
    "ms": ("millisecondo", "millisecondi", False),
    "€": ("euro", "euro", False),
    "$": ("dollaro", "dollari", False),
    "£": ("sterlina", "sterline", True),
}
_HZ_ALIASES = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
# Units after which the bare cardinal is read (uno per cento, not un per cento).
_BARE_UNITS = frozenset({"%", "‰"})
_CURRENCY_PREFIX = ("€", "$", "£")

_MONTHS = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
           "settembre", "ottobre", "novembre", "dicembre")


# --------------------------------------------------------------------------- regexes

_SP = "[   ]"  # space, no-break space, narrow no-break space
_NUM = r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+\.\d+|\d+)"
_NOT_BEFORE = r"(?<![\w.,°])"
_NOT_AFTER_NUM = r"(?![.,]?\d)"


def _unit_alternation() -> str:
    symbols = set(UNITS_IT)
    for unit in ("Hz", "kHz", "MHz", "GHz"):
        stem = unit[:-2]
        for h in ("H", "h"):
            for z in ("Z", "z"):
                for prefix in {stem, stem.lower(), stem.upper()}:
                    symbols.add(prefix + h + z)
    return "|".join(re.escape(s) for s in sorted(symbols, key=len, reverse=True))


_UNIT_ALT = _unit_alternation()
_ACRONYM_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(k) for k in sorted(ACRONYMS_IT, key=len, reverse=True))
    + r")(?![\w])"
)
_LETTER_RE = re.compile(r"\b(lettera|lettere)" + _SP + r"([A-Z])(?![\w])")
_DATE_RE = re.compile(r"(?<![\w/.])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/])")
_TIME_COLON_RE = re.compile(r"(?<![\w:.,])(\d{1,2}):(\d{2})(?![\w:]|[.,]\d)")
_TIME_DOT_RE = re.compile(r"\b(alle|dalle|ore|le)" + _SP + r"(\d{1,2})\.(\d{2})(?![\w.]|,\d)")
_FIRST_OF_MONTH_RE = re.compile(
    r"(?<![\w.,])1(?:°|º)?(" + _SP + "(?:" + "|".join(_MONTHS) + r"))\b", re.IGNORECASE
)
_DECADE_RE = re.compile(r"(?<![\w'’])['’](\d{2})(?![\w'’]|[.,]\d)")
_CURRENCY_PREFIX_RE = re.compile(
    r"(?<![\w])([€$£])" + _SP + r"?(" + _NUM + r")" + _NOT_AFTER_NUM + r"(?![\w])"
)
_UNIT_RE = re.compile(
    _NOT_BEFORE + r"([-−](?=\d))?(" + _NUM + r")"
    + _SP + r"?(" + _UNIT_ALT + r")(?![\w²³'\u2019])"
)
_ORDINAL_RE = re.compile(r"(?<![\w.,])(\d{1,6})([ºª])(?![\w])")
_GROUPED_RE = re.compile(
    _NOT_BEFORE + r"(\d{1,3}(?:\.\d{3})+)(?![\w]|[.,]\d)(" + _SP + r"[^\W\d_])?"
)
_IDENTIFIER_RE = re.compile(_NOT_BEFORE + r"(\d+(?:\.\d+)+)([a-z]{1,2})?(?![\w]|[.,]\d)")
_DECIMAL_RE = re.compile(_NOT_BEFORE + r"(\d+),(\d+)(?![\w]|[.,]\d)")
_INT_RE = re.compile(_NOT_BEFORE + r"(\d+)(?![\w]|[.,]\d)(" + _SP + r"[^\W\d_])?")


# --------------------------------------------------------------------------- number parsing

def _is_grouped(s: str) -> bool:
    parts = s.split(".")
    return len(parts) > 1 and 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:])


def _parse_number(s: str) -> tuple[int, str | None] | None:
    """Return (integer part, fraction digits or None) for a matched number token, or None."""
    if "," in s:
        whole, frac = s.split(",", 1)
    else:
        whole, frac = s, None
    if "." in whole:
        if _is_grouped(whole):
            whole = whole.replace(".", "")
        elif frac is None and whole.count(".") == 1:
            whole, frac = whole.split(".")  # English-style decimal; only reached with a unit
        else:
            return None
    if len(whole) > 1 and whole[0] == "0":
        return None
    n = int(whole)
    if n > _MAX_CARDINAL:
        return None
    return n, frac


def _number_words(n: int, frac: str | None) -> str:
    words = cardinal_it(n)
    return words if frac is None else f"{words} virgola {_fraction_words(frac)}"


def _quantity(n: int, frac: str | None, unit: str, negative: bool = False) -> str:
    sing, plural, feminine = UNITS_IT[unit]
    sign = "meno " if negative else ""
    if frac is None and n == 1 and unit not in _BARE_UNITS:
        if feminine:
            article = "un'" if sing[0] in "aeiou" else "una "
        else:
            article = "un "
        return f"{sign}{article}{sing}"
    words = _number_words(n, frac)
    if frac is None and n >= 10**6 and n % 10**6 == 0:
        return f"{sign}{words} di {plural}"
    return f"{sign}{words} {plural}"


def _canonical_unit(symbol: str) -> str:
    return symbol if symbol in UNITS_IT else _HZ_ALIASES[symbol.lower()]


# --------------------------------------------------------------------------- rewrite passes

def _sub_acronym(m: re.Match[str]) -> str:
    return ACRONYMS_IT[m.group(1)]


def _sub_letter(m: re.Match[str]) -> str:
    return f"{m.group(1)} {LETTER_NAMES_IT[m.group(2).lower()]}"


def _sub_date(m: re.Match[str]) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= day <= 31 and 1 <= month <= 12) or (len(year) == 4 and year[0] == "0"):
        return m.group(0)
    day_words = "primo" if day == 1 else cardinal_it(day)
    return f"{day_words} {_MONTHS[month - 1]} {year_it(int(year))}"


def _clock(hour: int, minute: int) -> str:
    hour_words = "una" if hour == 1 else cardinal_it(hour)
    return hour_words if minute == 0 else f"{hour_words} e {cardinal_it(minute)}"


def _sub_time_colon(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 24 or minute > 59:
        return m.group(0)
    return _clock(hour, minute)


def _sub_time_dot(m: re.Match[str]) -> str:
    hour, minute = int(m.group(2)), int(m.group(3))
    if hour > 24 or minute > 59:
        return m.group(0)
    return f"{m.group(1)} {_clock(hour, minute)}"


def _sub_first_of_month(m: re.Match[str]) -> str:
    return "primo" + m.group(1)


def _sub_decade(m: re.Match[str]) -> str:
    return cardinal_it(int(m.group(1)))


def _sub_currency_prefix(m: re.Match[str]) -> str:
    parsed = _parse_number(m.group(2))
    if parsed is None:
        return m.group(0)
    return _quantity(parsed[0], parsed[1], m.group(1))


def _sub_unit(m: re.Match[str]) -> str:
    parsed = _parse_number(m.group(2))
    if parsed is None:
        return m.group(0)
    return _quantity(parsed[0], parsed[1], _canonical_unit(m.group(3)), negative=bool(m.group(1)))


def _sub_ordinal(m: re.Match[str]) -> str:
    n = int(m.group(1))
    if n == 0 or (len(m.group(1)) > 1 and m.group(1)[0] == "0"):
        return m.group(0)
    return ordinal_it(n, feminine=m.group(2) == "ª")


def _sub_identifier(m: re.Match[str]) -> str:
    body, suffix = m.group(1), m.group(2)
    parts = body.split(".")
    if len(parts[0]) > 1 and parts[0][0] == "0":
        return m.group(0)
    spoken = []
    for p in parts:
        w = _int_words(p)
        spoken.append(w if w is not None else _digits(p))
    out = " punto ".join(spoken)
    if suffix:
        out += " " + " ".join(LETTER_NAMES_IT[c] for c in suffix)
    return out


def _sub_decimal(m: re.Match[str]) -> str:
    whole = _int_words(m.group(1))
    if whole is None:
        return m.group(0)
    return f"{whole} virgola {_fraction_words(m.group(2))}"


def _count_before_word(m: re.Match[str], digits: str, year: bool = False) -> str:
    following = m.group(2) or ""
    if digits == "1" and following:
        return m.group(0)  # uno/un/una depends on the noun's gender: leave it to the model
    words = _int_words(digits)
    if words is None:
        return m.group(0)
    n = int(digits)
    if year and len(digits) == 4 and digits[0] in "12":
        words = year_it(n)
    if following and n >= 10**6 and n % 10**6 == 0:
        return f"{words} di{following}"  # un milione di abitanti
    return words + following


def _sub_grouped(m: re.Match[str]) -> str:
    return _count_before_word(m, m.group(1).replace(".", ""))


_YEAR_CUE_RE = re.compile(
    r"(?:\b(?:nel|nell|del|dell|dal|dall|al|all|entro\s+il|fino\s+al|dopo\s+il|prima\s+del|"
    r"anno|anni|in|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)|\bdi)[\s'’]+$", re.IGNORECASE)
_QUANTITY_NOUNS = frozenset(
    "euro dollari persone abitanti copie anni giorni mesi ore minuti secondi metri chilometri "
    "km kg grammi litri studenti utenti voti punti pezzi unità casi morti vittime posti "
    "famiglie volte parole pagine righe file byte".split())


def _is_year_context(m: re.Match[str]) -> bool:
    """A 4-digit 1000-2999 integer is a year only in a date-like context (CS-8.L4a)."""
    digits = m.group(1)
    if len(digits) != 4 or digits[0] not in "12":
        return False
    text, start, end = m.string, m.start(1), m.end(1)
    following = (m.group(2) or "").strip().lower()
    if following in _QUANTITY_NOUNS:
        return False
    before, after = text[:start], text[end:]
    if re.search(r"\(\s*$", before) and re.match(r"\s*\)", after):
        return True
    if re.match(r"[-–—]\d{4}(?!\d)", after) or re.search(r"(?<!\d)\d{4}[-–—]$", before):
        return True
    return _YEAR_CUE_RE.search(before) is not None


def _sub_int(m: re.Match[str]) -> str:
    if len(m.group(1)) > 6:
        return m.group(0)  # Italian groups large quantities (1.000.000); a bare run is a code or phone number
    return _count_before_word(m, m.group(1), year=_is_year_context(m))


def normalize_italian(text: str) -> str:
    """Rewrite numbers, units and letter acronyms in Italian ``text`` into spoken words."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not any(c.isdigit() for c in text) and _ACRONYM_RE.search(text) is None \
            and _LETTER_RE.search(text) is None:
        return text
    # Order matters: quantities with a unit first (so "2.4 GHz" is a decimal, not a dotted
    # identifier, and "1 GHz" agrees as "un gigahertz"), then the standalone acronym table.
    out = _DATE_RE.sub(_sub_date, text)
    out = _TIME_DOT_RE.sub(_sub_time_dot, out)
    out = _TIME_COLON_RE.sub(_sub_time_colon, out)
    out = _FIRST_OF_MONTH_RE.sub(_sub_first_of_month, out)
    out = _DECADE_RE.sub(_sub_decade, out)
    out = _CURRENCY_PREFIX_RE.sub(_sub_currency_prefix, out)
    out = _UNIT_RE.sub(_sub_unit, out)
    out = _ACRONYM_RE.sub(_sub_acronym, out)
    out = _LETTER_RE.sub(_sub_letter, out)
    out = _ORDINAL_RE.sub(_sub_ordinal, out)
    out = _GROUPED_RE.sub(_sub_grouped, out)
    out = _IDENTIFIER_RE.sub(_sub_identifier, out)
    out = _DECIMAL_RE.sub(_sub_decimal, out)
    out = _INT_RE.sub(_sub_int, out)
    return out
