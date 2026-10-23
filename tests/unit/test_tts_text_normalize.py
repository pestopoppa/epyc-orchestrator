"""Golden tests for the Italian TTS text normalizer and its cascade wiring.

The first block is built from the tokens the operator flagged in the CS-8.L Italian listen
check (2026-10-22): the year 1963, the number 150, the 802.11n / 2.4Ghz / 5.0Ghz phrase and
pH (twice). All strings are short test strings written for this file.
"""

import pytest

from src.features import get_features

from src.voice.cascade import CascadeBackend
from src.voice.contracts import VoiceEvent, VoiceTurn
from src.voice.tts_text_normalize import (
    ACRONYMS_IT,
    canonical_language,
    cardinal_it,
    normalize_for_tts,
    normalize_italian,
    ordinal_it,
)


# --------------------------------------------------------------------------- CS-8.L goldens

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("la diga nel 1963, poi", "la diga nel millenovecento sessantatré, poi"),
        ("tra le 150 e le 200 copie", "tra le centocinquanta e le duecento copie"),
        (
            "lo standard 802.11n nella banda dei 2.4Ghz e dei 5.0Ghz.",
            "lo standard ottocentodue punto undici enne nella banda dei due virgola quattro "
            "gigahertz e dei cinque virgola zero gigahertz.",
        ),
        ("il cosiddetto pH della sostanza", "il cosiddetto pi acca della sostanza"),
        ("un indicatore di pH.", "un indicatore di pi acca."),
        ("(la lettera H di pH)", "(la lettera acca di pi acca)"),
    ],
)
def test_cs8l_flagged_tokens(text, expected):
    assert normalize_for_tts(text, "it") == expected


# --------------------------------------------------------------------------- cardinals

@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"), (1, "uno"), (3, "tre"), (11, "undici"), (17, "diciassette"),
        (21, "ventuno"), (23, "ventitré"), (28, "ventotto"), (33, "trentatré"),
        (80, "ottanta"), (99, "novantanove"), (100, "cento"), (101, "centouno"),
        (103, "centotré"), (108, "centootto"), (150, "centocinquanta"), (180, "centottanta"),
        (188, "centottantotto"), (283, "duecentottantatré"), (999, "novecentonovantanove"),
        (1000, "mille"), (1001, "milleuno"), (1003, "milletré"), (1940, "millenovecentoquaranta"),
        (1963, "millenovecentosessantatré"), (2002, "duemiladue"), (2006, "duemilasei"),
        (3000, "tremila"), (21000, "ventunmila"), (23000, "ventitremila"),
        (40000, "quarantamila"), (100000, "centomila"), (1_000_000, "un milione"),
        (1_200_000, "un milione duecentomila"), (2_000_000, "due milioni"),
        (21_000_000, "ventun milioni"), (1_000_000_003, "un miliardo tre"),
        (2_000_000_000, "due miliardi"), (-5, "meno cinque"),
    ],
)
def test_cardinals(n, words):
    assert cardinal_it(n) == words


def test_cardinal_rejects_out_of_range_and_non_int():
    with pytest.raises(ValueError):
        cardinal_it(10**12)
    with pytest.raises(TypeError):
        cardinal_it(True)
    with pytest.raises(TypeError):
        cardinal_it(1.5)


@pytest.mark.parametrize(
    ("n", "feminine", "words"),
    [
        (1, False, "primo"), (3, True, "terza"), (10, False, "decimo"),
        (11, False, "undicesimo"), (20, False, "ventesimo"), (21, False, "ventunesimo"),
        (23, False, "ventitreesimo"), (26, False, "ventiseiesimo"), (100, False, "centesimo"),
        (1000, False, "millesimo"), (2000, False, "duemillesimo"), (23, True, "ventitreesima"),
    ],
)
def test_ordinals(n, feminine, words):
    assert ordinal_it(n, feminine=feminine) == words


# --------------------------------------------------------------------------- text rules

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # thousands grouping, years, decades, dates, times
        ("meno di 40.000 abitanti", "meno di quarantamila abitanti"),
        ("1.000.000 abitanti", "un milione di abitanti"),
        ("il 15 agosto 1940", "il quindici agosto millenovecento quaranta"),
        ("negli anni '20, la", "negli anni venti, la"),
        ("negli anni ’20", "negli anni venti"),
        ("Angel (2006) illustra", "Angel (duemila sei) illustra"),
        ("il 1° maggio e il 1 giugno", "il primo maggio e il primo giugno"),
        ("il 15/08/1940", "il quindici agosto millenovecento quaranta"),
        ("alle 10.30 e alle 21:05", "alle dieci e trenta e alle ventuno e cinque"),
        ("alle 13:00", "alle tredici"),
        # decimals
        ("pesa 2,5 kg", "pesa due virgola cinque chilogrammi"),
        ("pi greco vale 3,14", "pi greco vale tre virgola quattordici"),
        ("solo 0,05 g", "solo zero virgola zero cinque grammi"),
        ("vale 3,14159", "vale tre virgola uno quattro uno cinque nove"),
        # units with agreement
        ("1 km", "un chilometro"),
        ("1 h di attesa", "un'ora di attesa"),
        ("1 GHz", "un gigahertz"),
        ("21 h", "ventuno ore"),
        ("50 Hz e 44,1 kHz", "cinquanta hertz e quarantaquattro virgola uno chilohertz"),
        ("100 MHz, 3 mhz", "cento megahertz, tre megahertz"),
        ("130 km/h", "centotrenta chilometri orari"),
        ("12 m² e 3 km²", "dodici metri quadrati e tre chilometri quadrati"),
        ("-5 °C e 20°C", "meno cinque gradi Celsius e venti gradi Celsius"),
        ("il 15% e l'1%", "il quindici per cento e l'uno per cento"),
        ("2.000.000 km", "due milioni di chilometri"),
        ("€5 e 3,50 € e £2", "cinque euro e tre virgola cinquanta euro e due sterline"),
        ("1.500 km", "millecinquecento chilometri"),
        # ordinal indicators
        ("il 3º posto, la 2ª volta", "il terzo posto, la seconda volta"),
        # dotted identifiers without a unit
        ("versione 3.10.2", "versione tre punto dieci punto due"),
        ("Wi-Fi 802.11ac", "Wi-Fi ottocentodue punto undici a ci"),
        # acronyms and letter names
        ("il DNA e l'RNA", "il di enne a e l'erre enne a"),
        ("le lettere X", "le lettere ics"),
        ("la banda dei GHz", "la banda dei gigahertz"),
    ],
)
def test_italian_rules(text, expected):
    assert normalize_italian(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Con il suo fucile M16 sparò",       # alphanumeric designations
        "in 4K e 5G, H2O, mp3",
        "FT-IR, l'IA, FKP, GBP, NATO",        # acronyms outside the table
        "il 1 caso su mille",                 # bare 1 before a noun: gender unknown
        "codice 007, prefisso 0039",          # zero-padded codes
        "telefono 1234567",                   # long bare digit run
        "punti 1,2,3",                        # comma list, not a decimal
        "tutti e tre l'hanno visto",          # no digits at all
        "",
        "Grazie al clima mite del Medio Oriente.",
    ],
)
def test_pass_through(text):
    assert normalize_italian(text) == text


def test_unit_symbol_needs_a_word_boundary():
    # "3 l'hanno" must not read the elided article as litri
    assert normalize_italian("tutti e 3 l'hanno visto") == "tutti e tre l'hanno visto"
    assert normalize_italian("tra 5 minuti") == "tra cinque minuti"


def test_compound_hyphen_keeps_number_reading():
    assert normalize_italian("COVID-19") == "COVID-diciannove"
    assert normalize_italian("1940-1945") == "millenovecento quaranta-millenovecento quarantacinque"


@pytest.mark.parametrize(
    "text",
    [
        "la diga nel 1963, poi 150 copie, 2.4Ghz, pH",
        "il 1° maggio alle 10.30, 1 h, 40.000 km, anni '20, 3º posto, 802.11n, 15/08/1940",
    ],
)
def test_idempotent(text):
    once = normalize_italian(text)
    assert normalize_italian(once) == once
    assert not any(c.isdigit() for c in once)


def test_acronym_table_is_extendable_and_letter_spelled():
    assert ACRONYMS_IT["pH"] == "pi acca"
    for spoken in ACRONYMS_IT.values():
        assert spoken == spoken.lower() and not any(c.isdigit() for c in spoken)


# --------------------------------------------------------------------------- language gate

@pytest.mark.parametrize("language", ["it", "IT", "it-IT", "it_it", "Italian", "italiano", " it "])
def test_italian_aliases_normalize(language):
    assert canonical_language(language) == "it"
    assert normalize_for_tts("nel 1963", language) == "nel millenovecento sessantatré"


@pytest.mark.parametrize("language", [None, "en", "English", "fr", "auto", "", 3])
def test_other_languages_are_byte_identical(language):
    text = "In 1963 the 2.4GHz pH was 150 km."
    assert normalize_for_tts(text, language) is text


def test_non_string_text_is_rejected():
    with pytest.raises(TypeError):
        normalize_for_tts(b"1963", "it")


# --------------------------------------------------------------------------- cascade wiring

class _Transcriber:
    def transcribe(self, audio, *, language=None):
        return "transcript"

    def health(self):
        return True


class _VoiceTurn:
    def __init__(self, text):
        self.text = text

    def stream_turn(self, transcript, *, session_id, turn_id, response_mode, must_preserve):
        yield VoiceEvent("text_delta", self.text)
        yield VoiceEvent("end")

    def inject(self, text):
        pass

    def cancel_generation(self):
        pass

    def cancel_orchestrator(self):
        pass

    def health(self):
        return True


class _Synth:
    def __init__(self):
        self.texts = []

    def stream_pcm(self, text, *, language=None):
        self.texts.append(text)
        yield b"pcm"

    def cancel(self):
        pass

    def health(self):
        return True


_ANSWER = "Il pH era 7 nel 1963."


def _run(backend, *, language=None, must_preserve=()):
    turn = VoiceTurn("t1", "s1", b"", language=language, transcript="q",
                     must_preserve=must_preserve)
    return list(backend.respond(turn))


def test_cascade_explicit_off_passes_the_answer_object_unchanged():
    synth = _Synth()
    events = _run(CascadeBackend(_Transcriber(), _VoiceTurn(_ANSWER), synth,
                                 tts_text_normalization=False, tts_respell=False), language="it")
    assert events[-1] == VoiceEvent("end")
    assert synth.texts == [_ANSWER]
    assert synth.texts[0].encode() == _ANSWER.encode()


def test_cascade_on_normalizes_only_the_synthesizer_text():
    synth = _Synth()
    backend = CascadeBackend(_Transcriber(), _VoiceTurn(_ANSWER), synth,
                             tts_text_normalization=True)
    events = _run(backend, language="it", must_preserve=("1963",))
    # the displayed text stays the route's own; must_preserve is checked on it
    assert [e.payload for e in events if e.kind == "text_delta"] == [_ANSWER]
    assert synth.texts == ["Il pi acca era sette nel millenovecento sessantatré."]


def test_cascade_on_uses_server_language_when_turn_has_none():
    synth = _Synth()
    backend = CascadeBackend(_Transcriber(), _VoiceTurn(_ANSWER), synth,
                             tts_text_normalization=True, tts_language="it")
    _run(backend)
    assert synth.texts == ["Il pi acca era sette nel millenovecento sessantatré."]


def test_cascade_on_leaves_other_languages_untouched():
    synth = _Synth()
    backend = CascadeBackend(_Transcriber(), _VoiceTurn(_ANSWER), synth,
                             tts_text_normalization=True, tts_language="it")
    _run(backend, language="en")
    assert synth.texts == [_ANSWER]


def test_cascade_flag_validation():
    with pytest.raises(TypeError):
        CascadeBackend(_Transcriber(), _VoiceTurn(""), _Synth(), tts_text_normalization="yes")
    with pytest.raises(ValueError):
        CascadeBackend(_Transcriber(), _VoiceTurn(""), _Synth(), tts_language=" ")


# --------------------------------------------------------------------------- CS-8.L4 live default

_LIVE = "Lo standard 802.11n usa 2.4Ghz. In maniera analoga, gli atomi di Giza e delle Marianne."


def test_cascade_live_default_applies_normalizer_and_respelling_for_italian():
    synth = _Synth()
    _run(CascadeBackend(_Transcriber(), _VoiceTurn(_LIVE), synth), language="it")
    out = synth.texts[0]
    assert "ottocentodue punto undici enne" in out and "due virgola quattro gigahertz" in out
    assert "anàloga" in out and "àtomi" in out and "Ghiza" in out
    assert "Mariànne" in out


def test_cascade_live_default_leaves_other_languages_byte_identical():
    synth = _Synth()
    _run(CascadeBackend(_Transcriber(), _VoiceTurn(_LIVE), synth), language="en")
    assert synth.texts == [_LIVE]


@pytest.mark.parametrize("var, kept, gone", [
    ("ORCHESTRATOR_TTS_IT_NORMALIZE", "anàloga", "due virgola quattro"),
    ("ORCHESTRATOR_TTS_IT_RESPELL", "due virgola quattro", "anàloga"),
])
def test_cascade_kill_switches_turn_each_step_off(monkeypatch, var, kept, gone):
    monkeypatch.setenv(var, "0")
    monkeypatch.setattr("src.features.features", lambda: get_features())
    synth = _Synth()
    _run(CascadeBackend(_Transcriber(), _VoiceTurn(_LIVE), synth), language="it")
    assert kept in synth.texts[0] and gone not in synth.texts[0]


def test_respell_it_57_and_it_26_style_inputs():
    from src.voice.tts_text_normalize import respell_italian
    assert normalize_for_tts("In maniera analoga, agli uomini", "it") == "In maniera anàloga, agli uomini"
    assert normalize_for_tts("nella banda dei 2.4Ghz", "it") == "nella banda dei due virgola quattro gigahertz"
    assert respell_italian("Giza") == "Ghiza" and respell_italian("GIZA") == "GHIZA"
    assert respell_italian("anàloga àtomi") == "anàloga àtomi"  # idempotent
    assert respell_italian("analogamente") == "analogamente"  # whole words only
