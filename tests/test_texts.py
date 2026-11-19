"""Texte: App und README tragen denselben Wortlaut der Annahmen (DEMO-PLAYBOOK §5), keine Platzhalter, Formelblöcke einzeln, Beispieltabellen stimmen mit der Rechnung."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fsp_constants as C
import fsp_game as G
import fsp_net as N
import fsp_scenario as S
from fsp_texts import ASSUMPTIONS_MARKDOWN, BEWEIS_H4, LITERATUR, TERMS_TABLE

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_readme_contains_the_assumptions_text_of_the_app_verbatim():
    assert ASSUMPTIONS_MARKDOWN.rstrip("\n") in (ROOT / "README.md").read_text(encoding="utf-8")


def test_app_shows_the_assumptions_text_in_its_expander():
    expander = next(e for e in _app().expander if e.label == "⚠️ Annahmen, Grenzen und bekannte Probleme")
    assert ASSUMPTIONS_MARKDOWN.strip() in " ".join(m.value for m in expander.markdown)


def test_app_has_the_required_sections_in_order():
    at = _app()
    labels = [e.label for e in at.expander]
    assert labels.index("⚠️ Annahmen, Grenzen und bekannte Probleme") < labels.index("📐 Mathematische Formulierung")
    assert labels[0] == "So funktioniert Flussspiel"
    assert any(m.value.startswith("## ❓ Kernfrage") for m in at.markdown)


def test_no_placeholders_or_forbidden_words_remain():
    for path in list(ROOT.glob("*.py")) + [ROOT / "README.md"] + list((ROOT / "tests").glob("*.py")) + [ROOT / "tools" / "make_messreihen.py"]:
        text = path.read_text(encoding="utf-8")
        if path.name == "README.md":
            text = re.sub(r"<!-- TODO nach dem Deploy.*?-->", "", text, flags=re.S)      # der Live-Link-Platzhalter des Gerüsts bleibt bis zum Deploy
        if path.name == "test_texts.py":
            continue
        assert "TODO" not in text, path.name
        assert "Saatwert" not in text and "Schwanz" not in text, path.name


def test_texts_avoid_the_figurative_word_traegt():
    """Nutzerwunsch 2026-11-09: reicht, gilt, zeigt, funktioniert statt „trägt“; auch in Kommentaren und Docstrings dieser Demo."""
    for path in list(ROOT.glob("*.py")) + [ROOT / "README.md", ROOT / "tools" / "make_messreihen.py"]:
        assert not re.search(r"\btr(ä|ae)g(t|en)\b", path.read_text(encoding="utf-8")), path.name


def test_math_expander_has_one_display_equation_per_block():
    """Je Display-Gleichung ein eigener $$-Block (sonst KaTeX-Fehler): kein Block enthält ein zweites $$ oder eine Leerzeile; kein ungeschütztes Prozentzeichen."""
    text = (ROOT / "app.py").read_text(encoding="utf-8") + BEWEIS_H4
    blocks = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
    assert len(blocks) >= 9
    for b in blocks:
        assert "\n\n" not in b and "$" not in b and "%" not in b


def test_math_expander_starts_with_a_reading_aid_and_a_symbol_table():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "**Lesehilfe.**" in text and "| Symbol | Bedeutung |" in text
    assert text.index("**Lesehilfe.**") < text.index("$$v(S) = \\max")


def test_literature_lines_have_authors_year_journal_and_pages():
    assert len(LITERATUR) == 3
    for line in LITERATUR:
        assert re.search(r"\(19\d\d\)", line) and re.search(r"\d{3}[–-]\d{3}", line)


# --------------------------------------------------------------------------- Handzahlen der Tabelle „Drei Kanten“ gegen die Rechnung
def test_terms_table_numbers_equal_the_computation():
    net = S.example_net("Drei Kanten (4 | 3, 5)")
    v = N.coalition_values(net)
    assert N.max_flow(net) == (4, [4, 3, 1])                                    # „4 Einheiten, davon 3 über B und 1 über C“ (ein größter Fluss unter mehreren)
    assert v[0b101] == 4 and v[0b110] == 0                                      # v({A, C}) = 4, v({B, C}) = 0
    assert sum(c for (a, b, c), i in zip(net.edges, range(3)) if i in (1, 2)) == 8     # Schnitt {B, C}
    assert [str(f) for f in G.shapley_fractions(v)] == ["5/2", "1/2", "1"]
    assert G.veto_edges(v) == [0] and G.surplus_edges(v) == [1]
    for needle in ("v({A, C}) = 4", "v({B, C}) = 0", "A: 5/2, B: 1/2, C: 1", "{B, C}: Kapazität 8"):
        assert needle in TERMS_TABLE
    app_text = " ".join(Path(APP).read_text(encoding="utf-8").split())
    assert "B bekommt daher 3/6 = 1/2" in app_text and "fünf Reihenfolgen" in app_text


def test_beispielspur_of_the_shapley_value_counts_one_order_in_six():
    """B fügt nur etwas hinzu, wenn A schon da ist und C noch nicht: genau die Reihenfolge (A, B, C) von 6 liefert Beitrag 3 (v(AB) - v(A) = 3)."""
    import itertools

    net = S.example_net("Drei Kanten (4 | 3, 5)")
    v = N.coalition_values(net)
    gains = []
    for perm in itertools.permutations(range(3)):
        mask = 0
        for i in perm:
            if i == 1:
                gains.append(float(v[mask | 2] - v[mask]))
            mask |= 1 << i
    assert sorted(gains) == [0.0] * 5 + [3.0] and sum(gains) / 6 == pytest.approx(0.5)


def test_assumption_text_numbers_are_settings():
    assert f"bis k = {C.K_MAX}" in ASSUMPTIONS_MARKDOWN and f"bis k = {C.K_BIG_MAX}" in ASSUMPTIONS_MARKDOWN
