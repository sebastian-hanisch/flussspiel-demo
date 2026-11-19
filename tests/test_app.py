"""AppTest-Rauchtests: Standard, jeder Knopf (auch die ▶️-Knöpfe), jeder Regler an Minimum und Maximum, jedes Preset, jede Netzform, Permalink.

Bei einem festen Beispielnetz sind Kantenzahl, Kapazitätsmodus und Zufalls-Seed gesperrt; die Randprüfung dieser Regler läuft deshalb an einem Zufallsnetz."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import fsp_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in query.items():
        at.query_params[key] = value
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _random_net_app(form="Zufalls-DAG") -> AppTest:
    at = _app()
    at.sidebar.selectbox(key="form_select").set_value(form).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception_and_has_the_required_sections():
    at = _app()
    subheads = [s.value for s in at.subheader]
    assert sum(1 for s in subheads if s.startswith("🔬")) == 5 and any(s.startswith("🚧") for s in subheads)
    labels = {e.label for e in at.expander}
    assert {"📐 Mathematische Formulierung", "⚠️ Annahmen, Grenzen und bekannte Probleme", "So funktioniert Flussspiel"} <= labels
    assert len(at.tabs) == 5


def test_every_button_click_does_not_raise():
    first = _app().button
    labels = [b.label for b in first]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if first[index].proto.disabled:
            continue                                  # gesperrt beim festen Beispielnetz (unten an einem Zufallsnetz geprüft)
        at = _app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_dice_button_gives_a_new_random_net():
    at = _random_net_app()
    next(b for b in at.sidebar.button if b.label.startswith("🎲")).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert 0 <= at.sidebar.number_input(key="seed_input").value <= C.SEED_MAX          # der gewürfelte Zufalls-Seed liegt in den Reglergrenzen


def test_the_live_buttons_show_a_result():
    at = _app()
    next(b for b in at.button if b.label.startswith("▶️ Kurzlauf")).click().run(timeout=TIMEOUT)
    assert not at.exception and any("Rechenzeit" in c.value for c in at.caption)
    at = _app()
    next(b for b in at.button if b.label.startswith("▶️ Größte Zahlung")).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Zeilen statt" in s.value for s in at.success), [s.value for s in at.success]


def test_pdf_button_then_download_button():
    at = _app()
    assert not at.get("download_button")                  # vor dem Klick gibt es nichts herunterzuladen (kein PDF bei jedem Rerun)
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert len(at.get("download_button")) == 1


@pytest.mark.parametrize("key", ["k_slider"])
@pytest.mark.parametrize("edge", ["min", "max"])
def test_k_slider_at_its_min_and_max_on_every_random_form(key, edge):
    for form in C.RANDOM_FORMS:
        at = _random_net_app(form)
        slider = at.sidebar.slider(key=key)
        slider.set_value(int(slider.min if edge == "min" else slider.max)).run(timeout=TIMEOUT)
        assert not at.exception, (form, edge, [str(e) for e in at.exception])


def test_seed_and_capmode_at_their_edges_and_all_minima_and_all_maxima_together():
    at = _random_net_app()
    at.sidebar.number_input(key="seed_input").set_value(0).run(timeout=TIMEOUT)
    assert not at.exception
    at.sidebar.number_input(key="seed_input").set_value(C.SEED_MAX).run(timeout=TIMEOUT)
    assert not at.exception
    for cap in C.CAPMODES:
        at.sidebar.selectbox(key="capmode_select").set_value(cap).run(timeout=TIMEOUT)
        assert not at.exception, cap
    for edge in ("min", "max"):
        at = _random_net_app()
        at.sidebar.slider(key="k_slider").set_value(C.K_MIN if edge == "min" else C.K_MAX)
        at.sidebar.number_input(key="seed_input").set_value(0 if edge == "min" else C.SEED_MAX)
        at.sidebar.selectbox(key="capmode_select").set_value("alle 1" if edge == "min" else "1 bis 20")
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_the_main_view_shows_its_net(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    assert any(m.label == "Betriebe (Kanten)" for m in at.metric)


@pytest.mark.parametrize("form", C.FORMS)
def test_every_net_form_runs(form):
    at = _app()
    at.sidebar.selectbox(key="form_select").set_value(form).run(timeout=TIMEOUT)
    assert not at.exception, (form, [str(e) for e in at.exception])


def test_tabs_work_for_a_big_net_and_for_a_net_beyond_the_live_limits():
    at = _random_net_app()
    at.sidebar.slider(key="k_slider").set_value(C.K_MAX).run(timeout=TIMEOUT)
    assert not at.exception
    assert any(b.label.startswith("▶️ Spielraum je Kante") for b in at.button)
    next(b for b in at.button if b.label.startswith("▶️ Spielraum je Kante")).click().run(timeout=TIMEOUT)
    assert not at.exception


def test_ausbau_tab_edge_choice_and_own_capacity():
    at = _app()
    at.selectbox(key="ausbau_kante_3").set_value("B").run(timeout=TIMEOUT)
    assert not at.exception
    at.number_input(key="ausbau_cap_3_1").set_value(0).run(timeout=TIMEOUT)
    assert not at.exception
    at.number_input(key="ausbau_cap_3_1").set_value(C.CAP_MAX).run(timeout=TIMEOUT)
    assert not at.exception


def test_coalition_choice_and_unit_toggle():
    at = _app()
    at.multiselect(key="coalition_3").set_value(["A", "C"]).run(timeout=TIMEOUT)
    assert not at.exception
    at.multiselect(key="coalition_3").set_value([]).run(timeout=TIMEOUT)
    assert not at.exception
    at.toggle(key="unit_toggle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception


def test_permalink_roundtrip_and_invalid_values_are_ignored():
    at = _app(form="Brücke (3, 2, 1, 2, 3)", k="7", cap="alle 1", seed="12")
    assert at.sidebar.selectbox(key="form_select").value == "Brücke (3, 2, 1, 2, 3)"
    assert at.sidebar.slider(key="k_slider").value == 7 and at.sidebar.number_input(key="seed_input").value == 12
    bad = _app(form="gibt es nicht", k="500", cap="x", seed="-4")
    assert bad.sidebar.selectbox(key="form_select").value == C.DEFAULT_FORM
    assert bad.sidebar.slider(key="k_slider").value == C.K_MAX
    assert bad.sidebar.selectbox(key="capmode_select").value == C.DEFAULT_CAPMODE
    assert bad.sidebar.number_input(key="seed_input").value == 0


def test_a_net_change_that_changes_the_edge_count_keeps_the_ausbau_tab_alive():
    at = _random_net_app()
    at.sidebar.slider(key="k_slider").set_value(4).run(timeout=TIMEOUT)
    at.sidebar.slider(key="k_slider").set_value(9).run(timeout=TIMEOUT)
    assert not at.exception
